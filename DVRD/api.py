import torch
from torchvision import transforms
import torch.nn as nn
import math
import torch.nn.functional as F

"""
DVRD（Dense Variation Region Detector）

- 作用：对潜变量（VAE 编码空间，通道数=4）中的变化/篡改区域进行密集检测与细化。
- 提供两种方式：
  1) TrainableDVRD：使用 UNet 结构、可加载 checkpoint 进行端到端细化。
  2) TrainfreeDVRD：无需训练，使用多尺度均值聚合 + 阈值化得到二值篡改图。

输入输出：
- 输入：形状 [B, 4, H, W] 的潜变量图或变化图（半精度/单精度）。
- 输出：形状 [B, 4, H, W] 的细化结果（trainable），或 [B, 1, H, W] 的二值结果（trainfree）。
"""
def from_pretrained(checkpoint_path, train_size=512, torch_dtype=torch.float16, strict=True, device='cpu'):
    """
    加载可训练 DVRD 模型：
    - 将 VAE 编码图下采样至 train_size/8（与潜空间尺度一致），再送入 UNet。
    - checkpoint_path：包含 'model_state_dict' 的权重文件。
    - 返回：已加载且 eval 的 TrainableDVRD 实例。
    """
    train_size = train_size // 8  # downsample 8x for VAE encoded images
    model = TrainableDVRD(train_size=train_size, torch_dtype=torch_dtype, device=device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    print("DVRD state dict loaded:", model.unet.load_state_dict(checkpoint['model_state_dict'], strict=strict))
    model.eval()
    return model

class TrainableDVRD(nn.Module):
    """可训练的密集变化区域检测器（UNet 细化）"""
    def __init__(self,
                 train_size=512,
                 torch_dtype=torch.float16,
                 device='cpu'):
        super(TrainableDVRD, self).__init__()
        if train_size is not None:
            self.transform = transforms.Compose([transforms.Resize(train_size)])
        else:
            self.transform = None
        self.unet = UNet(in_channels=4, out_channels=4).to(device)
        if torch_dtype == torch.float16:
            self.unet = self.unet.half()
            
    def forward(self, x):
        """
        前向：可选缩放至训练尺寸 -> UNet -> 还原回原尺寸。
        输入/输出形状均为 [B, 4, H, W]。
        """
        if self.transform:
            h, w = x.size()[-2:]
            x = self.transform(x)
            
        x = self.unet(x)
        
        if self.transform:
            x = transforms.Resize((h, w))(x)
            
        return x

class UNet(nn.Module):
    """标准 UNet：编码-解码并带跳连，用于潜空间细化。"""
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        
        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        
        def up_conv(in_channels, out_channels):
            return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        
        self.encoder1 = conv_block(in_channels, 64)
        self.encoder2 = conv_block(64, 128)
        self.encoder3 = conv_block(128, 256)
        self.encoder4 = conv_block(256, 512)
        self.encoder5 = conv_block(512, 1024)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.upconv4 = up_conv(1024, 512)
        self.decoder4 = conv_block(1024, 512)
        self.upconv3 = up_conv(512, 256)
        self.decoder3 = conv_block(512, 256)
        self.upconv2 = up_conv(256, 128)
        self.decoder2 = conv_block(256, 128)
        self.upconv1 = up_conv(128, 64)
        self.decoder1 = conv_block(128, 64)
        
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
    
    def forward(self, x):
        """
        UNet 编解码正向：保留多尺度信息并融合至输出。
        """
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool(enc1))
        enc3 = self.encoder3(self.pool(enc2))
        enc4 = self.encoder4(self.pool(enc3))
        enc5 = self.encoder5(self.pool(enc4))
        
        dec4 = self.upconv4(enc5)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        
        return self.final_conv(dec1)

class TrainfreeDVRD(torch.nn.Module):
    """
    免训练 DVRD：
    - 多尺度均值聚合（非重叠/可选重叠），再进行阈值化得到二值变化图；
    - 自适应 kernel 大小：随图像尺寸放大，以覆盖更大感受野；
    - overlapping=False 时使用 stride=k 的金字塔聚合以提速。
    """
    def __init__(self, 
                 max_kernel_size = 8,
                 adaptive_max_kernel_size = True,
                 overlapping = False,
                 ):
        super(TrainfreeDVRD, self).__init__()
        self.max_kernel_size = max_kernel_size
        self.adaptive_max_kernel_size = adaptive_max_kernel_size
        self.overlapping = overlapping
        
    def forward(self, change_img: torch.Tensor, confidence = 0.5):
        """
        输入：change_img（B,1,H,W 或 1, H, W），值域建议为 [0,1]。
        流程：多尺度平均 -> 聚合 -> 与 confidence 比较 -> 二值图。
        返回：与输入同分辨率的二值掩码（int）。
        """
        """
        input: 
            changed_img: [B, 1, H, W], img with changes
            confidence: a value to control the confidence of change detection
        return: 
            [B, 1, H, W]
        """
        if change_img.dim() == 3:
            change_img = change_img.unsqueeze(0)
            dim = 3
        change_img = change_img.float()
        addition_times = 1
        print('change_img(1):', change_img)

        # adaptively adjust the max_kernel_size
        if self.adaptive_max_kernel_size:
            # The max_kernel_size should be proportional to the image size.
            self.max_kernel_size = int(8 * (change_img.size(2) * change_img.size(3)) / (64 * 64))

        if self.overlapping:
            for k in range(3, self.max_kernel_size + 1, 2):
                change_img += F.avg_pool2d(change_img, kernel_size=k, stride=1, padding=k // 2)
                addition_times += 1
        else:
            for k in range(1, int(math.log2(self.max_kernel_size)) + 1):
                k = 2 ** k
                print('k:', k)
                temp_change = F.avg_pool2d(change_img, kernel_size=k, stride=k, padding=0)
                temp_change = torch.repeat_interleave(temp_change, repeats=k, dim=2)
                temp_change = torch.repeat_interleave(temp_change, repeats=k, dim=3)
                change_img += temp_change
                addition_times += 1
                
        change_img = change_img / addition_times
        
        # Apply the change confidence threshold
        change_img = (change_img >= confidence).int()
        print(f'change_img({addition_times}):', change_img)

        if dim == 3:
            change_img = change_img[0]
        return change_img