# TAG-WM: Tamper-Aware Generative Image Watermarking via Diffusion Inversion Sensitivity

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2506.23484-b31b1b.svg)](https://arxiv.org/pdf/2506.23484)
[![GitHub Stars](https://img.shields.io/github/stars/Suchenl/TAG-WM?style=social)](https://github.com/Suchenl/TAG-WM)

Official implementation of **TAG-WM**, a tamper-aware watermarking framework for diffusion-generated images.  
*Code release coming soon.*

---

## 🚀 Updates
- **[2025.08]** Code released! 🎉 We also updated paper's Abstract and Figure 1 for clarity and added GitHub link, Appendix and Acknowledgments.
- **[2025.07]** Code packaging in progress (Expected release: August 2025)
- **[2025.06]** Paper accepted to **ICCV 2025** and released on [arXiv](https://arxiv.org/pdf/2506.23484) 🎉

## Usage
### Install TAG-WM and environment dependencies
```bash
git clone https://github.com/Suchenl/TAG-WM.git
cd TAG-WM
conda create --name TAG-WM python=3.10
pip install -r requirements.txt
```
### Install image generation diffusion models
1. Use modelscope to download the models (example: SD 2.1 base model)
```bash
pip install modelscope
modelscope download --model AI-ModelScope/stable-diffusion-2-1-base
```

2. Use Huggingface to download the models (example: SD 2.1 base model)
```bash
# Make sure hf CLI is installed: pip install -U "huggingface_hub[cli]"
hf download stabilityai/stable-diffusion-2-1-base
```

### Test model's ability
```bash
python -m applied_to_sd2.test --model_path "SD_model_path" --start_sample_idx 0 --num 1000 --random_crop_ratio 0.3 --return_tamper_loc True --calc_wm_use_tamper_loc True

python -m applied_to_sd2.test --model_path "SD_model_path" --start_sample_idx 0 --num 1000 --logo_putting_num 2 --logo_ratio 0.5 --return_tamper_loc True --calc_wm_use_tamper_loc True
```

### Arguments: for testing image degradations
    --jpeg_ratio
    --gaussian_blur_r
    --median_blur_k
    --resize_ratio
    --gaussian_std
    --sp_prob
    --brightness_factor

## 🔍 Key Contributions
1. **Dual-Mark Joint Sampling**  
   - Simultaneously embeds copyright + localization watermarks in diffusion latent space  
   - Preserves standard normal distribution of latents (lossless visual quality)  
   - Enables tampering localization without degrading generation  

2. **Dense Variation Region Detector**  
   - Leverages diffusion inversion sensitivity to modifications  
   - Detects tampering via statistical deviations between original/reconstructed watermarks  
   - Achieves strong generalization across manipulation types  

3. **Tamper-Aware Message Decoding**  
   - Localization-guided decoding improves robustness against edits  
   - Maintains copyright extraction accuracy even under modifications  

## 📜 License
This project is licensed under the **MIT License** - a permissive free software license with minimal restrictions.  

### You are free to:
- ✅ Use the code **commercially or privately** (even in proprietary software)  
- ✅ Modify, adapt, or build upon the work  
- ✅ Distribute your modified versions  
- ✅ Use for research, education, or production systems  

### Requirements:
- 📝 Include original copyright notice  
- 📜 Retain license file  

[View full license text](LICENSE)

## 📄 Citation
If you find this work useful, please consider citing our paper and giving the repo a ⭐:
```bibtex
@article{chen2025tag,
  title={TAG-WM: Tamper-Aware Generative Image Watermarking via Diffusion Inversion Sensitivity}, 
  author={Chen, Yuzhuo and Ma, Zehua and Fang, Han and Zhang, Weiming and Yu, Nenghai},
  journal={arXiv preprint arXiv:2506.23484},
  year={2025}
}
