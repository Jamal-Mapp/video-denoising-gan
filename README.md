# 🎞️ Video Denoising with GANs

This project implements a generative adversarial network (GAN) for video denoising using the Vimeo-90K dataset. The goal is to clean noisy video frames and preserve temporal consistency.

## 🚀 Demo
![denoise](assets/demo.gif)

## 📌 Motivation
Low-light or compressed videos often suffer from temporal noise. This project applies GANs to enhance visual clarity while maintaining frame coherence.

## 🧪 Results
| Metric | Baseline | Our Model |
|--------|----------|-----------|
| PSNR   | 26.5     | 28.7      |
| SSIM   | 0.71     | 0.82      |

## 🔧 Tech Stack
- Python, PyTorch
- OpenCV, FFmpeg
- Vimeo-90K dataset

## 🧰 Folder Structure
- `src/` - training scripts, GAN model, data pipeline
- `notebooks/` - exploratory experiments
- `results/` - sample outputs & evaluation
- `assets/` - demo visuals (GIFs, PNGs)

## 🚀 Quickstart
```bash
git clone https://github.com/your-username/video-denoising-gan.git
cd video-denoising-gan
pip install -r requirements.txt
python src/train.py --config configs/config.yaml
