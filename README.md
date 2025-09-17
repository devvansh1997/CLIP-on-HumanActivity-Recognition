# INSTALLATION
Follow the colab notebook, but for convenience purposes I do have a requirements file. 

# Kaggle
For downloading the dataset, the colab nb have the commands present, but in order to download the dataset you will need to upload your own kaggle.json file when prompted as you need API access from kaggle using your own account to download the dataset locally. 

# Trained Model
Please download the models directly from my public [gdrive folder](https://drive.google.com/drive/folders/1u9cA9Qhw4u7N0cnG9TLh7oZvLc0_ArFD?usp=drive_link)

# Running
```
%cd /content/CLIP-on-HumanActivity-Recognition
!python -m har_clip.main --mode train+eval --config har_clip/config.yaml
```
and
```
%cd /content/CLIP-on-HumanActivity-Recognition
!python -m har_siglip.main --mode train+eval --config har_siglip/config.yaml
```

# Visualizations
```
!python compare_viz.py \
  --data_root /content/data/Structured \
  --siglip_ckpt runs/siglip/latest.pt \
  --clip_ckpt runs/clip/latest.pt
```