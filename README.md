# Prerequisites
- Linux
- Python 3.8
- NVIDIA GPU + CUDA CuDNN
- anaconda virtual environment
  ```
  conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia  
  
  conda install tqdm  
  
  conda install matplotlib==3.3.4  
  
  conda install seaborn  
  
  conda install scikit-learn  
  ```
# Dataset Discription
-covid_ct：
```
├─train├─X
│      │  001non-covid19.png
│      │  002non-covid19.png
│      │  003non-covid19.png
│      ├─Y
│      │  001covid19.png
│      │  002covid19.png
│      │  003covid19.png
├─valid├─X
│      │  001non-covid19.png
│      │  002non-covid19.png
│      │  003non-covid19.png
│      ├─Y
│      │  001covid19.png
│      │  002covid19.png
│      │  003covid19.png
├─test ├─X
│      │  001non-covid19.png
│      │  002non-covid19.png
│      │  003non-covid19.png
│      ├─Y
│      │  001covid19.png
│      │  002covid19.png
│      │  003covid19.png
```
# Pretrain the binary classification networks
classification_hingeloss_preaugment
  ```
python train.py --dataroot icassp2024/augmented_covid --dataset_name covid \
-project_name convnext_tiny --model_name convnext_tiny --gpu_ids 0,1

python test.py --dataroot icassp2024/augmented_covid --dataset_name covid \
-project_name convnext_tiny --model_name convnext_tiny --gpu_ids 0,1
  ```
