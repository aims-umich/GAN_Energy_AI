# Data Efficiency Assessment of Generative Adversarial Networks in Energy Applications

---

## 📄 Paper
Nabila, U.M., Lin, L., Zhao, X., Gurecky, W.L., Ramuhalli, P., Radaideh, M.I. (2025). “Data Efficiency Assessment of Generative Adversarial Networks in Energy Applications,” Preprint submitted to Energy and AI.


## ⚙️Environment Installation

This project uses PyTorch for most models (WGAN, FNN, GRU, etc.) and TensorFlow for the cGAN model. Please set up the environments accordingly:

🔵 PyTorch Environment (for WGAN, FNN, GRU, etc.)
```bash
# 1. Create a new conda environment with Python 3.11
conda create -n torchgpu python=3.11
# 2. Activate the environment
conda activate torchgpu
# 3. Install PyTorch and related libraries
pip install torch torchvision torchaudio
# 4. Install other relevant packages
pip install pandas matplotlib scikit-learn seaborn
```

🟠 TensorFlow Environment (for cGAN model)
```bash
# 1. Create a new conda environment with Python 3.11
conda create -n tfgpu python=3.11

# 2. Activate the environment
conda activate tfgpu

# 3. Install TensorFlow with CUDA support
pip install tensorflow[cuda]

# 4. Install other relevant packages
pip install pandas matplotlib scikit-learn seaborn
```

## 📂 Dataset Access

Due to GitHub's file size limit, the dataset `CAISO_zone_1_.csv` (187 MB) is hosted externally.

👉 [Download CAISO_zone_1_.csv from this link] (https://drive.google.com/file/d/1coOdL7Lq1hBkMSt8t9sRT3f5M7pPv7Jb/view?usp=sharing)

After downloading, place the file in the `data/` folder of this repository.
