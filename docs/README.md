# Data Efficiency Assessment of Generative Adversarial Networks in Energy Applications

---

## 📄 Paper
Nabila, U.M., Lin, L., Zhao, X., Gurecky, W.L., Ramuhalli, P., Radaideh, M.I. (2025). “Data Efficiency Assessment of Generative Adversarial Networks in Energy Applications,” Preprint submitted to Energy and AI.


## ⚙️Environment Installation

This project uses PyTorch for most models (WGAN, FNN, GRU, etc.) and TensorFlow for the cGAN model. Please set up the environments accordingly:

🔵 PyTorch Environment (for cGAN, FNN, GRU, etc.)
```bash
# 1. Create a new conda environment with Python 3.11
conda create -n torchgpu python=3.11

# 2. Activate the environment
conda activate torchgpu

# 3. Install PyTorch and related libraries
pip install torch torchvision torchaudio

# 4. Install other relevant packages
pip install pandas matplotlib scikit-learn seaborn
pip install optuna

```

🟠 TensorFlow Environment (only for cGAN CHF model)
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

## 📊 How to generate the results

- The folder `data` contains the train and test data files (with all the CHF data points) and PSML dataset.
- Datasets for different case scenarios are organized into appropriate case-specific folders under the model directory.
- Navigate to the desired case folder inside models and run the appropriate script to start training or evaluation (e.g., cGan_all.py for case1/all):
  
```bash
cd model/cGAN/CHF/case1/all/
python cGan_all.py
```

Results will be saved automatically in the working folder.
