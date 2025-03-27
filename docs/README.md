# Data Efficiency Assessment of Generative Adversarial Networks in Energy Applications

---

## 📄 Paper
Nabila, U. M., Lin, L., Zhao, X., Gurecky, W. L., Ramuhalli, P., Radaideh, M. I. (2025). Data efficiency assessment of generative adversarial networks in energy applications. *Energy and AI*, **20**, 100501. https://doi.org/10.1016/j.egyai.2025.100501


## ⚙️Environment Installation

This project uses PyTorch for most models (WGAN, FNN, GRU, etc.) and TensorFlow for the cGAN CHF model and FNN model. Please set up the environments accordingly:

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

🟠 TensorFlow Environment (only for cGAN CHF model and FNN model)
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

- The folder `data` contains the CHF test data file (with all the CHF data points) and PSML dataset.
- Datasets for different case scenarios are organized into appropriate case-specific folders under the model directory.
- Navigate to the desired case folder inside models and run the appropriate script to start training or evaluation (e.g., cGan_all.py for case1/all):
  
```bash
cd model/cGAN/CHF/case1/all/
python cGan_all.py
```

Results will be saved automatically in the working folder.

Note: The seasonal variation experiment (PSML case 3) is not included in a separate folder in this repository. This is because the results were briefly discussed in the paper without being presented in a dedicated table or figure.
