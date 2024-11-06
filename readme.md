# Cond-PDiff

This repository contains the code and resources for **Conditional Parameter Diffusion (Cond-PDiff)**, a framework designed for efficient parameter generation in neural networks.

---

## 🛠️ Installation

### 1. Clone the Repository

Clone the repository to your local environment:
```bash
git clone https://github.com/NUS-HPC-AI-Lab/Neural-Network-Diffusion.git
```

### 2. Set Up the Environment

Create a new Conda environment using the provided configuration file, or install necessary packages using `pip`:

#### Using Conda:
```bash
conda env create -f environment.yml
conda activate pdiff
```

#### Using pip:
```bash
pip install -r requirements.txt
```

---

Got it! Here’s the refined version with the note included:

---

## 📂 Prepare Datasets

To prepare **LoRA** datasets, execute the following scripts:

```bash
bash ./script/run_lora_bert.sh
bash ./script/run_lora_deberta-base.sh
bash ./script/run_lora_roberta-base.sh
```

The configuration for the **GLUE benchmark** can be found in [`config/multiple/glue.json`](). These scripts will generate the training data for the LoRA parameters required by Cond-PDiff.

> **Note:** 
> Our training data and model(autoencoder / diffusion model) are available in [cond-pdiff](https://purdue0-my.sharepoint.com/:f:/g/personal/jin509_purdue_edu/Eoig8tOYZuhGukKOjbVuJ7ABUHr0NFjQAbc7P2FgA23M9w?e=XGSMTQ).
> 
> Before running the scripts, download the datasets and place them in the following structure:
> ```plaintext
> dataset/
> ┣ bert-base-uncased/
> ┣ deberta-base/
> ┗ roberta-base/
> ```
> You may need to change `load_ae_checkpoint`, `load_ddpm_checkpoint`, `dataset_path`, according to dataset dir.

---

## 🏋️ Model Training

To train the autoencoder and diffusion model in **Cond-PDiff**, use the following command:
```bash
bash ae_train_multi_norm.sh
```

The training parameters are specified in [`config/multiple/ae_bash.yaml`](). 

> **Note:** If you encounter file path issues, use the `change_var.ipynb` notebook to adjust paths as needed.

---

## 📜 Citation

If you found this work useful, please consider citing us:

```bibtex
@misc{wang2024neural,
      title={Neural Network Diffusion}, 
      author={Kai Wang and Zhaopan Xu and Yukun Zhou and Zelin Zang and Trevor Darrell and Zhuang Liu and Yang You},
      year={2024},
      eprint={2402.13144},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
@article{jin2024conditional,
  title={Conditional lora parameter generation},
  author={Jin, Xiaolong and Wang, Kai and Tang, Dongwen and Zhao, Wangbo and Zhou, Yukun and Tang, Junshu and You, Yang},
  journal={arXiv preprint arXiv:2408.01415},
  year={2024}
}
```