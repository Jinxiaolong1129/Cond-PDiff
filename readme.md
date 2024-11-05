# Cond-PDiff

## Installation

1. Clone the repository:

```
git clone https://github.com/NUS-HPC-AI-Lab/Neural-Network-Diffusion.git
```

2. Create a new Conda environment and activate it: 

```
conda env create -f environment.yml
conda activate pdiff
```

or install necessary package by:

```
pip install -r requirements.txt
```

### **Prepare Datasets**

To prepare LoRA datasets, please run.
```
bash ./script/run_lora_bert.sh
bash ./script/run_lora_deberta-base.sh
bash ./script/run_lora_roberta-base.sh
```
The GLUE benchmark parameter is in [`config/multiple/glue.json`]().
The script is to obtain the training data of lora parameter of Cond-Pdiff. We provide our training data in [link]().


### **Model Training**

To get autoencoder and diffusion model in Cond-PDiff, please run.
The argument of training process is in [`config/multiple/ae_bash.yaml`]().

```
bash ae_train_multi_norm.sh
```
**Note:** If you encounter any issues related to file paths, you may need to use `change_var.ipynb` to adjust the paths accordingly.


### 

## Citation
If you found our work useful, please consider citing us.

```
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