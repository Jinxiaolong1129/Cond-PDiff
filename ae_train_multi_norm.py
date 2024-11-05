from omegaconf import OmegaConf
import argparse
from itertools import islice
import re
from datetime import datetime
import random
import wandb
import logging
import os
from tqdm import tqdm
import yaml
import copy


import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from glue_test import args_multiple_GLUE_test, GLUE_test

from dataset.weights_dataset import Multi_WeightDataset_norm

from model.od_encoder_decoder import medium, small
from model.ae_ddpm import DDPM_Trainer_multiple_cond_clip
from build_dataset import build_multi_para_diff_dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau


def prepare_data(dataset_path, batch_size=10):
    weight_matrix_dir = os.path.join(dataset_path, "weights_matrix.pt")
    positions_dir = os.path.join(dataset_path, "weights_positions.pt")
    matching_dirs_dict_dir = os.path.join(dataset_path, "checkpoints_dirs.pt")
    original_shapes_dir = os.path.join(dataset_path, "original_shapes.pt")

    weight_matrix = torch.load(weight_matrix_dir)
    positions = torch.load(positions_dir)
    matching_dirs = torch.load(matching_dirs_dict_dir)
    original_shapes = torch.load(original_shapes_dir)

    dataset = Multi_WeightDataset_norm(
        weight_matrix, matching_dirs, positions, original_shapes)

    train_loader = DataLoader(dataset, batch_size=batch_size,
                              shuffle=True, num_workers=1, pin_memory=True, drop_last=True)
    val_loader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True, num_workers=1, pin_memory=True, drop_last=True)

    return train_loader, val_loader


class AE_Trainer():
    def __init__(self, auto_encoder_model, train_loader, val_loader, device,
                 base_dir=None,
                 ae_config=None):
        self.auto_encoder_model = auto_encoder_model
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.device = device

        data_item = next(iter(train_loader))
        inputs = data_item['data']

        self.in_dim = inputs.shape[1]

        self.global_epoch = 0

        self.base_dir = base_dir

        self.already_validation_epoch = False

        self.best_metric = float('-inf')
        self.best_model_state = None

        self.ae_config = ae_config
        self.rank = ae_config["data_parameters"]["rank"]

        self.build_optimizer()

        self.ae_eval_epoch = ae_config["training_parameters"]["ae_eval_epoch"]
        self.ae_save_epoch = ae_config["training_parameters"]["ae_save_epoch"]

        self.average_input_accs = None
        self.max_accs = {}

        if ae_config["training_parameters"]["load_ae_checkpoint"] is not None:
            self.load_ae_checkpoint = ae_config["training_parameters"]["load_ae_checkpoint"]
            pattern = re.compile(r'auto_encoder_model_(\d+)\.pth')
            match = pattern.match(os.path.basename(self.load_ae_checkpoint))
            if match:
                self.start_epoch = int(match.group(1))
            else:
                raise ValueError("The load_ae_checkpoint is not valid")
        else:
            self.load_ae_checkpoint, self.start_epoch = self.check_checkpoint_dir(
                self.base_dir)

        if self.load_ae_checkpoint is not None:
            logging.info(f"Load model: {self.load_ae_checkpoint}")
            logging.info(f"Start epoch: {self.start_epoch}")

            self.load_checkpoint(self.load_ae_checkpoint)
            self.start_epoch = int(self.start_epoch)
            self.end_epoch = ae_config["training_parameters"]["epochs"]
        else:
            self.start_epoch = 0
            self.end_epoch = ae_config["training_parameters"]["epochs"]

        self.auto_encoder_model.to(self.device)
        if ae_config["training_parameters"]["ae_test"]:
            self.auto_encoder_model.to(self.device)
        else:
            self.auto_encoder_model.to(self.device)
            self.auto_encoder_model = torch.nn.DataParallel(
                self.auto_encoder_model)

        self.batch_size = ae_config["training_parameters"]["batch_size"]

        exp_dir = os.path.join(base_dir,
                               f'norm.exp_batch.{self.batch_size}_lr.{self.ae_config["training_parameters"]["lr"]}_patience.{self.ae_config["training_parameters"]["patience"]}_start_epoch.{self.start_epoch}_end_epoch.{self.end_epoch}')
        self.exp_dir = exp_dir
        self.model_save_dir = os.path.join(exp_dir, f"model")
        self.log_dir = os.path.join(exp_dir, "log")

        os.makedirs(self.model_save_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        self.log_file = os.path.join(self.log_dir, "para_ae_training_log.txt")

        logging.info(f"Model save dir: {self.model_save_dir}")
        logging.info(f"Log dir: {self.log_dir}")

    def train(self):
        self.log(f'Start Training!!!')
        for epoch in range(self.start_epoch, self.end_epoch):
            self.global_epoch = epoch
            self.log(f'Epoch {epoch+1}/{self.end_epoch}')
            self.train_epoch()

            if (epoch+1) % self.ae_save_epoch == 0:
                self.save_epoch_model(epoch)
                self.validate_ae_recon_epoch()
                self.log(f'Saved model at {self.log_file}')
            if (epoch+1) > (self.end_epoch - self.ae_eval_epoch):
                self.validate_ae_recon_epoch()
                best_ae = self.validate_param_epoch()
                self.save_model(best_ae)
                self.save_best_model()

        self.log('Training complete')
        self.save_best_model()

    def train_epoch(self):
        self.auto_encoder_model.train()
        train_loss = 0.0
        for item in self.train_loader:
            data = item['data']
            data = data.to(self.device)

            self.optimizer.zero_grad()
            reconstructed = self.auto_encoder_model(data)
            loss = self.criterion(reconstructed, data)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(self.train_loader)
        self.scheduler.step(avg_train_loss)

        # Log training loss and learning rate
        wandb.log({
            "autoencoder training loss": avg_train_loss,
            "autoencoder learning rate": self.optimizer.param_groups[0]['lr']
        })

        self.log(f'autoencoder training loss: {avg_train_loss}')
        self.log(
            f"autoencoder learning rate: {self.optimizer.param_groups[0]['lr']}")

    def validate_ae_recon_epoch(self):
        val_loss = 0.0
        with torch.no_grad():
            for item in self.val_loader:
                data = item['data']
                data = data.to(self.device)

                reconstructed = self.auto_encoder_model.module(data)
                # reconstructed = self.auto_encoder_model(data)

                loss = self.criterion(reconstructed, data)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(self.val_loader)
        wandb.log({"autoencoder recon val loss": avg_val_loss})
        self.log(f'autoencoder recon val loss: {avg_val_loss}')

    def validate_param_epoch(self):
        self.auto_encoder_model.eval()
        val_param_list = self.val_loader.dataset.get_GLUE_eval_from_each()

        if not self.already_validation_epoch:
            logging.info('---------------------------------')
            logging.info('Test the input auto_encoder_model')
            self.already_validation_epoch = True
            input_accs = []
            for index, item in enumerate(val_param_list):
                norm_param = item['data'].to(self.device)
                checkpoints_dir = item['checkpoint_dir']
                dataset = item['dataset']
                position = item['position']
                shape = item['shape']
                # # BUG
                # if dataset == 'sst2':
                #     continue
                param = self.val_loader.dataset.denormalize_data(
                    dataset_name=dataset, normalized_data=norm_param)
                eval_acc, eval_loss = self.task_func(param=param, checkpoints_dir=checkpoints_dir,
                                                     positions=position, original_shapes=shape,
                                                     dataset=dataset)
                input_accs.append((dataset, eval_acc))

            self.log("input auto_encoder_model accuracy:{}".format(input_accs))

            max_accs = {}
            for dataset, eval_acc in input_accs:
                if dataset not in max_accs or eval_acc > max_accs[dataset]:
                    max_accs[dataset] = eval_acc

            for dataset, max_acc in max_accs.items():
                self.log(f"best input auto_encoder_model {dataset}: {max_acc}")
            average_input_accs = sum(
                eval_acc for _, eval_acc in input_accs) / len(input_accs) if input_accs else 0
            self.log(f'average input auto_encoder_model: {average_input_accs}')

            self.max_accs = max_accs
            self.average_input_accs = average_input_accs

        """
        AE reconstruction parameters
        """
        self.log('---------------------------------')
        self.log('Test the AE auto_encoder_model')
        ae_val_param_list = copy.deepcopy(val_param_list)
        ae_val_param_origin = torch.stack(
            [item['data'] for item in ae_val_param_list])

        ae_recon_val_param_list = copy.deepcopy(val_param_list)

        val_param = torch.stack([item['data']
                                for item in ae_recon_val_param_list])

        processed_data = self.auto_encoder_model(val_param.to(self.device))

        for index, item in enumerate(ae_recon_val_param_list):
            ae_recon_val_param_list[index]['data'] = processed_data[index]

        if torch.equal(ae_recon_val_param_list[0]['data'], processed_data[0]):
            print("The tensors are equal.")
        else:
            print("The tensors are not equal.")

        # DDP change here
        with torch.no_grad():
            if isinstance(self.auto_encoder_model, torch.nn.DataParallel):
                latent = self.auto_encoder_model.module.encode(
                    val_param.to(self.device))
                self.log("latent shape:{}".format(latent.shape))
                ae_params = self.auto_encoder_model.module.decode(latent)
                self.log("ae params shape:{}".format(ae_params.shape))
            else:
                latent = self.auto_encoder_model.encode(val_param.to(self.device))
                self.log("latent shape:{}".format(latent.shape))
                ae_params = self.auto_encoder_model.decode(latent)
                self.log("ae params shape:{}".format(ae_params.shape))

        ae_rec_accs = []
        for index, item in enumerate(ae_recon_val_param_list):
            norm_param = item['data'].to(self.device)
            checkpoints_dir = item['checkpoint_dir']
            dataset = item['dataset']
            position = item['position']
            shape = item['shape']
            # # BUG sst2
            # if dataset == 'sst2':
            #     continue
            param = self.val_loader.dataset.denormalize_data(
                dataset_name=dataset, normalized_data=norm_param)
            origin_param = self.val_loader.dataset.denormalize_data(
                dataset_name=dataset, normalized_data=ae_val_param_list[0]['data'])
            eval_acc, eval_loss = self.task_func(param=param,
                                                 checkpoints_dir=checkpoints_dir,
                                                 positions=position, original_shapes=shape,
                                                 dataset=dataset)
            ae_rec_accs.append((dataset, eval_acc))
            print(f"ae_rec_accs: {ae_rec_accs}")

        for dataset, max_acc in self.max_accs.items():
            self.log(f"best input auto_encoder_model {dataset}: {max_acc}")
        self.log(
            f'average input auto_encoder_model: {self.average_input_accs}')

        max_ae_rec_accs = {}
        for dataset, eval_acc in ae_rec_accs:
            if dataset not in max_ae_rec_accs or eval_acc > max_ae_rec_accs[dataset]:
                max_ae_rec_accs[dataset] = eval_acc

        for dataset, max_acc in max_ae_rec_accs.items():
            self.log(
                f"AE reconstruction auto_encoder_models {dataset}: {max_acc}")

        self.log(f"ae_rec_accs: {ae_rec_accs}")
        average_ae_rec_accs = sum(
            eval_acc for _, eval_acc in ae_rec_accs) / len(ae_rec_accs) if ae_rec_accs else 0
        self.log(
            f'average AE reconstruction auto_encoder_models accuracy: {average_ae_rec_accs}')

        self.log('---------------------------------')

        wandb.log({
            "max_accs": self.max_accs,
            "average_input_accs": self.average_input_accs,
            "ae_rec_accs": ae_rec_accs,
            "max_ae_rec_accs": max_ae_rec_accs,
            "average_ae_rec_accs": average_ae_rec_accs,
            "epoch": self.global_epoch
        })

        return average_ae_rec_accs

    def test(self):
        self.log('Testing started!')
        self.validate_param_epoch()
        self.log('Testing complete!')

    def task_func(self, param, checkpoints_dir, positions, original_shapes, dataset):
        model_args, data_args, training_args = args_multiple_GLUE_test(self.ae_config['glue_test_parameters']['config_path'],
                                                                       dataset_name=dataset,
                                                                       model=self.ae_config['training_parameters']['model'],
                                                                       rank=self.rank
                                                                       )
        training_args.logging_dir = os.path.join(self.exp_dir, 'evaluate')
        training_args.output_dir = os.path.join(self.exp_dir, 'evaluate')

        eval_metric = GLUE_test(model_args, data_args, training_args,
                                param, checkpoints_dir, positions, original_shapes)
        if dataset == 'cola':
            eval_acc = eval_metric['eval_matthews_correlation']
            eval_loss = eval_metric['eval_loss']
        elif dataset in ['sst2', 'mnli', 'qnli', 'rte']:
            eval_acc = eval_metric['eval_accuracy']
            eval_loss = eval_metric['eval_loss']
        elif dataset == 'mrpc' or dataset == 'qqp':
            eval_acc = eval_metric['eval_f1']
            eval_loss = eval_metric['eval_loss']
        elif dataset == 'stsb':
            eval_acc = eval_metric['eval_pearson']
            eval_loss = eval_metric['eval_loss']
        else:
            raise ValueError("dataset name is not supported")

        return eval_acc, eval_loss

    def log(self, message):
        message = "[auto encoder][Epoch {}] {}".format(
            self.global_epoch, message)
        logging.info(message)
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)

        with open(self.log_file, "a") as log_file:
            log_file.write(message + "\n")

    def save_epoch_model(self, epoch, file_path='auto_encoder_model.pth'):
        file_path = os.path.join(
            self.model_save_dir, f'auto_encoder_model_{epoch}.pth')
        # Check if the model is wrapped with DataParallel
        if isinstance(self.auto_encoder_model, torch.nn.DataParallel):
            torch.save(self.auto_encoder_model.module.state_dict(), file_path)
        else:
            torch.save(self.auto_encoder_model.state_dict(), file_path)

        self.log(f'Saved model to {file_path}')

    def save_model(self, current_ae):
        if current_ae > self.best_metric:
            self.best_metric = current_ae
            # Check if the model is wrapped with DataParallel and save the underlying model
            if isinstance(self.auto_encoder_model, torch.nn.DataParallel):
                self.best_model_state = self.auto_encoder_model.module.state_dict()
            else:
                self.best_model_state = self.auto_encoder_model.state_dict()

    def save_best_model(self, file_path='auto_encoder_best_model.pth'):
        file_path = os.path.join(self.model_save_dir, file_path)
        if self.best_model_state:
            torch.save(self.best_model_state, file_path)
        else:
            if isinstance(self.auto_encoder_model, torch.nn.DataParallel):
                torch.save(
                    self.auto_encoder_model.module.state_dict(), file_path)
            else:
                torch.save(self.auto_encoder_model.state_dict(), file_path)

        self.log(f'Saved model to {file_path}')

    def build_optimizer(self):
        self.optimizer = optim.AdamW(self.auto_encoder_model.parameters(),
                                     lr=self.ae_config["training_parameters"]["lr"],
                                     betas=(0.9, 0.999),
                                     eps=1e-8,
                                     weight_decay=1e-2)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5,
                                           patience=self.ae_config["training_parameters"]["patience"], verbose=True)
        self.criterion = nn.MSELoss()

    def check_checkpoint_dir(self, base_dir):
        max_num = -1
        model_path = None
        # Match the model file name
        pattern = re.compile(r'auto_encoder_model_(\d+)\.pth')

        for filename in os.listdir(base_dir):
            match = pattern.match(filename)
            if match:
                num = int(match.group(1))
                if num > max_num:
                    max_num = num
                    model_path = os.path.join(base_dir, filename)

        return model_path, max_num

    def load_checkpoint(self, checkpoint_dir):
        checkpoint = torch.load(checkpoint_dir, map_location=self.device)
        self.auto_encoder_model.load_state_dict(checkpoint)
        logging.info(
            f'Loaded checkpoint from {checkpoint_dir} with best metric {self.best_metric}')


def ae_train_multi(ae_config: dict):
    # GPU
    gpu_id = ae_config['device']['gpu_id']
    device = torch.device(
        f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')

    # layer number
    if len(ae_config["data_parameters"]["layer_num"]) == 1:
        layer_number = str(ae_config["data_parameters"]["layer_num"][0])
    else:
        layer_number = '_'.join(str(num)
                                for num in ae_config["data_parameters"]["layer_num"])

    # wandb init
    dataset_name = '_'.join(ae_config["data_parameters"]['datasets'])
    wandb_name = ae_config['wandb_parameters']['run_name'] + '.' + layer_number + '.' + dataset_name + \
        '_' + ae_config["training_parameters"]['model'] + \
        '_' + f"rank_{ae_config['data_parameters']['rank']}"
    wandb_project_name = ae_config['wandb_parameters']['project'] + \
        '_' + ae_config["training_parameters"]['model']

    if os.environ.get('DEBUG') == "1":
        wandb_name = 'debug.' + layer_number
        wandb.init(project=wandb_project_name,
                   name=wandb_name,)
    else:
        wandb_name = ae_config['wandb_parameters']['run_name'] + \
            '.' + layer_number + '.' + dataset_name
        wandb.init(project=wandb_project_name,
                   name=wandb_name,)

    # NOTE  dataset size
    dataset_path = build_multi_para_diff_dataset(base_dir=ae_config["data_parameters"]["dataset_path"],
                                                 layer_number=ae_config["data_parameters"]["layer_num"],
                                                 datasets=ae_config["data_parameters"]["datasets"],
                                                 rank=ae_config["data_parameters"]["rank"],)

    # data loader
    train_loader, val_loader = prepare_data(
        dataset_path=dataset_path, batch_size=ae_config["training_parameters"]["batch_size"])

    # Model, loss, and optimizer
    if ae_config["model_parameters"]["model_type"] == "small":
        inputs, _ = next(iter(train_loader))
        in_dim = inputs.shape[-1]
        input_noise_factor = ae_config["model_parameters"]["input_noise_factor"]
        latent_noise_factor = ae_config["model_parameters"]["latent_noise_factor"]

        model = small(in_dim, input_noise_factor,
                      latent_noise_factor).to(device)
    elif ae_config["model_parameters"]["model_type"] == "medium":
        data_item = next(iter(train_loader))
        inputs = data_item['data']
        # inputs = next(iter(train_loader))
        in_dim = inputs.shape[-1]
        input_noise_factor = ae_config["model_parameters"]["input_noise_factor"]
        latent_noise_factor = ae_config["model_parameters"]["latent_noise_factor"]

        model = medium(in_dim, input_noise_factor,
                       latent_noise_factor).to(device)

    # AE training
    ae_trainer = AE_Trainer(model, train_loader, val_loader, device,
                            base_dir=dataset_path, ae_config=ae_config)

    if ae_config["training_parameters"]["ae_test"]:
        print('=====================================')
        print('Test the auto_encoder_model')
        ae_trainer.test()
    else:
        ae_trainer.train()

    # DDPM training
    ddpm_config = OmegaConf.load(args.ddpm_config)
    print(OmegaConf.to_yaml(ddpm_config))

    ddpm_config["training_parameters"]["load_ddpm_checkpoint"] = ae_bash[
        f'args_{args.ae_bash_args_num}']['load_ddpm_checkpoint']
    ddpm_config["training_parameters"]["ddpm_test"] = ae_bash[
        f'args_{args.ae_bash_args_num}']['ddpm_test']

    if f'args_{args.ae_bash_args_num}' in ae_bash and 'ddpm_end_epoch' in ae_bash[f'args_{args.ae_bash_args_num}']:
        ddpm_config["training_parameters"]["ddpm_end_epoch"] = ae_bash[
            f'args_{args.ae_bash_args_num}']['ddpm_end_epoch']

    ddpm_config.model_parameters.num_conditions = len(
        ae_config['data_parameters']['datasets'])
    ddpm_trainer = DDPM_Trainer_multiple_cond_clip(ddpm_config, ae_config, ae_trainer, train_loader, val_loader, device,
                                                   clip_model_name="openai/clip-vit-base-patch32",
                                                   base_dir=dataset_path)

    if ddpm_config["training_parameters"]["ddpm_test"]:
        print('=====================================')
        print('Test ddpm model')
        # if ddpm_config["training_parameters"]["test_similarity"]:
        #     ddpm_trainer.test()
        ddpm_trainer.test()
    else:
        ddpm_trainer.train()


if __name__ == '__main__':
    # os.environ['WANDB_DISABLED'] = 'false'
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    parser = argparse.ArgumentParser(
        description="Process a configuration file.")
    parser.add_argument('--ae_config', type=str,
                        help='Path to the ae_config file')
    parser.add_argument('--ddpm_config', type=str,
                        default='config/multiple/ddpm_training.yaml',
                        help='Path to the ddpm_config file')
    parser.add_argument('--ae_bash', type=str, help='Path to the ae_bash file')
    parser.add_argument('--ae_bash_args_num', type=str)
    args = parser.parse_args()

    with open(args.ae_config, 'r') as file:
        ae_config = yaml.safe_load(file)

    with open(args.ae_bash, 'r') as file:
        ae_bash = yaml.safe_load(file)

    layer_num = ae_bash[f'args_{args.ae_bash_args_num}']['layer_num']
    datasets_para = ae_bash[f'args_{args.ae_bash_args_num}']['datasets_para']

    ae_config['device']['gpu_id'] = ae_bash[f'args_{args.ae_bash_args_num}']['gpu_id']
    ae_config['training_parameters']['batch_size'] = ae_bash[
        f'args_{args.ae_bash_args_num}']['ae_batch_size']
    ae_config['training_parameters']['ae_test'] = ae_bash[
        f'args_{args.ae_bash_args_num}']['ae_test']
    ae_config["training_parameters"]["load_ae_checkpoint"] = ae_bash[
        f'args_{args.ae_bash_args_num}']['load_ae_checkpoint']
    ae_config["training_parameters"]["lr"] = ae_bash[f'args_{args.ae_bash_args_num}']['ae_lr']
    ae_config["training_parameters"]["patience"] = ae_bash[
        f'args_{args.ae_bash_args_num}']['patience']
    ae_config["training_parameters"]['model'] = ae_bash[
        f'args_{args.ae_bash_args_num}']['model']

    ae_config["training_parameters"]['ae_eval_epoch'] = ae_bash[
        f'args_{args.ae_bash_args_num}']['ae_eval_epoch']
    ae_config["training_parameters"]['ae_save_epoch'] = ae_bash[
        f'args_{args.ae_bash_args_num}']['ae_save_epoch']
    ae_config["training_parameters"]["epochs"] = ae_bash[
        f'args_{args.ae_bash_args_num}']['ae_epochs']

    logging.info(
        f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    logging.info(f'layer_num: {layer_num}')
    logging.info(f'datasets_para: {datasets_para}')
    logging.info(f'gpu_id: {ae_config["device"]["gpu_id"]}')
    logging.info(
        f'batch_size: {ae_config["training_parameters"]["batch_size"]}')
    logging.info(f'epochs: {ae_config["training_parameters"]["epochs"]}')
    logging.info(f'ae_test: {ae_config["training_parameters"]["ae_test"]}')
    logging.info(
        f'load_ae_checkpoint: {ae_config["training_parameters"]["load_ae_checkpoint"]}')
    logging.info(f'lr: {ae_config["training_parameters"]["lr"]}')
    logging.info(f'patience: {ae_config["training_parameters"]["patience"]}')

    for layer in layer_num:
        for datasets in datasets_para:
            print('=====================================')
            print(f'layer number: {layer}')
            print(f'datasets: {datasets}')
            print('=====================================')

            ae_config['data_parameters']['layer_num'] = layer
            ae_config['data_parameters']['datasets'] = datasets
            ae_config["data_parameters"]["dataset_path"] = ae_bash[
                f'args_{args.ae_bash_args_num}']['dataset_path']
            ae_config['data_parameters']['rank'] = ae_bash[f'args_{args.ae_bash_args_num}']['rank']
            ae_train_multi(ae_config)

    print('=====================================')
