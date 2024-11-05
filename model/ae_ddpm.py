import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import re
import copy
import logging
import wandb
from typing import Any


import pdb
import pickle
# import hydra.utils
import pytorch_lightning as pl
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .utils.ddpm import *
from .utils.utils import *
from .ddpm import DDPM
from .unet import AE_CNN_bottleneck, AE_CNN_bottleneck_cond, AE_CNN_bottleneck_cond_clip

from glue_test import args_multiple_GLUE_test, GLUE_test, GLUE_test_inference
from transformers import CLIPProcessor, CLIPModel

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def restore_from_vector_using_positions(vector, position, shapes):
    matrices = {}
    for key, shape in shapes.items():
        start_pos = position[key]
        size = torch.prod(torch.tensor(shape)).item()
        matrices[key] = vector[start_pos:start_pos+size].view(shape)
    return matrices


def get_update_adapter_weight(weight, position, shapes):
    state_dict = restore_from_vector_using_positions(weight, position, shapes)
    return state_dict


class DDPM_Trainer_multiple_cond_clip(DDPM):
    def __init__(self, ddpm_config, ae_config,
                 ae_trainer, train_loader, val_loader, device,
                 clip_model_name=None,
                 base_dir=None):
        super(DDPM_Trainer_multiple_cond_clip, self).__init__(ddpm_config)

        self.ae_config = ae_config
        self.rank = ae_config["data_parameters"]["rank"]

        self.auto_encoder_model = ae_trainer.auto_encoder_model
        input_dim = ae_trainer.in_dim
        input_noise = torch.randn((10, input_dim)).to(device)
        latent_dim = self.auto_encoder_model.encode(input_noise).shape

        self.ddpm_config = ddpm_config
        self.in_dim_diff = latent_dim[-1] * latent_dim[-2]
        self.in_channel_diff = ddpm_config.model_parameters.in_channel

        self.num_conditions = ddpm_config.model_parameters.num_conditions  # condition number
        logging.info(
            f'=========== ddpm diffusion model | num_conditions: {self.num_conditions} ===========')

        self.cond_emb_size = ddpm_config.model_parameters.cond_emb_size

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.device = device

        self.model = self.build_diff_model(self.in_dim_diff, self.in_channel_diff,
                                           num_conditions=self.num_conditions,
                                           cond_emb_size=self.cond_emb_size,
                                           time_step=1000)
        self.model = self.model.to(device)

        self.clip_model = CLIPModel.from_pretrained(clip_model_name)
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)

        self.clip_model = self.clip_model.to(device)
        self.clip_model.eval()

        self.global_epoch = 0
        self.already_validation_epoch = False

        self.best_metric = 0
        self.best_model_state = None

        self.base_dir = base_dir

        self.average_input_accs = None
        self.max_accs = None

        self.build_optimizer(ddpm_config)
        # training parameters
        self.ddpm_eval_epoch = ddpm_config.training_parameters.ddpm_eval_epoch
        self.ddpm_save_epoch = ddpm_config.training_parameters.ddpm_save_epoch

        if os.environ.get('DEBUG') == '1':
            self.log_file = os.path.join(
                base_dir, "log/cond_clip_ddpm_training_log_debug.txt")
        else:
            self.log_file = os.path.join(
                base_dir, "log/cond_clip_ddpm_training_log.txt")

        if ddpm_config["training_parameters"]["load_ddpm_checkpoint"] is not None:
            self.load_ddpm_checkpoint = ddpm_config["training_parameters"]["load_ddpm_checkpoint"]

            # match the model name
            pattern = re.compile(r'clip_cond_diffusion_model_(\d+)\.pth')
            match = pattern.match(os.path.basename(self.load_ddpm_checkpoint))
            if match:
                self.ddpm_start_epoch = int(match.group(1))
            else:
                raise ValueError("The load_ae_checkpoint is not valid")
        else:
            # dataset/bert-base-uncased/lora_r_1/para_dataset_sst2_rte_mrpc_cola_qnli_stsb/layer_11_10_9_8_7_6_5_4_3_2_1_0/clip_pdiff_exp_batch.256_lr.0.001_patience.5000_start_epoch.0_end_epoch.8000/clip_pdiff_model/clip_cond_diffusion_model_7999.pth
            self.load_ddpm_checkpoint = None  # TODO
            # start_epoch = self.load_ddpm_checkpoint.split(
            #     '/')[-1].split('_')[-1].split('.')[0]
            # self.ddpm_start_epoch = int(start_epoch)

        if self.load_ddpm_checkpoint is not None:
            self.log(f"Load model: {self.load_ddpm_checkpoint}")
            self.load_checkpoint(self.load_ddpm_checkpoint)
            self.ddpm_start_epoch = int(self.ddpm_start_epoch)
            self.ddpm_end_epoch = ddpm_config["training_parameters"]["ddpm_end_epoch"]
        else:
            self.ddpm_start_epoch = 0
            self.ddpm_end_epoch = ddpm_config["training_parameters"]["ddpm_end_epoch"]

        exp_dir = os.path.join(base_dir,
                               f'clip_pdiff_exp_batch.{self.train_loader.batch_size}_lr.{self.ddpm_config["training_parameters"]["lr"]}_patience.{self.ddpm_config["training_parameters"]["patience"]}_start_epoch.{self.ddpm_start_epoch}_end_epoch.{self.ddpm_end_epoch}')
        self.exp_dir = exp_dir
        self.model_save_dir = os.path.join(exp_dir, f"clip_pdiff_model")
        self.log_dir = os.path.join(exp_dir, "log")

        os.makedirs(self.model_save_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        logging.info(f"Model save dir: {self.model_save_dir}")
        logging.info(f"Log dir: {self.log_dir}")

        self.tsne_fig_dir = os.path.join(base_dir, "log")
        
        
        self.device_auto_encoder_model = torch.device("cuda:0")  
        self.device_diffusion = torch.device("cuda:0")  

        self.auto_encoder_model.encoder.to(self.device_auto_encoder_model)
        self.auto_encoder_model.decoder.to(self.device_auto_encoder_model)

        self.model.to(self.device_diffusion)
        self.clip_model.to(self.device_diffusion)
        
        


    def train(self):
        self.log('Training begin')
        for epoch in range(self.ddpm_start_epoch, self.ddpm_end_epoch):
            self.global_epoch = epoch
            self.log(f'Epoch {epoch+1}/{self.ddpm_end_epoch}')
            self.train_one_epoch()

            if (epoch + 1) % self.ddpm_save_epoch == 0:
                self.save_epoch_model(epoch)

            if (epoch + 1) > (self.ddpm_end_epoch-self.ddpm_eval_epoch):
                best_diff = self.validation_epoch()
                self.save_model(best_diff)
                self.save_best_model_to_disk()

            self.log(self.log_file)
        self.log('Training complete')
        self.save_best_model_to_disk()

    def generate_text_embedding(self, text):
        with torch.no_grad():
            inputs = self.clip_processor(
                text=text, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device_diffusion) for k, v in inputs.items()}

            clip_text_embeddings = self.clip_model.get_text_features(**inputs)
            clip_text_embeddings = clip_text_embeddings.to(self.device_diffusion)

        return clip_text_embeddings

    def train_one_epoch(self):
        train_loss = 0.0
        for item in self.train_loader:
            # ddpm training
            batch = item['data']
            condition = item['condition']
            description = item['description']
            batch = batch.to(self.device)
            condition = condition.to(self.device)
            # description = description.to(self.device)

            condition_text_embedding = self.generate_text_embedding(
                description)

            self.ddpm_optimizer.zero_grad()
            loss = self.forward(batch, condition_text_embedding)
            loss.backward()
            self.ddpm_optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(self.train_loader)
        self.scheduler.step(avg_train_loss)

        wandb.log({"diffusion train_loss": avg_train_loss,
                   "diffusion learning rate": self.ddpm_optimizer.param_groups[0]['lr']})
        self.log(f'diffusion training Loss: {avg_train_loss}')
        self.log(
            f"diffusion learning rate: {self.ddpm_optimizer.param_groups[0]['lr']}")

    def validation_epoch(self):
        self.auto_encoder_model.eval()
        self.model.eval()

        val_param_list = self.val_loader.dataset.get_GLUE_eval_from_each()
        self.already_validation_epoch = True
        if not self.already_validation_epoch:
            self.already_validation_epoch = True
            input_accs = []
            for index, item in enumerate(val_param_list):
                param = item['data'].to(self.device)
                checkpoints_dir = item['checkpoint_dir']
                dataset = item['dataset']
                position = item['position']
                shape = item['shape']
                eval_acc, eval_loss = self.task_func(param=param, checkpoints_dir=checkpoints_dir,
                                                     positions=position, original_shapes=shape,
                                                     dataset=dataset)
                input_accs.append((dataset, eval_acc))
                self.log("input auto_encoder_model accuracy:{}".format(input_accs))

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
            ae_recon_val_param_list = copy.deepcopy(val_param_list)

            val_param = torch.stack([item['data']
                                    for item in ae_recon_val_param_list])

            processed_data = self.auto_encoder_model(val_param.to(self.device))

            for index, item in enumerate(ae_recon_val_param_list):
                ae_recon_val_param_list[index]['data'] = processed_data[index]
                ae_recon_val_param_list[index]['label'] = processed_data[index]

            if torch.equal(ae_recon_val_param_list[0]['data'], processed_data[0]):
                print("The tensors are equal.")
            else:
                print("The tensors are not equal.")

            latent = self.auto_encoder_model.encode(val_param.to(self.device))
            self.log("latent shape:{}".format(latent.shape))

            ae_params = self.auto_encoder_model.decode(latent)
            self.log("ae params shape:{}".format(ae_params.shape))

            ae_rec_accs = []
            for index, item in enumerate(ae_recon_val_param_list):
                param = item['data'].to(self.device)
                checkpoints_dir = item['checkpoint_dir']
                dataset = item['dataset']
                position = item['position']
                shape = item['shape']
                eval_acc, eval_loss = self.task_func(param=param,
                                                     checkpoints_dir=checkpoints_dir,
                                                     positions=position, original_shapes=shape,
                                                     dataset=dataset)
                ae_rec_accs.append((dataset, eval_acc))

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

            average_ae_rec_accs = sum(
                eval_acc for _, eval_acc in ae_rec_accs) / len(ae_rec_accs) if ae_rec_accs else 0
            self.log(
                f'average AE reconstruction auto_encoder_models accuracy: {average_ae_rec_accs}')

            self.log('---------------------------------')

        def generate_in_validation_epoch(val_param_list):
            diff_recon_val_param_list = copy.deepcopy(val_param_list)

            val_param = torch.stack(
                [item['data'] for item in diff_recon_val_param_list]).to(self.device)
            description_param = [item['description']
                                 for item in diff_recon_val_param_list]

            condition_text_embedding = self.generate_text_embedding(
                description_param)

            batch = self.pre_process(val_param)

            outputs = []
            with torch.no_grad():  # Ensure no gradients are computed
                outputs = self.generate(
                    batch=batch, num=batch.shape[0], cond=condition_text_embedding)
            processed_data = self.post_process(outputs)

            for index, item in enumerate(diff_recon_val_param_list):
                diff_recon_val_param_list[index]['data'] = processed_data[index]
                diff_recon_val_param_list[index]['label'] = processed_data[index]

            if torch.equal(diff_recon_val_param_list[0]['data'], processed_data[0]):
                print("The tensors are equal.")
            else:
                print("The tensors are not equal.")

            return diff_recon_val_param_list

        diff_recon_val_param_list = generate_in_validation_epoch(
            val_param_list)

        diff_accs = []
        for index, item in enumerate(diff_recon_val_param_list):
            norm_param = item['data'].to(self.device)
            checkpoints_dir = item['checkpoint_dir']
            dataset = item['dataset']
            position = item['position']
            shape = item['shape']

            param = self.val_loader.dataset.denormalize_data(
                dataset_name=dataset, normalized_data=norm_param)

            eval_acc, eval_loss = self.task_func(param=param,
                                                 checkpoints_dir=checkpoints_dir,
                                                 positions=position, original_shapes=shape,
                                                 dataset=dataset)
            diff_accs.append((dataset, eval_acc))

        max_diff_accs = {}
        for dataset, eval_acc in diff_accs:
            if dataset not in max_diff_accs or eval_acc > max_diff_accs[dataset]:
                max_diff_accs[dataset] = eval_acc

        for dataset, max_acc in max_diff_accs.items():
            self.log(
                f"DIFF reconstruction auto_encoder_models {dataset}: {max_acc}")

        average_diff_accs = sum(
            eval_acc for _, eval_acc in diff_accs) / len(diff_accs) if diff_accs else 0

        self.log(
            f'average DIFF reconstruction auto_encoder_models accuracy: {average_diff_accs}')
        self.log(f'Diffusion reconstruction accuracy:{diff_accs}')
        self.log('---------------------------------')

        return average_diff_accs

    def test(self):
        self.log('Testing started!')
        # self.test_inference()
        self.test_validation()
        self.log('Testing complete!')

    def test_validation(self):
        self.auto_encoder_model.eval()
        self.model.eval()

        val_param_list = self.val_loader.dataset.get_GLUE_eval_from_each(
            num=10)

        def generate_in_validation_epoch(val_param_list):
            diff_recon_val_param_list = copy.deepcopy(val_param_list)
            
            with torch.no_grad():
                val_param = torch.stack(
                            [item['data'] for item in diff_recon_val_param_list]
                        ).to(self.device_auto_encoder_model)
                        
                description_param = [item['description'] for item in diff_recon_val_param_list]

                condition_text_embedding = self.generate_text_embedding(description_param).to(self.device_diffusion)
                
                batch = self.pre_process(val_param).to(self.device_diffusion)

                outputs = []
                with torch.no_grad():  # Ensure no gradients are computed
                    outputs = self.generate(
                        batch=batch, num=batch.shape[0], cond=condition_text_embedding)
                
                outputs = outputs.to(self.device_auto_encoder_model)
                processed_data = self.post_process(outputs)

                for index, item in enumerate(diff_recon_val_param_list):
                    diff_recon_val_param_list[index]['data'] = processed_data[index]
                    diff_recon_val_param_list[index]['label'] = processed_data[index]

                if torch.equal(diff_recon_val_param_list[0]['data'], processed_data[0]):
                    print("The tensors are equal.")
                else:
                    print("The tensors are not equal.")

                return diff_recon_val_param_list

        diff_recon_val_param_list = generate_in_validation_epoch(
            val_param_list)

        diff_accs = []
        for index, item in enumerate(diff_recon_val_param_list):
            norm_param = item['data'].to(self.device)
            checkpoints_dir = item['checkpoint_dir']
            dataset = item['dataset']
            position = item['position']
            shape = item['shape']
            # BUG sst2
            if dataset == 'sst2':
                continue
            param = self.val_loader.dataset.denormalize_data(
                dataset_name=dataset, normalized_data=norm_param)

            eval_acc, eval_loss = self.task_func(param=param,
                                                 checkpoints_dir=checkpoints_dir,
                                                 positions=position, original_shapes=shape,
                                                 dataset=dataset)
            diff_accs.append((dataset, eval_acc))

        max_diff_accs = {}
        for dataset, eval_acc in diff_accs:
            if dataset not in max_diff_accs or eval_acc > max_diff_accs[dataset]:
                max_diff_accs[dataset] = eval_acc

        for dataset, max_acc in max_diff_accs.items():
            self.log(
                f"DIFF reconstruction auto_encoder_models {dataset}: {max_acc}")

        average_diff_accs = sum(
            eval_acc for _, eval_acc in diff_accs) / len(diff_accs) if diff_accs else 0

        self.log(
            f'average DIFF reconstruction auto_encoder_models accuracy: {average_diff_accs}')
        self.log(f'Diffusion reconstruction accuracy:{diff_accs}')
        self.log('---------------------------------')

        return average_diff_accs

    def test_inference(self):
        self.auto_encoder_model.eval()
        self.model.eval()

        val_param_list = self.val_loader.dataset.get_GLUE_eval_from_each(num=1)

        def generate_in_validation_epoch(val_param_list):
            diff_recon_val_param_list = copy.deepcopy(val_param_list)

            val_param = torch.stack(
                [item['data'] for item in diff_recon_val_param_list]).to(self.device)
            description_param = [item['description']
                                 for item in diff_recon_val_param_list]

            condition_text_embedding = self.generate_text_embedding(
                description_param)

            batch = self.pre_process(val_param)

            outputs = []
            with torch.no_grad():  # Ensure no gradients are computed
                outputs = self.generate(
                    batch=batch, num=batch.shape[0], cond=condition_text_embedding)
            processed_data = self.post_process(outputs)

            for index, item in enumerate(diff_recon_val_param_list):
                diff_recon_val_param_list[index]['data'] = processed_data[index]
                diff_recon_val_param_list[index]['label'] = processed_data[index]

            if torch.equal(diff_recon_val_param_list[0]['data'], processed_data[0]):
                print("The tensors are equal.")
            else:
                print("The tensors are not equal.")

            return diff_recon_val_param_list

        diff_recon_val_param_list = generate_in_validation_epoch(
            val_param_list)

        diff_accs = []
        for index, item in enumerate(diff_recon_val_param_list):
            norm_param = item['data'].to(self.device)
            checkpoints_dir = item['checkpoint_dir']
            dataset = item['dataset']
            position = item['position']
            shape = item['shape']

            param = self.val_loader.dataset.denormalize_data(
                dataset_name=dataset, normalized_data=norm_param)

            eval_acc, eval_loss = self.task_func_test_inference(param=param,
                                                                checkpoints_dir=checkpoints_dir,
                                                                positions=position,
                                                                original_shapes=shape,
                                                                dataset=dataset)
            diff_accs.append((dataset, eval_acc))

        max_diff_accs = {}
        for dataset, eval_acc in diff_accs:
            if dataset not in max_diff_accs or eval_acc > max_diff_accs[dataset]:
                max_diff_accs[dataset] = eval_acc

        for dataset, max_acc in max_diff_accs.items():
            self.log(
                f"DIFF reconstruction auto_encoder_models {dataset}: {max_acc}")

        average_diff_accs = sum(
            eval_acc for _, eval_acc in diff_accs) / len(diff_accs) if diff_accs else 0

        self.log(
            f'average DIFF reconstruction auto_encoder_models accuracy: {average_diff_accs}')
        self.log(f'Diffusion reconstruction accuracy:{diff_accs}')
        self.log('---------------------------------')

        return average_diff_accs

    def check_checkpoint_dir(self, base_dir):
        max_num = -1
        model_path = None
        pattern = re.compile(r'clip_cond_diffusion_model_(\d+)\.pth')

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
        self.model.load_state_dict(checkpoint)

        self.log(f'Loaded checkpoint from {checkpoint_dir}')
        self.log('========================================')

    def build_diff_model(self, in_dim, in_channel, num_conditions=2, cond_emb_size=1024, time_step=1000):
        return AE_CNN_bottleneck_cond_clip(in_dim, in_channel,
                                           num_conditions=num_conditions, cond_emb_size=cond_emb_size,
                                           time_step=1000)

    def build_optimizer(self, ddpm_config):
        self.ddpm_optimizer = optim.AdamW(self.model.parameters(),
                                          lr=self.ddpm_config["training_parameters"]["lr"],
                                          betas=(0.9, 0.999),
                                          eps=1e-8,
                                          weight_decay=1e-2)
        self.scheduler = ReduceLROnPlateau(self.ddpm_optimizer, mode='min', factor=0.5,
                                           patience=self.ddpm_config["training_parameters"]["patience"], verbose=True)

    def save_best_model_to_disk(self, file_path='clip_cond_diffusion_best_model.pth'):
        file_path = os.path.join(self.model_save_dir, file_path)
        if self.best_model_state:
            torch.save(self.best_model_state, file_path)

    def save_model(self, current_ae):
        if current_ae > self.best_metric:
            self.best_metric = current_ae
            self.best_model_state = self.model.state_dict()

    def save_epoch_model(self, epoch, file_path='clip_cond_diffusion_model.pth'):
        file_path = os.path.join(
            self.model_save_dir, f'clip_cond_diffusion_model_{epoch}.pth')
        torch.save(self.model.state_dict(), file_path)

    def pre_process(self, batch):
        latent = self.auto_encoder_model.encode(batch)
        self.latent_shape = latent.shape[-2:]
        return latent

    def post_process(self, outputs):
        outputs = outputs.reshape(-1, *self.latent_shape)
        return self.auto_encoder_model.decode(outputs)

    def task_func(self, param, checkpoints_dir, positions, original_shapes, dataset):
        model_args, data_args, training_args = args_multiple_GLUE_test(self.ddpm_config['glue_test_parameters']['config_path'],
                                                                       dataset_name=dataset,
                                                                       model=self.ae_config['training_parameters']['model'],
                                                                       rank=self.rank)

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
        elif dataset == 'mrpc':
            eval_acc = eval_metric['eval_f1']
            eval_loss = eval_metric['eval_loss']
        elif dataset == 'stsb':
            eval_acc = eval_metric['eval_pearson']
            eval_loss = eval_metric['eval_loss']
        else:
            raise ValueError("dataset name is not supported")

        return eval_acc, eval_loss

    def task_func_test_inference(self, param, checkpoints_dir, positions, original_shapes, dataset):
        model_args, data_args, training_args = args_multiple_GLUE_test(self.ddpm_config['glue_test_parameters']['config_path'],
                                                                       dataset_name=dataset,
                                                                       model=self.ae_config['training_parameters']['model'],
                                                                       rank=self.rank)

        training_args.logging_dir = os.path.join(self.exp_dir, 'evaluate')
        training_args.output_dir = os.path.join(self.exp_dir, 'evaluate')

        eval_metric = GLUE_test_inference(
            model_args, data_args, training_args, param, checkpoints_dir, positions, original_shapes)
        if dataset == 'cola':
            eval_acc = eval_metric['eval_matthews_correlation']
            eval_loss = eval_metric['eval_loss']
        elif dataset in ['sst2', 'mnli', 'qnli', 'rte']:
            eval_acc = eval_metric['eval_accuracy']
            eval_loss = eval_metric['eval_loss']
        elif dataset == 'mrpc':
            eval_acc = eval_metric['eval_f1']
            eval_loss = eval_metric['eval_loss']
        elif dataset == 'stsb':
            eval_acc = eval_metric['eval_pearson']
            eval_loss = eval_metric['eval_loss']
        else:
            raise ValueError("dataset name is not supported")

        return eval_acc, eval_loss

    def log(self, message):
        message = "[diffusion][Epoch {}] {}".format(self.global_epoch, message)

        logging.info(message)

        # Ensure the directory of the log file exists
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)

        # Append message to the log file
        with open(self.log_file, "a") as log_file:
            log_file.write(message + "\n")
        log_file.close()
