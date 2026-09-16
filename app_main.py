#!/usr/bin/env python
# coding: utf-8
#!/usr/bin/python
# -*- coding: UTF-8 -*-
############### built-in modules ###############
import random
import os
import time
import warnings
from typing import Optional
import torchmetrics
import pandas as pd
import numpy as np
############### pytorch ###############
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from multiprocessing import Pool
############### utils ###############
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.pyplot import MultipleLocator
from datetime import datetime
import argparse
import json
from functools import partial
from torch.utils.tensorboard import SummaryWriter
import csv
import sys

warnings.filterwarnings('ignore')
ALL_ACT_LAYERS = {
    "leaky_relu": nn.LeakyReLU,
    "gelu": nn.GELU,
    "relu": nn.ReLU,
    'sigmoid': nn.Sigmoid,
    'tanh': nn.Tanh,
    'prelu': nn.PReLU,
}
ALL_LOSS_FUNC = {
    "mseloss": nn.MSELoss,
    "l1loss": nn.L1Loss,
    "smoothl1loss": nn.SmoothL1Loss
}
ALL_OPTIM_FUNC = {
    "adamw": torch.optim.AdamW,
    "adam": torch.optim.Adam,
    "sgd": torch.optim.SGD
}

class FcUnit(nn.Module): 
    def __init__(
        self, num_features, normalization="null_norm", activation='relu', dropout_prob=0.1):

        super().__init__()
        if normalization == "layer_norm":
            self.norm = nn.LayerNorm(num_features)
        elif normalization == "batch_norm":
            self.norm = nn.BatchNorm1d(num_features)
        elif normalization == "null_norm":
            self.norm = None
        else:
            raise ValueError(f"unknown normalization: {normalization}")

        self.fc = nn.LazyLinear(num_features) 
        self.act_fn = ALL_ACT_LAYERS[activation]()
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        # pre normalization
        if self.norm is not None:
            x = self.norm(x)
        else:
            pass
        x = self.fc(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        return x

class Cnn1dUnit(nn.Module):
    def __init__(self,out_channels, kernel_size=3, activation='relu', stride=1, padding=1):
        super().__init__()
        self.act_fn = ALL_ACT_LAYERS[activation]()

        self.conv1d = nn.LazyConv1d( 
            out_channels=out_channels,）
            kernel_size=kernel_size, 
            stride=stride,
            padding=padding 
        )
        self.batch_norm = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        x = self.conv1d(x)
        x = self.batch_norm(x)
        x = self.act_fn(x)
        return x

class Cnn2dUnit(nn.Module):

    def __init__(
        self,out_channels, kernel_size=2, activation='relu', stride=1, padding=1):
        super().__init__()

        self.act_fn = ALL_ACT_LAYERS[activation]()
        self.conv2d = nn.LazyConv2d(
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )
        self.batch_norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv2d(x)
        x = self.batch_norm(x)
        x = self.act_fn(x)
        return x

class CustomSimpleModel(nn.Module):
    def __init__(
            self,
            simple_layer_dict,
            seq_embed_layer_dict,
            common_embed_layer_dict,
            concat_layer_dict,
            mix_layer=3,
            norm='null_norm'
    ):
        super(CustomSimpleModel, self).__init__()
        
        self.norm = norm
        self.layer_norm_text = nn.LayerNorm(768)
        self.layer_norm_com = nn.LayerNorm(768)
        self.layer_norm_concat = nn.LayerNorm(768 * 2)
        
        self.mix_layer = mix_layer
        # simple layer
        simple_layer = []
        for num_fes in simple_layer_dict['simple_layer_list']:
            per_unit = FcUnit(
                num_features=num_fes,
                normalization=simple_layer_dict['normalization'],
                activation=simple_layer_dict['activation'],
                dropout_prob=simple_layer_dict['dropout_prob']
            )
            simple_layer.append(per_unit)
        self.simple_layer = nn.Sequential(*simple_layer)
        #seq_embed_layer
        seq_embed_layer = []
        for num_channels in seq_embed_layer_dict['seq_embed_con1d_list']:
            per_unit = Cnn1dUnit(
                out_channels=num_channels,
                kernel_size=seq_embed_layer_dict['kernel_size'],
                stride=seq_embed_layer_dict['stride'],
                padding=seq_embed_layer_dict['padding'],
            )
            seq_embed_layer.append(per_unit)
        seq_embed_layer.append(nn.Flatten())
        for num_fes in seq_embed_layer_dict['seq_embed_fc_list']:
            per_unit = FcUnit(
                num_features=num_fes,
                normalization=seq_embed_layer_dict['normalization'],
                activation=seq_embed_layer_dict['activation'],
                dropout_prob=seq_embed_layer_dict['dropout_prob']
            )
            seq_embed_layer.append(per_unit)
        self.seq_embed_layer = nn.Sequential(*seq_embed_layer)
        #common_embed_layer
        common_embed_layer = []
        for num_channels in common_embed_layer_dict['seq_embed_con2d_list']:
            per_unit = Cnn2dUnit(
                out_channels=num_channels,
                kernel_size=common_embed_layer_dict['kernel_size'],
                stride=common_embed_layer_dict['stride'],
                padding=common_embed_layer_dict['padding']
            )
            common_embed_layer.append(per_unit)
        common_embed_layer.append(nn.Flatten())
        for num_fes in common_embed_layer_dict['seq_embed_2d_fc_list']:
            per_unit = FcUnit(
                num_features=num_fes,
                normalization=common_embed_layer_dict['normalization'],
                activation=common_embed_layer_dict['activation'],
                dropout_prob=common_embed_layer_dict['dropout_prob']
            )
            common_embed_layer.append(per_unit)
        self.common_embed_layer = nn.Sequential(*common_embed_layer)
        #concat_layer
        concat_layer = []
        for num_fes in concat_layer_dict['concat_layer_list']:
            per_unit = FcUnit(
                num_features=num_fes,
                normalization=concat_layer_dict['normalization'],
                activation=concat_layer_dict['activation'],
                dropout_prob=concat_layer_dict['dropout_prob']
            )
            concat_layer.append(per_unit)
        self.concat_layer = nn.Sequential(*concat_layer)

    def forward(self, labels, text_embeds, com_embeds):
        # Normalize embeddings based on the specified normalization type
        if self.norm == 'concat':
            concat_embeds = torch.cat([com_embeds, text_embeds], dim=1)
            concat_embeds = self.layer_norm_concat(concat_embeds)
            com_embeds, text_embeds = concat_embeds[:, :768], concat_embeds[:, 768:]
        elif self.norm == 'single':
            text_embeds = self.layer_norm_text(text_embeds)
            com_embeds = self.layer_norm_com(com_embeds)

        # Prepare inputs for the common embedding layer
        inputs_addition = torch.cat([com_embeds.unsqueeze(1), text_embeds.unsqueeze(1)], dim=1)
        for _ in range(self.mix_layer):
            inputs_addition = torch.cat([inputs_addition, com_embeds.unsqueeze(1), text_embeds.unsqueeze(1)], dim=1)

        # Process individual embeddings
        com_dense_output = self.simple_layer(com_embeds)
        text_dense_output = self.simple_layer(text_embeds)

        com_embed_output = self.seq_embed_layer(com_dense_output.unsqueeze(1))
        text_embed_output = self.seq_embed_layer(text_dense_output.unsqueeze(1))

        # Process the combined inputs
        output_addition = self.common_embed_layer(inputs_addition.unsqueeze(1))

        # Concatenate all outputs and apply the final layer
        output = torch.cat([output_addition, com_embed_output, text_embed_output], dim=1)
        output = self.concat_layer(output)

        return output

class CustomSimpleDataset(Dataset):
    def __init__(self, text_embeds, com_embeds, targets):

        self.text_embeds = text_embeds.values if isinstance(text_embeds, pd.DataFrame) else text_embeds
        self.com_embeds = com_embeds.values if isinstance(com_embeds, pd.DataFrame) else com_embeds
        self.targets = targets.values if isinstance(targets, pd.DataFrame) else targets
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        text_embed = self.text_embeds[idx]
        com_embed = self.com_embeds[idx]
        target = self.targets[idx]

        return {
            "labels": torch.tensor(target, dtype=torch.float32).to(self.device),
            "text_embeds": torch.tensor(text_embed, dtype=torch.float32).squeeze(0).to(self.device),
            "com_embeds": torch.tensor(com_embed, dtype=torch.float32).squeeze(0).to(self.device),
        }


def gen_data_class(df):

    text_embeds = ['text'+str(i) for i in range(768)]
    com_embeds = ['ele'+str(i) for i in range(768)]

    dic = {
        "targets": df.iloc[:, 0],
        "text_embeds": df.loc[:, text_embeds],
        "com_embeds": df.loc[:, com_embeds]
    }
    return dic

def eval_model(y_true, y_pred):
    """
    Model evaluation for predicted results with "r2", "root mean squared error", and "mean absolute error"
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.is_tensor(y_true):
        y_true = torch.tensor(y_true)
    if not torch.is_tensor(y_pred):
        y_pred = torch.tensor(y_pred)

    # Ensure tensors are on the same device
    device = y_true.device
    y_pred = y_pred.to(device)

    r2_metric = torchmetrics.R2Score().to(device)
    rmse_metric = torchmetrics.MeanSquaredError(squared=False).to(device)
    mae_metric = torchmetrics.MeanAbsoluteError().to(device)

    r2 = r2_metric(y_pred, y_true).detach().cpu().item()
    rmse = rmse_metric(y_pred, y_true).detach().cpu().item()
    mae = mae_metric(y_pred, y_true).detach().cpu().item()

    result = {"r2": round(r2, 3), "rmse": round(rmse, 3), "mae": round(mae, 3)}
    return result

def plot_figure(x, y, z, save_fig=False, file_name='figure.png'):
    assert len(x) == len(y) == len(z) == 2, "x, y, and z must have 2 sublists each."

    labels = ['Training data', 'Test data']

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    font = {'family': 'Arial', 'size': 16}
    legend_font = FontProperties(**font)

    # Determine the range for the axes once
    all_data = x + y
    min_val = min([0.9 * tensor.min() for tensor in all_data])
    max_val = max([1.1 * tensor.max() for tensor in all_data])

    for i in range(len(x)):
        ax = axs[i]
        xi, yi, zi = x[i], y[i], z[i]
        label = labels[i]

        ax.scatter(xi, yi, color='#1F4B73', s=70, label=label)
        ax.plot([min_val, max_val], [min_val, max_val], color='red', label='Y=X')
        title_text = f'R2={zi["r2"]:.2f}, RMSE={zi["rmse"]:.2f}, MAE={zi["mae"]:.2f}'
        ax.set_title(rf'{title_text}', **font)  # LaTeX formatting
        ax.set_xlabel('True', **font)
        ax.set_ylabel('Predictions', **font)
        ax.set_xlim(min_val, max_val)
        ax.set_ylim(min_val, max_val)
        ax.legend(prop=legend_font)
        ax.grid(True)

    plt.tight_layout()

    if save_fig:
        plt.savefig(file_name, format='png')
        plt.close()

def util_data(y_data, max_len=10_000):
    y_data = y_data.T.tolist()[0]
    if len(y_data) < max_len:
        y_data += [6666.0 for i in range(max_len-len(y_data))]
    return y_data

def parse_arguments(arg_list=None):
 
    parser = argparse.ArgumentParser(description='SteelBERT Hyperparameter Optimization')

  
    parser.add_argument("--num_trial", type=int, default=20)
    parser.add_argument("--num_process", type=int, default=4)

    
    parser.add_argument("--model_name", type=str, default="SteelBERT")
    parser.add_argument("--target_property", type=str, default="Yield_value")
    parser.add_argument("--seed", type=int, default=301)
    parser.add_argument("--split_ratio", type=float, default=0.9)

    
    parser.add_argument("--norm", type=str, default='null_norm')
    parser.add_argument("--mix_layer", type=int, default=15)
    
    
    parser.add_argument("--dropout_list", type=str, default=None)
    default_simple = {'simple_layer_list':[512, 512, 512, 256], 'normalization':'null_norm', 'activation':'prelu', 'dropout_prob':0.0}
    default_seq_embed = {'seq_embed_con1d_list':[2, 2], 'kernel_size':2, 'stride':1, 'padding':1, 'seq_embed_fc_list':[32, 32, 32, 16, 16, 16, 16], 'normalization':'null_norm', 'activation':'prelu', 'dropout_prob':0.0}
    default_common_embed = {'seq_embed_con2d_list':[2, 2], 'kernel_size':2, 'stride':1, 'padding':2, 'seq_embed_2d_fc_list':[32, 32, 32, 16, 16, 16, 16], 'normalization':'null_norm', 'activation':'prelu', 'dropout_prob':0.0}
    default_concat = {'concat_layer_list':[64, 32, 8, 4, 1], 'normalization':'null_norm', 'activation':'prelu', 'dropout_prob':0.0}

    
    parser.add_argument("--simple_layer_dict", type=str, default=None)
    parser.add_argument("--seq_embed_layer_dict", type=str, default=None)
    parser.add_argument("--common_embed_layer_dict", type=str, default=None)
    parser.add_argument("--concat_layer_dict", type=str, default=None)

   
    parser.add_argument("--train_batch_list", type=str, default=None)
    parser.add_argument("--epoch_list", type=str, default=None)
    parser.add_argument("--lr_list", type=str, default=None)
    parser.add_argument("--step_list", type=str, default=None)
    parser.add_argument("--gamma_ratio_list", type=str, default=None)

   
    parser.add_argument("--embed_total_data_path", type=str, default='...')

    args = parser.parse_args(arg_list)

    def safe_json_parse(json_str, default_val):
        if json_str is None:
            return default_val
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"⚠️ Warning: Failed to parse JSON. Using default. Error: {e}")
            return default_val

    
    args.simple_layer_dict = safe_json_parse(args.simple_layer_dict, default_simple)
    args.seq_embed_layer_dict = safe_json_parse(args.seq_embed_layer_dict, default_seq_embed)
    args.common_embed_layer_dict = safe_json_parse(args.common_embed_layer_dict, default_common_embed)
    args.concat_layer_dict = safe_json_parse(args.concat_layer_dict, default_concat)
    
    args.train_batch_list = safe_json_parse(args.train_batch_list, [2, 4, 6, 8, 10])
    args.epoch_list = safe_json_parse(args.epoch_list, [150])
    args.lr_list = safe_json_parse(args.lr_list, [1e-4, 5e-5, 3e-5, 1e-5])
    args.step_list = safe_json_parse(args.step_list, [15, 20, 25])
    args.gamma_ratio_list = safe_json_parse(args.gamma_ratio_list, [0.9, 0.95])
    args.dropout_list = safe_json_parse(args.dropout_list, [0.0])

    
    s_dict = args.simple_layer_dict
    args.simple_layer_list = s_dict.get('simple_layer_list')
    args.simple_layer_drop_prob = s_dict.get('dropout_prob')
    
    seq_dict = args.seq_embed_layer_dict
    args.seq_embed_con1d_list = seq_dict.get('seq_embed_con1d_list')
    args.seq_embed_fc_list = seq_dict.get('seq_embed_fc_list')
    args.kernel_size = seq_dict.get('kernel_size')
    args.stride = seq_dict.get('stride')
    args.padding = seq_dict.get('padding')

    common_dict = args.common_embed_layer_dict
    args.seq_embed_con2d_list = common_dict.get('seq_embed_con2d_list')
    args.seq_embed_2d_fc_list = common_dict.get('seq_embed_2d_fc_list')

    c_dict = args.concat_layer_dict
    args.concat_layer_list = c_dict.get('concat_layer_list')
    args.concat_layer_drop_prob = c_dict.get('dropout_prob')
    
    return args


def creat_output_path_and_file():
    output_root_path = f"..."
    if not os.path.exists(output_root_path):
        os.makedirs(output_root_path, exist_ok=True)
    
    child_dir = ['reg_model_saved', 'preds', 'figs', 'runs']
    for d in child_dir:
        new_dir = os.path.join(output_root_path, d)
        if not os.path.exists(new_dir):
            os.makedirs(new_dir, exist_ok=True)
            
    outputs_cols = ['model_name', 'seed', 'prop', 'split_ratio', 'train_batch',
                    'epoch', 'lr', 'step', 'gamma_ratio', 'mix_layer', 'dropout',
                    'train_r2', 'test_r2',
                    'train_mae', 'test_mae',
                    'train_rmse', 'test_rmse', 'create_time']
    
    file_path = f"..."
    if not os.path.exists(file_path):
        with open(file_path, 'a', encoding='utf-8') as f:
            f.write(','.join(outputs_cols) + '\n')
    else:
        with open(file_path, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
        if first_line != ','.join(outputs_cols):
            with open(file_path, 'a', encoding='utf-8') as f:
                f.write(','.join(outputs_cols) + '\n')
                
    output_path = {
        'reg_csv': output_root_path,
        'reg_model_saved': os.path.join(output_root_path, 'reg_model_saved'),
        'preds': os.path.join(output_root_path, 'preds'),
        'figs': os.path.join(output_root_path, 'figs'),
        'runs': os.path.join(output_root_path, 'runs')
    }
    return output_path

def generate_parameter_dict(args):
    """
    修改为只接收一个 args 对象
    """
    config = {}
    config['model_name'] = args.model_name
    config['target_property'] = args.target_property
    config['seed'] = args.seed
    config['split_ratio'] = args.split_ratio
    config['norm'] = args.norm
    config['mix_layer'] = args.mix_layer
    
    config['simple_layer_list'] = args.simple_layer_list
    config["simple_layer_drop_prob"] = args.simple_layer_drop_prob
    config['seq_embed_con1d_list'] = args.seq_embed_con1d_list
    config['seq_embed_con2d_list'] = args.seq_embed_con2d_list
    config['seq_embed_fc_list'] = args.seq_embed_fc_list
    config['seq_embed_2d_fc_list'] = args.seq_embed_2d_fc_list
    config['concat_layer_list'] = args.concat_layer_list
    config["concat_layer_drop_prob"] = args.concat_layer_drop_prob
    
    config['train_batch'] = random.choice(args.train_batch_list)
    config['epoch'] = random.choice(args.epoch_list)
    config['lr'] = random.choice(args.lr_list)
    config['step'] = random.choice(args.step_list)
    config['gamma_ratio'] = random.choice(args.gamma_ratio_list)
    config['embed_total_data_path'] = args.embed_total_data_path
    
    dropout_probs = args.dropout_list
    
   
    selected_dropout = random.choice(dropout_probs)
    

    

    config["simple_layer_drop_prob"] = selected_dropout
    

    config["concat_layer_drop_prob"] = selected_dropout
    
    return config

def train_function(config, output_path):
    try:
 
        model_name = config['model_name']
        prop = config['target_property']
        seed = config['seed']
        split_ratio = config['split_ratio']
        train_batch = config['train_batch']
        epoch = config['epoch']
        lr = config['lr']
        step = config['step']
        gamma_ratio = config['gamma_ratio']
        embed_total_data_path = config['embed_total_data_path']
        mix_layer = config['mix_layer'] 
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
     
        norm = config['norm']
        

        simple_layer_dict = {
            'simple_layer_list': config['simple_layer_list'],
            'normalization': norm,
            'activation': 'prelu',
            'dropout_prob': config["simple_layer_drop_prob"]
        }
        
        seq_embed_layer_dict = {
            'seq_embed_con1d_list': config['seq_embed_con1d_list'],
            'kernel_size': 2,
            'stride': 1,
            'padding': 1,
            'seq_embed_fc_list': config['seq_embed_fc_list'],
            'normalization': norm,
            'activation': 'prelu',
            'dropout_prob': 0.0
        }
        
        common_embed_layer_dict = {
            'seq_embed_con2d_list': config['seq_embed_con2d_list'],
            'kernel_size': 2,
            'stride': 1,
            'padding': 2,
            'seq_embed_2d_fc_list': config['seq_embed_2d_fc_list'],
            'normalization': norm,
            'activation': 'prelu',
            'dropout_prob': 0.0
        }
        
        concat_layer_dict = {
            'concat_layer_list': config['concat_layer_list'],
            'normalization': norm,
            'activation': 'prelu',
            'dropout_prob': config["concat_layer_drop_prob"]
        }


        if not os.path.exists(embed_total_data_path):
            raise FileNotFoundError(f"❌：{embed_total_data_path}")
            
        model_train_data = pd.read_csv(embed_total_data_path, encoding="utf-8")
        

        drop_col = list(set(['Tensile_value', 'Yield_value', 'Elongation_value']) - set([prop]))
        existing_drop_cols = [c for c in drop_col if c in model_train_data.columns]
        if existing_drop_cols:
            model_train_data.drop(columns=existing_drop_cols, inplace=True)
            

        for i in range(10):
            model_train_data = model_train_data.sample(frac=1).reset_index(drop=True)
            

        shuffled = model_train_data.sample(frac=1, random_state=seed, ignore_index=True)
        split_idx = int(split_ratio * len(shuffled))
        train_data = shuffled.iloc[:split_idx].reset_index(drop=True)
        test_data = shuffled.iloc[split_idx:].reset_index(drop=True)
        
        
        

        train_dict = gen_data_class(train_data)
        test_dict = gen_data_class(test_data)
        

        train_dataset = CustomSimpleDataset(
            text_embeds=train_dict['text_embeds'],
            com_embeds=train_dict['com_embeds'],
            targets=train_dict['targets']
        )
        test_dataset = CustomSimpleDataset(
            text_embeds=test_dict['text_embeds'],
            com_embeds=test_dict['com_embeds'],
            targets=test_dict['targets']
        )


        train_dataloader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True, drop_last=False)
        batch_train_dataloader = DataLoader(train_dataset, batch_size=train_batch, shuffle=True, drop_last=False)
        test_dataloader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=True, drop_last=False)


        dt_string = datetime.now().strftime("_%Y.%m.%d.%H.%M.%S")
        

        current_dropout = config["simple_layer_drop_prob"]
        
        paras_li = [model_name, seed, prop, split_ratio, train_batch, epoch, lr, step, gamma_ratio, mix_layer, current_dropout]
        run_id = '_'.join([str(s) for s in paras_li]) + dt_string
        
        final_model_save_path = os.path.join(output_path['reg_model_saved'], f'{run_id}_ft_model.pt')
        runs_dir = os.path.join(output_path['runs'], f"run_{run_id}")
        
        
        reg_model = CustomSimpleModel(
            simple_layer_dict=simple_layer_dict,          
            seq_embed_layer_dict=seq_embed_layer_dict,    
            common_embed_layer_dict=common_embed_layer_dict, 
            concat_layer_dict=concat_layer_dict,         
            mix_layer=mix_layer,
            norm=norm
        ).to(device)

        loss_fn = nn.MSELoss()
        optimizer = torch.optim.AdamW(reg_model.parameters(), lr=lr, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step, gamma=gamma_ratio)
        
        writer = SummaryWriter(runs_dir, flush_secs=20)

        best_test_r2 = -1e5
        best_train_r2 = -1e5
        

        for epoch_num in tqdm(range(epoch), desc=f"Training {run_id[:20]}"):
            reg_model.train()
            total_train_loss = 0.0
            train_batch_loss = 0.0
            
            for batch, inputs in enumerate(batch_train_dataloader):
           
                for k in inputs:
                    if isinstance(inputs[k], torch.Tensor):
                        inputs[k] = inputs[k].to(device)
                        
                y = inputs['labels'].unsqueeze(1) # [Batch, 1]
                preds = reg_model(**inputs)       # [Batch, 1]
                
                train_loss = loss_fn(preds, y)
                
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()
                
                total_train_loss += train_loss.item()
                if batch == len(batch_train_dataloader) - 1:
                    train_batch_loss = train_loss.item()

            avg_train_loss = total_train_loss / max(len(batch_train_dataloader), 1)


            reg_model.eval()
            train_r2 = 0.0
            test_r2 = 0.0
            
            with torch.no_grad():

                r2_sum_train = 0; count_train = 0
                for inputs in train_dataloader:
                    for k in inputs:
                        if isinstance(inputs[k], torch.Tensor):
                            inputs[k] = inputs[k].to(device)
                    y = inputs['labels'].unsqueeze(1)
                    preds = reg_model(**inputs)
                    res = eval_model(y, preds)
                    r2_sum_train += res['r2']
                    count_train += 1
                train_r2 = r2_sum_train / count_train if count_train > 0 else 0


                r2_sum_test = 0; count_test = 0
                for inputs in test_dataloader:
                    for k in inputs:
                        if isinstance(inputs[k], torch.Tensor):
                            inputs[k] = inputs[k].to(device)
                    temp_y = inputs['labels'].unsqueeze(1)
                    temp_preds = reg_model(**inputs)
                    res = eval_model(temp_y, temp_preds)
                    r2_sum_test += res['r2']
                    count_test += 1
                test_r2 = r2_sum_test / count_test if count_test > 0 else 0


            writer.add_scalars(f'Reg_{dt_string}/Loss', {'train_loss': train_batch_loss}, epoch_num+1)
            writer.add_scalars(f'Reg_{dt_string}/R2', {'train_r2': train_r2, 'test_r2': test_r2}, epoch_num+1)


            if epoch_num == 0:
                best_test_r2 = test_r2
                torch.save(reg_model.state_dict(), final_model_save_path)

            
            if test_r2 > best_test_r2 and train_r2 > test_r2 > 0:
                best_test_r2 = test_r2
                torch.save(reg_model.state_dict(), final_model_save_path)

            if (epoch_num + 1) % 10 == 0 or epoch_num == epoch - 1:
                print(f"[PID {os.getpid()}] Epoch {epoch_num + 1}/{epoch}: Loss={avg_train_loss:.4f}, Train R2={train_r2:.4f}, Test R2={test_r2:.4f}")


        

        final_best_ft_model = CustomSimpleModel(
            simple_layer_dict=simple_layer_dict,
            seq_embed_layer_dict=seq_embed_layer_dict,
            common_embed_layer_dict=common_embed_layer_dict,
            concat_layer_dict=concat_layer_dict,
            mix_layer=mix_layer,
            norm=norm
        ).to(device)
        
        if os.path.exists(final_model_save_path):
            final_best_ft_model.load_state_dict(torch.load(final_model_save_path, map_location=device), strict=False)
        else:
            final_best_ft_model = reg_model

        final_best_ft_model.eval()
        
        y_train_all = []
        y_train_pred_all = []
        y_test_all = []
        y_test_pred_all = []
        
        with torch.no_grad():

            for inputs in train_dataloader:
                for k in inputs:
                    if isinstance(inputs[k], torch.Tensor):
                        inputs[k] = inputs[k].to(device)
                labels = inputs['labels'].unsqueeze(1).detach().cpu()
                preds = final_best_ft_model(**inputs).detach().cpu()
                y_train_all.append(labels)
                y_train_pred_all.append(preds)
            

            for inputs in test_dataloader:
                for k in inputs:
                    if isinstance(inputs[k], torch.Tensor):
                        inputs[k] = inputs[k].to(device)
                labels = inputs['labels'].unsqueeze(1).detach().cpu()
                preds = final_best_ft_model(**inputs).detach().cpu()
                y_test_all.append(labels)
                y_test_pred_all.append(preds)


        y_train = torch.cat(y_train_all, dim=0).numpy()
        y_train_preds = torch.cat(y_train_pred_all, dim=0).numpy()
        y_test = torch.cat(y_test_all, dim=0).numpy()
        y_test_preds = torch.cat(y_test_pred_all, dim=0).numpy()


        train_result = eval_model(torch.FloatTensor(y_train), torch.FloatTensor(y_train_preds))
        test_result = eval_model(torch.FloatTensor(y_test), torch.FloatTensor(y_test_preds))


        factor = 10 if prop in ['Yield_value', 'Tensile_value'] else 1
        y_train *= factor
        y_test *= factor
        y_train_preds *= factor
        y_test_preds *= factor


        plot_figure(
            [y_train, y_test],
            [y_train_preds, y_test_preds],
            [train_result, test_result],
            save_fig=True,
            file_name=os.path.join(output_path['figs'], f"fig_{run_id}.png")
        )


        min_len = min(len(y_train), len(y_train_preds), len(y_test), len(y_test_preds))
        plot_result = pd.DataFrame({
            'y_train': y_train[:min_len],
            'y_train_preds': y_train_preds[:min_len],
            'y_test': y_test[:min_len],
            'y_test_preds': y_test_preds[:min_len],
        })
        plot_result.to_csv(os.path.join(output_path['preds'], f"preds_{run_id}.csv"), index=None, float_format='%.4f')


        try:
            csv_file_path = "..."

            os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)
            

            current_dropout = config["simple_layer_drop_prob"]
            paras_li = [
                model_name, seed, prop, split_ratio, train_batch, 
                epoch, lr, step, gamma_ratio, mix_layer, current_dropout
            ]
            

            csv_row = paras_li + [
                train_result['r2'], test_result['r2'],
                train_result['mae'], test_result['mae'],
                train_result['rmse'], test_result['rmse'],
                dt_string
            ]


            with open(csv_file_path, 'a', newline='', encoding='utf-8') as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow(csv_row)
            

        except Exception as csv_err:
            print(f"❌ [PID {os.getpid()}]: {csv_err}")
            import traceback
            traceback.print_exc()

        writer.close()

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise e

def main(args=None):
    output_path = creat_output_path_and_file()
    
    all_args = parse_arguments(args)
    
    
    trials = [i for i in range(all_args.num_trial)]

    configs = [generate_parameter_dict(all_args) for _ in trials]
    

    debug_mode = False 
    
    if debug_mode or all_args.num_process == 1:

        for i in trials:
            try:
                train_function(configs[i], output_path)
            except Exception as e:

                break
    else:

        try:
            with Pool(processes=all_args.num_process) as pool:
                pool.starmap(train_function, [(configs[i], output_path) for i in trials])
        except Exception as e:

            import traceback
            traceback.print_exc()



if __name__ == "__main__":

    args = parse_arguments()
    print(f"{args.model_name}")
    print(f"{args.target_property}")
    print(f"{args.embed_total_data_path}")
    print("-" * 30)
    

    main()