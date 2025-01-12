import os
import torch
from torch.utils.data import DataLoader
import dill
from tqdm import tqdm
import numpy as np
import re

def get_optimizer(model: torch.nn.Module, param):
    if param['name'].lower() == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=param['lr'], momentum=param['momentum'])
    elif param['name'].lower() == 'adagrad':
        optimizer = torch.optim.Adagrad(model.parameters(), lr=param['lr'], eps=param['eps'])
    elif param['name'].lower() == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=param['lr'], momentum=param['momentum'], eps=param['eps'])
    elif param['name'].lower() == 'adadelta':
        optimizer = torch.optim.Adadelta(model.parameters(), lr=param['lr'], eps=param['eps'])
    elif param['name'].lower() == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=param['lr'], eps=param['eps'])
    elif param['name'].lower() == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=param['lr'], eps=param['eps'])
    else:
        print("Error: optimizer not recognized")
        quit()
    return optimizer

def get_criterion(name: str):
    name = name.lower()
    if name == 'crossentropy': # CrossEntropy
        return torch.nn.CrossEntropyLoss()
    elif name == 'bce': # Binary Cross Entropy
        return torch.nn.BCELoss()
    elif name == 'bcewithlogits': # Binary Cross Entropy with Logits
        return torch.nn.BCEWithLogitsLoss()
    elif name == 'mse': # Mean Squared Error
        return torch.nn.MSELoss()
    elif name == 'l1': # L1 Loss
        return torch.nn.L1Loss()
    elif name == 'smoothl1': # Smooth L1 Loss
        return torch.nn.SmoothL1Loss()
    elif name == 'hinge': # Hinge Loss
        return torch.nn.HingeEmbeddingLoss()
    elif name == 'kldiv': # KL Divergence
        return torch.nn.KLDivLoss()
    elif name == 'nll': # Negative Log Likelihood
        return torch.nn.NLLLoss()
    elif name == 'poissonnll': # Poisson Negative Log Likelihood
        return torch.nn.PoissonNLLLoss()
    elif name == 'cosineembedding': # Cosine Embedding
        return torch.nn.CosineEmbeddingLoss()
    elif name == 'huber': # Huber Loss
        return torch.nn.HuberLoss()
    elif name == 'multilabelmargin': # Multi Label Margin
        return torch.nn.MultiLabelMarginLoss()
    elif name == 'multilabelsoftmargin': # Multi Label Soft Margin
        return torch.nn.MultiLabelSoftMarginLoss()
    elif name == 'multimargin': # Multi Margin
        return torch.nn.MultiMarginLoss()
    elif name == 'marginranking': # Margin Ranking
        return torch.nn.MarginRankingLoss()
    elif name == 'ctc': # Connectionist Temporal Classification
        return torch.nn.CTCLoss()
    else:
        print("Error: loss function not recognized")
        quit()


def save_model(model, epoch, global_step, save_dir, optimizer = None, optimizer_info = None, epoch_zero_fill=4, iter_zero_fill=8):
    # Save the model and optimizer at the specified save_epoch
    epoch_str = str(epoch).zfill(epoch_zero_fill)
    iter_str = str(global_step).zfill(iter_zero_fill)

    save_path = os.path.join(save_dir, 'ckpt', f"model-e-{epoch_str}-i-{iter_str}.pt")
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    model_config = model.get_config_param()
    torch.save({
        'epoch': epoch,  # 現在のエポックを保存
        'model_config': model_config,  # モデルの設定を保存
        'model_state_dict': model.state_dict(),  # モデルの状態を保存
        'optimizer_state_dict': optimizer.state_dict(),  # オプティマイザの状態を保存
        'optimizer_info': optimizer_info,  # オプティマイザの設定を保存
        'global_step': global_step  # Tensorboardなどで使用するグローバルステップを保存
    }, save_path)
    return save_path

def load_model(func_create_model_form_config, load_dir, load_epoch, load_iter=None, device=None, extra_config_data=None):
    # Load the model and optimizer from the specified load_dir
    dirname = os.path.join(load_dir, 'ckpt')
    latent_iter = -1
    selected_file = None

    pattern = re.compile(rf"model-e-(\d+)-i-(\d+)\.pt")
    for filename in os.listdir(dirname):
        match = pattern.match(filename)
        if match:
            e_num = int(match.group(1))
            i_num = int(match.group(2))

            # 
            if load_epoch > 0 and (load_iter is not None and load_iter > 0):
                if e_num == load_epoch and i_num == load_iter:
                    selected_file = filename
                    latent_iter = i_num
                    break
            elif load_epoch > 0:
                if e_num == load_epoch and i_num > latent_iter:
                    selected_file = filename
                    latent_iter = i_num
            elif load_epoch == -1:
                if i_num > latent_iter:
                    selected_file = filename
                    latent_iter = i_num
            else:
                raise ValueError("load_epoch must be greater than 0 or -1")
    
    load_path = os.path.join(dirname, selected_file)

    checkpoint = torch.load(load_path)
    model_config = checkpoint['model_config']
    if extra_config_data is not None:
        model_config.update(extra_config_data)
    model = func_create_model_form_config(model_config)
    model.load_state_dict(checkpoint['model_state_dict'])

    if device is not None:
        model.to(device)
    if checkpoint['optimizer_info'] is not None and checkpoint['optimizer_state_dict'] is not None:
        optimizer_info = checkpoint['optimizer_info']
        optimizer = get_optimizer(model, optimizer_info)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    global_step = checkpoint['global_step']
    return model, epoch, global_step, optimizer, optimizer_info

def save_dataset(
        save_dir, dataset, train_loader, val_dataloader, test_dataloader, extra_data:dict=None, name=None):
    # Save the dataset to the specified save_dir
    if name is None:
        name = ''
    else:
        name = name + '_'
    save_dir = os.path.join(save_dir, "ds")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    dill.dump(dataset, open(os.path.join(save_dir, name + 'dataset.pkl'), 'wb'))
    dill.dump(train_loader, open(os.path.join(save_dir, name + 'train_loader.pkl'), 'wb'))
    dill.dump(val_dataloader, open(os.path.join(save_dir, name + 'val_loader.pkl'), 'wb'))
    dill.dump(test_dataloader, open(os.path.join(save_dir, name + 'test_loader.pkl'), 'wb'))
    if extra_data is not None:
        for key, value in extra_data.items():
            dill.dump(value, open(os.path.join(save_dir, name + key + '.pkl'), 'wb'))


def load_dataset(load_dir, batch_size=None, name=None, load_dataset=True, load_train_loader=True, load_val_dataloader=True, load_test_dataloader=True, extra_data_keys:list[str]=None):
    load_dir = os.path.join(load_dir, "ds")

    # Load the dataset from the specified save_dir
    if name is None:
        name = ''
    else:
        name = name + '_'
    if load_dataset:
        dataset = dill.load(open(os.path.join(load_dir, name + 'dataset.pkl'), 'rb'))
    else:
        dataset = None

    if load_train_loader:
        train_loader = dill.load(open(os.path.join(load_dir, name + 'train_loader.pkl'), 'rb'))
        if batch_size is not None:
            train_loader = DataLoader(train_loader.dataset, batch_size=batch_size, shuffle=True)
    else:
        train_loader = None

    if load_val_dataloader:
        val_dataloader = dill.load(open(os.path.join(load_dir, name + 'val_loader.pkl'), 'rb'))
        if batch_size is not None:
            val_dataloader = DataLoader(val_dataloader.dataset, batch_size=batch_size, shuffle=False)
    else:
        val_dataloader = None

    if load_test_dataloader:
        test_dataloader = dill.load(open(os.path.join(load_dir, name + 'test_loader.pkl'), 'rb'))
        if batch_size is not None:
            test_dataloader = DataLoader(test_dataloader.dataset, batch_size=batch_size, shuffle=False)
    else:
        test_dataloader = None

    extra_data = {}
    if extra_data_keys is not None:
        for key in extra_data_keys:
            extra_data[key] = dill.load(open(os.path.join(load_dir, name + key + '.pkl'), 'rb'))


    return dataset, train_loader, val_dataloader, test_dataloader, extra_data
