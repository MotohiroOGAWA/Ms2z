import os
import yaml
import time
from tqdm import tqdm
from datetime import datetime
import torch
import numpy as np
import dill
import re
from collections import defaultdict

from model.torch_utils import load_dataset, get_optimizer, get_criterion, save_dataset, save_model, load_model
from model.data_pipeline import get_ds
from model.ms2z import Ms2z
from model.logger import TSVLogger
from model.gradient_logger import GradientLogger

def main(work_dir, files, load_name, load_epoch, load_iter, batch_size, epochs,
         train_size, val_size, test_size, device,
         model_info,
         optimizer_info={'name':'Adam', 'lr':0.01, 'eps':0.00000001},
         save_epoch=1, save_iter=None,
         ):
    
    os.makedirs(work_dir, exist_ok=True)

    if load_name == '' or load_name is None:
        load_name = 'ckp' + datetime.now().strftime("%Y%m%d%H%M%S")
        # load_name = 'ckp' + 'test1224'
    
    load_dir = os.path.join(work_dir, 'trains', load_name)
    if not os.path.exists(load_dir):
        os.makedirs(load_dir)
    print(f"Working directory: {load_dir}")
    
    input_file = os.path.join(work_dir, files['tree'])

    # Load the model and optimizer if load_epoch > 0
    if load_epoch is not None and (load_epoch > 0 or load_epoch == -1):
        print(f"Loading model from {load_dir}, epoch: {load_epoch}, iter: {load_iter}")

        dataset, train_dataloader, val_dataloader, test_dataloader, ds_extra_data = \
            load_dataset(
                load_dir, batch_size=batch_size, name=None, 
                load_dataset=True, load_train_loader=True, 
                load_val_dataloader=True, load_test_dataloader=False,
                extra_data_keys=['vocab']
                )

        vocab_data = ds_extra_data['vocab']
        create_model_from_config = Ms2z.from_config_param

        model, load_epoch, global_step, optimizer, optimizer_info = \
            load_model(
                create_model_from_config, load_dir, load_epoch, load_iter, device,
                extra_config_data={'vocab_data': vocab_data}
                )
        
        initial_epoch = load_epoch
        max_epoch = epochs + load_epoch
        token_pre_train_epoch = 0

    # Create a new model if load_epoch = 0
    else:
        vocab_file = os.path.join(work_dir, files['vocab'])
        vocab_data =  dill.load(open(vocab_file, 'rb'))

        token_tensor, order_tensor, mask_tensor, max_seq_len, vocab_size\
             = read_tensor_file(input_file)
        
        variables = {
            'token': token_tensor,
            'order': order_tensor,
            'mask': mask_tensor,
        }
        ds_extra_data = {
            'vocab': vocab_data
        }
        dataset, train_dataloader, val_dataloader, test_dataloader \
              = get_ds(variables, mode='train', batch_size=batch_size, 
                        train_size=train_size, val_size=val_size, test_size=test_size,
                       device=torch.device('cpu'))
        # if dataset_save_dir == '':
        #     dataset_save_dir = os.path.join(load_dir, 'ds')
        save_dataset(load_dir, dataset, train_dataloader, val_dataloader, test_dataloader, extra_data=ds_extra_data)

        load_epoch = 0
        load_path = None
        initial_epoch = 0
        max_epoch = epochs
        global_step = 0

        model = Ms2z(
            vocab_data=vocab_data,
            max_seq_len=max_seq_len,
            node_dim=model_info['node_dim'],
            edge_dim=model_info['edge_dim'],
            atom_layer_lstm_iterations=model_info['atom_layer_lstm_iterations'],
            chem_encoder_h_size=model_info['chem_encoder_h_size'],
            latent_dim=model_info['latent_dim'],
            memory_seq_len=model_info['memory_seq_len'],
            decoder_layers=model_info['decoder_layers'],
            decoder_heads=model_info['decoder_heads'],
            decoder_ff_dim=model_info['decoder_ff_dim'],
            decoder_dropout=model_info['decoder_dropout'],
            target_var=model_info['target_var'],
        ).to(device)

        # define optimizer
        optimizer = get_optimizer(model, optimizer_info)

    print(f"train size: {len(train_dataloader.dataset)}, val size: {len(val_dataloader.dataset)}")  
    model.train()

    val_loss_list = np.zeros([0])
    logger = TSVLogger(os.path.join(load_dir, 'logs.tsv'), extra_columns=['level']) 
    gradient_logger = GradientLogger(os.path.join(load_dir, 'gradients.pkl'), save_interval=100)
    gradient_logger = None

    model.disable_sequential()

    start_time = time.time()
    for epoch in range(initial_epoch, max_epoch):
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{max_epoch}")

        for batch in batch_iterator:
            token_tensor = batch['token'].to(device)
            order_tensor = batch['order'].to(device)
            mask_tensor = batch['mask'].to(device)
            # param_backup = {name: param.clone().detach() for name, param in model.named_parameters()}

            input_tensor = {
                'token': token_tensor,
                'order': order_tensor,
                'mask': mask_tensor,
            }
            target_tensor = {
                'token': token_tensor,
                'order': order_tensor,
                'mask': mask_tensor,
            }
            loss_list, acc_list, target_data = \
                model(input_tensor, target_tensor)
            loss = calc_loss(loss_list)


            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if gradient_logger is not None:
                gradient_logger.log(model, global_step+1, epoch+1)
            optimizer.step()
            optimizer.zero_grad()

            # for name, param in model.named_parameters():
            #     if not torch.equal(param.data, param_backup[name]):
            #         print(f"Parameter '{name}' was updated.")
            #     else:
            #         print(f"Parameter '{name}' was NOT updated.")

            global_step += 1

            loss_items = {key: f"{value['loss']:6.3f}" for key, value in target_data.items()}
            for key, value in target_data.items():
                if value['accuracy'] is not None:
                    loss_items[key] = f'{loss_items[key]}({value["accuracy"]:.3f})'
            batch_iterator.set_postfix(loss_items)

            current_time = time.time() - start_time
            logger.log(
                "train", 
                epoch=epoch+1, global_step=global_step,
                data_size=token_tensor.shape[0], target_data=target_data,
                learning_rate=optimizer.param_groups[0]['lr'],
                timestamp=current_time,
                # extra_columns={'level': level}
                )

            # Save the model and optimizer
            if save_iter is not None and global_step % save_iter == 0:
                save_path = save_model(
                    model, epoch+1, global_step, load_dir, optimizer, optimizer_info, 
                    epoch_zero_fill=len(str(max_epoch))+1, iter_zero_fill=len(str(int(max_epoch*len(dataset)/batch_size)))+1)
                print(f"Model and optimizer saved at {save_path}")
                
        # Validation
        val_loss, val_target_data = run_validation(model, val_dataloader, logger, global_step, epoch+1, optimizer=optimizer, timestamp=current_time)
        val_loss_list = np.append(val_loss_list, val_loss)
        val_loss_items = {key: f"{value['loss']:6.3f}" for key, value in val_target_data.items()}
        for key, value in val_target_data.items():
            if value['accuracy'] is not None:
                val_loss_items[key] = f'{val_loss_items[key]}({value["accuracy"]:.3f})'
        print(f"Validation Loss after epoch {epoch+1}: {val_loss_items} {val_loss}")
        
        # Save the model and optimizer
        if save_epoch is not None and (epoch+1) % save_epoch == 0:
            save_path = save_model(
                model, epoch+1, global_step, load_dir, optimizer, optimizer_info, 
                epoch_zero_fill=len(str(max_epoch))+1, iter_zero_fill=len(str(int(max_epoch*len(dataset)/batch_size)))+1)
            print(f"Model and optimizer saved at {save_path}")

    # Save the model and optimizer at the end of training
    if save_epoch is not None and (epoch+1) % save_epoch != 0:
        save_path = save_model(
            model, epoch+1, global_step, load_dir, optimizer, optimizer_info, 
            epoch_zero_fill=len(str(max_epoch))+1, iter_zero_fill=len(str(int(max_epoch*len(dataset)/batch_size)))+1)
        print(f"Model and optimizer saved at {save_path}")

def run_validation(model, val_dataloader, logger, global_step, epoch, optimizer, timestamp=None, extra_columns=None):
    model.eval()
    val_loss = 0
    losses = defaultdict(lambda: {'loss': 0.0, 'accuracy': 0.0, 'criterion': ''})
    total_samples = 0

    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc="Running Validation"):
            token_tensor = batch['token'].to(device)
            order_tensor = batch['order'].to(device)
            mask_tensor = batch['mask'].to(device)

            input_tensor = {
                'token': token_tensor,
                'order': order_tensor,
                'mask': mask_tensor,
            }
            target_tensor = {
                'token': token_tensor,
                'order': order_tensor,
                'mask': mask_tensor,
            }
            loss_list, acc_list, target_data = \
                model(input_tensor, target_tensor)
            loss = calc_loss(loss_list)

            # param_backup = {name: param.clone().detach() for name, param in model.named_parameters()}

            samples = token_tensor.shape[0]

            total_samples += samples
            val_loss += loss.item() * samples
            for key, value in target_data.items():
                losses[key]['loss'] += value['loss'] * samples
                if value['accuracy'] is not None:
                    losses[key]['accuracy'] += value['accuracy'] * samples
                else:
                    losses[key]['accuracy'] = None
                losses[key]['criterion'] = value['criterion']

    avg_val_loss = val_loss / total_samples
    target_data = losses.copy()
    for key, value in target_data.items():
        value['loss'] /= total_samples
        if value['accuracy'] is not None:
            value['accuracy'] /= total_samples
    
    logger.log(
        "validation", 
        epoch=epoch, global_step=global_step,
        data_size=token_tensor.shape[0], target_data=target_data,
        learning_rate=optimizer.param_groups[0]['lr'],
        timestamp=timestamp,
        extra_columns=extra_columns
        )

    model.train()
    return avg_val_loss, target_data

def calc_loss(loss_list):
    kl_loss = loss_list['KL']
    pred_motif_loss = loss_list['pred_motif']

    loss = 0.1 * kl_loss + pred_motif_loss

    return loss

def read_tensor_file(input_file):
    input_tensor = torch.load(input_file)

    token_tensor = input_tensor['token']
    order_tensor = input_tensor['order']
    mask_tensor = input_tensor['mask']
    max_seq_len = input_tensor['length']
    vocab_size = input_tensor['vocab_size']

    return token_tensor, order_tensor, mask_tensor, max_seq_len, vocab_size

def get_tensor_files(directory):
    """
    Retrieve all tensor files with numbers in their names from the specified directory.
    Args:
        directory (str): Path to the directory containing the tensor files.

    Returns:
        dict: A dictionary where keys are numbers and values are corresponding file names.
    """
    tensor_files = {}
    pattern = re.compile(r'tensor_level(\d+)\.pt')  # Regex to match 'tensor_level{number}.pt'

    for file_name in os.listdir(directory):
        match = pattern.match(file_name)
        if match:
            level = int(match.group(1))  # Extract the number
            tensor_files[level] = os.path.join(directory, file_name)

    return tensor_files

def default_param():
    config = {
    }


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='training Ms2z model')
    parser.add_argument("-w", "--work_dir", type = str, required = True, help = "Working directory")
    parser.add_argument("-p", "--param_path", type = str, default='', help = "Parameter file (.yaml)")
    # parser.add_argument("-dso", "--dataset_save_dir", type = str, default='', help = "Dataset save directory")
    
    args = parser.parse_args()

    with open(args.param_path, 'r') as f:
        config = yaml.safe_load(f)

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(config['train']['device'])

    main(
        args.work_dir, 
        config['file'], 
        config['train']['load_name'], 
        config['train']['load_epoch'],
        config['train']['load_iter'],
        config['train']['batch_size'],
        config['train']['epochs'],
        config['train']['train_size'],
        config['train']['val_size'],
        config['train']['test_size'],
        device, 
        model_info=config['model'],
        optimizer_info=config['train']['optimizer'],
        save_epoch=config['train']['save_epoch'],
        save_iter=config['train']['save_iter'],
        )

