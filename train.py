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

def main(device, work_dir, files, load_name, load_epoch, load_iter, batch_size, epochs,
         train_size, val_size, test_size,
         model_info,
         optimizer_info={'name':'Adam', 'lr':0.01, 'eps':0.00000001},
         dataset_save_dir = '',
         save_epoch=1, save_iter=None,
         ):
    
    os.makedirs(work_dir, exist_ok=True)

    if load_name == '' or load_name is None:
        # load_name = 'ckp' + datetime.now().strftime("%Y%m%d%H%M%S")
        load_name = 'ckp' + 'test1224'
    
    load_dir = os.path.join(work_dir, 'projects', load_name)
    if not os.path.exists(load_dir):
        os.makedirs(load_dir)

    # Load the model and optimizer if load_epoch > 0
    if load_epoch is not None and (load_epoch > 0 or load_epoch == -1):
        dataset, train_dataloader, val_dataloader, _ = \
            load_dataset(load_dir, batch_size=batch_size, name=None, load_dataset=True, load_train_loader=True, load_val_dataloader=True, load_test_dataloader=False)
    
        create_model_from_config = Ms2z.from_config_param

        model, load_epoch, global_step, optimizer, optimizer_info = \
            load_model(create_model_from_config, load_dir, load_epoch, load_iter, device)
        
        initial_epoch = load_epoch
        max_epoch = epochs + load_epoch

    # Create a new model if load_epoch = 0
    else:
        input_file = files['tree']

        token_tensor, order_tensor, mask_tensor, max_seq_len, \
            vocab_size, fingerprint_tensor, fp_dim = read_tensor_file(input_file)
        
        variables = {
            'token': token_tensor,
            'order': order_tensor,
            'mask': mask_tensor,
            'fp': fingerprint_tensor
        }
        dataset, train_dataloader, val_dataloader, test_dataloader \
              = get_ds(variables, mode='train', batch_size=batch_size, 
                        train_size=train_size, val_size=val_size, test_size=test_size,
                       device=torch.device('cpu'))
        # if dataset_save_dir == '':
        #     dataset_save_dir = os.path.join(load_dir, 'ds')
        save_dataset(load_dir, dataset, train_dataloader, val_dataloader, test_dataloader)
        train_dataloader_by_level = {}
        train_dataloader_by_level[-1] = [dataset, train_dataloader, val_dataloader, test_dataloader]

        load_epoch = 0
        load_path = None
        initial_epoch = 0
        max_epoch = epochs
        global_step = 0

        vocab_file = files['vocab_file']
        vocab_data =  dill.load(open(vocab_file, 'rb'))
        model = Ms2z(
            vocab_data=vocab_data,
            max_seq_len=max_seq_len,
            node_dim=model_info['node_dim'],
            edge_dim=model_info['edge_dim'],
            latent_dim=model_info['latent_dim'],
            decoder_layers=model_info['decoder_layers'],
            decoder_heads=model_info['decoder_heads'],
            decoder_ff_dim=model_info['decoder_ff_dim'],
            decoder_dropout=model_info['decoder_dropout'],
            fp_dim=fp_dim,
        ).to(device)

        # define optimizer
        optimizer = get_optimizer(model, optimizer_info)

    # define loss function (criterion)
    # criterion = get_criterion(criterion_name)


    print(f"train size: {len(train_dataloader.dataset)}, val size: {len(val_dataloader.dataset)}")  
    model.train()

    val_loss_list = np.zeros([0])
    logger = TSVLogger(os.path.join(load_dir, 'logs.tsv')) 
    gradient_logger = GradientLogger(os.path.join(load_dir, 'gradients.pkl'), save_interval=100)


    level = 0
    input_files_by_level = get_tensor_files(os.path.dirname(input_file))
    start_time = time.time()
    for epoch in range(initial_epoch, max_epoch):
        model.train()
        if level not in train_dataloader_by_level:
            token_tensor, order_tensor, mask_tensor, max_seq_len, \
                vocab_size, fingerprint_tensor, fp_dim = read_tensor_file(input_files_by_level[level])
        
            _variables = {
                'token': token_tensor,
                'order': order_tensor,
                'mask': mask_tensor,
                'fp': fingerprint_tensor
            }
            _dataset, _train_dataloader, _val_dataloader, _test_dataloader \
              = get_ds(_variables, mode='train', batch_size=batch_size, 
                        train_size=train_size/(train_size+val_size), 
                        val_size=val_size/(train_size+val_size), 
                        test_size=None,
                       device=torch.device('cpu'))
            train_dataloader_by_level[level] = [_dataset, _train_dataloader, _val_dataloader, _test_dataloader]

        _train_dataloader = train_dataloader_by_level[level][1]
        batch_iterator = tqdm(_train_dataloader, desc=f"Epoch(lv.{level}) {epoch+1}/{max_epoch}")

        for batch in batch_iterator:
            token_tensor = batch['token'].to(device)
            order_tensor = batch['order'].to(device)
            mask_tensor = batch['mask'].to(device)
            fp_tensor = batch['fp'].to(device)

            # param_backup = {name: param.clone().detach() for name, param in model.named_parameters()}

            token_loss, token_sim_loss, kl_divergence_loss, fp_loss, = \
                model(token_tensor, order_tensor, mask_tensor, fp_tensor)
            
            loss = fp_loss + kl_divergence_loss
            # loss = fp_loss + token_sim_loss + kl_divergence_loss


            loss.backward()
            gradient_logger.log(model, global_step+1, epoch+1)
            optimizer.step()
            optimizer.zero_grad()

            # for name, param in model.named_parameters():
            #     if not torch.equal(param.data, param_backup[name]):
            #         print(f"Parameter '{name}' was updated.")
            #     else:
            #         print(f"Parameter '{name}' was NOT updated.")

            global_step += 1
    
            batch_iterator.set_postfix({
                "loss": f"{loss.item():6.3f}", 'token': f"{token_loss.item():6.3f}", 
                'ts': f"{token_sim_loss.item():6.3f}", 'KL': f"{kl_divergence_loss.item():6.3f}", 
                'FP': f"{fp_loss.item():6.3f}",
                })

            current_time = time.time() - start_time
            logger.log(
                "train", 
                epoch=epoch+1, global_step=global_step, 
                loss_type='', target_name='TotalLoss', loss_value=loss.item(), 
                data_size=token_tensor.shape[0], accuracy=None, 
                learning_rate=optimizer.param_groups[0]['lr'], timestamp=current_time)
            logger.log(
                "train", 
                epoch=epoch+1, global_step=global_step, 
                loss_type='', target_name='TokenLoss', loss_value=token_loss.item(), 
                data_size=token_tensor.shape[0], accuracy=None, 
                learning_rate=optimizer.param_groups[0]['lr'], timestamp=current_time)
            logger.log(
                "train", 
                epoch=epoch+1, global_step=global_step, 
                loss_type='', target_name='TokenSimLoss', loss_value=token_sim_loss.item(), 
                data_size=token_tensor.shape[0], accuracy=None, 
                learning_rate=optimizer.param_groups[0]['lr'], timestamp=current_time)
            logger.log(
                "train", 
                epoch=epoch+1, global_step=global_step, 
                loss_type='', target_name='KL_divergence_loss', loss_value=kl_divergence_loss.item(), 
                data_size=token_tensor.shape[0], accuracy=None, 
                learning_rate=optimizer.param_groups[0]['lr'], timestamp=current_time)
            logger.log(
                "train", 
                epoch=epoch+1, global_step=global_step, 
                loss_type='BCELoss', target_name='fingerprint', loss_value=fp_loss.item(), 
                data_size=token_tensor.shape[0], accuracy=None, 
                learning_rate=optimizer.param_groups[0]['lr'], timestamp=current_time)

            # Save the model and optimizer
            if save_iter is not None and global_step % save_iter == 0:
                save_path = save_model(
                    model, epoch+1, global_step, load_dir, optimizer, optimizer_info, 
                    epoch_zero_fill=len(str(max_epoch))+1, iter_zero_fill=len(str(int(max_epoch*len(dataset)/batch_size)))+1)
                print(f"Model and optimizer saved at {save_path}")
                
        # Validation
        # val_loss = run_validation(model, _val_dataloader, logger, global_step, epoch+1, timestamp=current_time)
        # val_loss_list = np.append(val_loss_list, val_loss)
        # print(f"Validation Loss after epoch {epoch+1}: {val_loss}")
        
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

def run_validation(model, val_dataloader, logger, global_step, epoch, timestamp=None):
    model.eval()
    val_loss = 0
    val_token_loss = 0
    val_bp_loss = 0
    val_kl_loss = 0
    val_fp_loss = 0
    val_atom_counter_loss = 0
    val_inner_bond_counter_loss = 0
    val_outer_bond_cnt_loss = 0
    total_samples = 0

    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc="Running Validation"):
            token_tensor = batch['token'].to(device)
            order_tensor = batch['order'].to(device)
            mask_tensor = batch['mask'].to(device)
            fp_tensor = batch['fp'].to(device)

            token_loss, token_sim_loss, kl_divergence_loss, fp_loss, = \
                model(token_tensor, order_tensor, mask_tensor, fp_tensor)
            
            loss = token_mismatch_loss + 0.5 * bp_loss + kl_divergence_loss + fp_loss + atom_counter_loss + inner_bond_counter_loss + outer_bond_cnt_loss
            samples = token_tensor.shape[0]

            total_samples += samples
            val_loss += loss.item() * samples
            val_token_loss += token_mismatch_loss.item() * samples
            val_bp_loss += bp_loss.item() * samples
            val_kl_loss += kl_divergence_loss.item() * samples
            val_fp_loss += fp_loss.item() * samples
            val_atom_counter_loss += atom_counter_loss.item() * samples
            val_inner_bond_counter_loss += inner_bond_counter_loss.item() * samples
            val_outer_bond_cnt_loss += outer_bond_cnt_loss.item() * samples


    avg_val_loss = val_loss / total_samples
    avg_val_token_loss = val_token_loss / total_samples
    avg_val_bp_loss = val_bp_loss / total_samples
    avg_val_kl_loss = val_kl_loss / total_samples
    avg_val_fp_loss = val_fp_loss / total_samples
    avg_val_atom_counter_loss = val_atom_counter_loss / total_samples
    avg_val_inner_bond_counter_loss = val_inner_bond_counter_loss / total_samples
    avg_val_outer_bond_cnt_loss = val_outer_bond_cnt_loss / total_samples

    logger.log(
        "validation", 
        epoch=epoch, global_step=global_step, 
        loss_type="", target_name='ValLoss', loss_value=avg_val_loss, 
        data_size=total_samples, accuracy=None, 
        learning_rate=None, timestamp=timestamp)
    logger.log(
        "validation", 
        epoch=epoch, global_step=global_step, 
        loss_type="", target_name='TokenLoss', loss_value=avg_val_token_loss, 
        data_size=total_samples, accuracy=None, 
        learning_rate=None, timestamp=timestamp)
    logger.log(
        "validation", 
        epoch=epoch, global_step=global_step, 
        loss_type="", target_name='BondPosLoss', loss_value=avg_val_bp_loss, 
        data_size=total_samples, accuracy=None, 
        learning_rate=None, timestamp=timestamp)
    logger.log(
        "validation", 
        epoch=epoch, global_step=global_step, 
        loss_type="", target_name='KL_divergence_loss', loss_value=avg_val_kl_loss, 
        data_size=total_samples, accuracy=None, 
        learning_rate=None, timestamp=timestamp)
    logger.log(
        "validation", 
        epoch=epoch, global_step=global_step, 
        loss_type='BCELoss', target_name='fingerprint', loss_value=avg_val_fp_loss, 
        data_size=total_samples, accuracy=None, 
        learning_rate=None, timestamp=timestamp)
    logger.log(
        "validation", 
        epoch=epoch, global_step=global_step, 
        loss_type='MSELoss', target_name='atom_counter', loss_value=avg_val_atom_counter_loss, 
        data_size=total_samples, accuracy=None, 
        learning_rate=None, timestamp=timestamp)
    logger.log(
        "validation", 
        epoch=epoch, global_step=global_step, 
        loss_type='MSELoss', target_name='inner_bond_counter', loss_value=avg_val_inner_bond_counter_loss, 
        data_size=total_samples, accuracy=None, 
        learning_rate=None, timestamp=timestamp)
    logger.log(
        "validation", 
        epoch=epoch, global_step=global_step, 
        loss_type='MSELoss', target_name='outer_bond_cnt', loss_value=avg_val_outer_bond_cnt_loss, 
        data_size=total_samples, accuracy=None, 
        learning_rate=None, timestamp=timestamp)
    model.train()
    return avg_val_loss


def read_tensor_file(input_file):
    input_tensor = torch.load(input_file)

    token_tensor = input_tensor['vocab']
    order_tensor = input_tensor['order']
    mask_tensor = input_tensor['mask']
    max_seq_len = input_tensor['length']
    vocab_size = input_tensor['vocab_size']
    fingerprint_tensor = input_tensor['fingerprints']
    fp_dim = input_tensor['fp_size']

    return token_tensor, order_tensor, mask_tensor, max_seq_len, vocab_size, fingerprint_tensor, fp_dim

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
    parser.add_argument("-dso", "--dataset_save_dir", type = str, default='', help = "Dataset save directory")
    
    args = parser.parse_args()

    with open(args.param_path, 'r') as f:
        config = yaml.safe_load(f)

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(config['train']['device'])

    main(
        device, 
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
        model_info=config['model'],
        optimizer_info=config['train']['optimizer'],
        dataset_save_dir = args.dataset_save_dir,
        save_epoch=config['train']['save_epoch'],
        save_iter=config['train']['save_iter'],
        )

