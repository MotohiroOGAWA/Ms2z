import os
import yaml
import time
from tqdm import tqdm
from datetime import datetime
import torch
import numpy as np
import dill

from lib.torch_utils import load_dataset, get_optimizer, get_criterion, save_dataset, save_model, load_model
from lib.data_pipeline import get_ds
from lib.ms2z import Ms2z
from lib.logger import TSVLogger

def main(device, work_dir, files, load_name, load_epoch, load_iter, batch_size, epochs,
         train_size, val_size, test_size,
         model_info,
         optimizer_info={'name':'Adam', 'lr':0.01, 'eps':0.00000001},
         criterion_name='MSELoss',
         dataset_save_dir = '',
         save_epoch=1, save_iter=None,
         ):
    
    os.makedirs(work_dir, exist_ok=True)

    if load_name == '' or load_name is None:
        # load_name = 'ckp' + datetime.now().strftime("%Y%m%d%H%M%S")
        load_name = 'ckp' + 'test1217'
    
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
        target_file = files['fp']

        input_tensor = torch.load(input_file)
        target_tensor = torch.load(target_file)

        vocab_tensor = input_tensor['vocab']
        order_tensor = input_tensor['order']
        mask_tensor = input_tensor['mask']
        max_seq_len = input_tensor['length']
        vocab_size = input_tensor['vocab_size']
        
        variables = {
            'vocab': vocab_tensor,
            'order': order_tensor,
            'mask': mask_tensor,
            'fp': target_tensor
        }
        dataset, train_dataloader, val_dataloader, test_dataloader \
              = get_ds(variables, mode='train', batch_size=batch_size, 
                        train_size=train_size, val_size=val_size, test_size=test_size,
                       device=device)
        if dataset_save_dir == '':
            dataset_save_dir = os.path.join(load_dir, 'ds')
        save_dataset(load_dir, dataset, train_dataloader, val_dataloader, test_dataloader)

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
        ).to(device)

        # define optimizer
        optimizer = get_optimizer(model, optimizer_info)

    # define loss function (criterion)
    criterion = get_criterion(criterion_name)


    print(f"train size: {len(train_dataloader.dataset)}, val size: {len(val_dataloader.dataset)}")  
    model.train()

    val_loss_list = np.zeros([0])
    logger = TSVLogger(os.path.join(load_dir, 'logs.tsv')) 

    start_time = time.time()
    for epoch in range(initial_epoch, max_epoch):
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{max_epoch}")
        for batch in batch_iterator:
            tree_tensor = batch['vocab'].to(device)
            order_tensor = batch['order'].to(device)
            mask_tensor = batch['mask'].to(device)
            # fp_tensor = batch['fp'].to(device)

            # param_backup = {name: param.clone().detach() for name, param in model.named_parameters()}

            token_loss, bp_loss, kl_divergence_loss = \
                model(tree_tensor, order_tensor, mask_tensor)
            loss = token_loss + 0.5 * bp_loss + kl_divergence_loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # for name, param in model.named_parameters():
            #     if not torch.equal(param.data, param_backup[name]):
            #         print(f"Parameter '{name}' was updated.")
            #     else:
            #         print(f"Parameter '{name}' was NOT updated.")

            global_step += 1
    
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}", 'token': f"{token_loss.item():6.3f}", 'bp': f"{bp_loss.item():6.3f}", 'KL': f"{kl_divergence_loss.item():6.3f}"})

            current_time = time.time() - start_time
            logger.log(
                "train", 
                epoch=epoch+1, global_step=global_step, 
                loss_type=criterion_name, target_name='TotalLoss', loss_value=loss.item(), 
                data_size=tree_tensor.shape[0], accuracy=None, 
                learning_rate=optimizer.param_groups[0]['lr'], timestamp=current_time)
            logger.log(
                "train", 
                epoch=epoch+1, global_step=global_step, 
                loss_type=criterion_name, target_name='TokenLoss', loss_value=token_loss.item(), 
                data_size=tree_tensor.shape[0], accuracy=None, 
                learning_rate=optimizer.param_groups[0]['lr'], timestamp=current_time)
            logger.log(
                "train", 
                epoch=epoch+1, global_step=global_step, 
                loss_type=criterion_name, target_name='BondPosLoss', loss_value=bp_loss.item(), 
                data_size=tree_tensor.shape[0], accuracy=None, 
                learning_rate=optimizer.param_groups[0]['lr'], timestamp=current_time)
            logger.log(
                "train", 
                epoch=epoch+1, global_step=global_step, 
                loss_type=criterion_name, target_name='KL_divergence_loss', loss_value=kl_divergence_loss.item(), 
                data_size=tree_tensor.shape[0], accuracy=None, 
                learning_rate=optimizer.param_groups[0]['lr'], timestamp=current_time)

            # Save the model and optimizer
            if save_iter is not None and global_step % save_iter == 0:
                save_path = save_model(
                    model, epoch+1, global_step, load_dir, optimizer, optimizer_info, 
                    epoch_zero_fill=len(str(max_epoch))+1, iter_zero_fill=len(str(int(max_epoch*len(dataset)/batch_size)))+1)
                print(f"Model and optimizer saved at {save_path}")
                
        # Validation
        val_loss = run_validation(model, val_dataloader, criterion, logger, global_step, epoch+1, timestamp=current_time)
        val_loss_list = np.append(val_loss_list, val_loss)
        print(f"Validation Loss after epoch {epoch+1}: {val_loss}")
        
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

def run_validation(model, val_dataloader, criterion, logger, global_step, epoch, timestamp=None):
    model.eval()
    val_loss = 0
    val_token_loss = 0
    val_bp_loss = 0
    val_kl_loss = 0
    total_samples = 0

    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc="Running Validation"):
            tree_tensor = batch['vocab'].to(device)
            order_tensor = batch['order'].to(device)
            mask_tensor = batch['mask'].to(device)
            # fp_tensor = batch['fp'].to(device)

            token_mismatch_loss, bp_loss, kl_divergence_loss = \
                model(tree_tensor, order_tensor, mask_tensor)
            
            loss = token_mismatch_loss + 0.5 * bp_loss + kl_divergence_loss
            samples = tree_tensor.shape[0]

            total_samples += samples
            val_loss += loss.item() * samples
            val_token_loss += token_mismatch_loss.item() * samples
            val_bp_loss += bp_loss.item() * samples
            val_kl_loss += kl_divergence_loss.item() * samples


    avg_val_loss = val_loss / total_samples
    avg_val_token_loss = val_token_loss / total_samples
    avg_val_bp_loss = val_bp_loss / total_samples
    avg_val_kl_loss = val_kl_loss / total_samples

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
    model.train()
    return avg_val_loss




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
        criterion_name=config['train']['criterion'],
        dataset_save_dir = args.dataset_save_dir,
        save_epoch=config['train']['save_epoch'],
        save_iter=config['train']['save_iter'],
        )

