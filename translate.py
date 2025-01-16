import os

from model.torch_utils import load_dataset, load_model
from model.ms2z import Ms2z


def load_ms2z_model(work_dir, load_name, load_epoch, load_iter, device):
    load_dir = os.path.join(work_dir, 'projects', load_name)
    if not os.path.exists(load_dir):
        raise FileNotFoundError(f"Directory {load_dir} does not exist.")
    print(f"Working directory: {load_dir}")

    # Load the model and optimizer if load_epoch > 0
    if load_epoch is not None and (load_epoch > 0 or load_epoch == -1):
        print(f"Loading model from {load_dir}, epoch: {load_epoch}, iter: {load_iter}")

        _, _, _, _, ds_extra_data = \
            load_dataset(
                load_dir, batch_size=1, name=None, 
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
        
        return model, load_epoch, global_step, vocab_data
    else:
        raise ValueError("load_epoch must be set to a positive integer.")

