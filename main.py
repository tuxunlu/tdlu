import os
import datetime
import yaml
import torch
import pytorch_lightning as pl
import pytorch_lightning.callbacks as plc
import inspect
from argparse import ArgumentParser
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

from models import ModelInterface, ModelInterfaceAux, ModelInterfaceAuxSaliency
from data import DataInterface


def load_callbacks(config):
    callbacks = []

    # Checkpointing callbacks.
    if config.get('enable_checkpointing', False):
        # Save best checkpoint (monitoring validation accuracy).
        callbacks.append(plc.ModelCheckpoint(
            monitor='val_f1',
            mode='max',
            save_top_k=1,
            filename='best-{epoch:03d}-{val_f1:.5f}-{val_acc:.5f}',
            verbose=True,
        ))
        # Save epoch checkpoint (latest model for each epoch).
        callbacks.append(plc.ModelCheckpoint(
            save_last=True,
            filename='last-{epoch:03d}-{val_f1:.5f}-{val_acc:.5f}',
        ))
        # Save all checkpoints.
        callbacks.append(plc.ModelCheckpoint(
            every_n_epochs=50,
            save_top_k=-1,
            save_on_train_epoch_end=True,
            filename='epoch-{epoch:03d}-{val_f1:.5f}-{val_acc:.5f}',
        ))
    
    # Learning rate monitor.
    callbacks.append(plc.LearningRateMonitor(
        logging_interval='epoch'
    ))
    
    return callbacks


def get_checkpoint_path(config):
    # Determine whether to resume from a manual checkpoint or automatically resume
    resume_from_manual_checkpoint = config.get('resume_from_manual_checkpoint', None)
    resume_from_last_checkpoint = config.get('resume_from_last_checkpoint', None)
    checkpoint_directory = None
    checkpoint_file_path = None

    if resume_from_manual_checkpoint:
        checkpoint_file_path = resume_from_manual_checkpoint
        truncated_path = resume_from_manual_checkpoint
        # Remove the last two path components to set the logging directory.
        for _ in range(2):
            truncated_path = truncated_path[:truncated_path.rfind(os.path.sep)]
        checkpoint_directory = os.path.dirname(truncated_path)
    elif resume_from_last_checkpoint:
        # Look into the log_dir and pick the latest log folder
        current_path = config['log_dir']
        log_dirs = os.listdir(current_path)
        log_dirs.sort(reverse=True)
        if len(log_dirs) == 0 or len(os.listdir(os.path.join(current_path, log_dirs[0]))) == 0:
            print(f"Warning: resume_from_last_checkpoint is True, but no checkpoint found at: {current_path}. Launching new training...")
            return None, None
        
        checkpoint_directory = os.path.join(current_path, log_dirs[0])
        version_dirs = os.listdir(checkpoint_directory)
        version_dirs.sort(reverse=True)
        checkpoint_file_path = os.path.join(checkpoint_directory, version_dirs[0], 'checkpoints')
        if not any(s.startswith('latest') and s.endswith('.ckpt') for s in os.listdir(checkpoint_file_path)):
            print(f"Warning: resume_from_last_checkpoint is True but no checkpoint file found at: {checkpoint_file_path}. Launching new training...")
            return None, None

        ckpt_files = sorted(
            filter(lambda s: s.startswith('latest') and s.endswith('.ckpt'), os.listdir(checkpoint_file_path)),
            reverse=True
        )
        checkpoint_file_path = os.path.join(checkpoint_file_path, ckpt_files[0])
    
    return checkpoint_directory, checkpoint_file_path


def main(config):
    # Set random seed for reproducibility.
    pl.seed_everything(config['seed'])
    
    # Instantiate the data module and Lightning model.
    data_module = DataInterface(**config)

    if config['model_interface'] == "ModelInterface":
        model_module = ModelInterface(**config)
    elif config['model_interface'] == "ModelInterfaceAux":
        model_module = ModelInterfaceAux(**config)
    elif config['model_interface'] == "ModelInterfaceAuxSaliency":
        model_module = ModelInterfaceAuxSaliency(**config)
    
    # Determine whether to resume from a checkpoint.
    checkpoint_directory, checkpoint_file_path = (None, None)
    if config.get('enable_checkpointing', False):
        checkpoint_directory, checkpoint_file_path = get_checkpoint_path(config)
    
    # Create a logger. If resuming from a checkpoint, reuse the same logger directory.
    if checkpoint_directory is not None:
        print(f"Resuming from checkpoint: {checkpoint_directory}, file: {checkpoint_file_path}")
        logger = TensorBoardLogger(save_dir='.', name=checkpoint_directory)
    else:
        print("Training from scratch...")
        log_dir_name_with_time = os.path.join(config['log_dir'], datetime.datetime.now().strftime("%Y%m%d-%H-%M-%S"))
        logger = TensorBoardLogger(save_dir='.', name=f"{log_dir_name_with_time}-{config['experiment_name']}")
    config['logger'] = logger

    # Load callbacks.
    config['callbacks'] = load_callbacks(config)
    
    # Filter Trainer keyword arguments from the config using inspect.
    signature = inspect.signature(Trainer.__init__)
    trainer_kwargs = {}
    for key in signature.parameters:
        if key in config:
            trainer_kwargs[key] = config[key]
    # Ensure our logger and callbacks are included.
    trainer_kwargs['logger'] = logger
    trainer_kwargs['callbacks'] = config['callbacks']
    trainer_kwargs['log_every_n_steps'] = config['log_every_n_steps']
    
    # Instantiate the Trainer.
    trainer = Trainer(accelerator="gpu", devices=1, strategy="ddp", **trainer_kwargs)

    trainer.fit(model=model_module, datamodule=data_module, ckpt_path=checkpoint_file_path)

    trainer.test(ckpt_path='best')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        '--config_path',
        default=os.path.join(os.getcwd(), 'config', 'config.yaml'),
        type=str,
        help='Path of config file'
    )
    parser.add_argument(
        '--resume_from_last_checkpoint',
        default=None,
        type=bool,
        help='Automatically search for and resume from the latest checkpoint'
    )
    parser.add_argument(
        '--resume_from_manual_checkpoint',
        default=None,
        type=str,
        help='Path to a checkpoint file (.ckpt) to resume training from'
    )

    parser.add_argument(
        '--cross_val_fold',
        type=int,
        default=None,
        help='Cross-validation fold number'
    )

    args = parser.parse_args()
    if not os.path.exists(args.config_path):
        raise FileNotFoundError(f'No config file found at {args.config_path}!')
    
    # Load configuration from YAML and normalize keys.
    with open(args.config_path) as f:
        config_dict = yaml.safe_load(f)
    # config_dict['resume_from_manual_checkpoint'] = args.resume_from_manual_checkpoint
    # config_dict['resume_from_last_checkpoint'] = args.resume_from_last_checkpoint
    # Convert all keys to lowercase to ensure consistency.
    config_dict = {k.lower(): v for k, v in config_dict.items()}

    if args.cross_val_fold is not None:
        config_dict['cross_val_fold'] = args.cross_val_fold
        # ensure experiment_name exists before using it
        exp_name = config_dict.get('experiment_name')
        exp_name = exp_name.replace('fold', f'fold{args.cross_val_fold}')
        config_dict['experiment_name'] = exp_name

    main(config_dict)
