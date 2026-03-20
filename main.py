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


class UnfreezeBackboneCallback(plc.Callback):
    """Unfreeze backbone at a given epoch for staged fine-tuning."""

    def on_train_epoch_start(self, trainer, pl_module):
        unfreeze_epoch = getattr(pl_module.hparams, "unfreeze_backbone_epoch", None)
        if unfreeze_epoch is None:
            return
        if trainer.current_epoch == unfreeze_epoch:
            model = getattr(pl_module, "model", pl_module)
            if hasattr(model, "_unfreeze_backbone"):
                model._unfreeze_backbone()


def load_callbacks(config):
    callbacks = []

    # Unfreeze backbone callback (staged fine-tuning).
    if config.get("unfreeze_backbone_epoch") is not None:
        callbacks.append(UnfreezeBackboneCallback())

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
    """
    Returns (checkpoint_directory, checkpoint_file_path, version).
    checkpoint_directory: experiment folder (e.g. runs/20260308-.../experiment_name)
    version: int or str to pass to TensorBoardLogger to reuse same folder (e.g. 0 for version_0)
    """
    resume_from_manual_checkpoint = config.get('resume_from_manual_checkpoint', None)
    resume_from_last_checkpoint = config.get('resume_from_last_checkpoint', None)
    checkpoint_directory = None
    checkpoint_file_path = None
    version = None

    if resume_from_manual_checkpoint:
        checkpoint_file_path = resume_from_manual_checkpoint
        # Path: .../experiment_name/version_0/checkpoints/file.ckpt
        path_parts = os.path.normpath(resume_from_manual_checkpoint).split(os.sep)
        for p in path_parts:
            if p.startswith('version_'):
                version = int(p.replace('version_', ''))
                break
        # checkpoint_directory = experiment folder (parent of version_0)
        checkpoint_directory = os.path.dirname(os.path.dirname(os.path.dirname(checkpoint_file_path)))
    elif resume_from_last_checkpoint:
        current_path = config['log_dir']
        log_dirs = os.listdir(current_path)
        log_dirs.sort(reverse=True)
        if len(log_dirs) == 0 or len(os.listdir(os.path.join(current_path, log_dirs[0]))) == 0:
            print(f"Warning: resume_from_last_checkpoint is True, but no checkpoint found at: {current_path}. Launching new training...")
            return None, None, None

        checkpoint_directory = os.path.join(current_path, log_dirs[0])
        version_dirs = os.listdir(checkpoint_directory)
        version_dirs.sort(reverse=True)
        version_folder = version_dirs[0]
        if version_folder.startswith('version_'):
            version = int(version_folder.replace('version_', ''))
        checkpoint_file_path = os.path.join(checkpoint_directory, version_folder, 'checkpoints')
        if not any(s.startswith('latest') and s.endswith('.ckpt') for s in os.listdir(checkpoint_file_path)):
            print(f"Warning: resume_from_last_checkpoint is True but no checkpoint file found at: {checkpoint_file_path}. Launching new training...")
            return None, None, None

        ckpt_files = sorted(
            filter(lambda s: s.startswith('latest') and s.endswith('.ckpt'), os.listdir(checkpoint_file_path)),
            reverse=True
        )
        checkpoint_file_path = os.path.join(checkpoint_file_path, ckpt_files[0])

    return checkpoint_directory, checkpoint_file_path, version


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
    checkpoint_directory, checkpoint_file_path, log_version = (None, None, None)
    if config.get('enable_checkpointing', False):
        checkpoint_directory, checkpoint_file_path, log_version = get_checkpoint_path(config)

    # Create a logger. If resuming, reuse the same version folder (no version_1, version_2, ...).
    if checkpoint_directory is not None:
        print(f"Resuming from checkpoint: {checkpoint_directory}, file: {checkpoint_file_path}")
        logger_kwargs = dict(save_dir='.', name=checkpoint_directory)
        if log_version is not None:
            logger_kwargs['version'] = log_version
        logger = TensorBoardLogger(**logger_kwargs)
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
    # Use "auto" for single GPU; "ddp" with devices=1 can cause unnecessary overhead
    trainer = Trainer(accelerator="gpu", **trainer_kwargs)

    trainer.fit(model=model_module, datamodule=data_module, ckpt_path=checkpoint_file_path)

    if len(data_module.test_set) > 0:
        trainer.test(ckpt_path='best')
    else:
        print("Skipping test (test set is empty).")


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
        action='store_true',
        help='Automatically search for and resume from the latest checkpoint in log_dir'
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
    config_dict = {k.lower(): v for k, v in config_dict.items()}

    if args.resume_from_manual_checkpoint is not None:
        config_dict['resume_from_manual_checkpoint'] = args.resume_from_manual_checkpoint
    if args.resume_from_last_checkpoint:
        config_dict['resume_from_last_checkpoint'] = True

    cross_val_fold = args.cross_val_fold if args.cross_val_fold is not None else config_dict.get('cross_val_fold')
    if cross_val_fold is not None:
        config_dict['cross_val_fold'] = cross_val_fold
        exp_name = config_dict.get('experiment_name', 'experiment')
        # Replace {fold} placeholder or append _fold{N}
        if '{fold}' in exp_name:
            exp_name = exp_name.replace('{fold}', str(cross_val_fold))
        else:
            exp_name = f"{exp_name}_fold{cross_val_fold}"
        config_dict['experiment_name'] = exp_name

    main(config_dict)
