import os
import datetime
import yaml
import pytorch_lightning as pl
import pytorch_lightning.callbacks as plc
import inspect
from argparse import ArgumentParser
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

from models import ModelInterface
from data import DataInterface


def load_callbacks(config):
    callbacks = []
    # # Early stopping: monitor validation accuracy and stop training if it stops improving.
    # callbacks.append(plc.EarlyStopping(
    #     monitor='validation_CrossEntropyLoss',
    #     mode='max',
    #     patience=10,
    #     min_delta=0.001
    # ))
    
    # Checkpointing callbacks.
    if config.get('enable_checkpointing', False):
        # Save best checkpoint (monitoring validation accuracy).
        callbacks.append(plc.ModelCheckpoint(
            monitor='val_acc',
            mode='max',
            save_top_k=1,
            filename='best-{epoch:03d}-{val_acc:.5f}',
            verbose=True,
        ))
        # Save epoch checkpoint (latest model for each epoch).
        callbacks.append(plc.ModelCheckpoint(
            save_last=True,
            filename='last-{epoch:03d}-{val_acc:.5f}',
        ))
        # Save all checkpoints.
        callbacks.append(plc.ModelCheckpoint(
            every_n_epochs=10,
            save_top_k=-1,
            save_on_train_epoch_end=True,
            filename='epoch-{epoch:03d}-{val_acc:.5f}',
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
    
    model = ModelInterface(**config)
    model.load_from_checkpoint("/fs/nexus-scratch/tuxunlu/git/tdlu/runs/20250424-21-15-05-smoothed_metafuse_mirai_freeze_2bins/version_0/checkpoints/best-epoch=492-val_acc=0.93868.ckpt")
    model.eval()
    
    # Load single image and normalize
    import torch
    from torchvision import transforms
    from PIL import Image

    # ------------------------------
    # 1. Specify your image path:
    # ------------------------------
    img_path = "/fs/nexus-scratch/tuxunlu/git/tdlu/dataset/WUSTL_png_nomarker_16/10003-2.png"

    # ------------------------------
    # 2. Build the same transforms
    #    you used during training:
    # ------------------------------
    mean = (0.111, 0.111, 0.111)
    std  = (0.185, 0.185, 0.185)

    preprocess = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),                    # or use your ToTensor16RGB()
        transforms.Normalize(mean, std),
    ])

    # ------------------------------
    # 3. Load & preprocess:
    # ------------------------------
    img = Image.open(img_path).convert("RGB")
    img_t = preprocess(img)                      # [C,H,W]
    img_t = img_t.unsqueeze(0).to(model.device)  # [1,C,H,W]

    # ------------------------------
    # 4. Forward pass & decode:
    # ------------------------------
    with torch.no_grad():
        out = model(img_t)   # adjust if your model returns multiple heads

    # Suppose `out` is logits over bins/classes:
    probs = torch.softmax(out, dim=1)            # [1, num_bins]
    pred_bin = probs.argmax(dim=1).item()

    print(f"Predicted bin: {pred_bin}, probs: {probs.squeeze().cpu().numpy()}")

    





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
    
    args = parser.parse_args()
    if not os.path.exists(args.config_path):
        raise FileNotFoundError(f'No config file found at {args.config_path}!')
    
    # Load configuration from YAML and normalize keys.
    with open(args.config_path) as f:
        config_dict = yaml.safe_load(f)
    config_dict['resume_from_manual_checkpoint'] = args.resume_from_manual_checkpoint
    config_dict['resume_from_last_checkpoint'] = args.resume_from_last_checkpoint
    # Convert all keys to lowercase to ensure consistency.
    config_dict = {k.lower(): v for k, v in config_dict.items()}
    
    main(config_dict)
