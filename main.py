import os
import argparse
from torch.utils.tensorboard import SummaryWriter
from utils.utils import get_run_number
from config.config import load_config
from dataset.TDLUDataset import TDLUDataset
from utils.model_factory import get_model_class
from utils.loss_factory import get_loss_function
from utils.trainer import train_and_evaluate

def main(config_file):
    config = load_config(config_file)
    model_config = config["model"]
    pretrained_path = model_config.get("pretrained_path", None)
    loss_config = config["loss"]

    hyperparameter_config = config["hyperparameters"]
    batch_size = hyperparameter_config["batch_size"]
    num_epochs = hyperparameter_config["num_epochs"]
    learning_rate = hyperparameter_config["learning_rate"]
    num_save = hyperparameter_config["num_save"]
    num_bins = hyperparameter_config["num_bins"]
    target = hyperparameter_config["target"]
    description = hyperparameter_config["description"]
    run_dir = hyperparameter_config["run_dir"]
    num_workers = hyperparameter_config["num_workers"]
    pin_memory = hyperparameter_config["pin_memory"]
    train_split = hyperparameter_config["train_split"]
    image_dir = hyperparameter_config["image_dir"]
    csv_path = hyperparameter_config["csv_path"]
    augment = hyperparameter_config["augment"]
    weights_json_path = hyperparameter_config["weights_json_path"]
    device_config = hyperparameter_config["device"]

    run_dir = os.path.join("runs", f"mg_experiment_{get_run_number()}_{description}_{target}_{num_bins}")
    os.makedirs(run_dir, exist_ok=True)
    writer = SummaryWriter(run_dir)

    # Dataset
    dataset = TDLUDataset(
        image_dir=image_dir,
        csv_path=csv_path,
        augment=augment,
        weights_json_path=weights_json_path,
        target=target,
        num_bins=num_bins,
    )
    train_dataloader, test_dataloader = dataset.get_dataloaders(
        batch_size=batch_size, train_split=train_split,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    # Model and Loss
    ModelClass = get_model_class(model_config["name"])
    model = ModelClass(num_bins=num_bins, pretrained_path=pretrained_path)
    criterion = get_loss_function(loss_config["name"])

    # Train and evaluate
    train_and_evaluate(
        model, train_dataloader, test_dataloader, criterion, writer, run_dir,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        num_save=num_save,
        num_bins=num_bins
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MGModule with configurable settings")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file (e.g., config1.yaml)")
    args = parser.parse_args()
    main(args.config)