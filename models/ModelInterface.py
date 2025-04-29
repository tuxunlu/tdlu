import inspect
import torch
import importlib
import torch.optim.lr_scheduler as lrs
import pytorch_lightning as pl
from typing import Callable, Dict, Tuple
import matplotlib.pyplot as plt

class ModelInterface(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.model = self.__load_model()
        self.loss_function = self.__configure_loss()
        # Initialize a list to collect predicted labels from each training batch.
        self._train_labels = []

    def forward(self, x, *args, **kwargs):
        return self.model(x, *args, **kwargs)

    def training_step(self, batch, batch_idx):
        # Expect batch to be: (train_input, *other_inputs, train_labels)
        train_input, *rest, train_labels = batch
        train_out = self(train_input, *rest)
        # Assume model returns logits (possibly with additional outputs)
        train_logits, *rest = train_out
        # Compute loss using the configured loss function.
        train_loss = self.loss_function(train_logits, train_labels, 'train')

        # Get predicted class labels.
        train_label = train_labels.argmax(dim=1)
        out_label = train_logits.argmax(dim=1)
        correct_num = torch.sum(train_label == out_label).float()
        batch_acc = correct_num / out_label.size(0)

        # Also log loss in a way that aggregates over the epoch if desired.
        self.log('train_loss', train_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', batch_acc, on_step=True, on_epoch=True, prog_bar=True)

        # Save the batch predictions for epoch-level aggregation.
        self._train_labels.append(train_label.detach().cpu())

        # Return the loss (Lightning uses this for optimization).
        return train_loss

    def on_train_epoch_end(self):
        """
        This hook aggregates information from all training steps of the epoch.
        It logs the overall training loss (if not already logged), computes the class distribution
        from batch predictions, logs a bar plot, and logs the current learning rate.
        """
        # Aggregate all training labels from the epoch.
        if self._train_labels:
            all_labels = torch.cat(self._train_labels)
            # Determine the number of bins from hyperparameters or infer from the data.
            num_bins = self.hparams.num_bins if hasattr(self.hparams, "num_bins") else (all_labels.max().item() + 1)
            class_counts = torch.bincount(all_labels, minlength=num_bins)

            # Create a bar plot for the class distribution.
            fig, ax = plt.subplots()
            ax.bar(range(num_bins), class_counts.numpy())
            ax.set_xlabel("Class")
            ax.set_ylabel("Count")
            ax.set_title(f"Class Distribution in Epoch {self.current_epoch}")
            # Log the figure to the logger (if supported).
            if self.logger is not None and hasattr(self.logger.experiment, "add_figure"):
                self.logger.experiment.add_figure("Class_Distribution", fig, global_step=self.current_epoch)
            plt.close(fig)
        
        # Clear the list for the next epoch.
        self._train_labels = []

    def validation_step(self, batch, batch_idx):
        val_input, *rest, val_labels = batch
        val_out = self(val_input, *rest)
        val_logits, *rest = val_out
        val_loss = self.loss_function(val_logits, val_labels, 'validation')

        val_label = val_labels.argmax(dim=1)
        out_label = val_logits.argmax(dim=1)
        correct_num = torch.sum(val_label == out_label).float()
        batch_acc = correct_num / out_label.size(0)

        self.log('val_loss', val_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val_acc', batch_acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        return val_loss

    def test_step(self, batch, batch_idx):
        test_input, *rest, test_labels = batch
        test_out = self(test_input, *rest)
        test_logits, *rest = test_out
        test_loss = self.loss_function(test_logits, test_labels, 'test')

        test_label = test_labels.argmax(dim=1)
        out_label = test_logits.argmax(dim=1)

        # Print sample predictions for debugging.
        print(f"Batch {batch_idx} Predictions: {out_label[:10].tolist()}, Labels: {test_label[:10].tolist()}")
        

        correct_num = torch.sum(test_label == out_label).float()
        batch_acc = correct_num / out_label.size(0)

        self.log('test_loss', test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test_acc', batch_acc, on_step=False, on_epoch=True, prog_bar=True)

        return test_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            params=self.model.parameters(),
            lr=float(self.hparams.lr),
            weight_decay=float(self.hparams.weight_decay)
        )

        # No learning rate scheduler provided.
        if self.hparams.lr_scheduler is None:
            return [optimizer]

        if self.hparams.lr_scheduler == 'step':
            scheduler = lrs.StepLR(
                optimizer,
                step_size=self.hparams.lr_decay_epochs,
                gamma=self.hparams.lr_decay_rate
            )
        elif self.hparams.lr_scheduler == 'cosine':
            scheduler = lrs.CosineAnnealingLR(
                optimizer,
                T_max=self.hparams.lr_decay_epochs,
                eta_min=self.hparams.lr_decay_min_lr
            )
        elif self.hparams.lr_scheduler == 'cosine_restart':
            scheduler = lrs.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=self.hparams.lr_decay_epochs,
                T_mult=1,
                eta_min=self.hparams.lr_decay_min_lr
            )
        else:
            raise ValueError('Invalid lr_scheduler type!')
        return [optimizer], [scheduler]

    def __calculate_loss_and_log(self, inputs, labels, loss_dict: Dict[str, Tuple[float, Callable]], stage: str):
        raw_loss_list = [func(inputs, labels) for _, func in loss_dict.values()]
        weighted_loss = [weight * raw_loss for (weight, _), raw_loss in zip(loss_dict.values(), raw_loss_list)]
        for name, raw_loss in zip(loss_dict.keys(), raw_loss_list):
            self.log(f'{stage}_{name}', raw_loss.item(), on_step=False, on_epoch=True, prog_bar=False)

        return sum(weighted_loss)

    def __configure_loss(self):
        # Configure loss functions based on hyperparameters.
        config_loss_weight = self.hparams.loss_weight
        config_loss_names = self.hparams.loss
        # config_label_smoothing = self.hparams.label_smoothing

        config_loss_funcs = [getattr(importlib.import_module('torch.nn'), name)() for name in self.hparams.loss]

        assert (len(config_loss_funcs) == len(config_loss_weight)
                and len(config_loss_funcs) == len(config_loss_names)
               ), "Loss function count and weight/name count mismatch!"

        config_loss_dict = {
            loss_name: (loss_weight, loss_func)
            for loss_name, loss_weight, loss_func in zip(config_loss_names, config_loss_weight, config_loss_funcs)
        }

        # Optional: add user-defined loss functions if needed.
        user_loss_dict = {}

        loss_dict = {**config_loss_dict, **user_loss_dict}

        def loss_func(inputs, labels, stage):
            return self.__calculate_loss_and_log(
                inputs=inputs,
                labels=labels,
                loss_dict=loss_dict,
                stage=stage
            )

        return loss_func

    def __load_model(self):
        name = self.hparams.model_class_name
        # Attempt to import the `CamelCase` class name from the `snake_case.py` module. The module should be placed
        # within the same folder as model_interface.py. Always name your model file name as `snake_case.py` and
        # model class name as corresponding `CamelCase`.
        camel_name = ''.join([i.capitalize() for i in name.split('_')])
        try:
            model_class = getattr(importlib.import_module('.' + name, package=__package__), camel_name)
        except Exception:
            raise ValueError(f'Invalid Module File Name or Invalid Class Name {name}.{camel_name}!')
        model = self.__instantiate(model_class)
        if self.hparams.use_compile:
            torch.compile(model)
        return model

    def __instantiate(self, model_class, **other_args):
        target_args = inspect.getfullargspec(model_class.__init__).args[1:]
        this_args = self.hparams.keys()
        merged_args = {}
        for arg in target_args:
            if arg in this_args:
                merged_args[arg] = getattr(self.hparams, arg)
        merged_args.update(other_args)
        return model_class(**merged_args)
