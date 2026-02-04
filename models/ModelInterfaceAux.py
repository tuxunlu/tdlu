import inspect
import torch
import importlib
import torch.optim.lr_scheduler as lrs
import pytorch_lightning as pl
from typing import Callable, Dict, Tuple
import matplotlib.pyplot as plt

from .loss.DICELoss import DICELoss
from .loss.FocalLoss import FocalLoss
from torchmetrics.classification import MulticlassF1Score, MulticlassAccuracy

class ModelInterfaceAux(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.model = self.__load_model()
        self.loss_fn_target = self.__configure_loss(prefix="")
        self.loss_fn_aux = self.__configure_loss(prefix="aux_")
        self.aux_loss_weight = float(getattr(self.hparams, "aux_loss_weight", 1.0))
        # Initialize a list to collect predicted labels from each training batch.
        self._train_labels = []

        # Num classes (allow separate counts)
        self.num_bins = int(getattr(self.hparams, "num_bins"))
        self.num_bins  = int(getattr(self.hparams, "num_bins"))

        # MAIN (TDLU) metrics
        self.train_f1_main = MulticlassF1Score(num_classes=self.num_bins, average='macro')
        self.val_f1_main   = MulticlassF1Score(num_classes=self.num_bins, average='macro')
        self.test_f1_main  = MulticlassF1Score(num_classes=self.num_bins, average='macro')

        self.train_acc_main = MulticlassAccuracy(num_classes=self.num_bins)
        self.val_acc_main   = MulticlassAccuracy(num_classes=self.num_bins)
        self.test_acc_main  = MulticlassAccuracy(num_classes=self.num_bins)

        # AUX (BD) metrics
        self.train_f1_aux = MulticlassF1Score(num_classes=self.num_bins, average='macro')
        self.val_f1_aux   = MulticlassF1Score(num_classes=self.num_bins, average='macro')
        self.test_f1_aux  = MulticlassF1Score(num_classes=self.num_bins, average='macro')

        self.train_acc_aux = MulticlassAccuracy(num_classes=self.num_bins)
        self.val_acc_aux   = MulticlassAccuracy(num_classes=self.num_bins)
        self.test_acc_aux  = MulticlassAccuracy(num_classes=self.num_bins)

        self.automatic_optimization = False

    def forward(self, x, *args, **kwargs):
        return self.model(x, *args, **kwargs)

    def training_step(self, batch, batch_idx):
        *train_input, target_label, aux_target_label, train_filenames = batch
        tdlu_logits, bd_logits, fused_feature = self(*train_input)

        # Compute loss using the configured loss function.
        loss_main = self.loss_fn_target(tdlu_logits, target_label, stage='train')
        loss_aux  = self.loss_fn_aux(bd_logits,  aux_target_label,  stage='train')
        train_loss = loss_main + self.aux_loss_weight * loss_aux

        # ---- manual opt: single forward, then backward+step ----
        opt_b, opt_h = self.optimizers()
        opt_b.zero_grad(set_to_none=True)
        opt_h.zero_grad(set_to_none=True)

        self.manual_backward(train_loss)
        self.clip_gradients(opt_b, gradient_clip_val=self.hparams.grad_clip_val, gradient_clip_algorithm="norm")
        self.clip_gradients(opt_h, gradient_clip_val=self.hparams.grad_clip_val, gradient_clip_algorithm="norm")

        opt_b.step()
        opt_h.step()

        # Get predicted class labels.
        pred_main = tdlu_logits.argmax(dim=1)
        pred_aux  = bd_logits.argmax(dim=1)

        self.train_f1_main.update(pred_main, target_label)
        self.train_acc_main.update(tdlu_logits, target_label)

        self.train_f1_aux.update(pred_aux, aux_target_label)
        self.train_acc_aux.update(bd_logits, aux_target_label)
    
        # Logging
        self.log('train_loss',      train_loss, on_step=False, on_epoch=True, prog_bar=True,  sync_dist=True)
        self.log('train_loss_main', loss_main,  on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log('train_loss_aux',  loss_aux,   on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)

        self.log('train_acc_main', self.train_acc_main, on_step=False, on_epoch=True, prog_bar=True,  sync_dist=True)
        self.log('train_f1_main',  self.train_f1_main,  on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)

        self.log('train_acc_aux',  self.train_acc_aux,  on_step=False, on_epoch=True, prog_bar=True,  sync_dist=True)
        self.log('train_f1_aux',   self.train_f1_aux,   on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)

        # For class distribution plot (keep using MAIN labels)
        self._train_labels.append(target_label.detach().cpu())

        return train_loss

    def on_train_epoch_end(self):
        # Step schedulers (epoch-based)
        sch_b, sch_h = self.lr_schedulers()
        sch_b.step(); sch_h.step()

        # Plot epoch class distribution for MAIN labels
        if self._train_labels:
            all_labels = torch.cat(self._train_labels)
            num_bins = self.num_bins if hasattr(self.hparams, "num_bins") else (all_labels.max().item() + 1)
            class_counts = torch.bincount(all_labels, minlength=num_bins)

            fig, ax = plt.subplots()
            ax.bar(range(num_bins), class_counts.numpy())
            ax.set_xlabel("Class"); ax.set_ylabel("Count")
            ax.set_title(f"Class Distribution (Main) - Epoch {self.current_epoch}")
            if self.logger is not None and hasattr(self.logger.experiment, "add_figure"):
                self.logger.experiment.add_figure("Train/Class_Distribution_Main", fig, global_step=self.current_epoch)
            plt.close(fig)

        self._train_labels = []

    def validation_step(self, batch, batch_idx):
        *val_input, target_label, aux_target_label, _ = batch
        tdlu_logits, bd_logits, _ = self(*val_input)

        with torch.no_grad():
            C_main = tdlu_logits.size(1)
            C_aux  = bd_logits.size(1)

            # dtype
            assert target_label.dtype == torch.long
            assert aux_target_label.dtype == torch.long

            # range + quick stats
            def check(name, y, C):
                u = torch.unique(y)
                msg = f"[{name}] C={C} uniques={u.tolist()} min={int(y.min())} max={int(y.max())}"
                assert (y.min() >= 0) and (y.max() < C), "OOB label! " + msg
                if self.global_rank == 0 and batch_idx == 0:
                    print(msg)

            check("MAIN", target_label, C_main)
            check("AUX",  aux_target_label, C_aux)


        loss_main = self.loss_fn_target(tdlu_logits, target_label, stage='validation')
        loss_aux  = self.loss_fn_aux(bd_logits,  aux_target_label,  stage='validation')
        val_loss = loss_main + loss_aux

        # Metrics (TorchMetrics supports logits for accuracy; F1 we compute at epoch end)
        self.val_acc_main.update(tdlu_logits, target_label)
        self.val_acc_aux.update(bd_logits, aux_target_label)

        self.val_f1_main.update(tdlu_logits.argmax(dim=1), target_label)
        self.val_f1_aux.update(bd_logits.argmax(dim=1), aux_target_label)

        self.log('val_loss',      val_loss,  on_step=False, on_epoch=True, prog_bar=True,  sync_dist=True)
        self.log('val_loss_main', loss_main, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log('val_loss_aux',  loss_aux,  on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        return val_loss

    def on_validation_epoch_end(self):
        f1_main  = self.val_f1_main.compute();  acc_main = self.val_acc_main.compute()
        f1_aux   = self.val_f1_aux.compute();   acc_aux  = self.val_acc_aux.compute()

        self.log('val_f1_main',  f1_main,  prog_bar=True,  sync_dist=True)
        self.log('val_acc_main', acc_main, prog_bar=True,  sync_dist=True)
        self.log('val_f1_aux',   f1_aux,   prog_bar=True,  sync_dist=True)
        self.log('val_acc_aux',  acc_aux,  prog_bar=True,  sync_dist=True)

        self.val_f1_main.reset();  self.val_acc_main.reset()
        self.val_f1_aux.reset();   self.val_acc_aux.reset()

    def test_step(self, batch, batch_idx):
        *test_input, target_label, aux_target_label, test_filenames = batch
        tdlu_logits, bd_logits, _ = self(*test_input)

        loss_main = self.loss_fn_target(tdlu_logits, target_label, stage='test')
        loss_aux  = self.loss_fn_aux(bd_logits,  aux_target_label,  stage='test')
        test_loss = loss_main + loss_aux

        pred_main = tdlu_logits.argmax(dim=1)
        pred_aux  = bd_logits.argmax(dim=1)

        print(f"[Batch {batch_idx}] MAIN preds: {pred_main[:10].tolist()}, labels: {target_label[:10].tolist()}")
        print(f"[Batch {batch_idx}]  AUX preds: {pred_aux[:10].tolist()},  labels: {aux_target_label[:10].tolist()}")

        self.test_acc_main.update(tdlu_logits, target_label)
        self.test_f1_main.update(pred_main, target_label)

        self.test_acc_aux.update(bd_logits, aux_target_label)
        self.test_f1_aux.update(pred_aux, aux_target_label)

        self.log('test_loss',      test_loss,  on_step=False, on_epoch=True, prog_bar=True)
        self.log('test_loss_main', loss_main,  on_step=False, on_epoch=True, prog_bar=False)
        self.log('test_loss_aux',  loss_aux,   on_step=False, on_epoch=True, prog_bar=False)

        self.log('test_acc_main', self.test_acc_main, on_step=False, on_epoch=True, prog_bar=True,  sync_dist=True)
        self.log('test_f1_main',  self.test_f1_main,  on_step=False, on_epoch=True, prog_bar=False)

        self.log('test_acc_aux',  self.test_acc_aux,  on_step=False, on_epoch=True, prog_bar=True,  sync_dist=True)
        self.log('test_f1_aux',   self.test_f1_aux,   on_step=False, on_epoch=True, prog_bar=False)

        return test_loss

    def configure_optimizers(self):
        optimizer_backbone = torch.optim.AdamW(
            self.model.backbone.parameters(),
            lr=float(self.hparams.backbone_lr),
            weight_decay=float(self.hparams.weight_decay)
        )
        optimizer_head = torch.optim.AdamW(
            list(self.model.tdlu_density_head.parameters()) + list(self.model.breast_density_head.parameters()),
            lr=float(self.hparams.head_lr),
            weight_decay=float(self.hparams.weight_decay)
        )

        # scheduler_backbone = lrs.CosineAnnealingLR(
        #     optimizer_backbone,
        #     T_max=self.hparams.lr_decay_epochs,
        #     eta_min=self.hparams.lr_decay_min_lr
        # )
        # scheduler_head = lrs.StepLR(
        #     optimizer_head,
        #     step_size=self.hparams.lr_decay_epochs,
        #     gamma=self.hparams.lr_decay_rate
        # )

        scheduler_backbone = lrs.CyclicLR(
            optimizer_backbone,
            base_lr=self.hparams.backbone_lr_decay_min_lr,
            max_lr=self.hparams.backbone_lr,
            step_size_up=self.hparams.backbone_lr_decay_epochs // 2,
            mode='triangular2'
        )

        scheduler_head = lrs.CyclicLR(
            optimizer_head,
            base_lr=self.hparams.head_lr_decay_min_lr,
            max_lr=self.hparams.head_lr,
            step_size_up=self.hparams.head_lr_decay_epochs // 2,
            mode='triangular2'
        )

        return (
            [optimizer_backbone, optimizer_head],
            [
                {"scheduler": scheduler_backbone, "interval": "epoch", "name": "lr_backbone"},
                {"scheduler": scheduler_head,     "interval": "epoch", "name": "lr_head"},
            ],
        )

    def __calculate_loss_and_log(self, inputs, labels,
                                 loss_dict: Dict[str, Tuple[float, Callable]],
                                 stage: str, name_space: str):
        """
        Apply a stack of losses to (inputs, labels). Logs each raw component as:
        {stage}_{name_space}_{loss_name}
        """
        raw_loss_list = [func(inputs, labels) for _, func in loss_dict.values()]
        weighted_loss = [weight * raw for (weight, _), raw in zip(loss_dict.values(), raw_loss_list)]

        for loss_name, raw in zip(loss_dict.keys(), raw_loss_list):
            self.log(f'{stage}_{name_space}_{loss_name}', raw.item(),
                     on_step=False, on_epoch=True, prog_bar=False)

        return sum(weighted_loss)

    def __configure_loss(self, prefix: str):
        """
        Build a composite loss function. If prefixed keys are present in hparams,
        e.g. target_loss / target_loss_weight, use them; otherwise fall back to
        loss / loss_weight.
        """
        names_key  = f'{prefix}_loss'
        weight_key = f'{prefix}_loss_weight'

        config_loss_names  = getattr(self.hparams, names_key,
                                     getattr(self.hparams, 'loss'))
        config_loss_weight = getattr(self.hparams, weight_key,
                                     getattr(self.hparams, 'loss_weight'))

        if isinstance(config_loss_names, str):
            config_loss_names = [config_loss_names]
        if isinstance(config_loss_weight, (int, float)):
            config_loss_weight = [config_loss_weight]

        # Build loss functions list
        config_loss_funcs = []
        for name in config_loss_names:
            if name == 'FocalLoss':
                config_loss_funcs.append(
                    FocalLoss(alpha=self.hparams.focal_loss_alpha, gamma=self.hparams.focal_loss_gamma)
                )
            elif hasattr(importlib.import_module('torch.nn'), name):
                config_loss_funcs.append(getattr(importlib.import_module('torch.nn'), name)())
            else:
                # Allow custom losses by fully-qualified import path, e.g. "mymod.MyLoss"
                mod, cls = name.rsplit('.', 1)
                config_loss_funcs.append(getattr(importlib.import_module(mod), cls)())

        assert len(config_loss_funcs) == len(config_loss_weight) == len(config_loss_names), \
            "Loss function count and weight/name count mismatch!"

        config_loss_dict = {
            loss_name: (loss_weight, loss_func)
            for loss_name, loss_weight, loss_func in zip(config_loss_names, config_loss_weight, config_loss_funcs)
        }

        def loss_func(inputs, labels, stage):
            return self.__calculate_loss_and_log(
                inputs=inputs,
                labels=labels,
                loss_dict=config_loss_dict,
                stage=stage,
                name_space=prefix if prefix else "main"  # default tag if no prefix
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
