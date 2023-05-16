import torch
from torch.optim import AdamW, Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, CyclicLR
from torch.nn.functional import (
    mse_loss,
    l1_loss,
    binary_cross_entropy_with_logits as bcn_loss,
)
from pytorch_lightning import LightningModule
from torchmdnet.models.model import create_model, load_model
from torchmdnet.utils import *
from torchmetrics import AUROC
import pandas as pd
import csv
import pickle
import numpy as np
import os


class LNNP(LightningModule):
    def __init__(self, hparams, data=None):
        super(LNNP, self).__init__()

        self.save_hyperparameters(hparams)

        self.tox_regression = self.hparams.train_type == "regression"
        if self.hparams.single_task_id is not None:
            assert (
                self.hparams.output_channels_toxicity == 1
            ), "Single task ID specified, but output_channels_toxicity > 1!"
            print(
                f"\nSingle task instead of multi-task training. Train on task with id {self.hparams.single_task_id}\n"
            )
        max_len_smiles = {
            "herg": 188,
            "ames": 175,
            "dili": 174,
            "ld50": 174,
            "skin_reaction": 98,
            "carcinogen": 292,
            "tox21": 325,
            "tox21_water": 342,
            "toxcast": 325,
            "sider": 413,
            "bbbp": 257,
            "clintox": 323,
            "bace": 194,
        }
        self.energies = {
            "herg": 5276.96435546875,
            "ames": 4932.81103515625,
            "dili": 8024.70556640625,
            "ld50": 7543.57666015625,
            "skin_reaction": 4704.2861328125,
            "carcinogen": 292,
            "tox21": 376.7106628417969,
            "tox21_water": 9314.572265625,
            "toxcast": 376.7106628417969,
            "sider": 429.65301513671875,
            "bbbp": 312.854736328125,
            "clintox": 379.4134521484375,
            "bace": 297.8172912597656,
            "mutagenicity": 3952.46923828125,
        }
        # assert self.hparams.max_len_smiles == max_len_smiles[self.hparams.dataset_arg['dataset']]

        if data is not None:
            self.data = data
            self.data.save_hyperparameters(hparams)

        if self.hparams.load_model:
            self.model = load_model(self.hparams.load_model, args=self.hparams)
        else:
            self.model = create_model(self.hparams)

        # initialize exponential smoothing
        self.ema = None
        self._reset_ema_dict()

        # initialize loss collection
        self.losses = None
        self._reset_losses_dict()

    def configure_optimizers(self):
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
            amsgrad=True,
        )
        if self.hparams.lr_schedule == "cosine":
            scheduler = CosineAnnealingLR(optimizer, self.hparams.lr_cosine_length)
            lr_scheduler = {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            }
        elif self.hparams.lr_schedule == "reduce_on_plateau":
            scheduler = ReduceLROnPlateau(
                optimizer,
                "min",
                factor=self.hparams.lr_factor,
                patience=self.hparams.lr_patience,
                min_lr=self.hparams.lr_min,
            )
            lr_scheduler = {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            }

        elif self.hparams.lr_schedule == "cyclic":
            scheduler = CyclicLR(
                optimizer,
                base_lr=self.hparams.lr_min,
                max_lr=self.hparams.lr,
                step_size_up=4,
                mode="exp_range",
                cycle_momentum=False,
            )
            lr_scheduler = {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            }
        else:
            raise ValueError(f"Unknown lr_schedule: {self.hparams.lr_schedule}")
        return [optimizer], [lr_scheduler]

    def train_dataloader(self):
        return self.data.train_dataloader

    def val_dataloader(self):
        return self.data.val_dataloader

    def test_dataloader(self):
        return self.data.test_dataloader

    def forward(self, z, pos, batch=None, Q=None, y=None, smiles=None):
        return self.model(z, pos, batch=batch, Q=Q, y=y, smiles=smiles)

    def training_step(self, batch, batch_idx):
        return self.step(batch, mse_loss, "train")

    def validation_step(self, batch, batch_idx, *args):
        if len(args) == 0 or (len(args) > 0 and args[0] == 0):
            # validation step
            return self.step(batch, mse_loss, "val")
        # test step
        return self.step(batch, l1_loss, "test")

    def test_step(self, batch, batch_idx):
        return self.step(batch, l1_loss, "test")

    def step(self, batch, loss_fn, stage):
        if self.hparams.single_task_id is not None:
            assert (
                self.hparams.output_channels_toxicity == 1
            ), "Output channels not 1, but single task ID specified"
            batch.tox_labels = batch.tox_labels[
                :, self.hparams.single_task_id
            ].unsqueeze(1)

        if self.hparams.use_energy_feature or self.hparams.energy_tox_multi_task:
            batch.y = batch.y / self.energies[self.hparams.dataset_arg["dataset"]]

        if "atom_props" in batch:
            assert batch.atom_props.shape[0] == batch.z.shape[0]
            assert batch.atom_props.shape[1] == 10
            batch.z = batch.atom_props

        with torch.set_grad_enabled(stage == "train" or self.hparams.derivative):
            # TODO: the model doesn't necessarily need to return a derivative once
            # Union typing works under TorchScript (https://github.com/pytorch/pytorch/pull/53180)
            pred, deriv = self(
                batch.z,
                batch.pos,
                batch=batch.batch,
                y=batch.y if self.hparams.use_energy_feature else None,
                smiles=batch.smiles
                if self.hparams.use_smiles or self.hparams.use_smiles_only
                else None,
                Q=batch.Q if self.hparams.use_total_charge else None,
            )

        loss_y, loss_toxicity = 0, 0

        if "y" in batch and self.hparams.energy_tox_multi_task:
            if batch.y.ndim == 1:
                batch.y = batch.y.unsqueeze(1)

            # energy/prediction loss
            loss_y = loss_fn(pred, batch.y)

            if stage in ["train", "val"] and self.hparams.ema_alpha_y < 1:
                if self.ema[stage + "_y"] is None:
                    self.ema[stage + "_y"] = loss_y.detach()
                # apply exponential smoothing over batches to y
                loss_y = (
                    self.hparams.ema_alpha_y * loss_y
                    + (1 - self.hparams.ema_alpha_y) * self.ema[stage + "_y"]
                )
                self.ema[stage + "_y"] = loss_y.detach()

            if self.hparams.energy_weight > 0:
                self.losses[stage + "_y"].append(loss_y.detach())

        tox_loss = (
            bcn_loss
            if self.hparams.train_type == "classification"
            else mse_loss
            if stage in ["train", "val"]
            else l1_loss
        )
        mask = batch.tox_labels != -100
        if self.hparams.energy_tox_multi_task:
            assert deriv is not None, "Energy-tox multi-task, but output None!"
            loss_toxicity = tox_loss(deriv, batch.tox_labels, reduction="none")
            predictions = deriv.detach().cpu()
        else:
            loss_toxicity = tox_loss(pred, batch.tox_labels, reduction="none")
            predictions = pred.detach().cpu()
        loss_toxicity = loss_toxicity * mask
        loss_toxicity = loss_toxicity.sum() / (mask.sum() + 1e-12)

        # total loss
        if self.hparams.energy_tox_multi_task:
            assert (
                self.hparams.energy_weight > 0.0 and self.hparams.toxicity_weight > 0.0
            )

        loss = (
            loss_y * self.hparams.energy_weight
            + loss_toxicity * self.hparams.toxicity_weight
        )

        self.losses[stage].append(loss.detach())

        labels = (
            batch.tox_labels.cpu()
            if self.tox_regression
            else batch.tox_labels.cpu().long()
        )
        return {"loss": loss, "labels": labels, "predictions": predictions}

    def multitask_auc(self, predicted, ground_truth, final_test=False):
        n_tasks = ground_truth.shape[1]
        ground_truth_np = ground_truth.cpu().numpy()
        auc = []
        auc_dict = {i: float for i in range(n_tasks)}
        for i in range(n_tasks):
            if np.any(ground_truth_np[:, i] == 0) and np.any(
                ground_truth_np[:, i] == 1
            ):
                auroc = AUROC(task="binary", ignore_index=-100)
                auc.append(auroc(predicted[:, i], ground_truth[:, i]))
                auc_dict[i] = auc
            else:
                continue

        if final_test:
            return auc_dict, sum(auc) / len(auc)
        return sum(auc) / len(auc)

    def optimizer_step(self, *args, **kwargs):
        optimizer = kwargs["optimizer"] if "optimizer" in kwargs else args[2]
        if self.trainer.global_step < self.hparams.lr_warmup_steps:
            lr_scale = min(
                1.0,
                float(self.trainer.global_step + 1)
                / float(self.hparams.lr_warmup_steps),
            )

            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.hparams.lr
        super().optimizer_step(*args, **kwargs)
        optimizer.zero_grad()

    def training_epoch_end(self, training_step_outputs):
        dm = self.trainer.datamodule
        if hasattr(dm, "test_dataset") and len(dm.test_dataset) > 0:
            should_reset = (
                self.current_epoch % self.hparams.test_interval == 0
                or (self.current_epoch - 1) % self.hparams.test_interval == 0
            )
            if should_reset:
                self.trainer.reset_val_dataloader(self)

        if not self.tox_regression:
            preds, targets = [], []
            for output in training_step_outputs:
                preds += output["predictions"]
                targets += output["labels"]

            targets = torch.stack(targets)
            preds = torch.stack(preds)

            auc = self.multitask_auc(preds, targets)
            self.losses["train_tox_auc"].append(auc)

    def test_epoch_end(self, test_step_outputs):
        if not self.trainer.sanity_checking:
            smiles = "_smiles" if self.hparams.use_smiles else ""
            smiles_only = "_smiles_only" if self.hparams.use_smiles_only else ""
            split = (
                "_random"
                if self.hparams.dataset_split == "random"
                else "_scaffold"
                if self.hparams.dataset_split == "scaffold"
                else "_random_scaffold"
                if self.hparams.dataset_split == "random_scaffold"
                else ""
            )
            conformer = (
                "_best"
                if self.hparams.dataset_arg["conformer"] == "best"
                and not self.hparams.use_smiles_only
                else "_xtb"
                if self.hparams.dataset_arg["conformer"] == "xtb"
                and not self.hparams.use_smiles_only
                else "_random"
                if self.hparams.dataset_arg["conformer"] == "random"
                and not self.hparams.use_smiles_only
                else ""
            )
            task_id = (
                f"_{str(self.hparams.single_task_id)}"
                if self.hparams.single_task_id is not None
                else ""
            )
            if not self.tox_regression:
                preds, targets = [], []
                for output in test_step_outputs:
                    preds += output["predictions"]
                    targets += output["labels"]

                targets = torch.stack(targets)
                preds = torch.stack(preds)

                aucs, mean_auc = self.multitask_auc(preds, targets, final_test=True)
                print(f"\n The AUC of the best model is: {mean_auc}\n")
                self.losses["final_test_tox_auc"].append(mean_auc)

                if len(aucs) > 2:
                    with open(
                        os.path.join(
                            self.hparams.log_dir,
                            f"{self.hparams.dataset_arg['dataset']}_{self.hparams.seed}{smiles}{split}{smiles_only}{conformer}.pickle",
                        ),
                        "wb",
                    ) as f:
                        pickle.dump(aucs, f)

                with open(
                    os.path.join(
                        self.hparams.log_dir,
                        f"{self.hparams.dataset_arg['dataset']}_{self.hparams.seed}{smiles}{split}{smiles_only}{conformer}{task_id}.csv",
                    ),
                    "w",
                ) as f:
                    writer = csv.writer(f)
                    writer.writerow([float(mean_auc)])
            else:
                losses, preds, targets = [], [], []
                for output in test_step_outputs:
                    preds += output["predictions"]
                    targets += output["labels"]
                    losses += output["loss"].detach().cpu().unsqueeze(0)
                loss = np.mean(losses)
                print(f"\n The loss of the best model is: {loss}\n")

                output_dict = {
                    "labels": torch.stack(targets),
                    "predictions": torch.stack(preds),
                }

                with open(
                    os.path.join(
                        self.hparams.log_dir,
                        f"{self.hparams.dataset_arg['dataset']}_{self.hparams.seed}{smiles}{pre_train}{split}{smiles_only}{conformer}.csv",
                    ),
                    "w",
                ) as f:
                    writer = csv.writer(f)
                    writer.writerow([float(loss)])
                self.losses["final_test_loss"].append(loss)

                with open(
                    os.path.join(
                        self.hparams.log_dir,
                        f"{self.hparams.dataset_arg['dataset']}_{self.hparams.seed}{smiles}{pre_train}{split}{smiles_only}{conformer}.pickle",
                    ),
                    "wb",
                ) as f:
                    pickle.dump(output_dict, f)

    def validation_epoch_end(self, validation_step_outputs):
        if not self.trainer.sanity_checking:
            # construct dict of logged metrics
            result_dict = {
                "epoch": float(self.current_epoch),
                "lr": self.trainer.optimizers[0].param_groups[0]["lr"],
                "train_loss": torch.stack(self.losses["train"]).mean(),
                "val_loss": torch.stack(self.losses["val"]).mean(),
            }

            if not self.tox_regression:
                preds_val, preds_test = [], []
                targets_val, targets_test = [], []

                val_and_test = len(validation_step_outputs) == 2
                for i, outputs in enumerate(validation_step_outputs):
                    if val_and_test:
                        for output in outputs:
                            if i == 0:
                                preds_val += output["predictions"]
                                targets_val += output["labels"]
                            else:
                                preds_test += output["predictions"]
                                targets_test += output["labels"]
                    else:
                        for output in validation_step_outputs:
                            preds_val += output["predictions"]
                            targets_val += output["labels"]

                targets_val = torch.stack(targets_val)
                preds_val = torch.stack(preds_val)

                auc_val = self.multitask_auc(preds_val, targets_val)

                result_dict["val_tox_auc"] = auc_val

                if val_and_test:
                    targets_test = torch.stack(targets_test)
                    preds_test = torch.stack(preds_test)
                    auc_test = self.multitask_auc(preds_test, targets_test)
                    result_dict["test_tox_auc"] = auc_test

                # reset validation dataloaders before and after testing epoch, which is faster
                # than skipping test validation steps by returning None
            # add test loss if available
            if len(self.losses["test"]) > 0:
                result_dict["test_loss"] = torch.stack(self.losses["test"]).mean()

            # if prediction is present, also log them separately
            if len(self.losses["train_y"]) > 0:
                result_dict["train_loss_y"] = torch.stack(self.losses["train_y"]).mean()

                result_dict["val_loss_y"] = torch.stack(self.losses["val_y"]).mean()

                if len(self.losses["test"]) > 0:
                    result_dict["test_loss_y"] = torch.stack(
                        self.losses["test_y"]
                    ).mean()

            if len(self.losses["train_tox_auc"]) > 0:
                result_dict["train_tox_auc"] = self.losses["train_tox_auc"][0]

            if len(self.losses["final_test_tox_auc"]) > 0:
                result_dict["final_test_tox_auc"] = self.losses["final_test_tox_auc"][0]

            self.log_dict(result_dict, sync_dist=True)
        self._reset_losses_dict()

    def _reset_losses_dict(self):
        self.losses = {
            "train": [],
            "val": [],
            "test": [],
            "train_y": [],
            "val_y": [],
            "test_y": [],
            "train_toxicity": [],
            "val_toxicity": [],
            "test_toxicity": [],
            "train_tox_auc": [],
            "val_tox_auc": [],
            "test_tox_auc": [],
            "final_test_tox_auc": [],
            "final_test_loss": [],
        }

    def _reset_ema_dict(self):
        self.ema = {"train_y": None, "val_y": None}
