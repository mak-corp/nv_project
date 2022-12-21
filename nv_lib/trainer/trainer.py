import random
from pathlib import Path
from random import shuffle
from itertools import chain
from collections import defaultdict

import numpy as np
import PIL
import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torchvision.transforms import ToTensor
from tqdm import tqdm

from .base_trainer import BaseTrainer

from nv_lib.logger.utils import plot_spectrogram_to_buf
from nv_lib.model import MelSpectrogramConfig
from nv_lib.utils import inf_loop, MetricTracker


class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(
            self,
            model,
            config,
            device,
            dataloaders,
            len_epoch=None,
            skip_oom=True,
    ):
        super().__init__(model, config, device)
        self.skip_oom = skip_oom
        self.config = config

        self.train_dataloader = dataloaders["train"]
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.train_dataloader)
        else:
            # iteration-based training
            self.train_dataloader = inf_loop(self.train_dataloader)
            self.len_epoch = len_epoch
        self.test_dataloader = dataloaders["test"]

        self.log_step = config["trainer"]["log_step"]
        self.grad_norm_clip = config["trainer"].get("grad_norm_clip", None)

        self.train_metrics = MetricTracker()

    def _clip_grad_norm(self):
        if self.grad_norm_clip is not None:
            clip_grad_norm_(self.model.parameters(), self.grad_norm_clip)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        print("Epoch:", epoch)
        self.model.train()
        self.train_metrics.reset()
        self.writer.add_scalar("epoch", epoch)
        for batch_idx, batch in enumerate(
            tqdm(self.train_dataloader, desc="train", total=self.len_epoch), start=1
        ):
            if 'error' in batch:
                continue

            if batch_idx > self.len_epoch:
                break
            try:
                batch = self.process_batch(
                    batch,
                    is_train=True,
                    metrics=self.train_metrics,
                )
            except RuntimeError as e:
                if "out of memory" in str(e) and self.skip_oom:
                    self.logger.warning("OOM on batch. Skipping batch.")
                    for p in self.model.parameters():
                        if p.grad is not None:
                            del p.grad  # free some memory
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e
            # self.train_metrics.update("grad norm", self.get_grad_norm())
            if batch_idx % self.log_step == 0 or batch_idx == 1:
                self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx - 1)
                self.writer.add_scalar(
                    "learning rate", self.model.get_last_lr()
                )
                self._log_predictions({
                    "train": batch,
                })
                self._log_scalars(self.train_metrics)
                # we don't want to reset train metrics at the start of every epoch
                # because we are interested in recent train metrics
                last_train_metrics = self.train_metrics.result()
                self.train_metrics.reset()
            
        self.model.scheduler_step()

        log = last_train_metrics

        self._test_epoch(epoch, "test")

        return log

    @staticmethod
    def move_batch_to_device(batch, device: torch.device):
        """
        Move all necessary tensors to the HPU
        """
        for tensor_for_gpu in ["gt_wav"]:
            if tensor_for_gpu in batch:
                batch[tensor_for_gpu] = batch[tensor_for_gpu].to(device)
        return batch

    def process_batch(self, batch, is_train: bool, metrics: MetricTracker):
        batch = self.move_batch_to_device(batch, self.device)
        batch = self.model.optimization_step(batch, metrics)
        return batch

    def _test_epoch(self, epoch, part):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()

        self.writer.set_step(epoch * self.len_epoch, part)
        
        batches = {}
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.test_dataloader, desc="test"), start=1):
                batch = Trainer.move_batch_to_device(batch, self.device)
                batch = self.model(batch)
                batches[f"test_{batch_idx}"] = batch
        self._log_predictions(batches)

    def _progress(self, batch_idx):
        base = "[{}/{} ({:.0f}%)]"
        if hasattr(self.train_dataloader, "n_samples"):
            current = batch_idx * self.train_dataloader.batch_size
            total = self.train_dataloader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

    def _log_predictions(self, batches):
        if self.writer is None:
            return

        for batch_name, batch in batches.items():
            idx = None
            for value_name in ["gt_wav", "gen_wav"]:
                values = batch[value_name]
                if idx is None:
                    idx = np.random.choice(len(values))
                self.writer.add_audio(f"{batch_name}/{value_name}", values[idx], MelSpectrogramConfig.sr)

    def _log_spectrogram(self, name, spectrogram_batch, idx=None):
        idx = idx if idx is not None else np.random.choice(len(spectrogram_batch))
        spectrogram = spectrogram_batch[idx].detach().cpu()
        image = PIL.Image.open(plot_spectrogram_to_buf(spectrogram))
        self.writer.add_image(name, ToTensor()(image))

    @torch.no_grad()
    def get_grad_norm(self, norm_type=2):
        parameters = self.model.parameters()
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]
        total_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad.detach(), norm_type).cpu() for p in parameters]
            ),
            norm_type,
        )
        return total_norm.item()

    def _log_scalars(self, metric_tracker: MetricTracker):
        if self.writer is None:
            return
        for metric_name, metric_avg in metric_tracker.result().items():
            self.writer.add_scalar(f"{metric_name}", metric_avg)
