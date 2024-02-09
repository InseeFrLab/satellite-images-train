"""
"""
from typing import Dict, Union, Optional
import torch
import pytorch_lightning as pl
from torch import nn, optim
import evaluate
from optim.metrics import IOU, pct_positive
from transformers import SegformerForSemanticSegmentation


class SegmentationModule(pl.LightningModule):
    """
    Pytorch Lightning Module for DeepLabv3.
    """

    def __init__(
        self,
        model: nn.Module,
        loss: Union[nn.Module],
        optimizer: Union[optim.SGD, optim.Adam],
        optimizer_params: Dict,
        scheduler: Union[optim.lr_scheduler.OneCycleLR, optim.lr_scheduler.ReduceLROnPlateau],
        scheduler_params: Dict,
        scheduler_interval: str,
    ):
        """
        Initialize TableNet Module.
        Args:
            model
            loss
            optimizer
            optimizer_params
            scheduler
            scheduler_params
            scheduler_interval
        """
        super().__init__()

        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.optimizer_params = optimizer_params
        self.scheduler = scheduler
        self.scheduler_params = scheduler_params
        self.scheduler_interval = scheduler_interval
        self.metric = evaluate.load("mean_iou")

    def forward(self, batch: torch.Tensor, labels: Optional[torch.Tensor] = None):
        """
        Perform forward-pass.

        Args:
            batch (torch.Tensor): Batch of images to perform forward-pass.
            labels (Optional[torch.Tensor]): Optional labels.
        Returns:
            Model output.
        """
        if labels is None:
            return self.model(batch)
        else:
            return self.model(batch, labels)

    @staticmethod
    def upsample_logits(logits: torch.Tensor, labels_shape: torch.Size) -> torch.Tensor:
        """
        Upsample Segformer logits to a given shape.

        Args:
            logits (torch.Tensor): Segformer logits.
            labels_shape (torch.Size): Labels shape.

        Returns:
            torch.Tensor: Upsampled logits
        """
        return nn.functional.interpolate(
            logits,
            size=labels_shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    def compute_iou_segformer(self, logits, labels) -> torch.Tensor:
        """
        Compute mean IOU for a SegFormer model output.

        Args:
            logits: Segformer logits model output.
            labels: Labels.

        Returns:
            torch.Tensor: IOU for the building class.
        """
        # scale the logits to the size of the label
        logits = self.upsample_logits(logits, labels.shape).argmax(dim=1)

        pred_labels = logits.detach().cpu().numpy()
        # currently using _compute instead of compute
        # see this issue for more info: https://github.com/huggingface/evaluate/pull/328#issuecomment-1286866576
        id2label = {0: "background", 1: "building"}
        metrics = self.metric._compute(
            predictions=pred_labels,
            references=labels,
            num_labels=len(id2label),
            ignore_index=0,
        )
        # add per category metrics as individual key-value pairs
        per_category_accuracy = metrics.pop("per_category_accuracy").tolist()
        per_category_iou = metrics.pop("per_category_iou").tolist()

        metrics.update({f"accuracy_{id2label[i]}": v for i, v in enumerate(per_category_accuracy)})
        metrics.update({f"iou_{id2label[i]}": v for i, v in enumerate(per_category_iou)})
        return metrics["iou_building"]

    def training_step(self, batch, batch_idx):
        """
        Training step.

        Args:
            batch: Data for training.
            batch_idx (int): batch index.

        Returns: Training loss.
        """
        images = batch["pixel_values"]
        labels = batch["labels"]

        if isinstance(self.model, SegformerForSemanticSegmentation):
            output = self.forward(images, labels)
            loss = output.loss
            logits = output.logits
            pct_building = pct_positive(logits, True)
            iou = self.compute_iou_segformer(logits, labels)
        else:
            output = self.forward(images)
            loss = self.loss(output, labels)
            pct_building = pct_positive(output, self.model.logits)
            iou = IOU(output, labels, self.model.logits)

        self.log("train_loss", loss, on_step=True, on_epoch=True)
        self.log("train_iou", iou, on_step=True, on_epoch=True)
        self.log("train_pct_building", pct_building, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step.

        Args:
            batch: Data for training.
            batch_idx (int): batch index.

        Returns: Validation loss.
        """
        images = batch["pixel_values"]
        labels = batch["labels"]

        if isinstance(self.model, SegformerForSemanticSegmentation):
            output = self.forward(images, labels)
            loss = output.loss
            logits = output.logits
            pct_building = pct_positive(logits, True)
            iou = self.compute_iou_segformer(logits, labels)
        else:
            output = self.forward(images)
            loss = self.loss(output, labels)
            pct_building = pct_positive(output, self.model.logits)
            iou = IOU(output, labels, self.model.logits)

        # Log on epoch, mean reduction
        self.log("validation_IOU", iou, on_step=True, on_epoch=True)
        self.log("validation_loss", loss, on_step=True, on_epoch=True)
        self.log("validation_pct_building", pct_building, on_step=True, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx):
        """
        Test step.

        Args:
            batch: Data for training.
            batch_idx (int): batch index.

        Returns: IOU on test data.
        """
        images = batch["pixel_values"]
        labels = batch["labels"]

        if isinstance(self.model, SegformerForSemanticSegmentation):
            output = self.forward(images, labels)
            loss = output.loss
            logits = output.logits
            pct_building = pct_positive(logits, True)
            iou = self.compute_iou_segformer(logits, labels)
        else:
            output = self.forward(images)
            loss = self.loss(output, labels)
            pct_building = pct_positive(output, self.model.logits)
            iou = IOU(output, labels, self.model.logits)

        self.log("test_loss", loss, on_epoch=True)
        self.log("test_IOU", iou, on_epoch=True)
        self.log("test_pct_building", pct_building, on_epoch=True)

        return IOU

    def configure_optimizers(self):
        """
        Configure optimizer for pytorch lighting.
        Returns: optimizer and scheduler for pytorch lighting.
        """
        optimizer = self.optimizer(self.parameters(), **self.optimizer_params)
        scheduler = self.scheduler(optimizer, self.scheduler_params["mode"])
        scheduler = {
            "scheduler": scheduler,
            "monitor": self.scheduler_params["monitor"],
            "interval": self.scheduler_interval,
        }

        return [optimizer], [scheduler]
