import argparse
from urllib import parse
import torch
import torch.nn.functional as F
import pytorch_lightning as pl


# Adapted from https://github.com/PyTorchLightning/deep-learning-project-template/blob/master/project/lit_image_classifier.py # noqa
class ClassificationTask(pl.LightningModule):
    def __init__(self, backbone, learning_rate=0.001, **kwargs):
        super(ClassificationTask, self).__init__()
        self.save_hyperparameters()
        self.backbone = backbone
        # metrics = torchmetrics.MetricCollection([])
        # self.val_metrics = metrics.clone(prefix="Val_")
        # self.test_metrics = metrics.clone(prefix="Test_")

    def forward(self, x):
        embedding = self.backbone(x)
        return embedding

    def shared_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.backbone(x)
        return y_hat, y

    def training_step(self, batch, batch_idx):
        y_hat, y = self.shared_step(batch, batch_idx)
        loss = F.cross_entropy(y_hat, y)
        self.log("Train_Loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        y_hat, y = self.shared_step(batch, batch_idx)
        loss = F.cross_entropy(y_hat, y)
        self.log("Val_Loss", loss)
        # self.log_dict(self.val_metrics(y_hat, y))
        return loss

    def test_step(self, batch, batch_idx):
        y_hat, y = self.shared_step(batch, batch_idx)
        loss = F.cross_entropy(y_hat, y)
        self.log("Test_Loss", loss)
        # self.log_dict(self.test_metrics(y_hat, y))
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    @staticmethod
    def add_task_specific_arguments(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--learning_rate", "--lr", type=float, default=0.001)
        return parser
