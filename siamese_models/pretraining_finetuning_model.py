import torch
import pytorch_lightning as pl

from .feature_encoder import FeatureEncoder
from .loss_functions import log_ratio_loss, cross_entropy_loss


class PretrainingFinetuningModel(pl.LightningModule):
    """Rotation-invariant implementation of Contrastive Similarity Learning."""

    def __init__(
        self,
        encoder_type="VNPointNet",
        pretrain=True,
        n_epochs: int = 5,
        file_path=None,
        batch_size=1,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.feat_enc = FeatureEncoder(encoder_type)

        # switch between pretraining and finetuning mode
        if pretrain:
            self.step = self.pretraining_step
        else:
            self.step = self.finetuning_step

    def configure_optimizers(self):
        """AdamW Optimizer"""
        optimizer = torch.optim.AdamW(self.parameters())
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams["n_epochs"])
        return [optimizer], [scheduler]

    def forward(self, x):
        """
        Input: [B x N x D]
        B: batch
        N: points (=512)
        D: dimension (=3)
        """
        assert len(x.shape) == 3
        return self.feat_enc(x)

    def pretraining_step(self, batch, batch_idx):
        """
        Input: [P x B x N x D]
        P: point clouds (A, B= 2)
        B: batch
        N: points (=512)
        D: dimension (=3)
        """
        assert len(batch) == 2
        pcl_0, pcl_1 = batch
        assert len(pcl_0.shape) in (3, 6)
        assert len(pcl_1.shape) in (3, 6)
        feat_0 = self(pcl_0)
        feat_1 = self(pcl_1)
        loss = cross_entropy_loss(feat_0, feat_1)

        return {
            "loss": loss,
            "embedding": feat_0,
            "feat_A": feat_0,
            "feat_B": feat_1,
        }

    def finetuning_step(self, batch, batch_idx):
        """
        Input: [P x B x N x D]
        P: point clouds (A, B= 2)
        B: batch
        N: points (=512)
        D: dimension (=3)
        """
        assert len(batch) == 2
        pcl_0, icp_dist = batch
        assert len(pcl_0.shape) in (3, 6)
        feat_0 = self(pcl_0)
        loss = log_ratio_loss(feat_0, icp_dist)

        return {
            "loss": loss,
            "embedding": feat_0,
            "feat_A": feat_0,
            "feat_B": None,
        }

    def training_step(self, batch, batch_idx):
        output = self.step(batch[:2], batch_idx)
        self.log("train_loss", output["loss"])
        return output

    def validation_step(self, batch, batch_idx):
        output = self.step(batch[:2], batch_idx)
        self.log("val_loss", output["loss"])
        return output

    def test_step(self, batch, batch_idx):
        return self.step(batch[:2], batch_idx)

    def predict_step(self, batch, batch_idx):
        return self.step(batch[:2], batch_idx)
