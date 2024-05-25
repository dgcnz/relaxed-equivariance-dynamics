from typing import Any, Dict, Tuple

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.regression import MeanAbsoluteError
import torch.nn as nn

class SuperResolutionLightningModule(LightningModule):
    def __init__(self, net,lr=1e-3,lossfunc=None):
        super().__init__()
        self.net = net
        self.lr = lr
        self.train_mae = MeanAbsoluteError()
        self.val_mae  = MeanAbsoluteError()
        self.test_mae = MeanAbsoluteError()
        if lossfunc == None:
            self.lossfunc = nn.L1Loss()
        

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        
        x = batch['lrs'] #Three timesteps of 16^3, each on a different channel
        y = batch['hr']  #One timestep of 64^3, just one channel
        pred = self.net(x)        
        loss = self.lossfunc(pred, y)
        self.train_mae(pred,y)        
        self.log("train/mae", self.train_mae, on_step=False, on_epoch=True, prog_bar=True)

        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        return loss
    def validation_step(self, batch, batch_idx):
        x = batch['lrs'] #Three timesteps of 16^3, each on a different channel
        y = batch['hr']  #One timestep of 64^3, just one channel
        pred = self.net(x)        
        loss = self.lossfunc(pred, y)
        self.val_mae(pred,y)        
        self.log("val/mae", self.test_mae, on_step=False, on_epoch=True, prog_bar=True)

        # Logging to TensorBoard (if installed) by default
        self.log("test_loss", loss)
        
    def test_step(self, batch, batch_idx):
        
        
        x = batch['lrs'] #Three timesteps of 16^3, each on a different channel
        y = batch['hr']  #One timestep of 64^3, just one channel
        pred = self.net(x)        
        loss = self.lossfunc(pred, y)
        self.test_mae(pred,y)        
        self.log("test/mae", self.test_mae, on_step=False, on_epoch=True, prog_bar=True)

        # Logging to TensorBoard (if installed) by default
        self.log("test_loss", loss)
        return loss


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
