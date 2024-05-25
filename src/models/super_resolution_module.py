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
        
    def process_batch(batch):
        # Processes a batch into x and y
        # That look like (Batch,Channel,X,Y,Z)
        # x has shape (Batch,Timestep/Channel,Timestep/Channel), X,Y,Z)
        # (unknown which is timestep and which is channel, but it shouldn't matter)
        x = batch['lrs'] 
        batch,timestep,channel,x_dim, y_dim, z_dim = x.shape
        x = x.reshape(batch,timestep*channel,x_dim,y_dim,z_dim)
        y = batch['hr']

        return x,y
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        
        x,y = self.process_batch(x,y)
        pred = self.net(x)        
        loss = self.lossfunc(pred, y)
        self.train_mae(pred,y)        
        self.log("train/mae", self.train_mae, on_step=False, on_epoch=True, prog_bar=True)

        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        return loss
    def validation_step(self, batch, batch_idx):
        x,y = self.process_batch(x,y)
        pred = self.net(x)        
        loss = self.lossfunc(pred, y)
        self.val_mae(pred,y)        
        self.log("val/mae", self.test_mae, on_step=False, on_epoch=True, prog_bar=True)

        # Logging to TensorBoard (if installed) by default
        self.log("test_loss", loss)
        
    def test_step(self, batch, batch_idx):
        
        
        x,y = self.process_batch(x,y)
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
