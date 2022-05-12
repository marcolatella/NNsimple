import pytorch_lightning as pl
from torchmetrics.functional import accuracy


class Predictor(pl.LightningModule):
    def __init__(self,
                 model,
                 loss_fn,
                 optimizer):
        super(Predictor, self).__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer

    def training_step(self, batch, batch_index):
        x, y = batch
        logits = self.model(x)
        train_loss = self.loss_fn(logits, y)
        acc = accuracy(logits, y)
        metrics = {'train_acc': acc}
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_loss', train_loss, prog_bar=True)
        return train_loss

    def validation_step(self, batch, batch_index):
        x, y = batch
        logits = self.model(x)
        val_loss = self.loss_fn(logits, y)
        acc = accuracy(logits, y)
        metrics = {'val_acc': acc}
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_loss', val_loss, prog_bar=True)
        return val_loss

    def test_step(self, batch, batch_index):
        x, y = batch
        logits = self.model(x)
        test_loss = self.loss_fn(logits, y)
        acc = accuracy(logits, y)
        metrics = {'Accuracy': acc}
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return test_loss

    def configure_optimizers(self):
        optimizer = self.optimizer
        return optimizer

