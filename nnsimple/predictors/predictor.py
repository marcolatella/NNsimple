import pytorch_lightning as pl


class Predictor(pl.LightningModule):
    def __init__(self,
                 model,
                 loss_fn,
                 optimizer,
                 metrics):
        super(Predictor, self).__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_metric = None
        self.val_metric = None
        self.test_metric = None

        if isinstance(metrics, dict):
            # TODO check if keys in 'metrics' exist ('train_metrics', ...)
            self.train_metric = metrics['train_metrics']
            self.val_metric = metrics['val_metrics']
            self.test_metric = metrics['test_metrics']
            # TODO else condition

    def training_step(self, batch, batch_index):
        x, y = batch
        logits = self.model(x)
        train_loss = self.loss_fn(logits, y)
        metrics = self.train_metric(logits, y)
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_loss', train_loss, prog_bar=True, on_step=True)
        return train_loss

    def validation_step(self, batch, batch_index):
        x, y = batch
        logits = self.model(x)
        val_loss = self.loss_fn(logits, y)
        metrics = self.val_metric(logits, y)
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_loss', val_loss, prog_bar=True)
        return val_loss

    def test_step(self, batch, batch_index):
        x, y = batch
        logits = self.model(x)
        test_loss = self.loss_fn(logits, y)
        metrics = self.test_metric(logits, y)
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return test_loss

    def configure_optimizers(self):
        optimizer = self.optimizer
        return optimizer

