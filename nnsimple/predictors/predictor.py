import pytorch_lightning as pl
import torch


class Predictor(pl.LightningModule):
    def __init__(self,
                 model = None,
                 model_params = None,
                 loss_fn = None,
                 optimizer = None,
                 optim_params = None,
                 metrics = None):
        super(Predictor, self).__init__()
        self.save_hyperparameters()
        self.model = model
        self.model_params = model_params
        self.loss_fn = loss_fn
        self.optim_params = optim_params
        self.train_metric = None
        self.val_metric = None
        self.test_metric = None

        if self.model is not None:
            self.model = self.model(**self.model_params)
        if optimizer is not None:
            self.optimizer = optimizer(self.parameters(), **self.optim_params)

        # Checking if the metrics is a dictionary.
        if isinstance(metrics, dict):
            # TODO check if keys in 'metrics' exist ('train_metrics', ...)
            self.train_metric = metrics['train_metrics']
            self.val_metric = metrics['val_metrics']
            self.test_metric = metrics['test_metrics']
            # TODO else condition

    def training_step(self, batch, batch_index):
        """
        We take a batch of data, pass it through the model, calculate the loss, calculate the metrics, and log the loss and
        metrics

        :param batch: The input batch
        :param batch_index: The index of the batch in the current epoch
        :return: The loss value
        """
        x, y = batch
        logits = self.model(x)
        train_loss = self.loss_fn(logits, y)
        metrics = self.train_metric(logits, y)
        if self.global_step % 20 == 0:
            self.log_dict(metrics, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            self.log('train_loss', train_loss, on_epoch=True)
        return train_loss

    def validation_step(self, batch, batch_index):
        """
        We pass in a batch of data, and the batch index, and we return the validation loss

        :param batch: The batch of data that was passed to the validation_step function
        :param batch_index: The index of the batch in the current epoch
        :return: The validation loss is being returned.
        """
        x, y = batch
        logits = self.model(x)
        val_loss = self.loss_fn(logits, y)
        metrics = self.val_metric(logits, y)
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_loss', val_loss, prog_bar=True, on_epoch=True)
        return val_loss

    def test_step(self, batch, batch_index):
        """
        The test_step function takes the test data and calculates the loss and
        metrics

        :param batch: The batch of data that was passed to the test_step function
        :param batch_index: The index of the batch in the current epoch
        :return: The test loss is being returned.
        """
        x, y = batch
        logits = self.model(x)
        test_loss = self.loss_fn(logits, y)
        metrics = self.test_metric(logits, y)
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return test_loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self.forward(batch)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def configure_optimizers(self):
        """
        The function takes in the optimizer that was defined in the init function and returns it
        :return: The optimizer
        """
        optimizer = self.optimizer
        return optimizer

