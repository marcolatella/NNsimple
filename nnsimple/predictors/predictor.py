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

        # Checking if the metrics is a dictionary.
        if isinstance(metrics, dict):
            # TODO check if keys in 'metrics' exist ('train_metrics', ...)
            self.train_metric = metrics['train_metrics']
            self.val_metric = metrics['val_metrics']
            #self.test_metric = metrics['test_metrics']
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
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_loss', train_loss, prog_bar=True, on_step=True)
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
        self.log('val_loss', val_loss, prog_bar=True)
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

    def configure_optimizers(self):
        """
        The function takes in the optimizer that was defined in the init function and returns it
        :return: The optimizer
        """
        optimizer = self.optimizer
        return optimizer

