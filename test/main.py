from pytorch_lightning import Trainer
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from pytorch_lightning.callbacks import ModelCheckpoint
from torchmetrics import MetricCollection, Accuracy

import nnsimple
from pytorch_lightning.loggers import NeptuneLogger
from nnsimple import nn
from nnsimple import Predictor
import numpy as np


def main():
    train_set = torchvision.datasets.CIFAR10(root='./ data', train=True, transform=transforms.ToTensor(), download=True)
    test_set = torchvision.datasets.CIFAR10(root='./ data', train=False, transform=transforms.ToTensor())

    mean = train_set.data.mean(axis=(0, 1, 2)) / 255
    std = train_set.data.std(axis=(0, 1, 2)) / 255
    print(f"Train_set > mean: {mean}, std: {std}")
    # mean: [0.49139968 0.48215841 0.44653091], std: [0.24703223 0.24348513 0.26158784]

    trans_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_set = torchvision.datasets.CIFAR10(root='./ data', train=True, transform=trans_train, download=True)
    test_set = torchvision.datasets.CIFAR10(root='./ data', train=False, transform=trans_train)

    indexes = np.arange(len(train_set))

    val_indexes = indexes[49000:]
    train_indexes = indexes[0:49000]

    train_sampler = torch.utils.data.SubsetRandomSampler(train_indexes)
    val_sampler = torch.utils.data.SubsetRandomSampler(val_indexes)

    # These are the hyperparameters for the model.
    batch_size = 32
    learning_rate = 0.001
    momentum = 0.9
    input_size = 3
    hidden_size = 32
    n_classes = 10
    dropout = 0.5
    kernel_size = 3

    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=batch_size,
                                               sampler=train_sampler,
                                               num_workers=2
                                               )
    val_loader = torch.utils.data.DataLoader(train_set,
                                             batch_size=batch_size,
                                             sampler=val_sampler,
                                             num_workers=2
                                             )
    test_loader = torch.utils.data.DataLoader(dataset=test_set,
                                              batch_size=32,
                                              shuffle=False)

    # image = train_set[0]
    # plt.imshow(image[0].permute(1, 2, 0))
    # plt.show()

    hparam = {"input_size": 3,
              "hidden_size_conv": 32,
              "hidden_size_fc": 512,
              "out_size": 10,
              "image_shape": None,
              "kernel_size": 3,
              "kernel_pooling": 2,
              "dropout": 0.5}

    model = nnsimple.nn.models.CNN
    loss_fn = torch.nn.CrossEntropyLoss()
    optim = torch.optim.SGD

    metrics = {
        'train_metrics': MetricCollection({'train_acc': Accuracy()}),
        'val_metrics': MetricCollection({'val_acc': Accuracy()}),
        'test_metrics': MetricCollection({'test_acc': Accuracy()})
    }

    optim_params = {'lr': 0.001,
                    'momentum': 0.9}

    predictor = Predictor(model=model,
                          model_params=hparam,
                          loss_fn=loss_fn,
                          optimizer=optim,
                          optim_params=optim_params,
                          metrics=metrics)

    checkpoint_callback = ModelCheckpoint(save_top_k=1,
                                          monitor='val_acc', mode='max')

    neptune_logger = NeptuneLogger(
        api_key="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIyMmU3OTdmOC1lNTMzLTQyZjAtOWQ4OS1hZWE0Y2FiZGIzOWUifQ==",
        # replace with your own
        project="marcolatella/Robotics",
        tags=["training", "CNN"],  # optional
    )

    trainer = Trainer(max_epochs=5,
                      gpus=1 if torch.cuda.is_available() else None,
                      callbacks=[checkpoint_callback],
                      accumulate_grad_batches=1,
                      logger=neptune_logger
                      )

    trainer.fit(predictor,
                train_dataloaders=train_loader,
                val_dataloaders=val_loader)

    trainer.test(predictor,
                 dataloaders=test_loader,
                 verbose=True)


if __name__ == "__main__":
    main()
