from pytorch_lightning import Trainer
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from pytorch_lightning.callbacks import ModelCheckpoint
from torchmetrics import MetricCollection, Accuracy

from nnsimple import nn
from nnsimple import Predictor
from nnsimple.utils.data.custom_dataset import CustomImageDataset
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



    model = nn.models.CNN(input_size, hidden_size, n_classes, kernel_size, dropout)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    metrics = {
        'train_metrics': MetricCollection({'train_acc': Accuracy()}),
        'val_metrics': MetricCollection({'val_acc': Accuracy()}),
        'test_metrics': MetricCollection({'test_acc': Accuracy()})
    }

    predictor = Predictor(model=model,
                          loss_fn=loss_fn,
                          optimizer=optimizer,
                          metrics=metrics)

    checkpoint_callback = ModelCheckpoint(save_top_k=1,
                                          monitor='val_acc', mode='max')

    trainer = Trainer(max_epochs=2,
                      gpus=1 if torch.cuda.is_available() else None,
                      callbacks=[checkpoint_callback],
                      )

    trainer.fit(predictor,
                train_dataloaders=train_loader,
                val_dataloaders=val_loader)

    trainer.test(predictor,
                 dataloaders=test_loader,
                 verbose=True)


if __name__ == "__main__":
    main()
