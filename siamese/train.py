# import the necessary libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as utils
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import torchvision.utils
import config
from utils import imshow, show_plot, ContrastiveLoss
import torchvision
from torch.autograd import Variable
from PIL import Image
import os
from tqdm import tqdm
import config
from torch.optim.lr_scheduler import StepLR

import argparse


# load the dataset
training_dir = config.training_dir
testing_dir = config.testing_dir
training_csv = config.training_csv
testing_csv = config.testing_csv
val_csv = config.val_csv
val_dir = config.val_dir

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# preprocessing and loading the dataset
class SiameseDataset:
    def __init__(self, training_csv=None, training_dir=None, transform=None):
        # used to prepare the labels and images path
        self.train_df = pd.read_csv(training_csv)
        self.train_df.columns = ["image1", "image2", "label"]
        self.train_dir = training_dir
        self.transform = transform

    def __getitem__(self, index):
        # getting the image path
        image1_path = os.path.join(self.train_dir, self.train_df.iat[index, 0])
        image2_path = os.path.join(self.train_dir, self.train_df.iat[index, 1])

        # Loading the image
        img0 = Image.open(image1_path)
        img1 = Image.open(image2_path)
        img0 = img0.convert("L")
        img1 = img1.convert("L")

        # Apply image transformations
        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        return (
            img0,
            img1,
            torch.from_numpy(
                np.array([int(self.train_df.iat[index, 2])], dtype=np.float32)
            ),
        )

    def __len__(self):
        return len(self.train_df)


class SiameseNetwork(nn.Module):
    """
    Siamese network for image similarity estimation.
    The network is composed of two identical networks, one for each input.
    The output of each network is concatenated and passed to a linear layer.
    The output of the linear layer passed through a sigmoid function.
    `"FaceNet" <https://arxiv.org/pdf/1503.03832.pdf>`_ is a variant of the Siamese network.
    This implementation varies from FaceNet as we use the `ResNet-18` model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_ as our feature extractor.
    In addition, we aren't using `TripletLoss` as the MNIST dataset is simple, so `BCELoss` can do the trick.
    """

    def __init__(self):
        super(SiameseNetwork, self).__init__()
        # get resnet model
        self.resnet = torchvision.models.resnet18(weights=None)

        # over-write the first conv layer to be able to read MNIST images
        # as resnet18 reads (3,x,x) where 3 is RGB channels
        # whereas MNIST has (1,x,x) where 1 is a gray-scale channel
        self.resnet.conv1 = nn.Conv2d(
            1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )
        self.fc_in_features = self.resnet.fc.in_features

        # remove the last layer of resnet18 (linear layer which is before avgpool layer)
        self.resnet = torch.nn.Sequential(*(list(self.resnet.children())[:-1]))

        # initialize the weights
        self.resnet.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward_once(self, x):
        output = self.resnet(x)
        output = output.view(output.size()[0], -1)
        return output

    def forward(self, input1, input2):
        # get two images' features
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        return output1, output2


# train the model
def train(model, device, optimizer, train_dataloader, train_losses, t_correct_set):
    criterion = nn.BCELoss()

    model.train()
    train_loss = 0
    correct = 0

    for _, (images_1, images_2, targets) in enumerate(train_dataloader, 0):
        images_1, images_2, targets = (
            images_1.to(device),
            images_2.to(device),
            targets.to(device),
        )
        optimizer.zero_grad()
        output1, output2 = model(images_1, images_2)
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss = criterion(output1, output2, targets)
        train_loss += loss
        loss.backward()
        optimizer.step()
        pred = torch.where(
            euclidean_distance > 0.5, 1, 0
        )  # get the index of the max log-probability
        correct += pred.eq(targets.view_as(pred)).sum().item()

    train_loss /= len(train_dataloader.dataset)
    print(
        "\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)".format(
            train_loss,
            correct,
            len(train_dataloader.dataset),
            100.0 * correct / len(train_dataloader.dataset),
        )
    )
    train_losses.append(train_loss.cpu().detach().numpy())
    t_correct_set.append(correct / len(train_dataloader.dataset))
    return train_loss


def test(model, device, eval_dataloader, val_losses, v_correct_set):
    criterion = nn.BCELoss()

    model.eval()
    val_loss = 0
    correct = 0

    with torch.no_grad():
        for images_1, images_2, targets in eval_dataloader:
            images_1, images_2, targets = (
                images_1.to(device),
                images_2.to(device),
                targets.to(device),
            )
            output1, output2 = model(images_1, images_2)
            loss = criterion(output1, output2, targets)
            val_loss += loss
            euclidean_distance = F.pairwise_distance(output1, output2)
            pred = torch.where(
                euclidean_distance > 0.5, 1, 0
            )  # get the index of the max log-probability
            correct += pred.eq(targets.view_as(pred)).sum().item()

    val_loss /= len(eval_dataloader.dataset)

    print(
        "\nVal set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)".format(
            val_loss,
            correct,
            len(eval_dataloader.dataset),
            100.0 * correct / len(eval_dataloader.dataset),
        )
    )
    val_losses.append(val_loss.cpu().detach().numpy())
    v_correct_set.append(correct / len(eval_dataloader.dataset))

    return val_loss


def main():
    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch Siamese network Example")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for testing (default: 64)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        metavar="N",
        help="number of epochs to train (default: 20)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        metavar="LR",
        help="learning rate (default: 0.001)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.1,
        metavar="M",
        help="Learning rate step gamma (default: 0.1)",
    )
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="quickly check a single pass",
    )
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        default=False,
        help="For Saving the current Model",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0005,
        metavar="L",
        help="Weight decay (default: 0.0005)",
    )
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    if use_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    train_kwargs = {"batch_size": args.batch_size}
    test_kwargs = {"batch_size": args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {"num_workers": 1, "pin_memory": True, "shuffle": True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    train_dataset = SiameseDataset(
        training_csv,
        training_dir,
        transform=transforms.Compose(
            [transforms.Resize((105, 105)), transforms.ToTensor()]
        ),
    )
    test_dataset = SiameseDataset(
        testing_csv,
        testing_dir,
        transform=transforms.Compose(
            [transforms.Resize((105, 105)), transforms.ToTensor()]
        ),
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, shuffle=True, **train_kwargs
    )
    test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=True, **test_kwargs)

    train_losses, test_losses, train_correct, test_correct = [], [], [], []
    best_eval_loss = 999

    # Declare Siamese Network
    model = SiameseNetwork().to(device)
    optimizer = torch.optim.Adadelta(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    scheduler = StepLR(optimizer, step_size=10, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        _ = train(
            args,
            model,
            device,
            train_loader,
            optimizer,
            epoch,
            train_losses,
            train_correct,
        )
        test_loss = test(model, device, test_loader, test_losses, test_correct)
        scheduler.step()

        print("-" * 20)

        if test_loss < best_eval_loss:
            best_eval_loss = test_loss
            print(f"Best Eval loss: {best_eval_loss}")
            torch.save(model.state_dict(), "content/model_contrastive.pth")
            print("Model Saved Successfully")
            print("-" * 20)


if __name__ == "__main__":
    main()
