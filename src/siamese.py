# import the necessary libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torchvision.utils
import config
from utils import imshow
from contrastive import ContrastiveLoss
import torchvision
from PIL import Image
import os
from tqdm import tqdm
import cv2

# load the dataset
training_dir = config.training_dir
testing_dir = config.testing_dir
training_csv = config.training_csv
testing_csv = config.testing_csv


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
        # self.resnet.apply(self.init_weights)

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
def train(
    model, device, criterion, optimizer, train_dataloader, train_losses, t_correct_set
):
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


def eval(model, device, criterion, eval_dataloader, val_losses, v_correct_set):
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
    acc = correct / len(eval_dataloader.dataset)

    print(
        "\nVal set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)".format(
            val_loss,
            correct,
            len(eval_dataloader.dataset),
            100.0 * acc,
        )
    )
    val_losses.append(val_loss.cpu().detach().numpy())
    v_correct_set.append(acc)

    return val_loss, acc

def main():
    # Load the the dataset from raw image folders
    siamese_dataset = SiameseDataset(
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the dataset as pytorch tensors using dataloader
    train_dataloader = DataLoader(
        siamese_dataset, shuffle=True, batch_size=config.batch_size
    )
    # eval_dataloader = DataLoader(eval_dataset, shuffle=True, batch_size=config.batch_size)
    test_dataloader = DataLoader(test_dataset, shuffle=True, batch_size=config.batch_size)

    # Declare Siamese Network
    model = SiameseNetwork().to(device)
    # Decalre Loss Function
    criterion = ContrastiveLoss()
    # Declare Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.0005)
    val_losses = []
    train_losses = []
    t_correct_set = []
    v_correct_set = []
    best_acc = 0
    # eval(model, device, eval_dataloader)
    
    for epoch in tqdm(range(1, 30)):
        train_loss = train(
            model,
            device,
            criterion,
            optimizer,
            train_dataloader,
            train_losses,
            t_correct_set,
        )
        val_loss, acc = eval(
            model, device, criterion, test_dataloader, val_losses, v_correct_set
        )

        print("-" * 20)

        # if acc > best_acc:
        #     best_acc = acc
        #     print(f"Best Accuracy: {best_acc}")
        torch.save(model.state_dict(), "../siamese/content/model_contrastive_new.pth")
        print("Model Saved Successfully")
        #     print("-" * 20)

    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for images_1, images_2, targets in test_dataloader:
            images_1, images_2, targets = (
                images_1.to(device),
                images_2.to(device),
                targets.to(device),
            )
            output1, output2 = model(images_1, images_2)
            loss = criterion(output1, output2, targets)
            test_loss += loss
            euclidean_distance = F.pairwise_distance(output1, output2)
            pred = torch.where(
                euclidean_distance > 1, 1, 0
            )  # get the index of the max log-probability
            correct += pred.eq(targets.view_as(pred)).sum().item()
    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss / len(test_dataloader.dataset),
            correct,
            len(test_dataloader.dataset),
            100.0 * correct / len(test_dataloader.dataset),
        )
    )
    
    plt.figure(figsize=(10, 5))
    plt.title("Training and Validation Loss")
    plt.plot(val_losses, label="val")
    plt.plot(train_losses, label="train")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig('loss.png')
    
    plt.figure(figsize=(10, 5))
    plt.title("Training and Validation Accuracy")
    plt.plot(v_correct_set, label="val")
    plt.plot(t_correct_set, label="train")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig('accuracy.png')
    
if __name__ == "__main__":
    main()
        