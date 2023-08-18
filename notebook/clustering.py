import os
import re
import warnings

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from skimage.metrics import structural_similarity
from skimage.registration import phase_cross_correlation
from tqdm import tqdm

from IPython.display import Image as IPythonImage

# Set pandas options
pd.options.mode.chained_assignment = None
warnings.filterwarnings("ignore")
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set random seed for reproducibility (if needed)
# torch.manual_seed(42)
# np.random.seed(42)

# Transformations
transform = transforms.Compose([transforms.Resize((105, 105)), transforms.ToTensor()])


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
    
transform = transforms.Compose([transforms.Resize((105, 105)), transforms.ToTensor()])


def transformation(transform, image1_path):
    img = Image.open(image1_path)
    img = img.convert("L")

    # Apply image transformations
    img = transform(img)
    return img


def register(img1_path, img2_path):
    imgRef = cv2.imread(img1_path)
    imgTest = cv2.imread(img2_path)
    # Convert to grayscale.
    imgTest_grey = cv2.cvtColor(imgTest, cv2.COLOR_BGR2GRAY)
    imgRef_grey = cv2.cvtColor(imgRef, cv2.COLOR_BGR2GRAY)

    height1, width1 = imgRef_grey.shape
    height2, width2 = imgTest_grey.shape

    height_max = max(height1, height2)
    width_max = max(width1, width2)

    # Pad the images to same size
    imgRef_grey = cv2.copyMakeBorder(
        imgRef_grey,
        height_max - height1,
        0,
        width_max - width1,
        0,
        cv2.BORDER_REPLICATE,
    )
    imgTest_grey = cv2.copyMakeBorder(
        imgTest_grey,
        height_max - height2,
        0,
        width_max - width2,
        0,
        cv2.BORDER_REPLICATE,
    )

    # pixel precision first
    shift, _, _ = phase_cross_correlation(imgRef_grey, imgTest_grey)
    # print(f"Detected subpixel offset (y, x): {shift}")

    height_offset, weith_offset = int(shift[0]), int(shift[1])

    # Compute Similarity
    if height_offset > 0 and weith_offset > 0:
        imgRef_grey = imgRef_grey[height_offset:, weith_offset:]
        imgTest_grey = imgTest_grey[:-height_offset, :-weith_offset]

    elif height_offset > 0 and weith_offset < 0:
        imgRef_grey = imgRef_grey[height_offset:, :weith_offset]
        imgTest_grey = imgTest_grey[:-height_offset, -weith_offset:]

    elif height_offset < 0 and weith_offset > 0:
        imgRef_grey = imgRef_grey[:height_offset, weith_offset:]
        imgTest_grey = imgTest_grey[-height_offset:, :-weith_offset]

    elif height_offset < 0 and weith_offset < 0:
        imgRef_grey = imgRef_grey[:height_offset, :weith_offset]
        imgTest_grey = imgTest_grey[-height_offset:, -weith_offset:]

    elif height_offset == 0 and weith_offset > 0:
        imgRef_grey = imgRef_grey[:, weith_offset:]
        imgTest_grey = imgTest_grey[:, :-weith_offset]

    elif height_offset == 0 and weith_offset < 0:
        imgRef_grey = imgRef_grey[:, :weith_offset]
        imgTest_grey = imgTest_grey[:, -weith_offset:]

    elif height_offset > 0 and weith_offset == 0:
        imgRef_grey = imgRef_grey[height_offset:, :]
        imgTest_grey = imgTest_grey[:-height_offset, :]

    elif height_offset < 0 and weith_offset == 0:
        imgRef_grey = imgRef_grey[:height_offset, :]
        imgTest_grey = imgTest_grey[-height_offset:, :]

    elif height_offset == 0 and weith_offset == 0:
        imgRef_grey = imgRef_grey
        imgTest_grey = imgTest_grey

    imgRef_grey = transform(Image.fromarray(imgRef_grey).convert("L"))
    imgTest_grey = transform(Image.fromarray(imgTest_grey).convert("L"))

    return imgRef_grey.to(device), imgTest_grey.to(device)

def computer_sim_score(img1_path, img2_path):
    imgRef_grey, imgTest_grey = registration(img1_path, img2_path)

    (score, diff) = structural_similarity(
        imgRef_grey,
        imgTest_grey,
        full=True,
    )

    return score

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SiameseNetwork().to(device)
    model.load_state_dict(torch.load("/fs9/yuxiao/nii_project/siamese/content/model_contrastive_test.pth"))
    model.eval()

    df = pd.read_csv("/fs9/yuxiao/nii_project/dataset/df/feature_data.csv")

    label_list = np.unique(df.label)
    df["block"] = np.nan

    for i in tqdm(range(len(label_list))):
        count = 0
        char = re.sub("[\W\d_]+", "", label_list[i])
        for idx1, row1 in df.loc[df["label"] == label_list[i]].iterrows():
            if np.isnan(df.loc[idx1, "block"]):
                count += 1
                df.loc[idx1, "block"] = count
                img1 = transformation(transform, "../dataset/2/" + row1.image_file).to(
                    device
                )
                directory = f"../dataset/char_block/{char}/{label_list[i]}/{count}"
                if not os.path.exists(directory):
                    os.makedirs(directory)
                Image.open(f"../dataset/1/{row1.image_file}").save(
                    f"{directory}/{row1.image_file}"
                )
                for idx2, row2 in df.loc[df["label"] == label_list[i]].iterrows():
                    if np.isnan(df.loc[idx2, "block"]):
                        img2 = transformation(
                            transform, "../dataset/2/" + row2.image_file
                        ).to(device)
                        # img1, img2 = register(
                        #     "../dataset/2/" + row1.image_file,
                        #     "../dataset/2/" + row2.image_file,
                        # )
                        output1, output2 = model(img1.unsqueeze(0), img2.unsqueeze(0))
                        euclidean_distance = F.pairwise_distance(output1, output2)
                        sim_score = computer_sim_score(
                            "../dataset/2/" + row1.image_file,
                            "../dataset/2/" + row2.image_file,
                        )
                        if ((euclidean_distance < 1.5) or (sim_score > 0.95)
                    ):
                            
                            df.loc[idx2, "block"] = count
                            Image.open(f"../dataset/1/{row2.image_file}").save(
                                f"{directory}/{row2.image_file}"
                            )

    df.to_csv("/fs9/yuxiao/nii_project/dataset/temp_df/feature_data_2.csv", index=False)


if __name__ == "__main__":
    main()




