import os
import pickle

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import Model
from tensorflow.keras.utils import load_img
from tqdm import tqdm

pd.options.mode.chained_assignment = None  # default='warn'
PATH_DF = "/Users/liyuxiao/Downloads/nii_project/dataset/1.json"
PATH = "/Users/liyuxiao/Downloads/nii_project/dataset/1"
PATH_FEATURES = "/Users/liyuxiao/Downloads/nii_project/dataset/feature_test_data.pkl"


def read_dataframe(path):
    df = pd.read_json(path)
    df["char"] = df["data"].apply(lambda x: x["char"])
    df["url"] = df["data"].apply(
        lambda x: "/Users/liyuxiao/Downloads/nii_project/" + x["url"][28:]
    )

    df.loc[:, "image_file"] = df["url"].apply(lambda x: x[48:])
    return df


def read_images_name(df, path):
    os.chdir(path)
    # this list holds all the image filename
    chars = []

    # creates a ScandirIterator aliased as files
    with os.scandir(path) as files:
        # loops through each file in the directory
        for file in files:
            if file.name.endswith(".jpg") and file.name in df["image_file"].values:
                chars.append(file.name)
    return chars


def blur_binarize_images(df):
    print("Start binarizing and blurring the images")
    for url in tqdm(df.url):
        img = cv2.imread(url)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, dst1 = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
        blur = cv2.blur(dst1, (2, 2))
        cv2.imwrite(url[:46] + "3/" + url[48:], blur)


def features_extractor(file, model):
    # load the image as a 224x224 array
    img = load_img(file, target_size=(224, 224))
    # convert from 'PIL.Image.Image' to numpy array
    img = np.array(img)
    # reshape the data for the model reshape(num_of_samples, dim 1, dim 2, channels)
    reshaped_img = img.reshape(1, 224, 224, 3)
    # prepare image for model
    imgx = preprocess_input(reshaped_img)
    # get the feature vector
    features = model.predict(imgx, use_multiprocessing=True, verbose=0)
    return features


def extract_features(chars, path):
    # load the model first and pass as an argument
    model = VGG16()
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)

    data = {}
    # lop through each image in the dataset
    for char in tqdm(chars):
        # try to extract the features and update the dictionary
        feat = extract_features(
            "/Users/liyuxiao/Downloads/nii_project/dataset/3/" + char, model
        )
        data[char] = feat

    with open(path, "wb") as fp:
        pickle.dump(data, fp)
        print("dictionary saved successfully to file")


def main():
    df = read_dataframe(PATH_DF)
    chars = read_images_name(df, PATH)
    blur_binarize_images(df)
    extract_features(chars, PATH_FEATURES)
