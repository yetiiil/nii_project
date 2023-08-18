import os
import pickle

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import Model
from tensorflow.keras.utils import load_img

from PIL import Image
from tqdm import tqdm

pd.options.mode.chained_assignment = None  # default='warn'
PATH_DF = "../dataset/1.json"
PATH = "../dataset/1"
PATH_BLUR_IMAGE = "../dataset/2/"
PATH_FEATURES = "../dataset/feature.pkl"


def try_convert_val(value):
    try:
        return value["jibo"]
    except:
        return np.nan


def read_dataframe(path):
    df = pd.read_json(path)
    df["char"] = df["data"].apply(lambda x: x["char"])
    df["url"] = df["data"].apply(lambda x: "" + x["url"][28:])
    df["image_file"] = df["url"].apply(lambda x: x[10:])
    df["jibo"] = df["data"].apply(lambda x: try_convert_val(x))
    df = df[["char", "url", "image_file", "jibo"]]
    return df


def read_images_name(df, path):
    chars = []

    # creates a ScandirIterator aliased as files
    with os.scandir(path) as files:
        # loops through each file in the directory
        chars.extend(
            file.name
            for file in files
            if file.name.endswith(".jpg") and file.name in df["image_file"].values
        )
    return chars


def blur_binarize_images(df, path_blur_image):
    print("Start binarizing and blurring the images")
    for url in tqdm(df.iloc[15000:].url):
        img = cv2.imread(url)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, dst1 = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
        blur = cv2.blur(dst1, (2, 2))
        cv2.imwrite(path_blur_image + url[10:], blur)


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


def extract_features(chars, path, path_blur_image):
    # load the model first and pass as an argument
    model = VGG16()
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)

    data = {}
    # lop through each image in the dataset
    for char in tqdm(chars):
        # try to extract the features and update the dictionary
        feat = features_extractor(path_blur_image + char, model)
        data[char] = feat

    with open(path, "wb") as fp:
        pickle.dump(data, fp)
        print("dictionary saved successfully to file")


def main():
    print("Start reading DataFrame")
    df = read_dataframe(PATH_DF)
    print("Start reading Images")
    chars = read_images_name(df, PATH)
    # print("Start binarizing Images")
    # blur_binarize_images(df, PATH_BLUR_IMAGE)
    print("Start extracting features")
    extract_features(chars, PATH_FEATURES, PATH_BLUR_IMAGE)


if __name__ == "__main__":
    main()
