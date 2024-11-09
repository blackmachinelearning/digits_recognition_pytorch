import os
import secrets

import cv2
import numpy as np
import torch as torch
from scipy.special import expit
from scipy.special import logit
import random
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
import pickle
import scipy.misc
import imageio
import glob
from PIL import Image
from PIL import ImageOps
from torch.utils.data import DataLoader

from configs.config import MODEL_PATH1
from neuralNetwork.CNN import CNN, CustomMNISTDataset, transform, PreprocessImages

# def resize_image(img_array, target_size=(28, 28)):
#     # Resizing the image array
#     resized_img = np.zeros(target_size)
#
#     # Resizing using simple interpolation (average pooling)
#     for i in range(target_size[0]):
#         for j in range(target_size[1]):
#             x_mapped = int(i * (img_array.shape[0] / target_size[0]))
#             y_mapped = int(j * (img_array.shape[1] / target_size[1]))
#             resized_img[i, j] = np.mean(img_array[x_mapped:x_mapped + 2, y_mapped:y_mapped + 2])
#
#     return resized_img.astype(np.uint8)


def resize_image(image):
    image = np.asfarray([float(i) for i in image])
    image = np.uint8(image)
    image = image.reshape(28, 28)  # cv2.cvtColor(, cv2.COLOR_GRAY2BGR)
    image_copy = image.copy()
    # Convert the image to grayscale
    if is_grayscale(image):
        gray_image = image
    else:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold the image to make everything white except black
    _, thresholded = cv2.threshold(gray_image, 1, 255, cv2.THRESH_BINARY)
    plt.imshow(thresholded)
    plt.show()

    # Find contours of the black regions
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    countour_image = cv2.drawContours(gray_image, contours, -1, (0, 255, 75), 2)
    plt.imshow(countour_image)
    plt.show()
    # Find the bounding box of the black regions
    x, y, w, h = cv2.boundingRect(contours[len(contours) - 1])
    # Crop the region of interest
    cropped_image = image_copy[y:y + h, x:x + w]
    resized_image = cv2.resize(cropped_image, (28, 28))
    plt.imshow(cropped_image)
    plt.show()
    return resized_image.flatten()


def is_grayscale(image_array):
    if len(image_array.shape) == 2:
        return True
    elif len(image_array.shape) == 3 and image_array.shape[2] == 1:
        return True
    else:
        return False


class NNPredictor:
    def __init__(self, fileNamePath: str):
        self.filePath = fileNamePath
        self.neuralNetwork = self.loadModel()

    def loadModel(self) -> Pipeline:
        with open(self.filePath, 'rb') as file:
            loaded_model = pickle.load(file)
        return loaded_model

    def predict(self, imageFileName: str):
        return self.predictSub(self.getImageData(imageFileName))

    def predictSub(self, X, **predict_params):
        return self.neuralNetwork.predict(X)

    def getImageData(self, imageFileName):
        image_file_name = imageFileName
        img_array = imageio.imread(image_file_name, mode='F')  # Load as grayscale

        folderName = 'reshaped'
        if not os.path.exists(folderName):
            os.mkdir(folderName)
        ppIm = PreprocessImages
        
        img = Image.open(imageFileName)

        img_name = secrets.token_hex(8)
        filePath1 = f'{folderName}/{img_name}.png'
        img.save(filePath1, format='PNG')
        # Resize the image to 28x28
        print(img_array)

        resized_img = ppIm.resize_image(img_array)
        # reshape from 28x28 to list of 784 values, invert values
        img_data = resized_img.reshape(784, 1)
        # then scale data to range from 0.01 to 1.0
        # img_data = img_data
        pixel_columns = [f'{i}x{j}' for i in range(1, 29) for j in range(1, 29)]
        return pd.DataFrame(img_data.T, columns=pixel_columns)


class CNNPredictor:
    def __init__(self, fileNamePath: str):
        self.filePath = fileNamePath
        self.cnn = self.loadModel()

    def loadModel(self):
        # with open(self.filePath, 'rb') as file:
        #     loaded_model = pickle.load(file)
        model = CNN()
        model.load_state_dict(torch.load(self.filePath))
        model.eval()
        return model

    def predict(self, imageFileName: str):
        return self.predictSub(self.getImageData(imageFileName))

    def predictSub(self, X, **predict_params):
        randy = np.random.randint(0, 10, (len(X),))
        X.insert(0, "label", randy)
        X['label'] = randy
        check_dataset = CustomMNISTDataset(dataFrame=X, transform=transform, target='label')
        check_data_loader = DataLoader(check_dataset, batch_size=512, shuffle=False, num_workers=0)

        with torch.no_grad():
            for data, target in check_data_loader:
                output = self.cnn(data)
        return output.argmax(dim=1, keepdim=True)

    def getImageData(self, imageFileName):
        image_file_name = imageFileName
        img_array = imageio.v2.imread(image_file_name, mode='F')  # Load as grayscale
        # imgshow(title=f'label', image=img_array, size=3)

        folderName = 'reshaped'
        if not os.path.exists(folderName):
            os.mkdir(folderName)
        
        img = Image.open(imageFileName)

        img_name = secrets.token_hex(8)
        filePath1 = f'{folderName}/{img_name}.png'
        img.save(filePath1, format='PNG')
        # Resize the image to 28x28

        ppIm = PreprocessImages(img_array)
        resized_img = ppIm.get_resized_image()
        # imgshow(title=f'label', image=resized_img.reshape((28, 28)), size=3)
        # reshape from 28x28 to list of 784 values, invert values
        img_data = resized_img.reshape(784, 1)
        # then scale data to range from 0.01 to 1.0
        # img_data = img_data
        pixel_columns = [f'{i}x{j}' for i in range(1, 29) for j in range(1, 29)]
        return pd.DataFrame(img_data.T, columns=pixel_columns)

def imgshow(title="", image = None, size = 6):
    w, h = image.shape[0], image.shape[1]
    aspect_ratio = w/h
    plt.figure(figsize=(size * aspect_ratio,size))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()
if __name__ == '__main__':
    filePath = "D:\Projects\Python\MachineLearning\JarvisGuessBot\downloaded\\f308101315282588.png"
    # nn = NNPredictor(fileNamePath=FILEPATH)
    # prediction = nn.predict(filePath)
    cnn = CNNPredictor(fileNamePath=MODEL_PATH1)
    prediction = cnn.predict(filePath)
    print(prediction)
