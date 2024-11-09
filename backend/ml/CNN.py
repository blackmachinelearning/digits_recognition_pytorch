import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchvision.transforms import ToTensor
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets
from torchvision.transforms import ToTensor
import cv2 as cv2
from sklearn.model_selection import train_test_split
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from backend.environment import Environment



class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=2, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=5, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(256, 512)  # Adjusted input size
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = x.view(-1, 256)  # Adjusted size here
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    # def __init__(self):
    #     super(CNN, self).__init__()
    #
    #     self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
    #     self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
    #     self.conv2_drop = nn.Dropout2d()
    #     self.fc1 = nn.Linear(320, 50)
    #     self.fc2 = nn.Linear(50, 10)
    #
    # def forward(self, x):
    #     x = x.view((-1, 1, 28, 28))  # Add the channel dimension
    #     x = F.relu(F.max_pool2d(self.conv1(x), 2))
    #     x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
    #     x = x.view(-1, 320)
    #     x = F.relu(self.fc1(x))
    #     x = F.dropout(x, training=self.training)
    #     x = self.fc2(x)
    #
    #     return F.softmax(x)


import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import MNIST

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                                ])


class CustomMNISTDataset(Dataset):
    def __init__(self, dataFrame=None, csv_file="", transform=None, target="label", torchDF=False):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        if dataFrame is None:
            self.mnist = pd.read_csv(csv_file)
        else:
            self.mnist = dataFrame
        self.transform = transform
        self.target = target
        self.torchDF = torchDF

    def __len__(self):
        return len(self.mnist)

    def resize_image(self, image):
        image = np.asfarray([float(i) for i in image])
        image = np.uint8(image)
        image = image.reshape(28, 28)  # cv2.cvtColor(, cv2.COLOR_GRAY2BGR)
        image_copy = image.copy()
        # Convert the image to grayscale
        if self.is_grayscale(image):
            gray_image = image
        else:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Threshold the image to make everything white except black
        _, thresholded = cv2.threshold(gray_image, 1, 255, cv2.THRESH_BINARY)

        # Find contours of the black regions
        contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # countour_image = cv2.drawContours(gray_image, contours, -1, (0, 255, 75), 2)
        # Find the bounding box of the black regions
        x, y, w, h = cv2.boundingRect(contours[len(contours) - 1])
        # Crop the region of interest
        cropped_image = image_copy[y:y + h, x:x + w]
        resized_image = cv2.resize(cropped_image, (28, 28))
        return resized_image.flatten()

    def __getitem__(self, idx):
        #         label = self.mnist.iloc[idx, :][self.target]  # Assuming the label is in the first column
        #         img = self.mnist.loc[:, self.mnist.columns != self.target].values.astype(np.uint8).reshape((28, 28, 1))
        #         print(img)
        if self.torchDF:
            img, label = self.mnist[idx]
            # label = self.mnist.iloc[idx, 0]  # Assuming the label is in the first column
            img1 = img[0].flatten()
            img = self.resize_image(img1).astype(np.uint8).reshape((28, 28, 1))
        else:
            label = self.mnist.iloc[idx, 0]  # Assuming the label is in the first column
            img = self.resize_image((self.mnist.iloc[idx, 1:])).astype(np.uint8).reshape((28, 28, 1))

        if self.transform:
            img = self.transform(img)

        return img, label

    def is_grayscale(self, image_array):
        if len(image_array.shape) == 2:
            return True
        elif len(image_array.shape) == 3 and image_array.shape[2] == 1:
            return True
        else:
            return False


# # Define your transformations (if any)
# transform = transforms.Compose([
#     transforms.ToTensor()
# ])

# Create instances of the custom dataset for training and testing
# train_dataset = CustomMNISTDataset(csv_file='/kaggle/input/digit-recognizer/train.csv', root='/kaggle/input/digit-recognizer/train.csv', transform=transform)
# test_dataset = CustomMNISTDataset(csv_file='/kaggle/input/digit-recognizer/test.csv', root='/kaggle/input/digit-recognizer/test.csv', transform=transform)


if __name__ == '__main__':
    # train_data = datasets.MNIST(
    #     root='data',
    #     train=True,
    #     transform=ToTensor(),
    #     download=True
    # )
    # test_data = datasets.MNIST(
    #     root='data',
    #     train=False,
    #     transform=ToTensor(),
    #     download=True
    # )
    # train_data = CustomMNISTDataset(dataFrame=train_data, transform=transform, target='label', torchDF=True)
    # test_data = CustomMNISTDataset(dataFrame=test_data, transform=transform, target='label', torchDF=True)
    #
    # loaders = {
    #     'train': DataLoader(
    #         train_data,
    #         batch_size=512,
    #         shuffle=True,
    #         num_workers=1
    #     ),
    #     'test': DataLoader(
    #         test_data,
    #         batch_size=512,
    #         shuffle=True,
    #         num_workers=1
    #     )
    # }
    trainDF = pd.read_csv("D:\Projects\Python\MachineLearning\JarvisGuessBot\\neuralNetwork\data\MNIST\\train.csv")
    # testDF = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")
    X = trainDF.loc[:, trainDF.columns != 'label']
    y = trainDF.label
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_train.insert(0, "label", y_train)
    X_test.insert(0, "label", y_test)
    train_dataset = CustomMNISTDataset(dataFrame=X_train, transform=transform, target='label')
    test_dataset = CustomMNISTDataset(dataFrame=X_test, transform=transform, target='label')
    loaders = {
        'train': DataLoader(
            train_dataset,
            batch_size=512,
            shuffle=True,
            num_workers=1
        ),
        'test': DataLoader(
            test_dataset,
            batch_size=512,
            shuffle=True,
            num_workers=1
        )
    }
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    _env = Environment()
    print(_env.models_paths_dict.get("mnist_pb"))
    print(f"File {_env.models_paths_dict.get('mnist_pb')} exists: {os.path.isfile(_env.models_paths_dict.get('mnist_pb'))}")
    
    if not os.path.isfile(_env.models_paths_dict.get("mnist_pb")):
        model = CNN().to(device)

        optimizer = optim.Adam(model.parameters(), lr=0.001)

        loss_fn = nn.CrossEntropyLoss()


        def train(epoch, loader_train=True):
            model.train()
            if loader_train:
                loader = loaders['train']
            else:
                loader = loaders['test']
            for batch_idx, (data, target) in enumerate(loader):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = loss_fn(output, target)
                loss.backward()
                optimizer.step()
                train_loader = loader
                if batch_idx % 20 == 0:
                    print(
                        f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\t{loss.item():.6f}')


        def test():
            model.eval()

            test_loss = 0
            correct = 0

            with torch.no_grad():
                for data, target in loaders['test']:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    test_loss += loss_fn(output, target).item()
                    pred = output.argmax(dim=1, keepdim=True)
                    correct += pred.eq(target.view_as(pred)).sum().item()

            test_loader = loaders['test']

            test_loss /= len(loaders['test'].dataset)
            print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%\n)')

        for epoch in range(1, 11):
            train(epoch)
            # train(epoch, loader_train=False)
        test()
        for epoch in range(1, 5):
            train(epoch, loader_train=False)

        torch.save(model.state_dict(), _env.models_paths_dict.get("mnist_pb"))
        # Define a dummy input that matches the model's input shape
    else:
        print("WE ARE IN ELSE STATE")
        model = CNN().to(device)
        model.load_state_dict(torch.load(_env.models_paths_dict.get("mnist_pb")))
        model.eval()
    
    dummy_input = torch.randn(1, 1, 28, 28, device=device)  # Batch size of 1, 1 channel, 28x28 image
    # Export the model to ONNX format
    torch.onnx.export(
        model,                      # Model to export
        dummy_input,                # Dummy input for tracing
        _env.models_paths_dict.get("mnist_onnx"),  # Path for saving ONNX model
        input_names=["input"],      # Name for the input layer
        output_names=["output"],    # Name for the output layer
        export_params=True          # Store the trained parameters in the model file
    )