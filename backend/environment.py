import os

from dotenv import dotenv_values

import logging

class Environment:
    def __init__(self):
        env: dict = {
            **dotenv_values(".env"),
            **os.environ
        }
        logging.info(f"Environment dict is: {env}")
        self.models_path: str = env.get("MODEL_PATH")
        
        # print(env.get("MODEL_NAME_MNIST"))
        # print(env.get("MODEL_NAME_ONNX"))
        # print(env.get("MODEL_NAME_PB"))
        self.model_name: dict = {
            "mnist": env.get("MODEL_NAME_MNIST"),
            "mnist_onnx": env.get("MODEL_NAME_ONNX"),
            "mnist_pb": env.get("MODEL_NAME_PB"),
        }
        self.models_paths_dict: dict = {
            "mnist": os.path.join(env.get("MODEL_PATH"), env.get("MODEL_NAME_MNIST")),
            "mnist_onnx": os.path.join(env.get("MODEL_PATH"), env.get("MODEL_NAME_ONNX")),
            "mnist_pb": os.path.join(env.get("MODEL_PATH"), env.get("MODEL_NAME_PB")),
        }

