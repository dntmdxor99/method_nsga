import torch

class SingletonMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]

class ModelHandler(metaclass=SingletonMeta):
    def __init__(self, model_path):
        self.model = torch.load(model_path)
        self.model.eval()

    def get_model(self):
        return self.model