import numpy as np

from skimage.color import rgb2gray
from skimage.transform import resize


class State:

    def __init__(self, height, width, depth):
        self.height = height
        self.width = width
        self.content = np.zeros((width, height, depth), dtype=np.uint8)

    def update(self, observation):
        last = self.pre_process(observation, self.height, self.width)
        self.content[:, :, :-1] = self.content[:, :, 1:]
        self.content[:, :, -1] = last

    @staticmethod
    def pre_process(observation, height, width):
        return np.uint8(resize(image=rgb2gray(observation),
                               output_shape=(height, width),
                               mode='constant')*255)
