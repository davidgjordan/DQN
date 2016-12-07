import numpy as np
import cv2
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

    # TODO: compare performance of cv2 to skimage

    @staticmethod
    def pre_process(observation, height, width):
        return cv2.resize(cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY), (height, width))
        # return np.uint8(resize(rgb2gray(observation), (width, height)) * 255)
