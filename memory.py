import random
from collections import deque


class ReplayMemory:
    """a circular buffer data structure for storing and sampling experiences"""

    def __init__(self, capacity, mini_batch_size):
        self.mini_batch_size = mini_batch_size
        self.memory = deque(maxlen=capacity)
        
    def insert(self, experience):
        if len(self.memory) >= self.memory.maxlen:
            self.memory.popleft()
        self.memory.append(experience)

    def draw(self):
        return random.sample(self.memory, self.mini_batch_size)

    def len(self):
        return len(self.memory)
