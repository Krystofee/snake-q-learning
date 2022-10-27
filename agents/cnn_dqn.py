import os
import random
from collections import deque
from copy import copy

import numpy as np
import torch
from torch import nn

from contants import GAME_SIZE
from trainer import Trainer

DIRECTION_UP = (0, -1)
DIRECTION_DOWN = (0, 1)
DIRECTION_LEFT = (-1, 0)
DIRECTION_RIGHT = (1, 0)


class CNNDQNetwork(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=(8, 8), stride=(4, 4), padding=7),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(16, 32, kernel_size=(4, 4), stride=(2, 2), padding=3),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Flatten(0, -1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, output_size)
        )

    def forward(self, x):
        return self.network(x)

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class CNNDQNAgent:
    MAX_MEMORY = 100000
    BATCH_SIZE = 10000
    LEARNING_RATE = 0.001

    def __init__(self):
        self.games_played = 0
        self.epsilon = 0
        self.gamma = 0.9
        self.memory = deque(maxlen=self.MAX_MEMORY)
        self.model = CNNDQNetwork(3)
        self.trainer = Trainer(self.model, learning_rate=self.LEARNING_RATE, gamma=self.gamma)

    def get_state(self, game):
        return list(game.pixels_memory)

    def remember(self, state, action, reward, next_state, is_game_over):
        self.memory.append((state, action, reward, next_state, is_game_over))

    def train_long_memory(self):
        if len(self.memory) > self.BATCH_SIZE:
            mini_sample = random.sample(self.memory, self.BATCH_SIZE)
        else:
            mini_sample = self.memory

        for state, action, reward, next_state, is_game_over in mini_sample:
            self.trainer.train_short_term(state, action, reward, next_state, is_game_over)

    def train_short_memory(self, state, action, reward, next_state, is_game_over):
        self.trainer.train_short_term(state, action, reward, next_state, is_game_over)

    def get_action(self, state, randomness=0):
        final_move = [0, 0, 0]

        if random.random() < randomness:
            move = random.randint(0, 1)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move