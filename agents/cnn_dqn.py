import os
import random
from collections import deque

import numpy as np
import torch
from torch import nn

from config import device
from trainer import Trainer

DIRECTION_UP = (0, -1)
DIRECTION_DOWN = (0, 1)
DIRECTION_LEFT = (-1, 0)
DIRECTION_RIGHT = (1, 0)


class CNNDQNetwork(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(5, 5), stride=(1, 1), padding=4),
            nn.ReLU(),
            nn.Flatten(0, -1),
            # nn.Linear(9216, 64),
            # nn.ReLU(),
            nn.Linear(30976, output_size),
        ).to(device)

    def forward(self, x):
        return self.network(x)

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class CNNDQNAgent:
    MAX_MEMORY = 1_000_000
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001

    def __init__(self):
        self.games_played = 0
        self.epsilon = 0
        self.gamma = 0.9
        self.memory = deque(maxlen=self.MAX_MEMORY)
        self.model = CNNDQNetwork(4)
        self.trainer = Trainer(self.model, learning_rate=self.LEARNING_RATE, gamma=self.gamma)

    def get_state(self, game):
        return np.array(game.pixels_memory)

    def remember(self, state, action, reward, next_state, is_game_over):
        self.memory.append((state, action, reward, next_state, is_game_over))

    def train_long_memory(self):
        if len(self.memory) > self.BATCH_SIZE:
            mini_sample = random.sample(self.memory, self.BATCH_SIZE)
        else:
            mini_sample = self.memory

        # for state, action, reward, next_state, is_game_over in mini_sample:
        for args in mini_sample:
            self.trainer.train_short_term(*args)

        # state, action, reward, next_state, is_game_over = zip(*mini_sample)
        # self.trainer.train_long_term(state, action, reward, next_state, is_game_over)

    def train_short_memory(self, state, action, reward, next_state, is_game_over):
        self.trainer.train_short_term(state, action, reward, next_state, is_game_over)

    def get_action(self, state, randomness=0):
        final_move = [0, 0, 0, 0]  # UP, DOWN, LEFT, RIGHT

        if random.random() < randomness:
            move = random.randint(0, 3)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float, device=device)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move