import os
import random
from collections import deque

import torch
from torch import nn
from torch.functional import F

from config import device
from trainer import Trainer


DIRECTION_UP = (0, -1)
DIRECTION_DOWN = (0, 1)
DIRECTION_LEFT = (-1, 0)
DIRECTION_RIGHT = (1, 0)


class SimpleDQNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.input_linear = nn.Linear(input_size, hidden_size).to(device)
        self.output_linear = nn.Linear(hidden_size, output_size).to(device)

    def forward(self, x):
        x = F.relu(self.input_linear(x))
        x = self.output_linear(x)
        return x

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class SimpleDQNAgent:
    MAX_MEMORY = 100_000
    BATCH_SIZE = 5_000
    LEARNING_RATE = 0.001

    def __init__(self):
        self.games_played = 0
        self.epsilon = 0
        self.gamma = 0.95
        self.memory = deque(maxlen=self.MAX_MEMORY)
        self.model = SimpleDQNetwork(19, 256, 3)
        self.trainer = Trainer(self.model, learning_rate=self.LEARNING_RATE, gamma=self.gamma)

    def get_state(self, game):
        snake_head = game.snake.positions[0]

        # Detect danger in the direction of snake
        danger_1_forward = game.check_collision(snake_head[0] + game.snake.direction[0], snake_head[1] + game.snake.direction[1])
        danger_1_left = game.check_collision(snake_head[0] - game.snake.direction[1], snake_head[1] + game.snake.direction[0])
        danger_1_right = game.check_collision(snake_head[0] + game.snake.direction[1], snake_head[1] - game.snake.direction[0])

        danger_directions = (
            (1, 0),
            (1, 1),
            (0, 1),
            (-1, 1),
            (-1, 0),
            (-1, -1),
            (0, -1),
            (1, -1),
        )
        dangers = []
        for danger_dir in danger_directions:
            danger = False
            for n in range(1, 3):
                danger |= game.check_snake_collision(snake_head[0] + danger_dir[0] * n, snake_head[1] + danger_dir[1] * n)
            dangers.append(danger)

        # Snake direction
        direction_up = game.snake.direction == DIRECTION_UP
        direction_down = game.snake.direction == DIRECTION_DOWN
        direction_left = game.snake.direction == DIRECTION_LEFT
        direction_right = game.snake.direction == DIRECTION_RIGHT

        # Detect apple direction
        apple_direction = [0, 0, 0, 0]  # up, down, left, right

        if game.apples:
            apple = game.apples[0]
            if apple.y < snake_head[1]:
                apple_direction[0] = 1
            elif apple.y > snake_head[1]:
                apple_direction[1] = 1
            if apple.x < snake_head[0]:
                apple_direction[2] = 1
            elif apple.x > snake_head[0]:
                apple_direction[3] = 1

        # State size is: 1171
        state = [
            int(danger_1_forward),
            int(danger_1_left),
            int(danger_1_right),

            *dangers,

            int(direction_up),
            int(direction_down),
            int(direction_left),
            int(direction_right),

            *apple_direction,
        ]

        # def print_state(state):
        #     print("danger", int(state[0]), int(state[1]), int(state[2]), "direction", int(state[3]), int(state[4]), int(state[5]), int(state[6]), "apple", state[7], state[8], state[9], state[10])
        #
        # print_state(state)

        return state

    def remember(self, state, action, reward, next_state, is_game_over):
        self.memory.append((state, action, reward, next_state, is_game_over))

    def train_long_memory(self):
        if len(self.memory) > self.BATCH_SIZE:
            mini_sample = random.sample(self.memory, self.BATCH_SIZE)
        else:
            mini_sample = self.memory

        state, action, reward, next_state, is_game_over = zip(*mini_sample)
        self.trainer.train_long_term(state, action, reward, next_state, is_game_over)

    def train_short_memory(self, state, action, reward, next_state, is_game_over):
        self.trainer.train_short_term(state, action, reward, next_state, is_game_over)

    def get_action(self, state, randomness=0):
        final_move = [0, 0, 0]

        if random.random() < randomness:
            move = random.randint(0, 1)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float, device=device)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move