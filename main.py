import os
import random
from collections import deque
from tkinter import *

import matplotlib.pyplot as plt
import torch
from IPython import display
from torch import nn
from torch import optim
from torch.nn import functional as F

SHOULD_RENDER = True

plt.ion()


def render_rect(canvas, x, y, width, height, color):
    canvas.create_rectangle(x, y, x + width, y + height, fill=color)


def plot(scores, mean_scores, randomness_list):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.plot(randomness_list)
    plt.ylim(ymin=0)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    plt.text(len(randomness_list)-1, randomness_list[-1], str(randomness_list[-1]))
    plt.show(block=False)
    plt.pause(.1)


DIRECTION_UP = (0, -1)
DIRECTION_DOWN = (0, 1)
DIRECTION_LEFT = (-1, 0)
DIRECTION_RIGHT = (1, 0)


class Snake:
    def __init__(self, canvas, x, y, block_size):
        self.canvas = canvas
        self.block_size = block_size
        self.positions = [(x, y), (x - 1, y), (x - 2, y)]
        self.direction = (1, 0)

    def render(self):
        for position in self.positions:
            render_rect(self.canvas, position[0] * self.block_size, position[1] * self.block_size, self.block_size, self.block_size, "green")

    def move(self, dx, dy):
        self.positions = [(self.positions[0][0] + dx, self.positions[0][1] + dy)] + self.positions[:-1]

    def move_forward(self):
        self.move(self.direction[0], self.direction[1])

    def move_left(self):
        self.direction = (self.direction[1], self.direction[0] * -1)
        self.move_forward()

    def move_right(self):
        self.direction = (self.direction[1] * -1, self.direction[0])
        self.move_forward()

    def grow(self):
        self.positions.append(self.positions[-1])

    def check_collision(self):
        return self.positions[0] in self.positions[1:]


class Apple:
    def __init__(self, canvas, x, y, block_size):
        self.canvas = canvas
        self.x = x
        self.y = y
        self.block_size = block_size

    def render(self):
        render_rect(self.canvas, self.x * self.block_size, self.y * self.block_size, self.block_size, self.block_size, "red")


class SnakeGame:
    SIZE_Y = 20
    SIZE_X = 20
    BLOCK_SIZE = 20
    MAX_STEPS_WITHOUT_EAT_UNTIL_GAMEOVER = 1000

    def __init__(self, tk, agent):
        self.agent = agent

        if tk:
            self.master = tk
            self.master.title("Snake AI")
            self.master.geometry(f"{self.SIZE_X*self.BLOCK_SIZE}x{self.SIZE_Y*self.BLOCK_SIZE}+1000+100")
            self.master.resizable(False, False)
            self.master.configure(bg="black")

            self.canvas = Canvas(self.master, width=self.SIZE_X*self.BLOCK_SIZE, height=self.SIZE_Y*self.BLOCK_SIZE, bg="black")
            self.canvas.pack()
        else:
            self.master = None
            self.canvas = None

        self.reset()

    def reset(self):
        self.steps_since_eaten_last_apple = 0
        self.snake = Snake(self.canvas, self.SIZE_X // 2, self.SIZE_Y // 2, self.BLOCK_SIZE)
        self.apples = []
        self.randomly_place_apple()
        self.render()

    def play_step(self, move):
        start_score = self.score
        reward = 0

        self.update(move)
        self.render()

        is_game_over = False
        if self.is_game_over():
            is_game_over = True
            reward = -10
        elif start_score < self.score:
            reward = 10

        return (reward, is_game_over, self.score)

    def update(self, move):
        self.steps_since_eaten_last_apple += 1

        # If on apple, then grow and remove apple
        for apple in self.apples:
            if apple.x == self.snake.positions[0][0] and apple.y == self.snake.positions[0][1]:
                self.steps_since_eaten_last_apple = 0
                self.snake.grow()
                self.apples.remove(apple)

        if move[0]:
            self.snake.move_forward()
        elif move[1]:
            self.snake.move_left()
        elif move[2]:
            self.snake.move_right()
        else:
            raise Exception("Invalid move")

        self.randomly_place_apple()

    def randomly_place_apple(self):
        if len(self.apples) < 1:
            coords = (random.randint(0, self.SIZE_X - 1), random.randint(0, self.SIZE_Y - 1))
            for apple in self.apples:
                if apple.x == coords[0] and apple.y == coords[1]:
                    self.randomly_place_apple()
            self.apples.append(Apple(self.canvas, *coords, self.BLOCK_SIZE))

    def check_collision(self, x, y):
        if x < 0 or x >= self.SIZE_X or y < 0 or y >= self.SIZE_Y:
            return True

        if self.check_snake_collision(x, y):
            return True

        return False

    def check_snake_collision(self, x, y):
        for position in self.snake.positions:
            if x == position[0] and y == position[1]:
                return True
        return False

    def render(self):
        if self.master:
            self.canvas.delete("all")
            for apple in self.apples:
                apple.render()
            self.snake.render()
            self.canvas.create_text(10, 10, text=f"Score: {self.score}", fill="white", anchor="nw")

    def is_game_over(self):
        # Check snake wall collision
        if self.snake.positions[0][0] < 0 or self.snake.positions[0][0] >= self.SIZE_X or self.snake.positions[0][1] < 0 or self.snake.positions[0][1] >= self.SIZE_Y:
            return True

        # Check snake self collision
        if self.snake.check_collision():
            return True

        if self.steps_since_eaten_last_apple > self.MAX_STEPS_WITHOUT_EAT_UNTIL_GAMEOVER:
            return True

        return False

    @property
    def score(self):
        return len(self.snake.positions) - 3


class Network(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.input_linear = nn.Linear(input_size, hidden_size)
        self.output_linear = nn.Linear(hidden_size, output_size)

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


class Trainer:
    def __init__(self, network, learning_rate=0.001, gamma=0.9):
        self.network = network
        self.optimizer = optim.Adam(network.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.criterion = nn.MSELoss()

    def train(self, state, action, reward, next_state, is_game_over):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        # (n, x)

        if len(state.shape) == 1:
            # (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            is_game_over = (is_game_over, )

        # 1: predicted Q values with current state
        pred = self.network(state)

        target = pred.clone()
        for idx in range(len(is_game_over)):
            Q_new = reward[idx]
            if not is_game_over[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.network(next_state[idx]))
            target[idx][torch.argmax(action[idx]).item()] = Q_new

        # 2: Q_new = r + y * max(next_predicted Q value) -> only do this if not is_game_over
        # pred.clone()
        # preds[argmax(action)] = Q_new
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()


class DQNAgent:
    MAX_MEMORY = 100_000
    BATCH_SIZE = 1_000
    LEARNING_RATE = 0.001
    MAX_RANDOMNESS = 50

    def __init__(self):
        self.games_played = 0
        self.epsilon = 0
        self.gamma = 0.9
        self.memory = deque(maxlen=self.MAX_MEMORY)
        self.model = Network(19, 256, 3)
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
        self.trainer.train(state, action, reward, next_state, is_game_over)

    def train_short_memory(self, state, action, reward, next_state, is_game_over):
        self.trainer.train(state, action, reward, next_state, is_game_over)

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


def train_loop(tk, game, agent, record, total_score, plot_scores, plot_mean_scores, randomness_list):
    # Get old state
    state_old = agent.get_state(game)

    # Get move
    coeff_until_game_over = game.steps_since_eaten_last_apple / game.MAX_STEPS_WITHOUT_EAT_UNTIL_GAMEOVER  # 0..1
    randomness = (1 / ((agent.games_played + 1) / 2) + coeff_until_game_over / 4) / max(game.score - 5, 1)
    final_move = agent.get_action(state_old, randomness)

    # Perform move and get new state
    reward, is_game_over, score = game.play_step(final_move)
    state_new = agent.get_state(game)

    # Train short memory
    agent.train_short_memory(state_old, final_move, reward, state_new, is_game_over)

    # Remember
    agent.remember(state_old, final_move, reward, state_new, is_game_over)

    if is_game_over:
        # Train long memory, plot result
        game.reset()
        agent.games_played += 1

        agent.train_long_memory()

        if score > record:
            record = score
            agent.model.save()

        print('Game', agent.games_played, 'Score', score, 'Record:', record, "Reward", reward)

        plot_scores.append(score)
        total_score += score
        mean_score = total_score / agent.games_played
        plot_mean_scores.append(mean_score)
        randomness_list.append(randomness)
        plot(plot_scores, plot_mean_scores, randomness_list)

    if SHOULD_RENDER:
        tk.after(10, lambda: train_loop(tk, game, agent, record, total_score, plot_scores, plot_mean_scores, randomness_list))
    else:
        train_loop(tk, game, agent, record, total_score, plot_scores, plot_mean_scores, randomness_list)


def train(tk):
    agent = DQNAgent()
    game = SnakeGame(tk, agent)
    train_loop(tk, game, agent, 0, 0, [], [], [])


if __name__ == '__main__':
    if SHOULD_RENDER:
        tk = Tk()
        tk.after(1000, lambda: train(tk))
        tk.mainloop()
    else:
        import sys
        sys.setrecursionlimit(100000)
        train(None)
