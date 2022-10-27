import random
from collections import deque
from tkinter import *

import matplotlib.pyplot as plt
import numpy as np
from IPython import display

from agents.cnn_dqn import CNNDQNAgent
from agents.simple_dqn import SimpleDQNAgent
from config import GAME_SIZE

SHOULD_RENDER = False

plt.ion()


def render_rect(canvas, x, y, width, height, color):
    canvas.create_rectangle(x, y, x + width, y + height, fill=color)


def plot(scores, mean_scores, randomness_list, reward_sum_list):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.plot(randomness_list)
    # plt.plot(reward_sum_list)
    plt.ylim(ymin=0)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    plt.text(len(randomness_list)-1, randomness_list[-1], str(randomness_list[-1]))
    # plt.text(len(reward_sum_list)-1, reward_sum_list[-1], str(reward_sum_list[-1]))
    plt.show(block=False)
    plt.pause(.1)


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

    def move_absolute_up(self):
        self.direction = (0, -1)
        self.move_forward()

    def move_absolute_down(self):
        self.direction = (0, 1)
        self.move_forward()

    def move_absolute_left(self):
        self.direction = (-1, 0)
        self.move_forward()

    def move_absolute_right(self):
        self.direction = (1, 0)
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
    BLOCK_SIZE = 20
    MAX_STEPS_WITHOUT_EAT_UNTIL_GAMEOVER = 1000

    def __init__(self, tk, agent):
        self.agent = agent

        if tk:
            self.master = tk
            self.master.title("Snake AI")
            self.master.geometry(f"{GAME_SIZE[0]*self.BLOCK_SIZE}x{GAME_SIZE[1]*self.BLOCK_SIZE}+1000+100")
            self.master.resizable(False, False)
            self.master.configure(bg="black")

            self.canvas = Canvas(self.master, width=GAME_SIZE[0]*self.BLOCK_SIZE, height=GAME_SIZE[1]*self.BLOCK_SIZE, bg="black")
            self.canvas.pack()
        else:
            self.master = None
            self.canvas = None

        self.reset()

    def reset(self):

        self.steps_since_eaten_last_apple = 0
        self.snake = Snake(self.canvas, GAME_SIZE[0] // 2 - 3, GAME_SIZE[1] // 2, self.BLOCK_SIZE)
        self.apples = []
        self.randomly_place_apple()

        self.pixels_memory = deque(maxlen=2)

        self.pixels_memory.append(self.get_pixels())
        self.snake.move_absolute_right()
        self.pixels_memory.append(self.get_pixels())
        self.snake.move_absolute_right()
        self.pixels_memory.append(self.get_pixels())
        self.snake.move_absolute_right()
        self.pixels_memory.append(self.get_pixels())

        self.render()

    @staticmethod
    def get_point_distance(x1, y1, x2, y2):
        return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def play_step(self, move):
        start_score = self.score
        reward = 0

        start_distance = self.get_point_distance(self.snake.positions[0][0], self.snake.positions[0][1], self.apples[0].x, self.apples[0].y)

        self.update(move)
        self.update_pixels_memory()
        self.render()

        end_distance = self.get_point_distance(self.snake.positions[0][0], self.snake.positions[0][1], self.apples[0].x, self.apples[0].y)

        is_game_over = False
        if self.is_game_over():
            is_game_over = True
            reward -= 1
        else:
            # if end_distance < start_distance:
            #     reward += 0.1
            # if end_distance > start_distance:
            #     reward -= 0.1
            if start_score < self.score:
                reward += 1

        return (reward, is_game_over, self.score)

    def update(self, move):
        self.steps_since_eaten_last_apple += 1

        # If on apple, then grow and remove apple
        for apple in self.apples:
            if apple.x == self.snake.positions[0][0] and apple.y == self.snake.positions[0][1]:
                self.steps_since_eaten_last_apple = 0
                self.snake.grow()
                self.apples.remove(apple)

        if len(move) == 3:
            if move[0]:
                self.snake.move_forward()
            elif move[1]:
                self.snake.move_left()
            elif move[2]:
                self.snake.move_right()
            else:
                raise Exception("Invalid move")
        if len(move) == 4:
            if move[0]:
                self.snake.move_absolute_up()
            elif move[1]:
                self.snake.move_absolute_left()
            elif move[2]:
                self.snake.move_absolute_down()
            elif move[3]:
                self.snake.move_absolute_right()
            else:
                raise Exception("Invalid move")

        self.randomly_place_apple()

    def randomly_place_apple(self):
        if len(self.apples) < 1:
            coords = (random.randint(0, GAME_SIZE[0] - 1), random.randint(0, GAME_SIZE[1] - 1))
            for apple in self.apples:
                if apple.x == coords[0] and apple.y == coords[1]:
                    self.randomly_place_apple()
            self.apples.append(Apple(self.canvas, *coords, self.BLOCK_SIZE))

    def check_collision(self, x, y):
        if x < 0 or x >= GAME_SIZE[0] or y < 0 or y >= GAME_SIZE[1]:
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
        if self.snake.positions[0][0] < 0 or self.snake.positions[0][0] >= GAME_SIZE[0] or self.snake.positions[0][1] < 0 or self.snake.positions[0][1] >= GAME_SIZE[1]:
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

    def update_pixels_memory(self):
        self.pixels_memory.append(self.get_pixels())

    def get_pixels(self):
        pixels = np.zeros((GAME_SIZE[0], GAME_SIZE[1]), dtype=np.uint8)

        try:
            for position in self.snake.positions:
                pixels[position[0]][position[1]] = 1
        except IndexError:
            pass

        for apple in self.apples:
            pixels[apple.x][apple.y] = 1

        return pixels


def train_loop(tk, game, agent, record, total_score, plot_scores, plot_mean_scores, randomness_list, reward_sum, reward_sum_list):
    # Get old state
    state_old = agent.get_state(game)

    # Get move
    # coeff_until_game_over = game.steps_since_eaten_last_apple / game.MAX_STEPS_WITHOUT_EAT_UNTIL_GAMEOVER  # 0..1
    # randomness = (1 / ((agent.games_played + 1) / 2) + coeff_until_game_over / 4) / max(game.score - 5, 1) / 5
    randomness = 0.1
    final_move = agent.get_action(state_old, randomness)

    # Perform move and get new state
    reward, is_game_over, score = game.play_step(final_move)
    reward_sum += reward
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

        print('Game', agent.games_played, 'Score', score, 'Record:', record, "Reward", reward_sum, len(agent.memory))

        plot_scores.append(score)
        total_score += score
        mean_score = total_score / agent.games_played
        plot_mean_scores.append(mean_score)
        randomness_list.append(randomness)
        reward_sum_list.append(reward_sum)
        plot(plot_scores, plot_mean_scores, randomness_list, reward_sum_list)
        reward_sum = 0

    if SHOULD_RENDER:
        tk.after(10, lambda: train_loop(tk, game, agent, record, total_score, plot_scores, plot_mean_scores, randomness_list, reward_sum, reward_sum_list))
    else:
        return tk, game, agent, record, total_score, plot_scores, plot_mean_scores, randomness_list, reward_sum, reward_sum_list


def train(tk):
    agent = CNNDQNAgent()
    # agent = SimpleDQNAgent()
    game = SnakeGame(tk, agent)
    return train_loop(tk, game, agent, 0, 0, [], [], [], 0, [])


if __name__ == '__main__':
    if SHOULD_RENDER:
        tk = Tk()
        tk.after(1000, lambda: train(tk))
        tk.mainloop()
    else:
        args = train(None)

        while True:
            args = train_loop(*args)
