import torch
from torch import optim, nn


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

    def train_short_term(self, state, action, reward, next_state, is_game_over):
        tensor_state = torch.tensor(state, dtype=torch.float)
        tensor_next_state = torch.tensor(next_state, dtype=torch.float)
        tensor_action = torch.tensor(action, dtype=torch.long)
        tensor_reward = torch.tensor(reward, dtype=torch.float)

        # 1: predicted Q values with current state
        pred = self.network(tensor_state)

        target = pred.clone()
        Q_new = tensor_reward
        if not is_game_over:
            Q_new = tensor_reward + self.gamma * torch.max(self.network(tensor_next_state))
        target[torch.argmax(tensor_action).item()] = Q_new

        # 2: Q_new = r + y * max(next_predicted Q value) -> only do this if not is_game_over
        # pred.clone()
        # preds[argmax(action)] = Q_new
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()

    def train_long_term(self, state, action, reward, next_state, is_game_over):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

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
