from agents.q_learning_agent import QLearningAgent
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch
import os
from copy import deepcopy


class ExperienceReplay(Dataset):
    def __init__(self, model, max_memory=50, gamma=0.99, transform=None, target_transform=None):
        self.model = model
        self.memory = []
        self.max_memory = max_memory
        self.gamma = gamma
        self.transform = transform
        self.target_transform = target_transform

    def remember(self, experience, game_over):
        # Save a state to memory
        self.memory.append([experience, game_over])
        # We don't want to store infinite memories, so if we have too many, we just delete the oldest one
        if len(self.memory) > self.max_memory:
            del self.memory[0]

    def update_model(self, model):
        self.model = model

    def __len__(self):
        return len(self.memory)

    def __getitem__(self, idx):
        s, a, r, s_new = self.memory[idx][0]
        goal_state = self.memory[idx][1]
        features = np.array(s)
        # init labels with old prediction (and later overwrite action a)
        label = self.model[s]
        if goal_state:
            label[a] = r
        else:
            label[a] = r + self.gamma * max(self.model[s_new])

        if self.transform:
            features = self.transform(features)
        if self.target_transform:
            label = self.target_transform(label)

        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        features = torch.from_numpy(features).float().to(device)
        label = torch.from_numpy(label).float().to(device)

        return features, label


class DeepQTable(nn.Module):

    def __init__(self, number_of_states, number_of_actions, Optimizer, loss_fn, transform):
        super(DeepQTable, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(number_of_states, 40),
            nn.ReLU(),
            nn.Linear(40, 20),
            nn.ReLU(),
            nn.Linear(20, number_of_actions), )
        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = "cuda"
        self.to(self.device)
        self.optimizer = Optimizer(self.parameters())
        self.loss_fn = loss_fn
        self.transform = transform

    def __getitem__(self, state):
        state = self.transform(np.array(state))
        state = torch.tensor(state).float().to(self.device)
        prediction = self(state)
        return prediction.cpu().detach().numpy()

    def __setitem__(self, state, value):
        # ignoring setting to values
        pass

    def forward(self, x):
        return self.model(x)

    def perform_training(self, dataloader):
        loss_history = []
        (X, y) = next(iter(dataloader))
        # Compute prediction and loss
        pred = self(X)
        loss = self.loss_fn(pred, y)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        loss_history.append(loss)
        return loss_history

    def save_model(self, file):
        torch.save(self.state_dict(), file)

    def load_model(self, file):
        self.load_state_dict(torch.load(file))


class DeepDuelingQTable(DeepQTable):
    def __init__(self, number_of_states, number_of_actions, Optimizer, loss_fn, transform):
        super(DeepDuelingQTable, self).__init__(number_of_states, number_of_actions, Optimizer, loss_fn, transform)
        self.input_network = nn.Sequential(
            nn.Linear(number_of_states, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU())
        self.value_network = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1))
        self.advantage_network = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, number_of_actions))
        self.to(self.device)
        self.optimizer = Optimizer(self.parameters())

    def forward(self, x):
        features = self.input_network(x)
        values = self.value_network(features)
        advantages = self.advantage_network(features)
        return values + (advantages - advantages.mean())


class DeepQLearningAgent(QLearningAgent):

    def __init__(self, problem, q_table=None, N_sa=None, gamma=0.99, max_N_exploration=100, R_Max=100,
                 q_table_file="deep_q_table.pth", batch_size=10, Optimizer=torch.optim.Adam, loss_fn=nn.MSELoss(),
                 ModelClass=DeepDuelingQTable):  # DeepQTable):
        super().__init__(problem, q_table=q_table, N_sa=N_sa, gamma=gamma, max_N_exploration=max_N_exploration,
                         R_Max=R_Max, q_table_file=q_table_file)
        all_states = np.array(self.states)
        # print(all_states)
        min_values = np.amin(all_states, axis=0)
        max_values = np.maximum(np.ones_like(self.states[0]), np.amax(all_states, axis=0))
        transform = lambda x: (x - min_values) / (max_values - min_values)
        if q_table is None:
            self.q_table = self.create_model(Optimizer, loss_fn, transform, ModelClass)
        self.batch_size = batch_size
        self.experience_replay = ExperienceReplay(self.q_table, transform=transform)
        self.loss_history = []

    def create_model(self, Optimizer, loss_fn, transform, ModelClass):
        return ModelClass(len(self.states[0]), len(self.actions), Optimizer, loss_fn, transform)

    def update_q_values(self, s, a, r, s_new, is_goal_state):
        self.experience_replay.remember((s, a, r, s_new), is_goal_state)
        if len(self.experience_replay) > self.batch_size:
            train_loader = DataLoader(self.experience_replay, batch_size=self.batch_size, shuffle=True)
            self.loss_history += self.q_table.perform_training(train_loader)

    def save(self):
        self.q_table.save_model(self.file)

    def load(self):
        self.q_table.load_model(self.file)


class DoubleDeepQLearningAgent(DeepQLearningAgent):

    def __init__(self, problem, extra=None, q_table=None, N_sa=None, gamma=0.9, max_N_exploration=100, R_Max=100,
                 q_table_file="double_deep_q_table.pth", batch_size=10, Optimizer=torch.optim.Adam,
                 loss_fn=nn.MSELoss(), ModelClass=DeepQTable,
                 update_interval=40):  # DeepQTableDeepDuelingQTable, update_interval=40):
        super().__init__(problem, q_table=q_table, N_sa=N_sa, gamma=gamma, max_N_exploration=max_N_exploration,
                         R_Max=R_Max, batch_size=batch_size, q_table_file=q_table_file, Optimizer=Optimizer,
                         loss_fn=loss_fn, ModelClass=ModelClass)
        self.online_q_table = deepcopy(self.q_table)
        self.update_count = 0
        self.update_interval = update_interval
        self.counter = 1
        self.rand_act = 0
        self.act = 0
        self.environment = extra

    def update_q_values(self, s, a, r, s_new, is_goal_state):
        self.experience_replay.remember((s, a, r, s_new), is_goal_state)
        if len(self.experience_replay) > self.batch_size:
            train_loader = DataLoader(self.experience_replay, batch_size=self.batch_size, shuffle=True)
            # training is now done on the online network
            self.loss_history += self.online_q_table.perform_training(train_loader)
            self.update_count += 1
            # update target network with online network
            if self.update_count % self.update_interval == 0:
                self.q_table = deepcopy(self.online_q_table)
                self.experience_replay.update_model(self.q_table)

    def act(self):
        # perception
        s = self.problem.get_current_state().to_state()
        # lookup in q_table
        action = self.actions[np.argmax(self.q_table[s])]
        return action

    def save(self, file="q_table.npy"):
        torch.save(self.online_q_table.state_dict(), file)
        print('Save successful')
        # self.online_q_table.save_model("q_table.npy")

    def load(self, file="q_table.npy"):
        #if os.path.exists("D:\Studium\Projekt\Methodenvergleich\PlantSimulationRL\q_table.npy"):
        if os.path.exists(file):
            self.q_table.load_model(file)
            self.online_q_table.load_model(file)
        else:
            print('Keine Datei')

    def train(self):
        action = None
        s_new = None
        step = 0
        self.act = 0
        self.rand_act = 0
        # self.load()
        while True:
            #self.environment.stop_simulation()
            current_state = self.problem.get_current_state()
            step += 1
            r = self.problem.get_reward(current_state)
            s = s_new
            s_new = current_state.to_state()
            # if self.problem.step % 2001 == 0:
            # self.save()
            if s_new not in self.N_sa.keys():
                self.N_sa[s_new] = np.zeros(len(self.actions))
                self.online_q_table[s_new] = np.zeros(len(self.actions))
            if action is not None:
                a = self.actions.index(action)
                self.N_sa[s][a] += 1
                self.update_q_values(s, a, r, s_new, self.problem.is_goal_state(current_state))
            if self.problem.is_goal_state(current_state):
                self.q_table = deepcopy(self.online_q_table)
                # self.save()
                return self.q_table, self.N_sa
            action = self.choose_GLIE_action(self.online_q_table[s_new], self.N_sa[s_new])
            # act
            self.problem.act(action)

    def choose_GLIE_action(self, q_values, N_s):
        self.counter += 1
        exploration_values = np.ones_like(q_values) * self.R_Max
        # which state/action pairs have been visited sufficiently
        no_sufficient_exploration = N_s < self.max_N_exploration
        # turn cost to a positive number
        q_values_pos = self.R_Max / 2 + q_values
        # select the relevant values (q or max value)
        max_values = np.maximum(exploration_values * no_sufficient_exploration, q_values_pos)
        # assure that we do not dived by zero
        if max_values.sum() == 0:
            probabilities = np.ones_like(max_values) / max_values.size
        else:
            probabilities = max_values / max_values.sum()
        # select action according to the (q) values
        # Aktionsauswahl mit epsilon greedy selection
        # Um zu trainieren das False: # entfernen
        if ((np.random.random()**2) - 0.5) > (((self.counter + 0)/ 20000)) * np.random.random():
            self.rand_act += 1
            print('Random'+'  -------  '+str(round(self.act/(self.act+self.rand_act),3)))
            action = np.random.choice(self.actions, p=probabilities)
        else:  # if True:
            action_indexes = np.argwhere(probabilities == np.amax(probabilities))
            action_indexes.shape = (action_indexes.shape[0])
            action_index = np.random.choice(action_indexes)
            action = self.actions[action_index]
            self.act += 1
            print('Not Random ' + action+' ------- '+str(round(self.act/(self.act+self.rand_act),3)))
        return action
