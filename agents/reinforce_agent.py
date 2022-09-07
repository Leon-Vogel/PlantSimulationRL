from agents.agent import Agent
import torch
import numpy as np
import torch.nn as nn


class PolicyNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, Optimizer=torch.optim.Adam, learning_rate=0.005,#3e-4,
                 gamma=0.99, transform=None):
        super(PolicyNetwork, self).__init__()

        self.num_actions = num_actions
        self.model = nn.Sequential(
            nn.Linear(num_inputs, 50),#128
            nn.ReLU(),
            nn.Linear(50, 20),#(128, 64)
            nn.ReLU(),
            nn.Linear(20, num_actions),
            nn.Softmax(1), )
        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = "cuda"
        self.to(self.device)
        self.optimizer = Optimizer(self.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.transform = transform

    def forward(self, state):
        state = self.transform(np.array(state))
        state = torch.tensor(state).float().unsqueeze(0).to(self.device)
        x = self.model(state)
        return x

    def get_max_action(self, state):
        # state = torch.from_numpy(np.array(state)).float().unsqueeze(0)
        probs = self.forward(state)
        highest_prob_action = np.argmax(np.squeeze(probs.cpu().detach().numpy()))
        return highest_prob_action

    def get_action(self, state):
        # state = torch.from_numpy(np.array(state)).float().unsqueeze(0)
        probs = self.forward(state)
        highest_prob_action = np.random.choice(self.num_actions, p=np.squeeze(probs.cpu().detach().numpy()))
        log_prob = torch.log(probs.squeeze(0)[highest_prob_action])
        return highest_prob_action, log_prob

    def update_policy(self, log_probabilities, rewards):
        discounted_rewards = []

        for t in range(len(rewards)):
            Gt = 0
            pw = 0
            for r in rewards[t:]:
                Gt = Gt + self.gamma ** pw * r
                pw = pw + 1
            discounted_rewards.append(Gt)

        discounted_rewards = torch.tensor(discounted_rewards)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (
                discounted_rewards.std() + 1e-9)  # normalize discounted rewards

        policy_gradient = []
        for log_prob, Gt in zip(log_probabilities, discounted_rewards):
            policy_gradient.append(-log_prob * Gt)

        self.optimizer.zero_grad()
        policy_gradient = torch.stack(policy_gradient).sum()
        policy_gradient.backward()
        self.optimizer.step()

    def save_model(self, file):
        print("---save---")
        torch.save(self.state_dict(), file)

    def load_model(self, file):
        print("---load---")
        self.load_state_dict(torch.load(file))


class ReinforceAgent(Agent):

    def __init__(self, problem, Optimizer=torch.optim.Adam, gamma=0.99, file="policy.npy", PolicyClass=PolicyNetwork):
        self.problem = problem
        self.actions = problem.get_all_actions()
        self.states = problem.get_all_states()
        self.file = file
        all_states = np.array(self.states)
        min_values = np.amin(all_states, axis=0)
        max_values = np.maximum(np.ones_like(self.states[0]), np.amax(all_states, axis=0))
        self.transform = lambda x: (x - min_values) / (max_values - min_values)
        self.policy = PolicyClass(len(self.states[0]), len(self.actions), Optimizer=Optimizer, gamma=gamma,
                                  transform=self.transform)

    def act(self):
        # perception
        s = self.problem.get_current_state().to_state()
        action_index = self.policy.get_max_action(s)
        action = self.actions[action_index]
        return action

    def train(self):
        log_probabilities = []
        rewards = []
        step = 0
        while True:
            step += 1
            current_state = self.problem.get_current_state()
            rewards.append(self.problem.get_reward(current_state))
            s = current_state.to_state()
            action_index, log_prob = self.policy.get_action(s)
            log_probabilities.append(log_prob)

            if self.problem.is_goal_state(current_state):
                if len(rewards) > 1:
                    print('\n Policy Update')
                    self.policy.update_policy(log_probabilities, rewards)
                    self.save()
                return
            if step > 600:
                return
            # act
            action = self.actions[action_index]
            self.problem.act(action)

    def save(self):
        self.policy.save_model(self.file)

    def load(self):
        self.policy.load_model(self.file)
