from agents.agent import Agent
import numpy as np


class QLearningAgent(Agent):

    def __init__(self, problem, q_table=None, N_sa=None, gamma=0.99, max_N_exploration=100, R_Max=100,
                 q_table_file="q_table.npy"):
        super().__init__(problem)
        self.actions = problem.get_all_actions()
        self.states = problem.get_all_states()
        if q_table is not None:
            self.q_table = q_table
        else:
            self.q_table = {}
        if N_sa is not None:
            self.N_sa = N_sa
        else:
            self.N_sa = {}
        self.gamma = gamma
        self.max_N_exploration = max_N_exploration
        self.R_Max = R_Max
        self.file = q_table_file

    def act(self):
        # perception
        s = self.problem.get_current_state().to_state()
        # lookup in q_table
        action = self.actions[np.argmax(self.q_table[s])]
        return action

    def train(self):
        action = None
        s_new = None
        while True:
            current_state = self.problem.get_current_state()
            r = self.problem.get_reward(current_state)
            s = s_new
            s_new = current_state.to_state()
            if s_new not in self.N_sa.keys():
                self.N_sa[s_new] = np.zeros(len(self.actions))
                self.q_table[s_new] = np.zeros(len(self.actions))
            if action is not None:
                a = self.actions.index(action)
                self.N_sa[s][a] += 1
                self.update_q_values(s, a, r, s_new, self.problem.is_goal_state(current_state))
            if self.problem.is_goal_state(current_state):
                return self.q_table, self.N_sa
            action = self.choose_GLIE_action(self.q_table[s_new], self.N_sa[s_new])
            # act
            self.problem.act(action)

    def update_q_values(self, s, a, r, s_new, is_goal_state):
        if is_goal_state:
            self.q_table[s][a] = self.q_table[s][a] + self.alpha(s, a) * (r-self.q_table[s][a])
        else:
            self.q_table[s][a] = self.q_table[s][a] + self.alpha(s, a) * (r + self.gamma * np.max(self.q_table[s_new]) -
                                                                      self.q_table[s][a])

    def choose_GLIE_action(self, q_values, N_s):
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
        if np.random.random() < (self.max_N_exploration+0.00001)/(np.max(N_s)+0.00001):
            action = np.random.choice(self.actions, p=probabilities)
        else:
            action_indexes = np.argwhere(probabilities == np.amax(probabilities))
            action_indexes.shape = (action_indexes.shape[0])
            action_index = np.random.choice(action_indexes)
            action = self.actions[action_index]
        return action

    def save_q_table(self):
        np.save(self.file, self.q_table)

    def load_q_table(self):
        self.q_table = np.load(self.file)

    def alpha(self, s, a):
        # learnrate alpha decreases with N_sa for convergence
        alpha = self.N_sa[s][a]**(-1/2)
        return alpha
