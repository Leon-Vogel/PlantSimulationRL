class Problem:

    def copy(self):
        """
        Creates a deep copy of itself
        :return: Problem
        """
        pass

    def act(self, action):
        """
        Peforms the action passed
        """
        pass

    def to_state(self):
        """
        Creates a tuple of the relevant state attributes
        :return: tuple()
        """
        pass

    def is_goal_state(self, state):
        """
        Checks if state is a goal state
        :param state: Problem
        :return: Boolean
        """
        pass

    def get_applicable_actions(self, state):
        """
        Returns a list of actions applicable in state
        :param state: Problem
        :return: list<String>
        """
        actions = []
        return actions

    def get_current_state(self):
        """
        returns itself and eventually performs an update first
        :return:
        """
        return self

    def eval(self, state):
        """
        Evaluates the state
        :param state: Problem
        :return: float
        """
        pass

    def get_all_actions(self):
        """
        returns a list of all actions
        :return: list<string>
        """
        actions = []
        return actions

    def get_all_states(self):
        """
        returns a list of all states
        :return: list<string>
        """
        states = []
        return states

    def get_reward(self, state):
        """
        Calulates a reward of the state for RL
        :param state: Problem
        :return: float
        """
        r = 0
        return r

    def reset(self):
        """
        resets the environment
        """
        pass






