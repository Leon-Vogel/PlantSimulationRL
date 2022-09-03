from ps_environment import Environment
from agents.q_learning_agent import QLearningAgent
from agents.deep_q_learning_agent import DeepQLearningAgent, DoubleDeepQLearningAgent
import matplotlib.pyplot as plt
import numpy as np
from plantsim.plantsim import Plantsim
from tqdm import tqdm

# doubleclick object in PlantSim and lookup the path_context
# socket is the name of the socket object in PlantSim or None if not used
pfad = 'D:\\Studium\Projekt\Methodenvergleich\PlantSimulationRL\simulations'
# model = pfad + '\MiniFlow_BE_based_MAS.spp'
# model = pfad + '\Methodenvergleich_20220902.spp'
model = pfad + '\PickandPlace_diL_20220903_mit_Lagerstand_neuer_R_mit_Durchlaufzeit.spp'
#model = pfad + '\PickandPlace_diL_20220902_mit_Lagerstand_neuer_R.spp'

plantsim = Plantsim(version='22.1', license_type='Educational', path_context='.Modelle.Modell', model=model,
                    socket=None, visible=True)

# set max number of iterations


max_iterations = 30
it = 0
env = Environment(plantsim)
# agent = QLearningAgent(env.problem)
# agent = DeepQLearningAgent(env.problem)
agent = DoubleDeepQLearningAgent(env.problem)
performance_train = []
q_table = agent.load()
# training
for it in tqdm(range(max_iterations), desc="Trainingsfortschritt: "):
    # while it < max_iterations:
    print(it)
    it += 1
    q_table, N_sa = agent.train()
    evaluation = env.problem.evaluation
    performance_train.append(evaluation)
    agent.save()
    env.reset()

#plantsim.quit()
# test_agent#
env = Environment(plantsim)
#agent = QLearningAgent(env.problem, q_table)
agent = DoubleDeepQLearningAgent(env.problem)
performance_test = []
number_of_tests = 20
it = 0
while it < number_of_tests:
    it += 1
    while not env.problem.is_goal_state(env.problem):
        action = agent.act()
        if action is not None:
            env.problem.act(action)
    evaluation = env.problem.evaluation
    performance_test.append(evaluation)
    env.reset()

# plot results
x = np.array(performance_train)
N = int(max_iterations / 10)
moving_average = np.convolve(x, np.ones(N) / N, mode='valid')
plt.plot(performance_train)
plt.plot(moving_average)
plt.show()

N = int(number_of_tests / 10)
x = np.array(performance_test)
moving_average = np.convolve(x, np.ones(N) / N, mode='valid')
plt.plot(performance_test)
plt.plot(moving_average)
plt.show()

# save q_table
# agent.save_q_table("agents/q_table.npy")
# plantsim.quit()
