from ps_environment import Environment
from agents.deep_q_learning_agent import DoubleDeepQLearningAgent
import matplotlib.pyplot as plt
import numpy as np
from plantsim.plantsim import Plantsim
from tqdm import tqdm

pfad = 'D:\\Studium\Projekt\Methodenvergleich\PlantSimulationRL\simulations'
# pfad = 'E:\\Studium\Projekt\Methodenvergleich\PlantSimulationRL_Git\simulations'
# model = pfad + '\PickandPlace_diL_20220906_mit_Produktanteilen.spp'
model = pfad + '\PickandPlace_diL_20220902_mit_Lagerstand_neuer_R.spp'
file = 'q_table.npy'

plantsim = Plantsim(version='22.1', license_type='Educational', path_context='.Modelle.Modell', model=model,
                    socket=None, visible=True)

max_iterations = 30
it = 0
env = Environment(plantsim)
agent = DoubleDeepQLearningAgent(env.problem)
performance_train = []
q_table = agent.load(file)
# training
for it in tqdm(range(max_iterations), desc="Trainingsfortschritt: "):
    # while it < max_iterations:
    print(it)
    it += 1
    q_table, N_sa = agent.train()
    evaluation = env.problem.evaluation
    performance_train.append(evaluation)
    agent.save(file)
    env.reset()

# test_agent#
env = Environment(plantsim)
agent = DoubleDeepQLearningAgent(env.problem)
q_table = agent.load()
performance_test = []
number_of_tests = 20
it = 0
env.reset()
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
