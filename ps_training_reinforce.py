from ps_environment import Environment
from agents.q_learning_agent import QLearningAgent
from agents.reinforce_agent import ReinforceAgent
import matplotlib.pyplot as plt
import numpy as np
from plantsim.plantsim import Plantsim

# doubleclick object in PlantSim and lookup the path_context
# socket is the name of the socket object in PlantSim or None if not used
pfad = 'D:\\Studium\Projekt\Methodenvergleich\PlantSimulationRL\simulations'
#model = pfad + '\MiniFlow_BE_based_MAS.spp'
model = pfad + '\Reihenfolgeplanung_diL_20220829_mit_Lagerstand.spp'
#model = pfad + '\Reihenfolgeplanung_diL_20220829_mit_komplettemLagerstand_SAC_neuer_Reward.spp'

#pfad = 'D:\\Studium\\3.Semester\DiskreteSimulation&RL\Projekt'
#model = pfad + '\game_assembly_20220825.2.spp'
plantsim = Plantsim(version='22.1', license_type='Educational', path_context='.Modelle.Modell', model=model,
                    socket=None, visible=True)

# set max number of iterations


max_iterations = 500
it = 0
env = Environment(plantsim)
agent = ReinforceAgent(env.problem, file='agents/policy.pth')
performance_train = []
# training
while it < max_iterations:
    complexity = max(1, env.problem.eval(env.problem))
    agent.train()
    evaluation = env.problem.evaluation
    performance_train.append(evaluation)
    print(it, performance_train[it])
    it += 1
    env.reset()
