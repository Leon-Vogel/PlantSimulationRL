from ps_environment import Environment
from agents.reinforce_agent import ReinforceAgent
from plantsim.plantsim import Plantsim

pfad = 'D:\\Studium\Projekt\Methodenvergleich\PlantSimulationRL\simulations'
model = pfad + '\PickandPlace_diL_20220906_mit_Produktanteilen_Reinforce.spp'
plantsim = Plantsim(version='22.1', license_type='Educational', path_context='.Modelle.Modell', model=model,
                    socket=None, visible=True)

max_iterations = 500
it = 0
env = Environment(plantsim)
agent = ReinforceAgent(env.problem, file='tmp/policy.pth')
performance_train = []
agent.load()
# training
while it < max_iterations:
    complexity = max(1, env.problem.eval(env.problem))
    agent.train()
    evaluation = env.problem.evaluation
    performance_train.append(evaluation)
    print(it, performance_train[it])
    it += 1
    env.reset()

'''
Reinforce trainiert die Policy erst, wenn der goal state erreicht wurde. 
Deswegen ist der goal state erstmal als min. 10 fertige Produkte definiert.
Trotzdem zeigt das Training keine Verbesserung. Wahrscheinlich muss erst mit einer Sim. 
trainiert werden, wo man nur 5 Produkte im Lager benötigt, damit sie fertig sind.
'''