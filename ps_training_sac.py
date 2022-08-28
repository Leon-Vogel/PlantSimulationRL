from ps_environment import Environment
import matplotlib.pyplot as plt
import numpy as np
from plantsim.plantsim import Plantsim
from agents.sac_agent import SAC_Agent
from utils import plot_learning_curve


pfad = 'D:\\Studium\Projekt\Methodenvergleich\PlantSimulationRL\simulations'
# model = pfad + '\MiniFlow_BE_based_MAS.spp'
model = pfad + '\Reihenfolgeplanung_diL_20220828_mit_Lagerstand_SAC.spp'
# model = pfad + '\Reihenfolgeplanung_diL_20220827.spp'

# pfad = 'D:\\Studium\\3.Semester\DiskreteSimulation&RL\Projekt'
# model = pfad + '\game_assembly_20220825.2.spp'
plantsim = Plantsim(version='22.1', license_type='Educational', path_context='.Modelle.Modell', model=model,
                    socket=None, visible=True)

if __name__ == '__main__':
    env = Environment(plantsim)  # env = gym.make('InvertedPendulumBulletEnv-v0')
    actions = env.problem.get_all_actions()
    observation = env.reset()
    env.problem.get_current_state()
    test = env.problem.state
    print(test)
    print(actions)
    agent = SAC_Agent(env.problem, input_dims=[len(test)],
                      n_actions=len(env.problem.get_all_actions()))
    # SAC_Agent(input_dims=env.observation_space.shape, env=env,
    #         n_actions=env.action_space.shape[0])
    max_iterations = 250
    filename = 'sac_training.png'
    figure_file = 'plots/' + filename
    best_score = 0  # env.reward_range[0]
    performance_train = []  # score_history = []
    load_checkpoint = False

    if load_checkpoint:
        agent.load_models()
        # env.render(mode='human')

    for i in range(max_iterations):
        observation = env.reset()
        action = None
        s_new = None
        done = False
        score = 0
        step = 0
        while True:
            current_state = env.problem.get_current_state()
            step += 1
            r = env.problem.get_reward(current_state)
            s = s_new
            s_new = current_state.to_state()
            a = actions.index(action) #Aktionen zu liste mit aktivierung transformieren
            done = env.problem.is_goal_state(current_state)
            score += r
            agent.remember(s, a, r, s_new, done)
            if not load_checkpoint:
                agent.learn()
            if done:
                break

            action = agent.choose_action(current_state)  # (observation)
            print(action)
            env.problem.act(action)

        performance_train.append(score)
        avg_score = np.mean(performance_train[-1000:])

        if avg_score > best_score:
            best_score = avg_score
            if not load_checkpoint:
                agent.save_models()

        print('episode ', i, 'score %.1f' % score, 'avg_score %.1f' % avg_score)

    if not load_checkpoint:
        x = [i + 1 for i in range(max_iterations)]
        plot_learning_curve(x, performance_train, figure_file)
