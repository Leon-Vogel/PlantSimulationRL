from ps_environment import Environment
import matplotlib.pyplot as plt
import numpy as np
from plantsim.plantsim import Plantsim
from agents.ppo_agent import PPOAgent
from utils import plot_learning_curve

pfad = 'D:\\Studium\Projekt\Methodenvergleich\PlantSimulationRL\simulations'
# model = pfad + '\MiniFlow_BE_based_MAS.spp'
model = pfad + '\Reihenfolgeplanung_diL_20220829_mit_komplettemLagerstand_SAC_neuer_Reward.spp'
# model = pfad + '\Reihenfolgeplanung_diL_20220827.spp'

# pfad = 'D:\\Studium\\3.Semester\DiskreteSimulation&RL\Projekt'
# model = pfad + '\game_assembly_20220825.2.spp'
plantsim = Plantsim(version='22.1', license_type='Educational', path_context='.Modelle.Modell', model=model,
                    socket=None, visible=True)


def get_actions2(act):
    a = [0, 0, 0]
    if act == 0:
        a = [1, 0, 0]
    elif act == 1:
        a = [0, 1, 0]
    elif act == 2:
        a = [0, 0, 1]
    return a

def get_actions(act):
    if np.argmax(act) == 1:
        a = 'Lager1'
    elif np.argmax(act) == 2:
        a = 'Lager2'
    else:
        a = 'Schleife'
    return a


if __name__ == '__main__':
    env = Environment(plantsim)  # env = gym.make('InvertedPendulumBulletEnv-v0')
    actions = env.problem.get_all_actions()
    observation = env.reset()
    env.problem.get_current_state()
    test = env.problem.state
    print(test)
    print(actions)
    agent = PPOAgent(input_dims=[len(test)], env=env.problem,
                      n_actions=len(actions))
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
            if action is not None:
                a = get_actions2(actions.index(action))  # Aktionen zu liste mit aktivierung transformieren
                done = env.problem.is_goal_state(current_state)
                score += r
                agent.remember(s, a, r, s_new, done)
                if not load_checkpoint:
                    agent.learn()
            if done:
                break

            action = get_actions(agent.choose_action(s_new))  # (observation)
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






'''
import gym
import numpy as np
from ppo_torch import Agent
from utils import plot_learning_curve

if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    N = 20
    batch_size = 5
    n_epochs = 4
    alpha = 0.0003
    agent = Agent(n_actions=env.action_space.n, batch_size=batch_size, 
                    alpha=alpha, n_epochs=n_epochs, 
                    input_dims=env.observation_space.shape)
    n_games = 300

    figure_file = 'plots/cartpole.png'

    best_score = env.reward_range[0]
    score_history = []

    learn_iters = 0
    avg_score = 0
    n_steps = 0

    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0
        while not done:
            action, prob, val = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            n_steps += 1
            score += reward
            agent.remember(observation, action, prob, val, reward, done)
            if n_steps % N == 0:
                agent.learn()
                learn_iters += 1
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

        print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score,
                'time_steps', n_steps, 'learning_steps', learn_iters)
    x = [i+1 for i in range(len(score_history))]
    plot_learning_curve(x, score_history, figure_file)



'''


