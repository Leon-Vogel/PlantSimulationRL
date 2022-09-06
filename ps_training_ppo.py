from ps_environment import Environment
import matplotlib.pyplot as plt
import numpy as np
from plantsim.plantsim import Plantsim
# from agents.ppo_agent import PPOAgent
from agents.ppo_torch import PPOAgent
from utils import plot_learning_curve

# pfad = 'D:\\Studium\Projekt\Methodenvergleich\PlantSimulationRL\simulations'
pfad = 'E:\\Studium\Projekt\Methodenvergleich\PlantSimulationRL_Git\simulations'
# model = pfad + '\MiniFlow_BE_based_MAS.spp'
# model = pfad + '\PickandPlace_diL_20220828_mit_Lagerstand_SAC_neuer_Reward.spp'
# model = pfad + '\PickandPlace_diL_20220903_mit_Lagerstand_neuer_R_mit_Durchlaufzeit.spp'
# model = pfad + '\PickandPlace_diL_20220906_mit_Lagerstand_neuer_R_mit_Durchlaufzeit.spp'
model = pfad + '\PickandPlace_diL_20220906_mit_Lagerstand_wenigR.spp'

if __name__ == '__main__':
    plantsim = Plantsim(version='22.1', license_type='Educational', path_context='.Modelle.Modell', model=model,
                        socket=None, visible=True)
    env = Environment(plantsim)  # env = gym.make('InvertedPendulumBulletEnv-v0')
    N = 50
    batch_size = 64
    n_epochs = 30
    alpha = 0.0001

    actions = env.problem.get_all_actions()
    observation = env.reset()
    env.problem.get_current_state()
    test = env.problem.state

    print(test)
    print(actions)

    agent = PPOAgent(input_dims=[len(test)], env=env.problem,
                     n_actions=len(actions), batch_size=batch_size,
                     alpha=alpha, n_epochs=n_epochs)

    max_iterations = 50
    filename = 'PPO_training.png'
    figure_file = 'plots/' + filename
    best_score = 1  # env.reward_range[0]
    performance_train = []  # score_history = []
    learn_iters = 0
    avg_score = 0
    n_steps = 0
    load_checkpoint = False  # False
    save_changes = True

    if load_checkpoint:
        agent.load_models()
        # env.render(mode='human')

    for i in range(max_iterations):
        env.reset()  # observation = env.reset()
        action = None
        s_new = None
        done = False
        score = 0
        step = 0
        current_state = env.problem.get_current_state()
        observation = current_state.to_state()
        # while True:
        while not done:
            action, prob, val = agent.choose_action(observation)
            a = env.problem.actions[action]
            env.problem.act(a)
            current_state = env.problem.get_current_state()
            observation_ = current_state.to_state()
            reward = env.problem.get_reward(current_state)
            done = env.problem.is_goal_state(current_state)
            n_steps += 1
            score += reward
            agent.remember(observation, action, prob, val, reward, done)
            if n_steps % N == 0:
                agent.learn()
                learn_iters += 1
                print('learning')
            observation = observation_
        performance_train.append(score)
        avg_score = np.mean(performance_train[-100:])

        if avg_score > best_score:
            best_score = avg_score
            if save_changes:
                agent.save_models()

        print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score,
              'time_steps', n_steps, 'learning_steps', learn_iters)
    x = [i + 1 for i in range(len(performance_train))]
    plot_learning_curve(x, performance_train, figure_file)
