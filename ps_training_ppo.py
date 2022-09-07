from ps_environment import Environment
import numpy as np
from plantsim.plantsim import Plantsim
from agents.ppo_torch import PPOAgent
from utils import plot_learning_curve

pfad = 'D:\\Studium\Projekt\Methodenvergleich\PlantSimulationRL\simulations'
model = pfad + '\PickandPlace_diL_20220906_mit_Produktanteilen.spp'

if __name__ == '__main__':
    plantsim = Plantsim(version='22.1', license_type='Educational', path_context='.Modelle.Modell', model=model,
                        socket=None, visible=False)
    env = Environment(plantsim)  # env = gym.make('InvertedPendulumBulletEnv-v0')
    # Nachkommastelle an N anfügen, um das Lernen auszuschalten
    N = 1000  # 30 to 5000 steps between training
    batch_size = 512  # 4 to 4096
    n_epochs = 15  # 3 to 30 epochs in training
    alpha = 0.0002  # 0.0003

    actions = env.problem.get_all_actions()
    observation = env.reset()
    env.problem.get_current_state()
    test = env.problem.state

    agent = PPOAgent(input_dims=[len(test)], env=env.problem,
                     n_actions=len(actions), batch_size=batch_size,
                     alpha=alpha, n_epochs=n_epochs)

    max_iterations = 150
    filename = 'PPO_training.png'
    figure_file = 'plots/' + filename
    best_score = 10008
    performance_train = []
    learn_iters = 0
    avg_score = 0
    n_steps = 0
    load_checkpoint = True
    save_changes = True

    if load_checkpoint:
        agent.load_models()

    for i in range(max_iterations):
        env.reset()
        action = None
        s_new = None
        done = False
        score = 0
        step = 0
        count = 1
        current_state = env.problem.get_current_state()
        observation = current_state.to_state()
        # while True:
        while not done:
            if step > 10000:  # Reset des Agenten bei zu großer trajectory
                step = 0
                count = 1
                if load_checkpoint:
                    agent.load_models()
                env.reset()
                current_state = env.problem.get_current_state()
                observation = current_state.to_state()
            step += 1
            action, prob, val = agent.choose_action(observation)
            a = env.problem.actions[action]
            env.problem.act(a)
            current_state = env.problem.get_current_state()
            observation_ = current_state.to_state()
            reward = env.problem.get_reward(current_state)
            done = env.problem.is_goal_state(current_state)
            if reward == 10:
                count += 10
            print("Step " + str(step) + ": " + a + " - Reward: " + str(reward) + " - finished: " + str(
                count - 1) + " - " + str(round((step / count), 3)) + "\n")
            n_steps += 1
            score += reward
            agent.remember(observation, action, prob, val, reward, done)
            if n_steps % N == 0:
                agent.learn()
                learn_iters += 1
                print('---------learning--------')
            observation = observation_
        performance_train.append(score)
        avg_score = np.mean(performance_train[-1:])

        if avg_score > best_score:
            best_score = avg_score
            if save_changes:
                agent.save_models()

        print('///Episode', i, 'Return %.1f' % score, 'avg Return %.1f' % avg_score, 'best score %.1f' % best_score,
              'time_steps', n_steps, 'learning_steps', learn_iters)
        print('//////////////////////////////////////////////////////////////////////////////////////////////////// \n '
              '//////////////////////////////////////////////////////////////////////////////////////////////////// \n '
              '//////////////////////////////////////////////////////////////////////////////////////////////////// \n '
              '//////////////////////////////////////////////////////////////////////////////////////////////////// n ')
        # N += 50
    x = [i + 1 for i in range(len(performance_train))]
    plot_learning_curve(x, performance_train, figure_file)
