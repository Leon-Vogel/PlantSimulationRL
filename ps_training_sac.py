from ps_environment import Environment
import numpy as np
from plantsim.plantsim import Plantsim
from agents.sac_agent import SAC_Agent
from utils import plot_learning_curve, get_actions, get_actions2

pfad = 'D:\\Studium\Projekt\Methodenvergleich\PlantSimulationRL\simulations'
#model = pfad + '\PickandPlace_diL_20220906_mit_Produktanteilen.spp'
model = pfad + '\Methodenvergleich_20220909.spp'

plantsim = Plantsim(version='22.1', license_type='Educational', path_context='.Modelle.Modell', model=model,
                    socket=None, visible=False)

if __name__ == '__main__':
    env = Environment(plantsim)  # env = gym.make('InvertedPendulumBulletEnv-v0')
    actions = env.problem.get_all_actions()
    observation = env.reset()
    env.problem.get_current_state()
    test = env.problem.state
    print(test)
    print(actions)
    agent = SAC_Agent(input_dims=[len(test)], env=env.problem,
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
        count = 1
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
            if r == 10:
                count += 10
            print("Step " + str(step) + ": " + action + " - Reward: " + str(r) + " - finished: " + str(
                count - 1) + " - " + str(round((step / count), 3)) + "\n")

            env.problem.act(action)

        performance_train.append(score)
        avg_score = np.mean(performance_train[-1:])

        if avg_score > best_score:
            best_score = avg_score
            if not load_checkpoint:
                agent.save_models()

        print('episode ', i, 'score %.1f' % score, 'avg_score %.1f' % avg_score)

    if not load_checkpoint:
        x = [i + 1 for i in range(max_iterations)]
        plot_learning_curve(x, performance_train, figure_file)
