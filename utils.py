import numpy as np
import matplotlib.pyplot as plt


def plot_learning_curve(x, scores, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i - 100):(i + 1)])
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
    plt.savefig(figure_file)


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
    # probs = softmax(act)
    # a = random.choices(population=['Schleife','Lager1','Lager2'], weights=probs)
    if np.argmax(act) == 1:
        a = 'Lager1'
    elif np.argmax(act) == 2:
        a = 'Lager2'
    else:
        a = 'Schleife'
    return a  # [0]
