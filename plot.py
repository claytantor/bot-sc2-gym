import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display


def show_screen(screen, name):
    plt.figure(2)
    plt.imshow(screen.cpu().squeeze(0).permute(1, 2, 0).numpy(),
            interpolation='none')
    plt.title(name)
    plt.show()

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


def plot_scores(episode_scores, map_name="scores"):
    plt.figure(1)
    plt.clf()
    # scores_t = torch.tensor(episode_scores, dtype=torch.float)
    plt.title('Training {}...'.format(map_name))
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.plot(moving_average(episode_scores, 2))
    plt.plot(moving_average(episode_scores, 20))

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())                       

