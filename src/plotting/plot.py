import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns
import pandas as pd
from matplotlib.patches import Rectangle


OUTPUT_PATH = Path.joinpath(Path(__file__).parent.parent.parent.resolve(), "output")


def plot_cumulative_losses(backtest, config_name, save=False):
    plt.figure(figsize=(16, 9))
    for key, algorithm in backtest.algorithms.items():
        plt.plot(algorithm.losses.cumsum(), label=algorithm.name)

    # plot switching lines
    n_segments = (backtest.n_switches_per_epoch + 1)*(backtest.n_epochs)
    for i in [x*np.ceil(backtest.total_trials_global / n_segments) for x in range(1, n_segments+1)]:
        plt.axvline(x=i, color='gray', linestyle=':', alpha=0.5)

    # Plot benchmarks
    # plt.plot(backtest.losses[:, -1].cumsum(), label='Typical Action')
    # plt.plot(backtest.best_action_losses.cumsum(), label='Best Action Sequence')

    plt.xlabel('t')
    plt.ylabel('Cumulative loss')
    plt.legend(fontsize=8)
    plt.gca().set_xlim(left=0)
    plt.gca().set_ylim(bottom=0)

    if save:
        plt.savefig(Path.joinpath(OUTPUT_PATH, config_name, "cumulative_loss.png"))
    else:
        plt.show()

def plot_task_matrix(config_name, n_tasks, task_matrix, n_epochs, total_trials_global, super_pool_size, fig_height=9,
                     fig_width=16, cmap='rainbow', save=True):
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)

    im = plt.imshow(task_matrix.T, interpolation='nearest', aspect="auto", cmap=cmap)

    # Colorbar
    # observed_actions = list(np.unique(task_matrix.values))
    # if len(observed_actions) > 1:
    #     boundaries = [observed_actions[0] -1] + observed_actions
    #     plt.colorbar(im, boundaries=np.array(boundaries) + 0.5, ticks=np.array(range(super_pool_size)), spacing='proportional', label='Good policies')

    # Epoch Switches
    epoch_length = np.ceil(total_trials_global / n_epochs).astype(int)
    for e in range(1, n_epochs):
        plt.axvline(x=e*epoch_length, color='k', linestyle='--', lw=1)

    # Task ticks
    ax.set_yticks([i for i in range(n_tasks)])
    ax.set_yticklabels([f"{i+1}" for i in range(n_tasks)][::-1])

    # Labels
    ax.set_ylabel('Task', fontsize=9)
    ax.set_xlabel('Epoch', fontsize=9)
    ax.tick_params(axis='y', which='both', length=0)

    # Second x-axis
    # ax2 = ax.twiny()
    ax.set_xlim(ax.get_xlim())
    ax.set_xticks([int((i*total_trials_global / n_epochs) /2) for i in range(1,2*n_epochs) if i%2 !=0])
    ax.set_xticklabels([f"{i}" for i in range(1,n_epochs+1)])
    ax.tick_params(axis='x', which='both', length=0)

    fig.tight_layout()

    if save:
        fig.savefig(Path.joinpath(OUTPUT_PATH, config_name, "task_matrix.pdf"), format='pdf', bbox_inches='tight')
    else:
        plt.show()


def plot_single_parameter_tuning_results(result_dict,  parameter_ranges, params_to_tune, save_to=None):
    fig, ax = plt.subplots()
    for algorithm, res in result_dict.items():
        df = pd.DataFrame(res.mean(axis=2)).melt()
        sns.lineplot(x="variable", y="value", data=df, label=algorithm, ax=ax)
    plt.ylabel('Total Loss')
    plt.xlabel(params_to_tune[0])
    ax.set_xticks([i for i in range(len(parameter_ranges[params_to_tune[0]]))])
    ax.set_xticklabels([np.round(i, 3) for i in parameter_ranges[params_to_tune[0]]])
    if save_to is not None:
        plt.savefig(Path.joinpath(save_to, f"{params_to_tune[0]}_tuning.png"))
    else:
        plt.show()


def plot_heatmap_tuning_results(result_dict, parameter_ranges, params_to_tune, save_to=None):
    fig, (ax, ax2) = plt.subplots(ncols=2, figsize=(12, 6))
    for algorithm, res in result_dict.items():

        r = res.shape[0]
        # Take an average across iterations
        std_ = res.std(axis=0)
        res = res.mean(axis=0)

        sns.heatmap(data=res, cbar_kws={'label': 'Total Loss'}, ax=ax, annot=False)
        sns.heatmap(data=std_, cbar_kws={'label': 'Standard Deviation'}, ax=ax2, annot=True)

        for i in range(np.where(res == res.min())[0].shape[0]):
            ax.add_patch(
                Rectangle((np.where(res == res.min())[1][i], np.where(res == res.min())[0][i]), 1, 1, fill=False,
                          edgecolor='blue', lw=3))

        ax.set_xlabel(params_to_tune[1])
        ax.set_ylabel(params_to_tune[0])
        ax.set_xticks([i + 0.5 for i in range(len(parameter_ranges[params_to_tune[1]]))])
        ax.set_xticklabels([np.round(i, 3) for i in parameter_ranges[params_to_tune[1]]])
        ax.set_yticks([i + 0.5 for i in range(len(parameter_ranges[params_to_tune[0]]))])
        ax.set_yticklabels([np.round(i, 3) for i in parameter_ranges[params_to_tune[0]]])

        ax2.set_xticks([])
        ax2.set_yticks([])

        ax.set_title(algorithm)
        ax2.set_title(f'Std. Dev. (runs={r})')
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        if save_to is not None:
            plt.savefig(Path.joinpath(save_to, f"{algorithm}_tuning_{params_to_tune[0]}_{params_to_tune[1]}.png"))
        else:
            plt.show()
