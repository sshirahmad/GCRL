import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from parser_file import get_evaluation_parser
import tensorflow as tf
from torch.serialization import save
import numpy as np
import pandas as pd
import os
import io
import warnings
from visualization import exp_StyleDomainShift_IM


from utils import NUMBER_PERSONS, set_name_method, set_name_env, set_name_finetune


def main(args):
    # create folder images
    if not os.path.exists(f'images/{args.dataset_name}'):
        os.makedirs(f'images/{args.dataset_name}')

    if args.dataset_name in ["eth", "hotel", "zara1", "zara2", "univ"]:
        print(f'\nRESULTS\nDataset: {args.dataset_name}\n\nPretrain: ')
        result = pd.read_csv(f'results/{args.dataset_name}/pretrain/summary.csv', sep=', ', engine='python')
        result = result[result.split == 'test']
        result = result.drop(['seed', 'envs', 'split'], axis=1)
        result = pd.pivot_table(result,
                                values=['ADE', 'FDE'],
                                columns=['shifts'],
                                aggfunc={'ADE': [np.mean, np.std],
                                         'FDE': [np.mean, np.std]},
                                sort=True
                                ).round(decimals=3)
        if result.shape[0] == 0:
            warnings.warn("No 'pretrain' experiments available.")
        else:
            print(result)

    # pretrain exp
    if args.dataset_name == "v4":
        if args.exp == 'pretrain' or args.exp == 'all':
            print(f'\nRESULTS\nDataset: {args.dataset_name}\n\nPretrain: ')
            result = pd.read_csv(f'results/{args.dataset_name}/pretrain/summary_bestN.csv', sep=', ', engine='python')
            result = result[result.split == 'test']
            result = result.drop(['seed', 'split'], axis=1)
            result = result[result.envs.astype(bool)]
            result = pd.pivot_table(result,
                                    values=['ADE', 'FDE'],
                                    columns=['N'],
                                    aggfunc={'ADE': [np.mean, np.std],
                                             'FDE': [np.mean, np.std]},
                                    sort=True
                                    ).round(decimals=3)
            best_N = result.columns.values
            plt.figure(1)
            fig, ax = plt.subplots(figsize=(9, 6))
            for label in (ax.get_xticklabels() + ax.get_yticklabels()):
                label.set_fontsize(16)
            plt.plot(best_N, result.loc[('ADE', 'mean')].values, "o-g", label="ADE")
            plt.fill_between(best_N, result.loc[('ADE', 'mean')].values - result.loc[('ADE', 'std')].values,
                             result.loc[('ADE', 'mean')].values + result.loc[('ADE', 'std')].values, alpha=.4,
                             color='green')
            plt.plot(best_N, result.loc[('FDE', 'mean')].values, "o-r", label="FDE")
            plt.fill_between(best_N, result.loc[('FDE', 'mean')].values - result.loc[('FDE', 'std')].values,
                             result.loc[('FDE', 'mean')].values + result.loc[('FDE', 'std')].values, alpha=.4,
                             color='red')
            plt.xlabel("N", fontsize=18)
            plt.ylabel("ADE/FDE", fontsize=18)
            plt.legend(loc="upper left", fontsize=15)

            result = pd.read_csv(f'results/{args.dataset_name}/pretrain/summary.csv', sep=', ', engine='python')
            result = result[result.split == 'test']
            result = result.drop(['seed', 'split'], axis=1)
            result = result[result.envs.astype(bool)]
            result = pd.pivot_table(result,
                                    values=['ADE', 'FDE'],
                                    columns=['envs'],
                                    aggfunc={'ADE': [np.mean, np.std],
                                             'FDE': [np.mean, np.std]},
                                    sort=True
                                    ).round(decimals=3)
            m_ade_im, m_fde_im, s_ade_im, s_fde_im = exp_StyleDomainShift_IM()
            domain_shifts = result.columns.values
            plt.figure(2)
            fig, ax = plt.subplots(figsize=(9, 6))
            for label in (ax.get_xticklabels() + ax.get_yticklabels()):
                label.set_fontsize(16)
            plt.plot(domain_shifts, result.values[0], "o-g", label="VCRL")
            plt.fill_between(domain_shifts, result.values[0] - result.values[1],
                             result.values[0] + result.values[1], alpha=.4, color='green')
            plt.plot(domain_shifts, m_ade_im, "o-r", label="IM")
            plt.fill_between(domain_shifts, np.array(m_ade_im) - np.array(s_ade_im),
                             np.array(m_ade_im) + np.array(s_ade_im), alpha=.4, color='red')
            plt.xlabel("Style Domain Shifts", fontsize=18)
            plt.ylabel("ADE", fontsize=18)
            plt.legend(loc="upper left", fontsize=15)
            plt.show()
            if result.shape[0] == 0:
                warnings.warn("No 'pretrain' experiments available.")
            else:
                print(result)

        else:
            # finetune exp
            print('\n\Finetune: ')
            result = pd.read_csv(f'results/{args.dataset_name}/finetune/summary.csv', sep=', ', engine='python')
            result = result[result.split == 'test']
            result = result[result.envs == args.env]
            result = result.drop(['step', 'irm', 'envs', 'seed', 'split'], axis=1)
            reduce = sorted(result.reduce.unique())
            result['finetune'] = result['finetune'].apply(set_name_finetune)

            if result.shape[0] == 0:
                warnings.warn("No 'Finetune' experiments available.")
            else:
                f, ax = plt.subplots(figsize=(5.5, 5))
                sns.despine(f)
                sns.lineplot(data=result, x="reduce", y="ADE", hue='finetune', marker='o')
                ax.legend_.set_title(None)
                ax.set_xlabel('# Batches')
                ax.set_xticks(list(reduce), list([int(elem / 64) for elem in reduce]))
                plt.savefig(f'images/{args.dataset_name}/finetune_ade.png', bbox_inches='tight', pad_inches=0)

                f, ax = plt.subplots(figsize=(5.5, 5))
                sns.despine(f)
                sns.lineplot(data=result, x="reduce", y="FDE", hue='finetune', marker='o')
                ax.legend_.set_title(None)
                ax.set_xlabel('# Batches')
                ax.set_xticks(list(reduce), list([int(elem / 64) for elem in reduce]))
                plt.savefig(f'images/{args.dataset_name}/finetune_fde.png', bbox_inches='tight', pad_inches=0)

                print(
                    f'see plots `images/{args.dataset_name}/finetune_ade.png` and `images/{args.dataset_name}/finetune_fde.png`')


def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
  returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image


def draw_solo(saved_pred, wto):
    # all on cpu
    num_env = len(saved_pred)
    for i in range(num_env):
        saved_pred[i] = [x.cpu() for x in saved_pred[i]]

    X = 1
    Y = len(saved_pred) // X + 1 if len(saved_pred) % X != 0 else len(saved_pred) // X
    # create the plot 
    figure, axes = plt.subplots(X, Y)
    figure.set_size_inches(2 * Y, 2 * X)
    colors = ['red', 'blue', 'green', 'black', 'yellow', 'purple']

    num_seq = saved_pred[0][0].shape[1] // NUMBER_PERSONS
    a = np.arange(num_seq)

    for k, (obs, fut, pred) in enumerate(saved_pred):
        for i, seq in enumerate(a[wto:wto + 1]):
            for j in range(NUMBER_PERSONS):
                axes[k // X].plot(obs[:, NUMBER_PERSONS * seq + j, 0], obs[:, NUMBER_PERSONS * seq + j, 1], label='obs',
                                  color=colors[j])
                axes[k // X].plot(fut[:, NUMBER_PERSONS * seq + j, 0], fut[:, NUMBER_PERSONS * seq + j, 1], label='fut',
                                  color=colors[j])
                axes[k // X].plot(pred[:, NUMBER_PERSONS * seq + j, 0], pred[:, NUMBER_PERSONS * seq + j, 1], '--',
                                  label='pred', color=colors[j])

    # convert it to numpy array
    cm_image = plot_to_image(figure)
    array = (cm_image.numpy()[0])[:, :, :3]
    array = np.transpose(array, (2, 0, 1))

    return figure, array


def draw_solo_all(saved_pred):
    # all on cpu
    for i in range(len(saved_pred)):
        for j in range(len(saved_pred[0])):
            saved_pred[i][j] = [x.cpu() for x in saved_pred[i][j]]

    X = len(saved_pred)
    Y = len(saved_pred[0])
    # create the plot 
    figure, axes = plt.subplots(X, Y)
    figure.set_size_inches(2 * Y, 2 * X)
    colors = ['red', 'blue', 'green', 'black', 'yellow', 'purple']

    num_seq = saved_pred[0][0][0].shape[1] // NUMBER_PERSONS
    a = np.arange(num_seq)

    for m, saved_pred_ in enumerate(saved_pred):
        for k, (obs, fut, pred) in enumerate(saved_pred_):
            for i, seq in enumerate(a[m:m + 1]):
                for j in range(NUMBER_PERSONS):
                    axes[m][k].set_xticks([])
                    axes[m][k].set_yticks([])
                    axes[m][k].plot(obs[:, NUMBER_PERSONS * seq + j, 0], obs[:, NUMBER_PERSONS * seq + j, 1],
                                    label='obs', color=colors[j])
                    axes[m][k].plot(fut[:, NUMBER_PERSONS * seq + j, 0], fut[:, NUMBER_PERSONS * seq + j, 1],
                                    label='fut', color=colors[j])
                    axes[m][k].plot(pred[:, NUMBER_PERSONS * seq + j, 0], pred[:, NUMBER_PERSONS * seq + j, 1], '--',
                                    label='pred', color=colors[j])

    # convert it to numpy array
    cm_image = plot_to_image(figure)
    array = (cm_image.numpy()[0])[:, :, :3]
    array = np.transpose(array, (2, 0, 1))

    return figure, array


def draw_image(saved_pred):
    # all on cpu
    num_env = len(saved_pred)
    for i in range(num_env):
        saved_pred[i] = [x.cpu() for x in saved_pred[i]]

    # create the plot 
    figure, axes = plt.subplots(3, 2 * num_env)
    figure.set_size_inches(4 * num_env, 6)
    colors = ['red', 'blue', 'green', 'black', 'yellow', 'purple']

    num_seq = saved_pred[0][0].shape[1] // NUMBER_PERSONS
    a = np.arange(num_seq)
    # np.random.shuffle(a)

    for k, (obs, fut, pred) in enumerate(saved_pred):
        for i, seq in enumerate(a[:6]):
            for j in range(NUMBER_PERSONS):
                axes[i % 3][i // 3 + 2 * k].set_xticks([])
                axes[i % 3][i // 3 + 2 * k].set_yticks([])
                axes[i % 3][i // 3 + 2 * k].plot(obs[:, NUMBER_PERSONS * seq + j, 0],
                                                 obs[:, NUMBER_PERSONS * seq + j, 1], label='obs', color=colors[j])
                axes[i % 3][i // 3 + 2 * k].plot(fut[:, NUMBER_PERSONS * seq + j, 0],
                                                 fut[:, NUMBER_PERSONS * seq + j, 1], label='fut', color=colors[j])
                # axes[i%3][i//3+2*k].plot(pred[:,NUMBER_PERSONS*seq+j,0], pred[:,NUMBER_PERSONS*seq+j,1], '--', label='pred', color=colors[j])

    # convert it to numpy array
    cm_image = plot_to_image(figure)
    array = (cm_image.numpy()[0])[:, :, :3]
    array = np.transpose(array, (2, 0, 1))

    return figure, array


if __name__ == "__main__":
    args = get_evaluation_parser().parse_args()
    main(args)
