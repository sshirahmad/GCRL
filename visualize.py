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
from visualization import exp_StyleDomainShift_IM, exp_DomianAdaptation_IM_seeds12345


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
            best_N = sorted(result.N.unique())
            result = pd.pivot_table(result,
                                    values=['ADE', 'FDE'],
                                    columns=['N'],
                                    aggfunc={'ADE': [np.mean, np.std],
                                             'FDE': [np.mean, np.std]},
                                    sort=True
                                    ).round(decimals=3)
            plt.figure()
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
            plt.legend(loc="upper right", fontsize=15)
            plt.savefig(f'images/{args.dataset_name}/Best_N.png', bbox_inches='tight', pad_inches=0)

            result = pd.read_csv(f'results/{args.dataset_name}/pretrain/summary.csv', sep=', ', engine='python')
            result = result[result.split == 'test']
            result = result.drop(['seed', 'split'], axis=1)
            domain_shifts = sorted(result.envs.unique())
            result = result[result.envs.astype(bool)]
            result = pd.pivot_table(result,
                                    values=['ADE', 'FDE'],
                                    columns=['envs'],
                                    aggfunc={'ADE': [np.mean, np.std],
                                             'FDE': [np.mean, np.std]},
                                    sort=True
                                    ).round(decimals=3)
            m_ade_im, m_fde_im, s_ade_im, s_fde_im = exp_StyleDomainShift_IM()

            plt.figure()
            fig, ax = plt.subplots(figsize=(9, 6))
            for label in (ax.get_xticklabels() + ax.get_yticklabels()):
                label.set_fontsize(16)
            plt.plot(domain_shifts, result.values[0], "o-g", label="GCRL")
            plt.fill_between(domain_shifts, result.values[0] - result.values[1],
                             result.values[0] + result.values[1], alpha=.4, color='green')
            plt.plot(domain_shifts, m_ade_im, "o-r", label="IM")
            plt.fill_between(domain_shifts, np.array(m_ade_im) - np.array(s_ade_im),
                             np.array(m_ade_im) + np.array(s_ade_im), alpha=.4, color='red')
            plt.xlabel("Style Domain Shifts", fontsize=18)
            plt.ylabel("ADE", fontsize=18)
            plt.legend(loc="upper left", fontsize=15)
            plt.savefig(f'images/{args.dataset_name}/DG.png', bbox_inches='tight', pad_inches=0)
            if result.shape[0] == 0:
                warnings.warn("No 'pretrain' experiments available.")
            else:
                print(
                    f'see plots `images/{args.dataset_name}/DG.png` and `images/{args.dataset_name}/Best_N.png`')

        elif args.exp == 'finetune' or args.exp == 'all':
            # finetune exp
            print('\n\Finetune: ')
            result = pd.read_csv(f'results/{args.dataset_name}/finetune/{args.finetune}/summary.csv', sep=', ', engine='python')
            result = result[result.split == 'test']
            result = result.drop(['envs', 'split'], axis=1)
            batches = sorted(result.batches.unique())
            result = pd.pivot_table(result,
                                    values=['ADE', 'FDE'],
                                    columns=['batches'],
                                    aggfunc={'ADE': [np.mean, np.std],
                                             'FDE': [np.mean, np.std]},
                                    sort=True
                                    ).round(decimals=3)

            m_ade_IM, s_ade_IM, m_fde_IM, s_fde_IM = exp_DomianAdaptation_IM_seeds12345()

            plt.figure()
            fig, ax = plt.subplots(figsize=(9, 6))
            for label in (ax.get_xticklabels() + ax.get_yticklabels()):
                label.set_fontsize(16)

            plt.plot(batches, result.values[0], "o-g", label="GCRL")
            plt.fill_between(batches, result.values[0] - result.values[1],
                             result.values[0] + result.values[1], alpha=.4, color='green')

            plt.plot(batches, m_ade_IM, "o-r", label="IM")
            plt.fill_between(batches, np.array(m_ade_IM) - np.array(s_ade_IM),
                             np.array(m_ade_IM) + np.array(s_ade_IM), alpha=.4, color='red')
            plt.xlabel("Number of Batches", fontsize=18)
            plt.ylabel("ADE", fontsize=18)
            plt.legend(loc="upper right", fontsize=15)
            plt.savefig(f'images/{args.dataset_name}/DA.png', bbox_inches='tight', pad_inches=0)

            if result.shape[0] == 0:
                warnings.warn("No 'Finetune' experiments available.")
            else:
                print(
                    f'see plots `images/{args.dataset_name}/DA.png`')


if __name__ == "__main__":
    args = get_evaluation_parser().parse_args()
    main(args)
