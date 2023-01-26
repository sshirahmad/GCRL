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

from utils import NUMBER_PERSONS, set_name_method, set_name_env, set_name_finetune


def exp_StyleDomainShift_IM():

    """
        In this function Domain Generalization experiments' results by style-domain-shifts
        of 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8 at inference for
        5 seeds; 1,2,3,4,5 are presented for IM-Synthetic dataset with IM model.
        :return ADE-MEAN, ADE-STD, FDE-MEAN, FDE-STD:
        """

    # ##### EIM (seed = 1) epochs=470 ##### #
    ADE_EIM_S1_ds1 = 0.0498
    FDE_EIM_S1_ds1 = 0.0708

    ADE_EIM_S1_ds2 = 0.0576
    FDE_EIM_S1_ds2 = 0.0731

    ADE_EIM_S1_ds3 = 0.0611
    FDE_EIM_S1_ds3 = 0.0777

    ADE_EIM_S1_ds4 = 0.0686
    FDE_EIM_S1_ds4 = 0.0865

    ADE_EIM_S1_ds5 = 0.0806
    FDE_EIM_S1_ds5 = 0.0995

    ADE_EIM_S1_ds6 = 0.1056
    FDE_EIM_S1_ds6 = 0.1300

    ADE_EIM_S1_ds7 = 0.1522
    FDE_EIM_S1_ds7 = 0.1868

    ADE_EIM_S1_ds8 = 0.2174
    FDE_EIM_S1_ds8 = 0.2671

    # ##### EIM_S2 (seed = 2) epochs=470 ##### #
    ADE_EIM_S2_ds1 = 0.0484
    FDE_EIM_S2_ds1 = 0.0694

    ADE_EIM_S2_ds2 = 0.0534
    FDE_EIM_S2_ds2 = 0.0692

    ADE_EIM_S2_ds3 = 0.0575
    FDE_EIM_S2_ds3 = 0.0732

    ADE_EIM_S2_ds4 = 0.0671
    FDE_EIM_S2_ds4 = 0.0838

    ADE_EIM_S2_ds5 = 0.0794
    FDE_EIM_S2_ds5 = 0.0996

    ADE_EIM_S2_ds6 = 0.1090
    FDE_EIM_S2_ds6 = 0.1317

    ADE_EIM_S2_ds7 = 0.1578
    FDE_EIM_S2_ds7 = 0.1842

    ADE_EIM_S2_ds8 = 0.2253
    FDE_EIM_S2_ds8 = 0.2575

    # ##### EIM_S3 (seed = 3) epochs=470 ##### #
    ADE_EIM_S3_ds1 = 0.0515
    FDE_EIM_S3_ds1 = 0.0767

    ADE_EIM_S3_ds2 = 0.0550
    FDE_EIM_S3_ds2 = 0.0771

    ADE_EIM_S3_ds3 = 0.0582
    FDE_EIM_S3_ds3 = 0.0779

    ADE_EIM_S3_ds4 = 0.0696
    FDE_EIM_S3_ds4 = 0.0845

    ADE_EIM_S3_ds5 = 0.0752
    FDE_EIM_S3_ds5 = 0.0961

    ADE_EIM_S3_ds6 = 0.1034
    FDE_EIM_S3_ds6 = 0.1240

    ADE_EIM_S3_ds7 = 0.1494
    FDE_EIM_S3_ds7 = 0.1669

    ADE_EIM_S3_ds8 = 0.2091
    FDE_EIM_S3_ds8 = 0.2255

    # ##### EIM_S4 (seed = 4) epochs=470 ##### #
    ADE_EIM_S4_ds1 = 0.0486
    FDE_EIM_S4_ds1 = 0.0825

    ADE_EIM_S4_ds2 = 0.0536
    FDE_EIM_S4_ds2 = 0.0805

    ADE_EIM_S4_ds3 = 0.0572
    FDE_EIM_S4_ds3 = 0.0840

    ADE_EIM_S4_ds4 = 0.0689
    FDE_EIM_S4_ds4 = 0.0939

    ADE_EIM_S4_ds5 = 0.0768
    FDE_EIM_S4_ds5 = 0.1022

    ADE_EIM_S4_ds6 = 0.1001
    FDE_EIM_S4_ds6 = 0.1231

    ADE_EIM_S4_ds7 = 0.1460
    FDE_EIM_S4_ds7 = 0.1672

    ADE_EIM_S4_ds8 = 0.2107
    FDE_EIM_S4_ds8 = 0.2339

    # ##### EIM_S5 (seed = 5) epochs=470 ##### #
    ADE_EIM_S5_ds1 = 0.0490
    FDE_EIM_S5_ds1 = 0.0717

    ADE_EIM_S5_ds2 = 0.0594
    FDE_EIM_S5_ds2 = 0.0755

    ADE_EIM_S5_ds3 = 0.0624
    FDE_EIM_S5_ds3 = 0.0796

    ADE_EIM_S5_ds4 = 0.0696
    FDE_EIM_S5_ds4 = 0.0879

    ADE_EIM_S5_ds5 = 0.0799
    FDE_EIM_S5_ds5 = 0.1008

    ADE_EIM_S5_ds6 = 0.1094
    FDE_EIM_S5_ds6 = 0.1320

    ADE_EIM_S5_ds7 = 0.1670
    FDE_EIM_S5_ds7 = 0.1959

    ADE_EIM_S5_ds8 = 0.2383
    FDE_EIM_S5_ds8 = 0.2782

    ADE_seeds_ds1 = [ADE_EIM_S1_ds1, ADE_EIM_S2_ds1, ADE_EIM_S3_ds1, ADE_EIM_S4_ds1, ADE_EIM_S5_ds1]
    FDE_seeds_ds1 = [FDE_EIM_S1_ds1, FDE_EIM_S2_ds1, FDE_EIM_S3_ds1, FDE_EIM_S4_ds1, FDE_EIM_S5_ds1]
    m_ade1, m_fde1, s_ade1, s_fde1 = get_mean_std_over_seeds(ADE_seeds_ds1, FDE_seeds_ds1, ds=0.1, model='IM')

    ADE_seeds_ds2 = [ADE_EIM_S1_ds2, ADE_EIM_S2_ds2, ADE_EIM_S3_ds2, ADE_EIM_S4_ds2, ADE_EIM_S5_ds2]
    FDE_seeds_ds2 = [FDE_EIM_S1_ds2, FDE_EIM_S2_ds2, FDE_EIM_S3_ds2, FDE_EIM_S4_ds2, FDE_EIM_S5_ds2]
    m_ade2, m_fde2, s_ade2, s_fde2 = get_mean_std_over_seeds(ADE_seeds_ds2, FDE_seeds_ds2, ds=0.2, model='IM')

    ADE_seeds_ds3 = [ADE_EIM_S1_ds3, ADE_EIM_S2_ds3, ADE_EIM_S3_ds3, ADE_EIM_S4_ds3, ADE_EIM_S5_ds3]
    FDE_seeds_ds3 = [FDE_EIM_S1_ds3, FDE_EIM_S2_ds3, FDE_EIM_S3_ds3, FDE_EIM_S4_ds3, FDE_EIM_S5_ds3]
    m_ade3, m_fde3, s_ade3, s_fde3 = get_mean_std_over_seeds(ADE_seeds_ds3, FDE_seeds_ds3, ds=0.3, model='IM')

    ADE_seeds_ds4 = [ADE_EIM_S1_ds4, ADE_EIM_S2_ds4, ADE_EIM_S3_ds4, ADE_EIM_S4_ds4, ADE_EIM_S5_ds4]
    FDE_seeds_ds4 = [FDE_EIM_S1_ds4, FDE_EIM_S2_ds4, FDE_EIM_S3_ds4, FDE_EIM_S4_ds4, FDE_EIM_S5_ds4]
    m_ade4, m_fde4, s_ade4, s_fde4 = get_mean_std_over_seeds(ADE_seeds_ds4, FDE_seeds_ds4, ds=0.4, model='IM')

    ADE_seeds_ds5 = [ADE_EIM_S1_ds5, ADE_EIM_S2_ds5, ADE_EIM_S3_ds5, ADE_EIM_S4_ds5, ADE_EIM_S5_ds5]
    FDE_seeds_ds5 = [FDE_EIM_S1_ds5, FDE_EIM_S2_ds5, FDE_EIM_S3_ds5, FDE_EIM_S4_ds5, FDE_EIM_S5_ds5]
    m_ade5, m_fde5, s_ade5, s_fde5 = get_mean_std_over_seeds(ADE_seeds_ds5, FDE_seeds_ds5, ds=0.5, model='IM')

    ADE_seeds_ds6 = [ADE_EIM_S1_ds6, ADE_EIM_S2_ds6, ADE_EIM_S3_ds6, ADE_EIM_S4_ds6, ADE_EIM_S5_ds6]
    FDE_seeds_ds6 = [FDE_EIM_S1_ds6, FDE_EIM_S2_ds6, FDE_EIM_S3_ds6, FDE_EIM_S4_ds6, FDE_EIM_S5_ds6]
    m_ade6, m_fde6, s_ade6, s_fde6 = get_mean_std_over_seeds(ADE_seeds_ds6, FDE_seeds_ds6, ds=0.6, model='IM')

    ADE_seeds_ds7 = [ADE_EIM_S1_ds7, ADE_EIM_S2_ds7, ADE_EIM_S3_ds7, ADE_EIM_S4_ds7, ADE_EIM_S5_ds7]
    FDE_seeds_ds7 = [FDE_EIM_S1_ds7, FDE_EIM_S2_ds7, FDE_EIM_S3_ds7, FDE_EIM_S4_ds7, FDE_EIM_S5_ds7]
    m_ade7, m_fde7, s_ade7, s_fde7 = get_mean_std_over_seeds(ADE_seeds_ds7, FDE_seeds_ds7, ds=0.7, model='IM')

    ADE_seeds_ds8 = [ADE_EIM_S1_ds8, ADE_EIM_S2_ds8, ADE_EIM_S3_ds8, ADE_EIM_S4_ds8, ADE_EIM_S5_ds8]
    FDE_seeds_ds8 = [FDE_EIM_S1_ds8, FDE_EIM_S2_ds8, FDE_EIM_S3_ds8, FDE_EIM_S4_ds8, FDE_EIM_S5_ds8]
    m_ade8, m_fde8, s_ade8, s_fde8 = get_mean_std_over_seeds(ADE_seeds_ds8, FDE_seeds_ds8, ds=0.8, model='IM')

    means_ade = [m_ade1, m_ade2, m_ade3, m_ade4, m_ade5, m_ade6, m_ade7, m_ade8]
    stds_ade = [s_ade1, s_ade2, s_ade3, s_ade4, s_ade5, s_ade6, s_ade7, s_ade8]

    means_fde = [m_fde1, m_fde2, m_fde3, m_fde4, m_fde5, m_fde6, m_fde7, m_fde8]
    stds_fde = [s_fde1, s_fde2, s_fde3, s_fde4, s_fde5, s_fde6, s_fde7, s_fde8]

    return means_ade, means_fde, stds_ade, stds_fde


def get_mean_std_over_seeds(ADE_seeds, FDE_seeds, ds=0.6, model='IM'):

    ave_ADE_over_seeds = np.mean(ADE_seeds)
    ave_FDE_over_seeds = np.mean(FDE_seeds)

    std_ADE_over_seeds = np.std(ADE_seeds)
    std_FDE_over_seeds = np.std(FDE_seeds)

    return ave_ADE_over_seeds, ave_FDE_over_seeds, std_ADE_over_seeds, std_FDE_over_seeds


def exp_DomianAdaptation_IM_seeds12345():

    """
    In this function Domain Adaptation experiments' results by fine-tuning for 6 batches at inference for
    5 seeds; 1,2,3,4,5 are presented for IM-Synthetic dataset with IM model.
    :return ADE-MEAN, ADE-STD, FDE-MEAN, FDE-STD:

    # \Finetune:
    reduce       64x0=0     64x1=64   64x2=128    64x3=192    64x4=256    64x5=320    64x6=384
    ADE mean     0.106      0.102     0.098       0.096       0.096       0.094       0.092
         std     0.004      0.005     0.004       0.004       0.004       0.003       0.004
     FDE mean    0.128      0.123     0.120       0.118       0.118       0.115       0.114
         std     0.004      0.005     0.005       0.005       0.005       0.005       0.005
    """

    m_ade_IM = [0.106,  0.102,  0.098,  0.096,  0.096,  0.094,  0.092]
    s_ade_IM = [0.004,  0.005,  0.004,  0.004,  0.004,  0.003,  0.004]

    m_fde_IM = [0.128,  0.123,  0.120,  0.118,  0.118,  0.115,  0.114]
    s_fde_IM = [0.004,  0.005,  0.005,  0.005,  0.005,  0.005,  0.005]

    return m_ade_IM, s_ade_IM, m_fde_IM, s_fde_IM


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
        if args.exp == 'pretrainN' or args.exp == 'all':
            print(f'\nRESULTS\nDataset: {args.dataset_name}\n\nPretrainN: ')
            result = pd.read_csv(f'results/{args.dataset_name}/pretrainN/summary.csv', sep=', ', engine='python')
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

            if result.shape[0] == 0:
                warnings.warn("No 'pretrain' experiments available.")
            else:
                print(f'see plot `images/{args.dataset_name}/Best_N.png`')

        elif args.exp == 'pretrain' or args.exp == 'all':
            print(f'\nRESULTS\nDataset: {args.dataset_name}\n\nPretrain: ')
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
                print(f'see plot `images/{args.dataset_name}/DG.png`')

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
