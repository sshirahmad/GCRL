import argparse
import logging
import os

import torch
import numpy as np
import math

import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')

from loader import data_loader
from parser_file import get_evaluation_parser
from models import CRMF
from utils import *


def get_generator(args):
    """
    Upload model
    """
    args.n_units = (
            [args.traj_lstm_hidden_size]
            + [int(x) for x in args.hidden_units.strip().split(",")]
            + [args.graph_lstm_hidden_size]
    )
    args.n_heads = [int(x) for x in args.heads.strip().split(",")]
    model = CRMF(args)
    load_all_model(args, model, None)
    model.cuda()
    model.eval()

    return model


def cal_ade_fde(fut_traj, pred_fut_traj):
    """
    Compute the ADE and FDE
    """
    ade = displacement_error(pred_fut_traj, fut_traj, mode="raw")
    fde = final_displacement_error(pred_fut_traj[-1], fut_traj[-1], mode="raw")
    return ade, fde


def evaluate(args, loader, generator, training_step):
    """
    Evaluate the performances
    """
    ade_outer, fde_outer = [], []
    total_traj = 0
    step = 0
    with torch.no_grad():
        for batch in loader:
            batch = [tensor.cuda() for tensor in batch]
            if args.dataset_name in ('eth', 'hotel', 'univ', 'zara1', 'zara2'):
                (
                    obs_traj,
                    fut_traj,
                    obs_traj_rel,
                    fut_traj_rel,
                    seq_start_end,
                ) = batch
            elif 'synthetic' in args.dataset_name or args.dataset_name in ['synthetic', 'v2', 'v2full', 'v4']:
                (
                    obs_traj,
                    fut_traj,
                    obs_traj_rel,
                    fut_traj_rel,
                    seq_start_end,
                    _,
                    _
                ) = batch
            else:
                raise ValueError('Unrecognized dataset name "%s"' % args.dataset_name)


            step += seq_start_end.shape[0]
            ade, fde = [], []
            total_traj += fut_traj.size(1)
            for k in range(args.best_k):
                pred_fut_traj_rel = generator(batch, training_step)

                pred_fut_traj = relative_to_abs(pred_fut_traj_rel, obs_traj[-1, :, :2])

                ade_, fde_ = cal_ade_fde(fut_traj[:, :, :2], pred_fut_traj)

                ade.append(ade_)
                fde.append(fde_)

            ade_sum_batch = evaluate_helper(ade, seq_start_end)
            fde_sum_batch = evaluate_helper(fde, seq_start_end)

            ade_outer.append(ade_sum_batch)
            fde_outer.append(fde_sum_batch)
        ade_sum = sum(ade_outer)
        fde_sum = sum(fde_outer)
        return ade_sum, fde_sum, total_traj


def sceneplot(obsv_scene, pred_scene, gt_scene, figname='scene.png', lim=9.0):
    """
    Plot a scene
    """
    num_traj = pred_scene.shape[0]
    obsv_frame = obsv_scene.shape[1]
    pred_frame = pred_scene.shape[1]
    cm_subsection = np.linspace(0.0, 1.0, num_traj)
    colors = [matplotlib.cm.jet(x) for x in cm_subsection]

    for i in range(num_traj):
        for k in range(1, obsv_frame):
            plt.plot(obsv_scene[i, k - 1:k + 1, 0], obsv_scene[i, k - 1:k + 1, 1],
                     '-o', color=colors[i], alpha=1.0)

        plt.plot([obsv_scene[i, -1, 0], pred_scene[i, 0, 0]], [obsv_scene[i, -1, 1], pred_scene[i, 0, 1]],
                 '--', color=colors[i], alpha=1.0, linewidth=1.0)
        for k in range(1, pred_frame):
            alpha = 1.0 - k / pred_frame
            width = (1.0 - alpha) * 24.0
            plt.plot(pred_scene[i, k - 1:k + 1, 0], pred_scene[i, k - 1:k + 1, 1],
                     '--', color=colors[i], alpha=alpha, linewidth=width)

        for k in range(1, pred_frame):
            plt.plot(gt_scene[i, k - 1:k + 1, 0], gt_scene[i, k - 1:k + 1, 1],
                     '--', color=colors[i], alpha=1.0)

    xc = obsv_scene[:, -1, 0].mean()
    yc = obsv_scene[:, -1, 1].mean()
    plt.xlim(xc - lim, xc + lim)
    plt.ylim(yc - lim / 2.0, yc + lim / 2.0)

    plt.gca().set_aspect('equal', adjustable='box')
    plt.gca().get_xaxis().set_visible(False)
    plt.gca().get_yaxis().set_visible(False)
    plt.savefig(figname, bbox_inches='tight', pad_inches=.1)
    plt.close()


def visualize(args, loader, generator, training_step):
    """
    Viasualize some scenes
    """
    keywords = args.resume.split('_')
    suffix = 'ds_' + args.domain_shifts + '_' + keywords[1] + '.png'

    # range of idx for visualization
    lb_idx = 44
    ub_idx = 44

    with torch.no_grad():
        for b, data in enumerate(loader):
            batch = [tensor.cuda() for tensor in data]
            (
                obs_traj,
                fut_traj,
                obs_traj_rel,
                _,
                seq_start_end,
            ) = batch

            for k in range(args.best_k):
                pred_fut_traj_rel = generator(batch, training_step)
                pred_fut_traj = relative_to_abs(pred_fut_traj_rel, obs_traj[-1, :, :2])
                idx_sample = seq_start_end.shape[0]
                for i in range(idx_sample):
                    if i < lb_idx or i > ub_idx:
                        continue  # key scenes
                    idx_start, idx_end = seq_start_end[i][0], seq_start_end[i][1]
                    obsv_scene = obs_traj[:, idx_start:idx_end, :]
                    pred_scene = pred_fut_traj[:, idx_start:idx_end, :]
                    gt_scene = fut_traj[:, idx_start:idx_end, :]

                    figname = './images/visualization/scene_{:02d}_{:02d}_sample_{:02d}_{}'.format(i, b, k, suffix)
                    sceneplot(obsv_scene.permute(1, 0, 2).cpu().detach().numpy(),
                              pred_scene.permute(1, 0, 2).cpu().detach().numpy(),
                              gt_scene.permute(1, 0, 2).cpu().detach().numpy(), figname)


def compute_col(predicted_traj, predicted_trajs_all, thres=0.2, num_interp=4):
    '''
    Compute the collisions
    '''
    dense_all = interpolate_traj(predicted_trajs_all, num_interp)
    dense_ego = interpolate_traj(predicted_traj[None, :], num_interp)
    distances = np.linalg.norm(dense_all - dense_ego, axis=-1)
    mask = distances[:, 0] > 0
    return distances[mask].min(axis=0) < thres


def main(args):
    print('Using GPU: ' + str(torch.cuda.is_available()))
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num
    generator = get_generator(args)
    valo_envs_path, valo_envs_name = get_envs_path(args.dataset_name, "test", '0.6')  # +'-'+args.filter_envs_pretrain)
    loaders = [data_loader(args, valo_env_path, valo_env_name, test=True) for valo_env_path, valo_env_name in zip(valo_envs_path, valo_envs_name)]
    logging.info('Model: {}'.format(args.resume))
    logging.info('Dataset: {}'.format(args.dataset_name))
    logging.info('Eval shift: {}'.format(args.domain_shifts))
    logging.info('Dataset type: {}'.format(args.dset_type))

    # quantitative
    if args.metrics == 'accuracy':

        ade_outer = []
        fde_outer = []
        for _ in range(100):
            ade = 0
            fde = 0
            total_traj = 0
            for loader in loaders:

                ade_sum_i, fde_sum_i, total_traj_i = evaluate(args, loader, generator, training_step="P6")
                ade += ade_sum_i
                fde += fde_sum_i
                total_traj += total_traj_i
            ade = ade / (total_traj * args.fut_len)
            fde = fde / total_traj
            logging.info('ADE: {:.4f}\tFDE: {:.4f}'.format(ade, fde))
            ade_outer += [ade]
            fde_outer += [fde]

        ade_outer = torch.stack(ade_outer).mean(0)
        fde_outer = torch.stack(fde_outer).mean(0)
        print(ade_outer, fde_outer)

    # qualitative
    if args.metrics == 'qualitative':
        for loader in loaders:
            visualize(args, loader, generator, training_step="P3")

    # collisions [to be implemented]
    if args.metrics == 'collision':
        for loader in loaders:
            visualize(args, loader, generator, training_step="P3")


if __name__ == "__main__":
    args = get_evaluation_parser().parse_args()
    set_logger(os.path.join(args.log_dir, args.dataset_name,
                            args.resume[:-8] + "_" + args.dset_type + "_ds_" + str(args.domain_shifts) + ".log"))
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    main(args)
