import numpy as np

from trajectories import TrajectoryDataset, SynTrajectoryDataset
from utils import *
from parser_file import get_training_parser
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, ConcatDataset


def main(args):
    train_envs_path, train_envs_name = get_envs_path(args.dataset_name, "train", args.filter_envs)
    valo_envs_path, valo_envs_name = get_envs_path(args.dataset_name, "test", args.filter_envs)
    train_envs_path += valo_envs_path
    train_envs_name += valo_envs_name

    obs_traj = []
    fut_traj = []
    hue = []
    for i, (train_env_path, train_env_name) in enumerate(zip(train_envs_path, train_envs_name)):
        alpha_e = set_domain_shift(args.domain_shifts, train_env_name)

        dset = SynTrajectoryDataset(
            train_env_path,  # path
            obs_len=args.obs_len,
            fut_len=args.fut_len,
            n_coordinates=args.n_coordinates,
            add_confidence=args.add_confidence,
            alpha_e=alpha_e,
            reduce=20,
            finetune_ratio=args.finetune_ratio,
        )

        obs_traj.append(dset.obs_traj)
        fut_traj.append(dset.fut_traj)
        hue_env = (i + 1) * np.ones(dset.fut_traj.shape[0])
        hue.append(hue_env)

    obs_traj = np.concatenate(obs_traj)
    fut_traj = np.concatenate(fut_traj)
    hue = np.concatenate(hue)
    sns.set_style('whitegrid')
    print(obs_traj.shape)
    sns.jointplot(x=obs_traj[:, 0, 0], y=fut_traj[:, 0, 0], palette="bright", hue=hue, kind="kde", common_norm=False)
    sns.jointplot(x=obs_traj[:, 1, 0], y=fut_traj[:, 1, 0], palette="dark", hue=hue, kind="kde", common_norm=False)
    sns.jointplot(x=obs_traj[:, 1, 0], y=fut_traj[:, 1, 0], kind="kde", common_norm=False)
    plt.show()


if __name__ == "__main__":
    input_args = get_training_parser().parse_args()
    main(input_args)
