import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from loader import data_loader
from parser_file import get_training_parser
from utils import *
from models import CRMF
from losses import erm_loss, irm_loss
from torch.nn.utils import clip_grad_norm_
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from visualize import draw_image, draw_solo, draw_solo_all


def main(args):
    # Set environment variables
    set_seed_globally(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num
    model_name = set_name_experiment(args)
    print('model name: ', model_name)
    if not os.path.exists(args.tfdir + '/' + model_name):
        os.makedirs(args.tfdir + '/' + model_name)

    training_step = "P7"

    logging.info("Initializing Training Set")
    train_envs_path, train_envs_name = get_envs_path(args.dataset_name, "train", args.filter_envs)
    train_loaders = [data_loader(args, train_env_path, train_env_name) for train_env_path, train_env_name in
                     zip(train_envs_path, train_envs_name)]

    logging.info("Initializing Validation Set")
    val_envs_path, val_envs_name = get_envs_path(args.dataset_name, "val",
                                                 args.filter_envs)  # +'-'+args.filter_envs_pretrain)
    val_loaders = [data_loader(args, val_env_path, val_env_name) for val_env_path, val_env_name in
                   zip(val_envs_path, val_envs_name)]

    logging.info("Initializing Validation O Set")
    valo_envs_path, valo_envs_name = get_envs_path(args.dataset_name, "test", args.filter_envs)
    valo_loaders = [data_loader(args, valo_env_path, valo_env_name) for valo_env_path, valo_env_name in zip(valo_envs_path, valo_envs_name)]

    # training routine length
    num_batches_train = min([len(train_loader) for train_loader in train_loaders])
    num_batches_val = min([len(val_loader) for val_loader in val_loaders])
    num_batches_valo = min([len(valo_loader) for valo_loader in valo_loaders])

    # bring different dataset all together for simplicity of the next functions
    train_dataset = {'loaders': train_loaders, 'names': train_envs_name, 'num_batches': num_batches_train}
    valid_dataset = {'loaders': val_loaders, 'names': val_envs_name, 'num_batches': num_batches_val}
    valido_dataset = {'loaders': valo_loaders, 'names': valo_envs_name, 'num_batches': num_batches_valo}

    for dataset, ds_name in zip((train_dataset, valid_dataset, valido_dataset),
                                ('Train', 'Validation', 'Validation O')):
        print(ds_name + ' dataset: ', dataset)

    args.n_units = (
            [args.traj_lstm_hidden_size]
            + [int(x) for x in args.hidden_units.strip().split(",")]
            + [args.graph_lstm_hidden_size]
    )
    args.n_heads = [int(x) for x in args.heads.strip().split(",")]

    # create the model
    model = CRMF(args).cuda()
    if training_step == "P7":
        calculate_distance_posteriors(model, valid_dataset, valido_dataset)
    else:
        calculate_distance_priors(model, valid_dataset, valido_dataset)


def calculate_distance_priors(model, valid_dataset, valido_dataset=None):
    """
    Evaluate the performances on the validation set

    Args:
        - stage (str): either 'validation' or 'training': says on which dataset the metrics are computed
    """
    model.eval()

    fig, ax = plt.subplots(1, 2)
    with torch.no_grad():
        for val_idx, (loader, loader_name) in enumerate(zip(valid_dataset['loaders'], valid_dataset['names'])):
            for batch_idx, batch in enumerate(loader):
                batch = [tensor.cuda() for tensor in batch]
                (obs_traj, fut_traj, _, _, _) = batch

                pz, z_vec, ps, s_vec = model(batch, "P8", env_idx=val_idx)
                ax[0].scatter(z_vec[:, 0].cpu(), pz.cpu())
                ax[1].scatter(s_vec[:, 0].cpu(), ps.cpu())

        for val_idx, (loader, loader_name) in enumerate(zip(valido_dataset['loaders'], valido_dataset['names'])):
            for batch_idx, batch in enumerate(loader):
                batch = [tensor.cuda() for tensor in batch]
                (obs_traj, fut_traj, _, _, _) = batch

                pz, z_vec, ps, s_vec = model(batch, "P8")
                ax[0].scatter(z_vec[:, 0].cpu(), pz.cpu())
                ax[1].scatter(s_vec[:, 0].cpu(), ps.cpu())

        plt.savefig('fig.png')


def calculate_distance_posteriors(model, valid_dataset, valido_dataset=None):
    """
    Evaluate the performances on the validation set

    Args:
        - stage (str): either 'validation' or 'training': says on which dataset the metrics are computed
    """
    model.eval()
    mean_z_env = []
    cov_z_env = []
    mean_s_env = []
    cov_s_env = []
    with torch.no_grad():
        for val_idx, (loader, loader_name) in enumerate(zip(valid_dataset['loaders'], valid_dataset['names'])):
            for batch_idx, batch in enumerate(loader):
                batch = [tensor.cuda() for tensor in batch]
                (obs_traj, fut_traj, _, _, _) = batch

                z, s = model(batch, "P7", env_idx=val_idx)
                mean_z = torch.mean(z.mean, dim=0)
                covariance_z = torch.mean(z.covariance_matrix, dim=0)
                mean_s = torch.mean(s.mean, dim=0)
                covariance_s = torch.mean(s.covariance_matrix, dim=0)

            mean_z_env += [mean_z]
            cov_z_env += [covariance_z]
            mean_s_env += [mean_s]
            cov_s_env += [covariance_s]

        for val_idx, (loader, loader_name) in enumerate(zip(valido_dataset['loaders'], valido_dataset['names'])):
            for batch_idx, batch in enumerate(loader):
                batch = [tensor.cuda() for tensor in batch]
                (obs_traj, fut_traj, _, _, _) = batch

                z, s = model(batch, "P7")
                mean_z = torch.mean(z.mean, dim=0)
                covariance_z = torch.mean(z.covariance_matrix, dim=0)
                mean_s = torch.mean(s.mean, dim=0)
                covariance_s = torch.mean(s.covariance_matrix, dim=0)

            mean_z_env += [mean_z]
            cov_z_env += [covariance_z]
            mean_s_env += [mean_s]
            cov_s_env += [covariance_s]

    dist_z_mean = torch.zeros(len(mean_z_env), len(mean_z_env))
    dist_z_cov = torch.zeros(len(mean_z_env), len(mean_z_env))
    dist_s_mean = torch.zeros(len(mean_z_env), len(mean_z_env))
    dist_s_cov = torch.zeros(len(mean_z_env), len(mean_z_env))
    for i in range(len(mean_z_env)):
        for j in range(len(mean_z_env)):
            dist_s_mean[i, j] = torch.norm(mean_s_env[i] - mean_s_env[j], p=2)
            dist_s_cov[i, j] = torch.norm(cov_s_env[i] - cov_s_env[j])
            dist_z_mean[i, j] = torch.norm(mean_z_env[i] - mean_z_env[j], p=2)
            dist_z_cov[i, j] = torch.norm(cov_z_env[i] - cov_z_env[j])

    print("Distance between mean of Z distributions across environments:\n", dist_z_mean)
    print("Distance between covariance of Z distributions across environments:\n", dist_z_cov)
    print("Distance between mean of S distributions across environments:\n", dist_s_mean)
    print("Distance between covariance of S distributions across environments:\n", dist_s_cov)


if __name__ == "__main__":
    print('Using GPU: ' + str(torch.cuda.is_available()))
    input_args = get_training_parser().parse_args()
    print('Arguments for training: ', input_args)
    set_logger(os.path.join(input_args.log_dir, "train.log"))
    main(input_args)
