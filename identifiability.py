import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from loader import data_loader
from parser_file import get_training_parser
from utils import *
from models import CRMF
from scipy.optimize import least_squares


def main(args):
    # Set environment variables
    set_seed_globally(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num
    model_name = set_name_experiment(args)
    print('model name: ', model_name)
    if not os.path.exists(args.tfdir + '/' + model_name):
        os.makedirs(args.tfdir + '/' + model_name)

    args.batch_size = "10000"
    args.shuffle = False

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
    valo_envs_path, valo_envs_name = get_envs_path(args.dataset_name, "test", "0.6")
    valo_loaders = [data_loader(args, valo_env_path, valo_env_name) for valo_env_path, valo_env_name in
                    zip(valo_envs_path, valo_envs_name)]

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

    paths = [
             "./models/E25/P6/CRMF_epoch_626.pth.tar",
             "./models/E25_S2/P6/CRMF_epoch_714.pth.tar",
             "./models/E25_S3/P6/CRMF_epoch_614.pth.tar",
             "./models/E25_S4/P6/CRMF_epoch_700.pth.tar",
             "./models/E25_S5/P6/CRMF_epoch_624.pth.tar",
             # "./models/E12/P6/CRMF_epoch_703.pth.tar",
             # "./models/E13/P6/CRMF_epoch_669.pth.tar",
             # "./models/E14/P6/CRMF_epoch_573.pth.tar",
             ]

    z_vec = []
    s_vec = []
    for i in range(len(paths)):
        args.resume = paths[i]
        # create the model
        model = CRMF(args).cuda()
        load_all_model(args, model, None)
        model.cuda()
        model.eval()
        with torch.no_grad():
            z_vec_seed = []
            s_vec_seed = []
            mean_z = []
            mean_s = []
            covariance_z = []
            covariance_s = []
            for val_idx, (loader, loader_name) in enumerate(zip(valid_dataset['loaders'], valid_dataset['names'])):
                for batch_idx, batch in enumerate(loader):
                    batch = [tensor.cuda() for tensor in batch]
                    (obs_traj, fut_traj, _, _, _, _, _) = batch

                    qz, qs = model(batch, "P8", env_idx=val_idx)
                    z_vec_seed += [qz.rsample([20, ]).flatten(start_dim=0, end_dim=1)]
                    s_vec_seed += [qs.rsample([20, ]).flatten(start_dim=0, end_dim=1)]

                mean_z += [qz.mean]
                covariance_z += [qz.covariance_matrix.flatten(start_dim=1, end_dim=2)]
                mean_s += [qs.mean]
                covariance_s += [qs.covariance_matrix.flatten(start_dim=1, end_dim=2)]

        s_vec += [torch.cat(s_vec_seed)]
        z_vec += [torch.cat(z_vec_seed)]

    s_vec = torch.stack(s_vec).cpu().numpy()
    z_vec = torch.stack(z_vec).cpu().numpy()
    # mean_z = torch.stack(mean_z).cpu().numpy()
    # mean_s = torch.stack(mean_s).cpu().numpy()
    # covariance_z = torch.stack(covariance_z).cpu().numpy()
    # covariance_s = torch.stack(covariance_s).cpu().numpy()
    MCC(z_vec, s_vec, mode="weak")
    # MCC(covariance_z, covariance_s, mode="strong")


def MCC(z_vec, s_vec, mode="weak"):
    ccz = []
    ccs = []
    distaffs = []
    distaffz = []
    for i in range(len(z_vec)):
        for j in range(i + 1, len(z_vec)):
            if mode == "weak":
                w, _, _, _ = np.linalg.lstsq(z_vec[j], z_vec[i])
                affine = z_vec[j] @ w
            else:
                affine = z_vec[j]

            ccall = np.corrcoef(z_vec[i], affine, rowvar=False)
            ccz += [(ccall[0, -2] + ccall[1, -1]) / 2]
            # distaffz += [np.linalg.norm(affine - z_vec[i], ord=2) / (
            #             np.linalg.norm(affine, ord=2) ** 0.5 * np.linalg.norm(z_vec[i], ord=2) ** 0.5)]

            if mode == "weak":
                w, _, _, _ = np.linalg.lstsq(s_vec[j], s_vec[i])
                affine = s_vec[j] @ w
            else:
                affine = s_vec[j]

            ccall = np.corrcoef(s_vec[i], affine, rowvar=False)
            ccs += [(ccall[0, -2] + ccall[1, -1]) / 2]
            # distaffs += [np.linalg.norm(affine - s_vec[i], ord=2) / (
            #             np.linalg.norm(affine, ord=2) ** 0.5 * np.linalg.norm(s_vec[i], ord=2) ** 0.5)]

    ccz = np.mean(np.stack(ccz))
    # distaffz = np.mean(np.stack(distaffz))
    ccs = np.mean(np.stack(ccs))
    # distaffs = np.mean(np.stack(distaffs))
    print(f"{mode} MCC of Z:", ccz)
    print(f"{mode} DistAff of Z:", distaffz)
    print(f"{mode} MCC of S:", ccs)
    print(f"{mode} DistAff of S:", distaffs)


if __name__ == "__main__":
    print('Using GPU: ' + str(torch.cuda.is_available()))
    input_args = get_training_parser().parse_args()
    print('Arguments for training: ', input_args)
    set_logger(os.path.join(input_args.log_dir, "train.log"))
    main(input_args)
