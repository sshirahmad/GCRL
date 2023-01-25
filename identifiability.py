
from loader import data_loader
from parser_file import get_evaluation_parser
from utils import *
from models import VCRL


def main(args):
    # Set environment variables
    set_seed_globally(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num

    args.batch_size = "10000"
    args.shuffle = False

    logging.info("Initializing Sets")
    envs_path, envs_name = get_envs_path(args.dataset_name, args.dset_type, args.filter_envs)
    loaders = [data_loader(args, env_path, env_name) for env_path, env_name in
                     zip(envs_path, envs_name)]

    # training routine length
    num_batches = min([len(loader) for loader in loaders])

    # bring different dataset all together for simplicity of the next functions
    dataset = {'loaders': loaders, 'names': envs_name, 'num_batches': num_batches}

    z_vec = []
    s_vec = []
    for i in range(len(args.paths)):
        args.resume = args.paths[i]
        # create the model
        model = VCRL(args).cuda()
        load_all_model(args, model, None)
        model.cuda()
        model.eval()
        with torch.no_grad():
            z_vec_seed = []
            s_vec_seed = []
            for _, (loader, loader_name) in enumerate(zip(dataset['loaders'], dataset['names'])):
                for batch_idx, batch in enumerate(loader):
                    batch = [tensor.cuda() for tensor in batch]
                    (obs_traj, fut_traj, _, _, _, _, _) = batch

                    qz, qs = model(batch, identify=True)
                    z_vec_seed += [qz.rsample([20, ]).flatten(start_dim=0, end_dim=1)]
                    s_vec_seed += [qs.rsample([20, ]).flatten(start_dim=0, end_dim=1)]

        s_vec += [torch.cat(s_vec_seed)]
        z_vec += [torch.cat(z_vec_seed)]

    s_vec = torch.stack(s_vec).cpu().numpy()
    z_vec = torch.stack(z_vec).cpu().numpy()
    MCC(z_vec, s_vec, mode=args.mcc)


def MCC(z_vec, s_vec, mode="weak"):
    ccz = []
    ccs = []
    for i in range(len(z_vec)):
        for j in range(i + 1, len(z_vec)):
            if mode == "weak":
                w, _, _, _ = np.linalg.lstsq(z_vec[j], z_vec[i])
                affine = z_vec[j] @ w
            else:
                affine = z_vec[j]

            ccall = np.corrcoef(z_vec[i], affine, rowvar=False)
            ccz += [(ccall[0, -2] + ccall[1, -1]) / 2]

            if mode == "weak":
                w, _, _, _ = np.linalg.lstsq(s_vec[j], s_vec[i])
                affine = s_vec[j] @ w
            else:
                affine = s_vec[j]

            ccall = np.corrcoef(s_vec[i], affine, rowvar=False)
            ccs += [(ccall[0, -2] + ccall[1, -1]) / 2]

    ccz = np.mean(np.stack(ccz))
    ccs = np.mean(np.stack(ccs))
    print(f"{mode} MCC of Z:", ccz)
    print(f"{mode} MCC of S:", ccs)


if __name__ == "__main__":
    print('Using GPU: ' + str(torch.cuda.is_available()))
    input_args = get_evaluation_parser().parse_args()
    print('Arguments for training: ', input_args)
    set_logger(os.path.join(input_args.log_dir, "train.log"))
    main(input_args)
