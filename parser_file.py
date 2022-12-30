import argparse
from utils import int_tuple


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", default="./log/", help="Directory containing logging file")
    parser.add_argument("--model_dir", default="./models/E7/", help="Directory containing logging file")
    parser.add_argument("--tfdir", default='./runs/E7/', type=str)
    parser.add_argument("--dataset_name", default="v4", type=str)
    parser.add_argument("--model_name", default="mlp", type=str)
    parser.add_argument("--resume", default="./models/E1/P6/CRMF_epoch_889.pth.tar",
                        type=str, metavar="PATH", help="path to latest checkpoint (default: none)")

    # randomness
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to calculate MC expectations")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")

    # computation
    parser.add_argument("--gpu_num", default="1", type=str)
    parser.add_argument("--loader_num_workers", default=6, type=int)

    # architecture (STGAT)
    parser.add_argument("--traj_lstm_hidden_size", default=32, type=int)
    parser.add_argument("--hidden-units", type=str, default="16",
                        help="Hidden units in each hidden layer, splitted with comma")
    parser.add_argument("--graph_lstm_hidden_size", default=32, type=int)
    parser.add_argument("--heads", type=str, default="4,1", help="Heads in each layer, splitted with comma")
    parser.add_argument("--graph_network_out_dims", type=int, default=32,
                        help="dims of every node after through GAT module")
    parser.add_argument("--dropout", type=float, default=0, help="Dropout rate (1 - keep probability)")
    parser.add_argument("--alpha", type=float, default=0.2, help="Alpha for the leaky_relu")
    parser.add_argument('--teachingratio', default=0.0, type=float,
                        help="The probability of using ground truth future trajectories instead of model predictions during training")

    # dataset
    parser.add_argument("--obs_len", default=8, type=int)
    parser.add_argument("--fut_len", default=12, type=int)
    parser.add_argument("--n_coordinates", type=int, default=2, help="Number of coordinates")
    parser.add_argument("--filter_envs", type=str, default="0.1-0.3-0.5",
                        help="Filter only certain environments (i.e 0.1-0.3-0.5)")
    parser.add_argument("--skip", default=1, type=int)
    parser.add_argument("--delim", default="\t")
    parser.add_argument("--finetune_ratio", default=0.1, type=float, help="Number of batches to be used in finetuning")
    parser.add_argument("--batch_method", default='hom', type=str,
                        help='Use Homogeneous (hom), Heterogeneous (het) or alternated homogeneous (alt) batches during training')
    parser.add_argument("--contrastive", default=False, type=bool)
    parser.add_argument("--batch_size", default='64', type=str)
    parser.add_argument("--shuffle", default=True, type=bool)
    parser.add_argument('--reduce', default=0, type=int)
    parser.add_argument('--reduceall', default=9000, type=int)


    # architecture (VE)
    parser.add_argument("--z_dim", type=int, default=2, help="Dimension of z latent variable")
    parser.add_argument("--s_dim", type=int, default=2, help="Dimension of s latent variable")
    parser.add_argument("--num_envs", default=3, type=int, help="Number of environments in the dataset")

    # spurious feature
    parser.add_argument("--add_confidence", default=False, type=bool)
    parser.add_argument("--domain_shifts", default='0', type=str,
                        help='domain_shifts per environment: hotel,univ,zara1,zara2,eth')

    return parser


def get_evaluation_parser():
    parser = get_parser()
    parser.add_argument("--dset_type", default="test", type=str)
    parser.add_argument("--best_k", default=20, type=int)
    parser.add_argument('--metrics', type=str, default='accuracy', choices=['accuracy', 'collision', 'qualitative'],
                        help='evaluate metrics')

    return parser


def get_training_parser():
    parser = get_parser()

    # dataset
    parser.add_argument("--filter_envs_pretrain", type=str, default="",
                        help="Say which env were used during pretraining (for contrastive loss) (i.e 0.1-0.3-0.5)")

    # training
    parser.add_argument("--best_k", default=20, type=int)
    parser.add_argument("--start-epoch", default=1, type=int, metavar="N",
                        help="manual epoch number (useful on restarts)")
    parser.add_argument("--use_gpu", default=1, type=int)

    # general training
    parser.add_argument("--finetune", default="", type=str)
    parser.add_argument("--num_epochs", default='50-20-20-1-20-100', type=lambda x: int_tuple(x, '-'))  # '150-100-150',

    # learning rates
    parser.add_argument("--lr_scheduler", default=False, type=bool)  # '150-100-150',

    parser.add_argument("--lrvar", default=1e-3, type=float,
                        help="initial learning rate for variant encoder optimizer")
    parser.add_argument('--lrinv', default=1e-3, type=float,
                        help="initial learning rate for the invariant encoder optimizer")
    parser.add_argument('--lrfut', default=1e-3, type=float,
                        help="initial learning rate for the future decoder optimizer")
    parser.add_argument('--lrpast', default=1e-3, type=float,
                        help="initial learning rate for the past decoder optimizer")
    parser.add_argument('--lrmap', default=1e-3, type=float,
                        help="initial learning rate for the regressor optimizer")
    parser.add_argument('--lrpar', default=1e-3, type=float,
                        help="initial learning rate for the parameters optimizer")

    return parser
