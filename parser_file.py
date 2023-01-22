import argparse
from utils import int_tuple


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", default="./log/", help="Directory containing logging file")
    parser.add_argument("--model_dir", default="", help="Directory containing logging file")
    parser.add_argument("--tfdir", default="", type=str)
    parser.add_argument("--dataset_name", default="eth", type=str)
    parser.add_argument("--model_name", default="lstm", type=str)
    parser.add_argument("--resume", default="",
                        type=str, metavar="PATH", help="path to latest checkpoint (default: none)")

    # randomness
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to calculate MC expectations")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")

    # computation
    parser.add_argument("--gpu_num", default="1", type=str)
    parser.add_argument("--loader_num_workers", default=2, type=int)

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
    parser.add_argument("--filter_envs", type=str, default="0.6",
                        help="Filter only certain environments (i.e 0.1-0.3-0.5)")
    parser.add_argument("--skip", default=1, type=int)
    parser.add_argument("--delim", default="\t")
    parser.add_argument("--batch_method", default='hom', type=str,
                        help='Use Homogeneous (hom), Heterogeneous (het) or alternated homogeneous (alt) batches during training')
    parser.add_argument("--decoupled_loss", action=argparse.BooleanOptionalAction, help='decouple ELBO from y')

    parser.add_argument("--finetune", default="", type=str, help="Select the components of S prior to finetune")
    parser.add_argument("--batch_size", default='64', type=str)
    parser.add_argument("--shuffle", default=True, type=bool)
    parser.add_argument('--reduce', default=0, type=int)
    parser.add_argument('--reduceall', default=0, type=int, help="all data samples: 9000")

    # architecture (VE)
    parser.add_argument('--rel_recon', action='store_true', help="Whether to reconstruct relative trajectories or absolute trajectories")
    parser.add_argument('--no-rel_recon', dest='rel_recon', action='store_false', help="Whether to reconstruct relative trajectories or absolute trajectories")
    parser.add_argument("--z_dim", type=int, default=2, help="Dimension of z latent variable")
    parser.add_argument("--s_dim", type=int, default=2, help="Dimension of s latent variable")
    parser.add_argument("--mlp_latent_dim", type=int, default=8, help="Dimension of mlp encoders outputs")
    parser.add_argument("--num_envs", default=5, type=int, help="Number of environments in the dataset")
    parser.add_argument('--coupling', action='store_true', help="Whether to use coupling layers or not")
    parser.add_argument('--no-coupling', dest='coupling', action='store_false', help="Whether to use coupling layers or not")


    # spurious feature
    parser.add_argument('--add_confidence', action='store_true')
    parser.add_argument('--no-add_confidence', dest='add_confidence', action='store_false')
    parser.add_argument("--domain_shifts", default='0', type=str,
                        help='domain_shifts per environment: hotel,univ,zara1,zara2,eth')

    return parser


def get_evaluation_parser():
    parser = get_parser()
    parser.add_argument("--exp", default='all', choices=['pretrain', 'finetune', 'all'], help="Select Experiment")
    parser.add_argument("--dset_type", default="test", type=str)
    parser.add_argument("--best_k", default=20, type=int)
    parser.add_argument('--metrics', type=str, default='accuracy', choices=['accuracy', 'collision', 'qualitative'],
                        help='evaluate metrics')

    return parser


def get_training_parser():
    parser = get_parser()

    # training
    parser.add_argument("--best_k", default=1, type=int)
    parser.add_argument("--start-epoch", default=1, type=int, metavar="N",
                        help="manual epoch number (useful on restarts)")
    parser.add_argument("--use_gpu", default=1, type=int)

    # general training
    parser.add_argument("--num_epochs", default=1000, type=int)

    # learning rates
    parser.add_argument('--lr_scheduler', action='store_true')
    parser.add_argument('--no-lr_scheduler', dest='lr_scheduler', action='store_false')

    parser.add_argument("--lrvar", default=5e-3, type=float,
                        help="initial learning rate for variant encoder optimizer")
    parser.add_argument('--lrinv', default=5e-3, type=float,
                        help="initial learning rate for the invariant encoder optimizer")
    parser.add_argument('--lrfut', default=5e-3, type=float,
                        help="initial learning rate for the future decoder optimizer")
    parser.add_argument('--lrpast', default=5e-3, type=float,
                        help="initial learning rate for the past decoder optimizer")
    parser.add_argument('--lrmap', default=5e-3, type=float,
                        help="initial learning rate for the regressor optimizer")
    parser.add_argument('--lrpar', default=5e-3, type=float,
                        help="initial learning rate for the parameters optimizer")

    return parser
