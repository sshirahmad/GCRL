import os
import logging
import random
from re import L
import torch
import numpy as np
from torch.distributions import MultivariateNormal, Gamma, Poisson

NUMBER_PERSONS = 2
NUMBER_COUPLES = 2

from datetime import datetime


class AverageMeter(object):
    """
    Computes and stores the average and current value of a specific metric
    """

    def __init__(self, name, fmt=":.4f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"

        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        logging.info("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "Batch: " + fmt + "/" + fmt.format(num_batches)


def set_logger(log_path):
    """
    Set the logger to log info in terminal and file `log_path`.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        fold = log_path.rsplit('/', 1)[0]
        if not os.path.exists(fold):
            os.makedirs(fold)
        open(log_path, "w+")
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s:%(levelname)s: %(message)s")
        )
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(stream_handler)


def relative_to_abs(rel_traj, start_pos):
    """
    Convert relative coordinates in 'natural' coordinates

    Inputs:
    - rel_traj: pytorch tensor of shape (seq_len, batch, 2)
    - start_pos: pytorch tensor of shape (batch, 2)
    Outputs:
    - abs_traj: pytorch tensor of shape (seq_len, batch, 2)
    """
    rel_traj = rel_traj.permute(1, 0, 2)  # --> (batch, seq_len, 2)
    displacement = torch.cumsum(rel_traj, dim=1)
    start_pos = torch.unsqueeze(start_pos, dim=1)
    abs_traj = displacement + start_pos
    return abs_traj.permute(1, 0, 2)


def erm_loss(l2_loss_rel, seq_start_end):
    loss_sum = torch.zeros(1).cuda()
    for start, end in seq_start_end.data:
        _l2_loss_rel = torch.narrow(l2_loss_rel, 0, start, end - start)
        _l2_loss_rel = torch.sum(_l2_loss_rel, dim=0)  # [best_k elements]
        _l2_loss_rel = torch.max(_l2_loss_rel) / (end - start)
        loss_sum += _l2_loss_rel

    return loss_sum


def get_dset_path(dset_name, dset_type):
    _dir = os.path.dirname(__file__)
    return os.path.join(_dir, "datasets", dset_name, dset_type)


def get_envs_path(dataset_name, dset_type, filter_envs):
    dset_path = get_dset_path(dataset_name, dset_type)

    # ETH-UCY Dataset
    ETH_UCY = ['eth', 'hotel', 'univ', 'zara1', 'zara2']
    if dataset_name in ETH_UCY:
        files_name = os.listdir(dset_path)
        if dset_type == 'test':
            ETH_UCY = [dataset_name]
        else:
            ETH_UCY.remove(dataset_name)
        envs_names = []
        for i, env in enumerate(ETH_UCY):
            envs_names.append([])
            if env == 'eth':
                for file_name in files_name:
                    if 'biwi_eth' in file_name:
                        envs_names[i].append(file_name)
            elif env == 'hotel':
                for file_name in files_name:
                    if 'biwi_hotel' in file_name:
                        envs_names[i].append(file_name)
            elif env == 'univ':
                for file_name in files_name:
                    if ('students' in file_name) or ('uni_examples' in file_name):
                        envs_names[i].append(file_name)
            elif env == 'zara1':
                for file_name in files_name:
                    if 'crowds_zara01' in file_name:
                        envs_names[i].append(file_name)
            elif env == 'zara2':
                for file_name in files_name:
                    if ('crowds_zara02' in file_name) or ('crowds_zara03' in file_name):
                        envs_names[i].append(file_name)
        envs_paths = [[os.path.join(dset_path, env_name) for env_name in env_names] for env_names in envs_names]
        return envs_paths, ETH_UCY

    # Synthetic Dataset
    elif dataset_name in ['synthetic', 'v2', 'v2full', 'v4'] or 'synthetic' in dataset_name:
        envs_name = os.listdir(dset_path)
        if filter_envs != '':
            filter_envs = [i for i in filter_envs.split('-')]
            envs_name_ = []
            for env_name in envs_name:
                for filter_env in filter_envs:
                    if filter_env + '_radius' in env_name:
                        envs_name_.append(env_name)
            envs_name = envs_name_
        envs_path = [os.path.join(dset_path, env_name) for env_name in envs_name]
        return envs_path, envs_name

    else:
        logging.raiseExceptions(dataset_name + ' dataset doesn\'t exists')


def int_tuple(s, delim=','):
    return tuple(int(i) for i in s.strip().split(delim))


def l2_loss(pred_fut_traj, fut_traj, mode="average"):
    """
    Compute L2 loss

    Input:
    - pred_fut_traj: Tensor of shape (seq_len, batch, 2). Predicted trajectory.
    - fut_traj: Tensor of shape (seq_len, batch, 2). Ground truth future trajectory.
    - mode: Can be one of sum, average, raw
    Output:
    - loss: l2 loss depending on mode
    """
    if len(pred_fut_traj.size()) == 4:
        loss = (fut_traj[:, :, :2].repeat(pred_fut_traj.shape[1], 1, 1, 1).permute(0, 2, 1, 3) - pred_fut_traj.permute(1, 2, 0, 3)) ** 2
    else:
        loss = (fut_traj[:, :, :2].permute(1, 0, 2) - pred_fut_traj.permute(1, 0, 2)) ** 2

    if mode == "sum":
        return torch.sum(loss)
    elif mode == "average":
        return torch.mean(loss)
    elif mode == "raw":
        if len(pred_fut_traj.size()) == 3:
            return loss.sum(dim=2).sum(dim=1)
        else:
            return loss.sum(dim=3).sum(dim=2)


def displacement_error(pred_fut_traj, fut_traj, consider_ped=None, mode="sum"):
    """
    Compute ADE

    Input:
    - pred_fut_traj: Tensor of shape (seq_len, batch, 2). Predicted trajectory. [12, person_num, 2]
    - fut_traj: Tensor of shape (seq_len, batch, 2). Groud truth future trajectory.
    - consider_ped: Tensor of shape (batch)
    - mode: Can be one of sum, raw
    Output:
    - loss: gives the Euclidean displacement error
    """

    loss = (fut_traj.permute(1, 0, 2) - pred_fut_traj.permute(1, 0, 2)) ** 2
    if consider_ped is not None:
        loss = torch.sqrt(loss.sum(dim=2)).sum(dim=1) * consider_ped
    else:
        loss = torch.sqrt(loss.sum(dim=2)).sum(dim=1)
    if mode == "sum":
        return torch.sum(loss)
    elif mode == "mean":
        return torch.mean(loss)
    elif mode == "raw":
        return loss


def final_displacement_error(pred_fut_pos, fut_pos, consider_ped=None, mode="sum"):
    """
    Compute FDE

    Input:
    - pred_fut_pos: Tensor of shape (batch, 2). Predicted last pos.
    - fut_pos: Tensor of shape (seq_len, batch, 2). Groud truth last pos.
    - consider_ped: Tensor of shape (batch)
    Output:
    - loss: gives the eculidian displacement error
    """

    loss = (fut_pos - pred_fut_pos) ** 2
    if consider_ped is not None:
        loss = torch.sqrt(loss.sum(dim=1)) * consider_ped
    else:
        loss = torch.sqrt(loss.sum(dim=1))
    if mode == "raw":
        return loss
    else:
        return torch.sum(loss)


def set_domain_shift(domain_shifts, env_name):
    """
    Set the domain shift
    """
    domain_shifts = [int(i) for i in domain_shifts.split('-')]
    if len(domain_shifts) == 5:
        if env_name == 'hotel' or 'env1' in env_name:
            alpha_e = domain_shifts[0]
        elif env_name == 'univ' or 'env2' in env_name:
            alpha_e = domain_shifts[1]
        elif env_name == 'zara1' or 'env3' in env_name:
            alpha_e = domain_shifts[2]
        elif env_name == 'zara2' or 'env4' in env_name:
            alpha_e = domain_shifts[3]
        elif env_name == 'eth' or 'env5' in env_name:
            alpha_e = domain_shifts[4]
        else:
            raise ValueError('Unkown Environment!')
    elif len(domain_shifts) == 1:
        alpha_e = domain_shifts[0]
    else:
        raise ValueError('Express a domain_shift for each of the 5 enviroment or 1 for all.')
    return alpha_e


def set_name_experiment(args, name='VCRL'):

    return f'{name}_data_{args.dataset_name}_ds_{args.domain_shifts}_bk_{args.best_k}_ns_{args.num_samples}_ep_{args.num_epochs}_seed_{args.seed}_cl_{args.coupling}_dc_{args.decoupled_loss}_latentdim_{args.z_dim}_cluster_{args.num_envs}'


def set_batch_size(batch_method, batch_sizes, env_name):
    '''
    Set the batch size
    '''
    # heterogenous batches
    if batch_method == 'het' or batch_method == 'alt':
        if batch_sizes == '':
            # ETH-UCY Dataset
            if env_name == 'hotel':
                return 7
            elif env_name == 'univ':
                return 30
            elif env_name == 'zara1':
                return 16
            elif env_name == 'zara2':
                return 38
            elif env_name == 'eth':
                return 1
            # Synthetic Dataset
            else:
                return 64
        else:
            batch_sizes = [int(i) for i in batch_sizes.split('-')]
            if len(batch_sizes) == 5:
                if env_name == 'hotel' or 'env1' in env_name:
                    return batch_sizes[0]
                elif env_name == 'univ' or 'env2' in env_name:
                    return batch_sizes[1]
                elif env_name == 'zara1' or 'env3' in env_name:
                    return batch_sizes[2]
                elif env_name == 'zara2' or 'env4' in env_name:
                    return batch_sizes[3]
                elif env_name == 'eth' or 'env5' in env_name:
                    return batch_sizes[4]
                else:
                    raise ValueError('Unkown Environment!')
            elif len(batch_sizes) == 1:
                return batch_sizes[0]
            else:
                raise ValueError('Express a batch_size for each of the 5 enviroment or 1 for all.')

    # homogeneous batches
    elif batch_method == 'hom':
        if batch_sizes == '':
            return 64
        else:
            return int(batch_sizes)
    else:
        raise ValueError('Unkown batch method')


def interpolate_traj(traj, num_interp=4):
    """
    Add linearly interpolated points of a trajectory
    """
    sz = traj.shape
    dense = np.zeros((sz[0], (sz[1] - 1) * (num_interp + 1) + 1, 2))
    dense[:, :1, :] = traj[:, :1]

    for i in range(num_interp + 1):
        ratio = (i + 1) / (num_interp + 1)
        dense[:, i + 1::num_interp + 1, :] = traj[:, 0:-1] * (1 - ratio) + traj[:, 1:] * ratio

    return dense


def evaluate_helper(error, seq_start_end):
    sum_ = 0
    error = torch.stack(error, dim=1)
    for (start, end) in seq_start_end:
        start = start.item()
        end = end.item()
        _error = error[start:end]
        _error = torch.sum(_error, dim=0)
        _error = torch.min(_error)
        sum_ += _error
    return sum_


def cal_ade_fde(fut_traj, pred_fut_traj, mode='sum'):
    """
    Compute the ADE and FDE
    """
    ade = displacement_error(pred_fut_traj, fut_traj, mode=mode)
    fde = final_displacement_error(pred_fut_traj[-1], fut_traj[-1], mode=mode)
    return ade, fde


best_ade = 100

def set_seed_globally(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_model_name(args, name='VCRL', epoch=None, time=False):
    if time:
        name = datetime.now().strftime("%m-%d_%H:%M_") + name

    if epoch:
        name += f'_epoch_{epoch}'

    if args.finetune:
        name = f'model_t{args.reduce}'

    return name


def set_name_method(method):
    if 's_3' in method:
        model = 'Baseline'
    elif 's_6' in method:
        model = 'Modular'

    if 'i_0.0' in method:
        risk = 'ERM'
    else:
        risk = f'IRM (λ={method[6:]})'
    return f'{model} {risk}'


def set_name_env(env):
    if env in [0.1, 0.3, 0.5]:
        return 'IID'
    elif env == 0.4:
        return 'OoD-Inter'
    elif env == 0.6:
        return 'OoD-Extra'


def set_name_finetune(finetune):
    if 'integ' in finetune:
        return 'Update f only'
    elif 'all' in finetune:
        return 'Update Ψ,f,g'
    elif 'refinement' in finetune:
        return 'Update f + Refinement'


def save_all_model(args, model, model_name, optimizers, metric, epoch):

    checkpoint = {
        'epoch': epoch + 1,
        'state_dicts': {
            'variant_encoder': model.variant_encoder.state_dict(),
            'invariant_encoder': model.invariant_encoder.state_dict(),
            'x_to_z': model.x_to_z.state_dict(),
            'x_to_s': model.x_to_s.state_dict(),
            'future_decoder': model.future_decoder.state_dict(),
            'past_decoder': model.past_decoder.state_dict(),
            'pi_priore': model.pi_priore,
        },
        'optimizers': {
            key: val.state_dict() for key, val in optimizers.items()
        },
        'metric': metric,
    }

    if args.coupling:
        checkpoint['state_dicts']['coupling_layers_z'] = model.coupling_layers_z.state_dict()
        checkpoint['state_dicts']['coupling_layers_s'] = model.coupling_layers_s.state_dict()
    else:
        checkpoint['state_dicts']['mean_priors'] = model.mean_priors
        checkpoint['state_dicts']['logvar_priors'] = model.logvar_priors
        checkpoint['state_dicts']['mean_priorz'] = model.mean_priorz
        checkpoint['state_dicts']['logvar_priorz'] = model.logvar_priorz

    if args.model_dir:
        filefolder = f'{args.model_dir}'
    else:
        if args.finetune:
            phase = 'finetune'
        else:
            phase = 'pretrain'
        filefolder = f'./models/{args.dataset_name}/{phase}'

        if args.finetune:
            filefolder += f'/{args.finetune}/{args.seed}'

    # Check whether the specified path exists or not
    if not os.path.exists(filefolder):
        os.makedirs(filefolder)

    filename = f'{filefolder}/{get_model_name(args, model_name, epoch=epoch)}.pth.tar'
    torch.save(checkpoint, filename)
    logging.info(f" --> Model Saved in {filename}")


def load_all_model(args, model, optimizers, lr_schedulers=None, num_batches=0):
    model_path = args.resume

    if os.path.isfile(model_path):
        checkpoint = torch.load(model_path, map_location='cpu')
        args.start_epoch = checkpoint['epoch']

        models_checkpoint = checkpoint['state_dicts']

        if lr_schedulers != None:
            for name, lr_scheduler_optim in lr_schedulers.items():
                if lr_scheduler_optim != None:
                    lr_scheduler_optim.last_epoch = (args.start_epoch - 1) * num_batches

        # future decoder
        model.future_decoder.load_state_dict(models_checkpoint['future_decoder'])
        if optimizers != None:
            optimizers['future_decoder'].load_state_dict(checkpoint['optimizers']['future_decoder'])
            update_lr(optimizers['future_decoder'], args.lrfut)

        # past decoder
        model.past_decoder.load_state_dict(models_checkpoint['past_decoder'])
        if optimizers != None:
            optimizers['past_decoder'].load_state_dict(checkpoint['optimizers']['past_decoder'])
            update_lr(optimizers['past_decoder'], args.lrpast)

        # invariant encoder
        if args.coupling:
            model.coupling_layers_z.load_state_dict(models_checkpoint['coupling_layers_z'])
        else:
            model.mean_priorz.data = models_checkpoint['mean_priorz'].data.cuda()
            model.logvar_priorz.data = models_checkpoint['logvar_priorz'].data.cuda()
        model.invariant_encoder.load_state_dict(models_checkpoint['invariant_encoder'])
        model.x_to_z.load_state_dict(models_checkpoint['x_to_z'])
        if optimizers != None:
            optimizers['inv'].load_state_dict(checkpoint['optimizers']['inv'])
            update_lr(optimizers['inv'], args.lrinv)

        # variant encoder
        if args.coupling:
            model.coupling_layers_s.load_state_dict(models_checkpoint['coupling_layers_s'])
        else:
            model.mean_priors.data = models_checkpoint['mean_priors'].data.cuda()
            model.logvar_priors.data = models_checkpoint['logvar_priors'].data.cuda()
        model.pi_priore.data = models_checkpoint['pi_priore'].data.cuda()
        model.variant_encoder.load_state_dict(models_checkpoint['variant_encoder'])
        model.x_to_s.load_state_dict(models_checkpoint['x_to_s'])
        if optimizers != None:
            optimizers['var'].load_state_dict(checkpoint['optimizers']['var'])
            update_lr(optimizers['var'], args.lrvar)

        logging.info("=> loaded checkpoint '{}' (epoch {})".format(model_path, checkpoint["epoch"]))

    else:
        logging.info('model {} not found'.format(model_path))


def get_fake_optim():
    import torch.nn as nn
    l = nn.Linear(1, 1)
    return torch.optim.Adam(l.parameters())


def freeze(freez, models):
    for model in models:
        if model != None:
            for p in model.parameters():
                p.requires_grad = not freez


def update_lr(opt, lr):
    for param_group in opt.param_groups:
        param_group["lr"] = lr
