import os
import logging
import random
from re import L
import numpy as np
import cv2
from collections import defaultdict

from typing import Optional

import torch
import torch.nn as nn

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

    elif dataset_name in ['sdd_domain0', 'sdd_domain1', 'sdd_domain2', 'sdd_domain3']:
        envs_path = [dset_path]
        envs_name = [dataset_name]

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
        loss = (fut_traj[:, :, :2].repeat(pred_fut_traj.shape[1], 1, 1, 1).permute(0, 2, 1, 3) - pred_fut_traj.permute(
            1, 2, 0, 3)) ** 2
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


def set_name_experiment(args, name='GCRL'):
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


def get_model_name(args, name='GCRL', epoch=None, time=False):
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

    filename = f'{filefolder}/{get_model_name(args, model_name)}.pth.tar'
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


def gkern(kernlen=31, nsig=4):
    """	creates gaussian kernel with side length l and a sigma of sig """
    ax = np.linspace(-(kernlen - 1) / 2., (kernlen - 1) / 2., kernlen)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(nsig))
    return kernel / np.sum(kernel)


def create_gaussian_heatmap_template(size, kernlen=81, nsig=4, normalize=True):
    """ Create a big gaussian heatmap template to later get patches out """
    template = np.zeros([size, size])
    kernel = gkern(kernlen=kernlen, nsig=nsig)
    m = kernel.shape[0]
    x_low = template.shape[1] // 2 - int(np.floor(m / 2))
    x_up = template.shape[1] // 2 + int(np.ceil(m / 2))
    y_low = template.shape[0] // 2 - int(np.floor(m / 2))
    y_up = template.shape[0] // 2 + int(np.ceil(m / 2))
    template[y_low:y_up, x_low:x_up] = kernel
    if normalize:
        template = template / template.max()
    return template


def create_dist_mat(size, normalize=True):
    """ Create a big distance matrix template to later get patches out """
    middle = size // 2
    dist_mat = np.linalg.norm(np.indices([size, size]) - np.array([middle, middle])[:, None, None], axis=0)
    if normalize:
        dist_mat = dist_mat / dist_mat.max() * 2
    return dist_mat


def get_patch(template, traj, H, W):
    x = np.round(traj[:, 0]).astype('int')
    y = np.round(traj[:, 1]).astype('int')

    x_low = template.shape[1] // 2 - x
    x_up = template.shape[1] // 2 + W - x
    y_low = template.shape[0] // 2 - y
    y_up = template.shape[0] // 2 + H - y

    patch = [template[y_l:y_u, x_l:x_u] for x_l, x_u, y_l, y_u in zip(x_low, x_up, y_low, y_up)]

    return patch


def preprocess_image_for_segmentation(images, encoder='resnet101', encoder_weights='imagenet', seg_mask=False,
                                      classes=6):
    """
     Preprocess image for pretrained semantic segmentation,
     input is dictionary containing images
     In case input is segmentation map,
     then it will create one-hot-encoding from discrete values
     """
    import segmentation_models_pytorch as smp

    preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder, encoder_weights)

    for key, im_dict in images.items():
        for key2, im in im_dict.items():
            if seg_mask:
                im = [(im == v) for v in range(classes)]
                im = np.stack(im, axis=-1)  # .astype('int16')
            else:
                im = preprocessing_fn(im)
            im = im.transpose(2, 0, 1).astype('float32')
            im = torch.Tensor(im)
            images[key][key2] = im


def resize_image(images, factor, seg_mask=False):
    for key, image_dict in images.items():
        for key2, image in image_dict.items():
            if seg_mask:
                images[key][key2] = cv2.resize(image, (0, 0), fx=factor, fy=factor, interpolation=cv2.INTER_NEAREST)
            else:
                images[key][key2] = cv2.resize(image, (0, 0), fx=factor, fy=factor, interpolation=cv2.INTER_AREA)


def pad(images, division_factor=32):
    """
    Pad image so that it can be divided by division_factor, as many architectures such as UNet needs a specific size
    at it's bottlenet layer
    """

    for key, im_dict in images.items():
        for key2, im in im_dict.items():
            if im.ndim == 3:
                H, W, C = im.shape
            else:
                H, W = im.shape
            H_new = int(np.ceil(H / division_factor) * division_factor)
            W_new = int(np.ceil(W / division_factor) * division_factor)
            im = cv2.copyMakeBorder(im, 0, H_new - H, 0, W_new - W, cv2.BORDER_CONSTANT)
            images[key][key2] = im


def sampling(probability_map, num_samples, rel_threshold=None, replacement=False):
    # new view that has shape=[batch*timestep, H*W]
    prob_map = probability_map.view(probability_map.size(0) * probability_map.size(1), -1)
    if rel_threshold is not None:
        thresh_values = prob_map.max(dim=1)[0].unsqueeze(1).expand(-1, prob_map.size(1))
        mask = prob_map < thresh_values * rel_threshold
        prob_map = prob_map * (~mask).int()
        prob_map = prob_map / prob_map.sum()

    # samples.shape=[batch*timestep, num_samples]
    samples = torch.multinomial(prob_map, num_samples=num_samples, replacement=replacement)
    # samples.shape=[batch, timestep, num_samples]

    # unravel sampled idx into coordinates of shape [batch, time, sample, 2]
    samples = samples.view(probability_map.size(0), probability_map.size(1), -1)
    idx = samples.unsqueeze(3)
    preds = idx.repeat(1, 1, 1, 2).float()
    preds[:, :, :, 0] = (preds[:, :, :, 0]) % probability_map.size(3)
    preds[:, :, :, 1] = torch.floor((preds[:, :, :, 1]) / probability_map.size(3))

    return preds


def image2world(image_coords, scene, homo_mat, resize):
    """
	Transform trajectories of one scene from image_coordinates to world_coordinates
	:param image_coords: torch.Tensor, shape=[num_person, (optional: num_samples), timesteps, xy]
	:param scene: string indicating current scene, options=['eth', 'hotel', 'student01', 'student03', 'zara1', 'zara2']
	:param homo_mat: dict, key is scene, value is torch.Tensor containing homography matrix (data/eth_ucy/scene_name.H)
	:param resize: float, resize factor
	:return: trajectories in world_coordinates
	"""
    traj_image2world = image_coords.clone()
    if traj_image2world.dim() == 4:
        traj_image2world = traj_image2world.reshape(-1, image_coords.shape[2], 2)
    if scene in ['eth', 'hotel']:
        # eth and hotel have different coordinate system than ucy data
        traj_image2world[:, :, [0, 1]] = traj_image2world[:, :, [1, 0]]
    traj_image2world = traj_image2world / resize
    traj_image2world = F.pad(input=traj_image2world, pad=(0, 1, 0, 0), mode='constant', value=1)
    traj_image2world = traj_image2world.reshape(-1, 3)
    traj_image2world = torch.matmul(homo_mat[scene], traj_image2world.T).T
    traj_image2world = traj_image2world / traj_image2world[:, 2:]
    traj_image2world = traj_image2world[:, :2]
    traj_image2world = traj_image2world.view_as(image_coords)
    return traj_image2world


def read_images(data, image_path, image_file, seg_mask=False):

    images = defaultdict(dict)
    for env in data.envId.unique():
        for scene in data.sceneId.unique():
            im_path = os.path.join(image_path, f"{env}_{scene}_{image_file}")
            if seg_mask:
                im = cv2.imread(im_path, 0)
            else:
                im = cv2.imread(im_path)
            if im is not None:
                images[env][scene] = im

    return images


def augment_data(data, image_path='data/SDD/train', images={}, image_file='reference.jpg', seg_mask=False):
    """
    Perform data augmentation
    :param data: Pandas df, needs x,y,metaId,sceneId columns
    :param image_path: example - 'data/SDD/val'
    :param images: dict with key being sceneId, value being PIL image
    :param image_file: str, image file name
    :param seg_mask: whether it's a segmentation mask or an image file
    :return:
    """

    ks = [1, 2, 3]
    data_ = data.copy()  # data without rotation, used so rotated data can be appended to original df
    k2rot = {1: '_rot90', 2: '_rot180', 3: '_rot270'}
    for k in ks:
        metaId_max = data['metaId'].max()
        for env in data_.envId.unique():
            for scene in data_.sceneId.unique():
                im_path = os.path.join(image_path, f"{env}_{scene}_{image_file}")
                if seg_mask:
                    im = cv2.imread(im_path, 0)
                else:
                    im = cv2.imread(im_path)

                if im is not None:
                    data_rot, im = rot(data_[(data_.sceneId == scene) & (data_.envId == env)], im, k)
                    # image
                    rot_angle = k2rot[k]
                    images[env][scene + rot_angle] = im

                    data_rot['sceneId'] = scene + rot_angle
                    data_rot['metaId'] = data_rot['metaId'] + metaId_max + 1
                    data = data.append(data_rot)

    metaId_max = data['metaId'].max()
    for env in data.envId.unique():
        for scene in data.sceneId.unique():
            try:
                im = images[env][scene]
                data_flip, im_flip = fliplr(data[(data.sceneId == scene) & (data.envId == env)], im)
                data_flip['sceneId'] = data_flip['sceneId'] + '_fliplr'
                data_flip['metaId'] = data_flip['metaId'] + metaId_max + 1
                data = data.append(data_flip)
                images[env][scene + '_fliplr'] = im_flip
            except:
                continue

    return data, images


def rot(df, image, k=1):
    """
    Rotates image and coordinates counter-clockwise by k * 90° within image origin
    :param df: Pandas DataFrame with at least columns 'x' and 'y'
    :param image: PIL Image
    :param k: Number of times to rotate by 90°
    :return: Rotated Dataframe and image
    """

    xy = df.copy()
    if image.ndim == 3:
        y0, x0, channels = image.shape
    else:
        y0, x0 = image.shape

    xy.loc()[:, 'x'] = xy['x'] - x0 / 2
    xy.loc()[:, 'y'] = xy['y'] - y0 / 2
    c, s = np.cos(-k * np.pi / 2), np.sin(-k * np.pi / 2)
    R = np.array([[c, s], [-s, c]])
    xy.loc()[:, ['x', 'y']] = np.dot(xy[['x', 'y']], R)
    for i in range(k):
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

    if image.ndim == 3:
        y0, x0, channels = image.shape
    else:
        y0, x0 = image.shape

    xy.loc()[:, 'x'] = xy['x'] + x0 / 2
    xy.loc()[:, 'y'] = xy['y'] + y0 / 2
    return xy, image


def fliplr(df, image):
    """
    Flip image and coordinates horizontally
    :param df: Pandas DataFrame with at least columns 'x' and 'y'
    :param image: PIL Image
    :return: Flipped Dataframe and image
    """

    xy = df.copy()
    if image.ndim == 3:
        y0, x0, channels = image.shape
    else:
        y0, x0 = image.shape

    xy.loc()[:, 'x'] = xy['x'] - x0 / 2
    xy.loc()[:, 'y'] = xy['y'] - y0 / 2
    R = np.array([[-1, 0], [0, 1]])
    xy.loc()[:, ['x', 'y']] = np.dot(xy[['x', 'y']], R)
    image = cv2.flip(image, 1)

    if image.ndim == 3:
        y0, x0, channels = image.shape
    else:
        y0, x0 = image.shape

    xy.loc()[:, 'x'] = xy['x'] + x0 / 2
    xy.loc()[:, 'y'] = xy['y'] + y0 / 2
    return xy, image


def create_meshgrid(x: torch.Tensor, normalized_coordinates: Optional[bool]) -> torch.Tensor:

    assert len(x.shape) == 4, x.shape

    _, _, height, width = x.shape
    _device, _dtype = x.device, x.dtype
    if normalized_coordinates:
        xs = torch.linspace(-1.0, 1.0, width, device=_device, dtype=_dtype)
        ys = torch.linspace(-1.0, 1.0, height, device=_device, dtype=_dtype)
    else:
        xs = torch.linspace(0, width - 1, width, device=_device, dtype=_dtype)
        ys = torch.linspace(0, height - 1, height, device=_device, dtype=_dtype)

    return torch.meshgrid(ys, xs)  # pos_y, pos_x


class SoftArgmax2D(nn.Module):
    r"""Creates a module that computes the Spatial Soft-Argmax 2D
    of a given input heatmap.
    Returns the index of the maximum 2d coordinates of the give map.
    The output order is x-coord and y-coord.
    Arguments:
        normalized_coordinates (Optional[bool]): wether to return the
          coordinates normalized in the range of [-1, 1]. Otherwise,
          it will return the coordinates in the range of the input shape.
          Default is True.
    Shape:
        - Input: :math:`(B, N, H, W)`
        - Output: :math:`(B, N, 2)`
    Examples::
        >>> input = torch.rand(1, 4, 2, 3)
        >>> m = tgm.losses.SpatialSoftArgmax2d()
        >>> coords = m(input)  # 1x4x2
        >>> x_coord, y_coord = torch.chunk(coords, dim=-1, chunks=2)
    """

    def __init__(self, normalized_coordinates: Optional[bool] = True) -> None:
        super(SoftArgmax2D, self).__init__()
        self.normalized_coordinates: Optional[bool] = normalized_coordinates
        self.eps: float = 1e-6

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(input):
            raise TypeError("Input input type is not a torch.Tensor. Got {}"
                            .format(type(input)))
        if not len(input.shape) == 4:
            raise ValueError("Invalid input shape, we expect BxCxHxW. Got: {}"
                             .format(input.shape))
        # unpack shapes and create view from input tensor
        batch_size, channels, height, width = input.shape
        x: torch.Tensor = input.view(batch_size, channels, -1)

        # compute softmax with max substraction trick
        exp_x = torch.exp(x - torch.max(x, dim=-1, keepdim=True)[0])
        exp_x_sum = 1.0 / (exp_x.sum(dim=-1, keepdim=True) + self.eps)

        # create coordinates grid
        pos_y, pos_x = create_meshgrid(input, self.normalized_coordinates)
        pos_x = pos_x.reshape(-1)
        pos_y = pos_y.reshape(-1)

        # compute the expected coordinates
        expected_y: torch.Tensor = torch.sum(
            (pos_y * exp_x) * exp_x_sum, dim=-1, keepdim=True)
        expected_x: torch.Tensor = torch.sum(
            (pos_x * exp_x) * exp_x_sum, dim=-1, keepdim=True)
        output: torch.Tensor = torch.cat([expected_x, expected_y], dim=-1)

        return output.view(batch_size, channels, 2)  # BxNx2