import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from loader import data_loader
from parser_file import get_training_parser
from utils import *
from models import VCRL
import math
from torch.optim.lr_scheduler import OneCycleLR


def main(args):
    # Set environment variables
    set_seed_globally(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num
    model_name = set_name_experiment(args)
    print('model name: ', model_name)
    if not os.path.exists(args.tfdir + '/' + model_name):
        os.makedirs(args.tfdir + '/' + model_name)

    writer = SummaryWriter(log_dir=args.tfdir + '/' + model_name, flush_secs=10)

    logging.info("Initializing Training Set")
    train_envs_path, train_envs_name = get_envs_path(args.dataset_name, "train", args.filter_envs)
    train_loaders = [data_loader(args, train_env_path, train_env_name) for train_env_path, train_env_name in
                     zip(train_envs_path, train_envs_name)]

    logging.info("Initializing Validation Set")
    val_envs_path, val_envs_name = get_envs_path(args.dataset_name, "val", args.filter_envs)
    val_loaders = [data_loader(args, val_env_path, val_env_name) for val_env_path, val_env_name in
                   zip(val_envs_path, val_envs_name)]

    logging.info("Initializing Validation O Set")
    valo_envs_path, valo_envs_name = get_envs_path(args.dataset_name, "test", '0.6')

    valo_loaders = [data_loader(args, valo_env_path, valo_env_name) for valo_env_path, valo_env_name in
                    zip(valo_envs_path, valo_envs_name)]

    # training routine length
    num_batches_train = min([len(train_loader) for train_loader in train_loaders])
    num_batches_val = min([len(val_loader) for val_loader in val_loaders])
    num_batches_valo = min([len(valo_loader) for valo_loader in valo_loaders])

    # get labels of envs and create dic linking env name and env label
    if args.dataset_name in ('eth', 'hotel', 'univ', 'zara1', 'zara2'):
        # assert (all_train_labels == all_valid_labels)
        train_labels = {name: train_envs_name.index(name) for name in train_envs_name}
        val_labels = {name: val_envs_name.index(name) for name in val_envs_name}
        valo_labels = {name: valo_envs_name.index(name) for name in valo_envs_name}

    elif 'synthetic' in args.dataset_name or args.dataset_name in ['synthetic', 'v2', 'v2full', 'v4']:
        all_train_labels = sorted([float(d.split('_')[7]) for d in train_envs_name])
        all_valid_labels = sorted([float(d.split('_')[7]) for d in val_envs_name])
        all_valid_labelso = sorted([float(d.split('_')[7]) for d in valo_envs_name])
        # assert (all_train_labels == all_valid_labels)
        train_labels = {name: all_train_labels.index(float(name.split('_')[7])) for name in train_envs_name}
        val_labels = {name: all_valid_labels.index(float(name.split('_')[7])) for name in val_envs_name}
        valo_labels = {name: all_valid_labelso.index(float(name.split('_')[7])) for name in valo_envs_name}
    else:
        raise ValueError('Unrecognized dataset name "%s"' % args.dataset_name)

    # bring different dataset all together for simplicity of the next functions
    train_dataset = {'loaders': train_loaders, 'names': train_envs_name, 'labels': train_labels,
                     'num_batches': num_batches_train}
    valid_dataset = {'loaders': val_loaders, 'names': val_envs_name, 'labels': val_labels,
                     'num_batches': num_batches_val}
    valido_dataset = {'loaders': valo_loaders, 'names': valo_envs_name, 'labels': valo_labels,
                      'num_batches': num_batches_valo}

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
    model = VCRL(args).cuda()

    # optimizers
    optimizers = {
        'var': torch.optim.Adam(
            [
                {"params": model.pi_priore, 'lr': args.lrvar},
                {"params": model.variant_encoder.parameters(), 'lr': args.lrvar},
                {"params": model.x_to_s.parameters(), 'lr': args.lrvar},
            ]
        ),
        'inv': torch.optim.Adam(
            [
                {"params": model.x_to_z.parameters(), 'lr': args.lrinv},
                {"params": model.invariant_encoder.parameters(), 'lr': args.lrinv},
            ]
        ),
        'future_decoder': torch.optim.Adam(
            model.future_decoder.parameters(),
            lr=args.lrfut,
        ),
        'past_decoder': torch.optim.Adam(
            model.past_decoder.parameters(),
            lr=args.lrpast,
        )
    }

    if args.coupling:
        optimizers["var"].add_param_group({"params": model.coupling_layers_s.parameters(), 'lr': args.lrvar})
        optimizers["inv"].add_param_group({"params": model.coupling_layers_z.parameters(), 'lr': args.lrinv})
    else:
        optimizers["var"].add_param_group({"params": model.mean_priors, 'lr': args.lrvar})
        optimizers["var"].add_param_group({"params": model.logvar_priors, 'lr': args.lrvar})
        optimizers["inv"].add_param_group({"params": model.mean_priorz, 'lr': args.lrinv})
        optimizers["inv"].add_param_group({"params": model.logvar_priorz, 'lr': args.lrinv})

    num_batches = 0
    for train_loader in train_loaders:
        num_batches += len(train_loader)

    total_steps = num_batches * args.num_epochs

    if args.lr_scheduler:
        lr_schedulers = {
            'var': OneCycleLR(optimizers['var'], max_lr=1e-3, div_factor=25.0, total_steps=int(total_steps),
                              pct_start=0.3),
            'inv': OneCycleLR(optimizers['inv'], max_lr=1e-3, div_factor=25.0, total_steps=int(total_steps),
                              pct_start=0.3),
            'future_decoder': OneCycleLR(optimizers['future_decoder'], max_lr=1e-3, div_factor=25.0,
                                         total_steps=int(total_steps), pct_start=0.3),
            'past_decoder': OneCycleLR(optimizers['past_decoder'], max_lr=1e-3, div_factor=25.0,
                                       total_steps=int(total_steps), pct_start=0.3),
            }
    else:
        lr_schedulers = {
            'var': None,
            'inv': None,
            'future_decoder': None,
            'past_decoder': None
            }

    if args.resume:
        load_all_model(args, model, optimizers, lr_schedulers, num_batches)
        model.cuda()

    min_metric = 1e10
    for epoch in range(args.start_epoch, args.num_epochs + 1):
        logging.info(f"\n===> EPOCH: {epoch}")

        if args.finetune == "all" and args.coupling:
            freeze(True, (model.invariant_encoder, model.x_to_z, model.coupling_layers_z))
            freeze(False, (model.variant_encoder, model.x_to_s, model.future_decoder, model.past_decoder, model.coupling_layers_s))
        elif args.finetune == "weights+s" and args.coupling:
            freeze(True, (model.invariant_encoder, model.x_to_z, model.coupling_layers_z, model.coupling_layers_s))
            freeze(False, (model.variant_encoder, model.x_to_s, model.future_decoder, model.past_decoder))

        train_all(args, model, optimizers, train_dataset, epoch, train_envs_name, writer, lr_schedulers, stage='training')

        with torch.no_grad():
            validate_ade(args, model, train_dataset, epoch, writer, stage='training')
            metric = validate_ade(args, model, valid_dataset, epoch, writer, stage='validation')
            validate_ade(args, model, valido_dataset, epoch, writer, stage='validation o')

        if args.finetune:
            if metric < min_metric:
                min_metric = metric
                save_all_model(args, model, model_name, optimizers, metric, epoch)
                print(f'\n{"_" * 150}\n')
        else:
            save_all_model(args, model, model_name, optimizers, metric, epoch)

    writer.close()


def train_all(args, model, optimizers, train_dataset, epoch, train_envs_name, writer, lr_schedulers, stage):
    """
    Train the entire model for an epoch

    Args:
        - model (CausalMotionModel): model to train
        - optimizers: inv and style optimizers to use
        - datasets: train dataset (and pretrain dataset if finetuning)
        - stage (str): either 'validation' or 'training': says on which dataset we calculate the loss (and only backprop on 'training')
    """
    model.train()
    logging.info(f"- Computing loss ({stage})")

    assert (stage == 'training')

    if args.batch_method == "het" or args.batch_method == "alt":
        train_iter = [iter(train_loader) for train_loader in train_dataset['loaders']]
        loss_meter = AverageMeter("Loss", ":.4f")
        e1_loss_meter = AverageMeter("ELBO Loss", ":.4f")
        e2_loss_meter = AverageMeter("ELBO Loss", ":.4f")
        e3_loss_meter = AverageMeter("ELBO Loss", ":.4f")
        e4_loss_meter = AverageMeter("ELBO Loss", ":.4f")
        p_loss_meter = AverageMeter("Prediction Loss", ":.4f")
        progress = ProgressMeter(train_dataset['num_batches'], [loss_meter], prefix="")
        for batch_idx in range(train_dataset['num_batches']):

            batch_loss = []
            ped_tot = torch.zeros(1).cuda()

            # COMPUTE LOSS ON EACH OF THE ENVIRONMENTS
            for train_idx, (env_iter, env_name) in enumerate(zip(train_iter, train_dataset['names'])):
                try:
                    batch = next(env_iter)
                except StopIteration:
                    raise RuntimeError()

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

                # reset gradients
                for opt in optimizers.values():
                    opt.zero_grad()
                ped_tot += fut_traj_rel.shape[1]

                if args.finetune:
                    l2_loss_rel = []
                    l2_loss_elbo1 = []
                    l2_loss_elbo2 = []

                    log_py, E1, E2, _ = model(batch)

                    if args.decoupled_loss:
                        for i in range(args.best_k):
                            log_qy = - l2_loss(log_py[:, i, :, :], fut_traj_rel, mode="raw")
                            l2_loss_rel.append(log_qy)

                        l2_loss_elbo1.append(E1)
                        l2_loss_elbo2.append(E2)

                    else:

                        log_qy = torch.log(torch.exp(log_py).mean(0))
                        l2_loss_rel.append(log_qy)
                        l2_loss_elbo1.append(E1 / torch.exp(log_qy))
                        l2_loss_elbo2.append(E2 / torch.exp(log_qy))

                    l2_loss_rel = torch.stack(l2_loss_rel, dim=1)
                    predict_loss = erm_loss(l2_loss_rel, seq_start_end)

                    l2_loss_elbo1 = torch.stack(l2_loss_elbo1, dim=1)
                    l2_loss_elbo2 = torch.stack(l2_loss_elbo2, dim=1)

                    elbo_loss1 = erm_loss(l2_loss_elbo1, seq_start_end)

                    if args.model_name == "lstm":
                        elbo_loss2 = erm_loss(l2_loss_elbo2, seq_start_end)
                    if args.model_name == "mlp":
                        elbo_loss2 = l2_loss_elbo2.mean()

                    batch_loss.append((- predict_loss) + (- elbo_loss1) + (- elbo_loss2))

                    e1_loss_meter.update(elbo_loss1.item(), fut_traj_rel.shape[1])
                    e2_loss_meter.update(elbo_loss2.item(), fut_traj_rel.shape[1])
                    p_loss_meter.update(predict_loss.item(), fut_traj_rel.shape[1])

                else:
                    l2_loss_rel = []
                    l2_loss_elbo1 = []
                    l2_loss_elbo2 = []
                    l2_loss_elbo3 = []

                    log_py, E1, E2, E3 = model(batch)

                    if args.decoupled_loss:
                        for i in range(args.best_k):
                            log_qy = - l2_loss(log_py[:, i, :, :], fut_traj_rel, mode="raw")
                            l2_loss_rel.append(log_qy)

                        l2_loss_elbo1.append(E1)
                        l2_loss_elbo2.append(E2)
                        l2_loss_elbo3.append(E3)

                    else:

                        log_qy = torch.log(torch.exp(log_py).mean(0))
                        l2_loss_rel.append(log_qy)
                        l2_loss_elbo1.append(E1 / torch.exp(log_qy))
                        l2_loss_elbo2.append(E2 / torch.exp(log_qy))
                        l2_loss_elbo3.append(E3 / torch.exp(log_qy))

                    l2_loss_rel = torch.stack(l2_loss_rel, dim=1)
                    predict_loss = erm_loss(l2_loss_rel, seq_start_end)

                    l2_loss_elbo1 = torch.stack(l2_loss_elbo1, dim=1)
                    l2_loss_elbo2 = torch.stack(l2_loss_elbo2, dim=1)
                    l2_loss_elbo3 = torch.stack(l2_loss_elbo3, dim=1)

                    elbo_loss1 = erm_loss(l2_loss_elbo1, seq_start_end)

                    if args.model_name == "lstm":
                        elbo_loss2 = erm_loss(l2_loss_elbo2, seq_start_end)
                    if args.model_name == "mlp":
                        elbo_loss2 = l2_loss_elbo2.mean()

                    elbo_loss3 = erm_loss(l2_loss_elbo3, seq_start_end)

                    batch_loss.append((- predict_loss) + (- elbo_loss1) + (- elbo_loss2) + (- elbo_loss3))

                    e1_loss_meter.update(elbo_loss1.item(), fut_traj_rel.shape[1])
                    e2_loss_meter.update(elbo_loss2.item(), fut_traj_rel.shape[1])
                    e3_loss_meter.update(elbo_loss3.item(), fut_traj_rel.shape[1])
                    p_loss_meter.update(predict_loss.item(), fut_traj_rel.shape[1])

            loss = torch.zeros(()).cuda()
            loss += torch.stack(batch_loss).sum()

            # backpropagate the loss
            loss.backward()

            # choose which optimizer to use depending on the training step
            if args.finetune:
                if lr_schedulers['future_decoder'] is not None:
                    lr_schedulers['future_decoder'].step()
                optimizers['future_decoder'].step()

                if lr_schedulers['past_decoder'] is not None:
                    lr_schedulers['past_decoder'].step()
                optimizers['past_decoder'].step()

                if lr_schedulers['var'] is not None:
                    lr_schedulers['var'].step()
                optimizers['var'].step()
            else:
                if lr_schedulers['inv'] is not None:
                    lr_schedulers['inv'].step()
                optimizers['inv'].step()

                if lr_schedulers['future_decoder'] is not None:
                    lr_schedulers['future_decoder'].step()
                optimizers['future_decoder'].step()

                if lr_schedulers['past_decoder'] is not None:
                    lr_schedulers['past_decoder'].step()
                optimizers['past_decoder'].step()

                if lr_schedulers['var'] is not None:
                    lr_schedulers['var'].step()
                optimizers['var'].step()

            loss_meter.update(loss.item(), ped_tot.item())
            progress.display(batch_idx + 1)

        writer.add_scalar(f"total_loss/{stage}", loss_meter.avg, epoch)
        writer.add_scalar(f"recon_loss/{stage}", e1_loss_meter.avg, epoch)
        writer.add_scalar(f"sreg/{stage}", e2_loss_meter.avg, epoch)
        writer.add_scalar(f"zreg/{stage}", e3_loss_meter.avg, epoch)
        writer.add_scalar(f"pred_loss/{stage}", p_loss_meter.avg, epoch)

    else:

        # Homogenous batches
        total_loss_meter = AverageMeter("Total Loss", ":.4f")
        e1_loss_meter = AverageMeter("ELBO Loss", ":.4f")
        e2_loss_meter = AverageMeter("ELBO Loss", ":.4f")
        e3_loss_meter = AverageMeter("ELBO Loss", ":.4f")
        p_loss_meter = AverageMeter("Prediction Loss", ":.4f")
        for train_idx, train_loader in enumerate(train_dataset['loaders']):
            loss_meter = AverageMeter("Loss", ":.4f")
            progress = ProgressMeter(len(train_loader), [loss_meter],
                                     prefix="Dataset: {:<20}".format(train_envs_name[train_idx]))
            for batch_idx, batch in enumerate(train_loader):
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
                        _,
                        obs_traj_rel,
                        fut_traj_rel,
                        seq_start_end,
                        _,
                        _
                    ) = batch
                else:
                    raise ValueError('Unrecognized dataset name "%s"' % args.dataset_name)

                # reset gradients
                for opt in optimizers.values():
                    opt.zero_grad()

                if args.finetune:
                    l2_loss_rel = []
                    l2_loss_elbo1 = []
                    l2_loss_elbo2 = []

                    log_py, E1, E2, _ = model(batch)
                    if args.decoupled_loss:
                        for i in range(args.best_k):
                            log_qy = - l2_loss(log_py[:, i, :, :], fut_traj_rel, mode="raw")
                            l2_loss_rel.append(log_qy)

                        l2_loss_elbo1.append(E1)
                        l2_loss_elbo2.append(E2)

                    else:
                        log_qy = torch.log(torch.exp(log_py).mean(0))
                        l2_loss_rel.append(log_qy)
                        l2_loss_elbo1.append(E1 / torch.exp(log_qy))
                        l2_loss_elbo2.append(E2 / torch.exp(log_qy))

                    l2_loss_rel = torch.stack(l2_loss_rel, dim=1)
                    l2_loss_elbo1 = torch.stack(l2_loss_elbo1, dim=1)
                    l2_loss_elbo2 = torch.stack(l2_loss_elbo2, dim=1)
                    predict_loss = erm_loss(l2_loss_rel, seq_start_end)
                    elbo_loss1 = erm_loss(l2_loss_elbo1, seq_start_end)
                    elbo_loss2 = erm_loss(l2_loss_elbo2, seq_start_end)

                    loss = (- predict_loss) + (- elbo_loss1) + (- elbo_loss2)

                    e1_loss_meter.update(elbo_loss1.item(), obs_traj.shape[1])
                    e2_loss_meter.update(elbo_loss2.item(), obs_traj.shape[1])
                    p_loss_meter.update(predict_loss.item(), obs_traj.shape[1])

                else:
                    l2_loss_rel = []
                    l2_loss_elbo1 = []
                    l2_loss_elbo2 = []
                    l2_loss_elbo3 = []

                    log_py, E1, E2, E3 = model(batch)
                    if args.decoupled_loss:
                        for i in range(args.best_k):
                            log_qy = - l2_loss(log_py[:, i, :, :], fut_traj_rel, mode="raw")
                            l2_loss_rel.append(log_qy)

                        l2_loss_elbo1.append(E1)
                        l2_loss_elbo2.append(E2)
                        l2_loss_elbo3.append(E3)

                    else:

                        log_qy = torch.log(torch.exp(log_py).mean(0))
                        l2_loss_rel.append(log_qy)
                        l2_loss_elbo1.append(E1 / torch.exp(log_qy))
                        l2_loss_elbo2.append(E2 / torch.exp(log_qy))
                        l2_loss_elbo3.append(E3 / torch.exp(log_qy))

                    l2_loss_rel = torch.stack(l2_loss_rel, dim=1)
                    l2_loss_elbo1 = torch.stack(l2_loss_elbo1, dim=1)
                    l2_loss_elbo2 = torch.stack(l2_loss_elbo2, dim=1)
                    l2_loss_elbo3 = torch.stack(l2_loss_elbo3, dim=1)
                    predict_loss = erm_loss(l2_loss_rel, seq_start_end)
                    elbo_loss1 = erm_loss(l2_loss_elbo1, seq_start_end)
                    elbo_loss2 = erm_loss(l2_loss_elbo2, seq_start_end)
                    elbo_loss3 = erm_loss(l2_loss_elbo3, seq_start_end)

                    loss = (- predict_loss) + (- elbo_loss1) + (- elbo_loss2) + (- elbo_loss3)

                    e1_loss_meter.update(elbo_loss1.item(), obs_traj.shape[1])
                    e2_loss_meter.update(elbo_loss2.item(), obs_traj.shape[1])
                    e3_loss_meter.update(elbo_loss3.item(), obs_traj.shape[1])
                    p_loss_meter.update(predict_loss.item(), obs_traj.shape[1])

                # backpropagate the loss
                loss.backward()

                # choose which optimizer to use depending on the training step
                if args.finetune:
                    if lr_schedulers['future_decoder'] is not None:
                        lr_schedulers['future_decoder'].step()
                    optimizers['future_decoder'].step()

                    if lr_schedulers['past_decoder'] is not None:
                        lr_schedulers['past_decoder'].step()
                    optimizers['past_decoder'].step()

                    if lr_schedulers['var'] is not None:
                        lr_schedulers['var'].step()
                    optimizers['var'].step()

                else:
                    if lr_schedulers['inv'] is not None:
                        lr_schedulers['inv'].step()
                    optimizers['inv'].step()

                    if lr_schedulers['future_decoder'] is not None:
                        lr_schedulers['future_decoder'].step()
                    optimizers['future_decoder'].step()

                    if lr_schedulers['past_decoder'] is not None:
                        lr_schedulers['past_decoder'].step()
                    optimizers['past_decoder'].step()

                    if lr_schedulers['var'] is not None:
                        lr_schedulers['var'].step()
                    optimizers['var'].step()

                total_loss_meter.update(loss.item(), obs_traj.shape[1])
                loss_meter.update(loss.item(), obs_traj.shape[1])
                progress.display(batch_idx + 1)

        writer.add_scalar(f"total_loss/{stage}", total_loss_meter.avg, epoch)
        writer.add_scalar(f"recon_loss/{stage}", e1_loss_meter.avg, epoch)
        writer.add_scalar(f"sreg/{stage}", e2_loss_meter.avg, epoch)
        writer.add_scalar(f"zreg/{stage}", e3_loss_meter.avg, epoch)
        writer.add_scalar(f"pred_loss/{stage}", p_loss_meter.avg, epoch)


def validate_ade(args, model, valid_dataset, epoch, writer, stage, write=True):
    """
    Evaluate the performances on the validation set

    Args:
        - stage (str): either 'validation' or 'training': says on which dataset the metrics are computed
    """
    model.eval()

    assert (stage in ['training', 'validation', 'validation o'])

    logging.info(f"- Computing ADE ({stage})")
    with torch.no_grad():
        ade = 0
        fde = 0
        total_traj = 0
        for val_idx, (loader, loader_name) in enumerate(zip(valid_dataset['loaders'], valid_dataset['names'])):
            ade_outer, fde_outer = [], []
            total_traj_i = 0
            for batch_idx, batch in enumerate(loader):
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

                ade_list, fde_list = [], []
                total_traj_i += fut_traj.size(1)

                for k in range(args.best_k):
                    pred_fut_traj_rel = model(batch)

                    # from relative path to absolute path
                    pred_fut_traj = relative_to_abs(pred_fut_traj_rel, obs_traj[-1, :, :2])

                    # compute ADE and FDE metrics
                    ade_, fde_ = cal_ade_fde(fut_traj[:, :, :2], pred_fut_traj)

                    ade_list.append(ade_)
                    fde_list.append(fde_)

                ade_sum_batch = evaluate_helper(ade_list, seq_start_end)
                fde_sum_batch = evaluate_helper(fde_list, seq_start_end)
                ade_outer.append(ade_sum_batch)
                fde_outer.append(fde_sum_batch)

            ade_sum = sum(ade_outer)
            fde_sum = sum(fde_outer)
            ade += ade_sum
            fde += fde_sum
            total_traj += total_traj_i

            logging.info(f'\t\t ADE on {loader_name:<25} dataset:\t {ade_sum / (total_traj_i * args.fut_len)}')

        ade = ade / (total_traj * args.fut_len)
        fde = fde / total_traj

    logging.info(f"Average {stage}:\tADE  {ade:.4f}\tFDE  {fde:.4f}")
    repoch = epoch
    if write:
        writer.add_scalar(f"ade/{stage}", ade, repoch)
        writer.add_scalar(f"fde/{stage}", fde, repoch)

    return ade


def cal_ade_fde(fut_traj, pred_fut_traj):
    """
    Compute the ADE and FDE
    """
    ade = displacement_error(pred_fut_traj, fut_traj, mode="raw")
    fde = final_displacement_error(pred_fut_traj[-1], fut_traj[-1], mode="raw")
    return ade, fde


if __name__ == "__main__":
    print('Using GPU: ' + str(torch.cuda.is_available()))
    input_args = get_training_parser().parse_args()
    print('Arguments for training: ', input_args)
    set_logger(os.path.join(input_args.log_dir, "train.log"))
    main(input_args)
