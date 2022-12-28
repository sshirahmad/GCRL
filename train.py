import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from loader import data_loader
from parser_file import get_training_parser
from utils import *
from models import CRMF
from losses import erm_loss, irm_loss
import math
from torch.optim.lr_scheduler import OneCycleLR
from torch.nn.utils import clip_grad
from scipy.optimize import linear_sum_assignment as linear_assignment


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

    valo_loaders = [data_loader(args, valo_env_path, valo_env_name, test=True) for valo_env_path, valo_env_name in
                    zip(valo_envs_path, valo_envs_name)]
    finetune_loaders = [data_loader(args, valo_env_path, valo_env_name, finetune=True) for valo_env_path, valo_env_name
                        in zip(valo_envs_path, valo_envs_name)]

    # training routine length
    num_batches_train = min([len(train_loader) for train_loader in train_loaders])
    num_batches_val = min([len(val_loader) for val_loader in val_loaders])
    num_batches_valo = min([len(valo_loader) for valo_loader in valo_loaders])
    num_batches_finetune = min([len(finetune_loader) for finetune_loader in finetune_loaders])

    # bring different dataset all together for simplicity of the next functions
    train_dataset = {'loaders': train_loaders, 'names': train_envs_name, 'num_batches': num_batches_train}
    valid_dataset = {'loaders': val_loaders, 'names': val_envs_name, 'num_batches': num_batches_val}
    valido_dataset = {'loaders': valo_loaders, 'names': valo_envs_name, 'num_batches': num_batches_valo}
    finetune_dataset = {'loaders': finetune_loaders, 'names': valo_envs_name, 'num_batches': num_batches_finetune}

    for dataset, ds_name in zip((train_dataset, valid_dataset, valido_dataset, finetune_dataset),
                                ('Train', 'Validation', 'Validation O', 'Finetune')):
        print(ds_name + ' dataset: ', dataset)

    args.n_units = (
            [args.traj_lstm_hidden_size]
            + [int(x) for x in args.hidden_units.strip().split(",")]
            + [args.graph_lstm_hidden_size]
    )
    args.n_heads = [int(x) for x in args.heads.strip().split(",")]

    # create the model
    model = CRMF(args).cuda()

    # style related optimizer
    optimizers = {
        'par': torch.optim.Adam(
            [
                {"params": model.pi_priore, 'lr': args.lrpar},
                {"params": model.coupling_layers_s.parameters(), 'lr': args.lrpar},
                # {"params": model.logvar_priors, 'lr': args.lrpar},
                # {"params": model.mean_priors, 'lr': args.lrpar},
            ]
        ),
        'var': torch.optim.Adam(
            [
                {"params": model.variant_encoder.parameters(), 'lr': args.lrvar},
                {"params": model.x_to_s.parameters(), 'lr': args.lrvar},
            ]
        ),
        'inv': torch.optim.Adam(
            [
                {"params": model.coupling_layers_z.parameters(), 'lr': args.lrinv},
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

    num_batches = 0
    for train_loader in train_loaders:
        num_batches += len(train_loader)

    training_steps = np.array([sum(args.num_epochs[:i]) - sum(args.num_epochs[:i - 1]) for i in range(1, 8)])
    total_steps = num_batches * training_steps
    # lr_schedulers = {
    #     "P1": {
    #         'var': OneCycleLR(optimizers['var'], max_lr=1e-3, div_factor=25.0, total_steps=int(total_steps[0]),
    #                           pct_start=0.3),
    #         'inv': OneCycleLR(optimizers['inv'], max_lr=1e-3, div_factor=25.0, total_steps=int(total_steps[0]),
    #                           pct_start=0.3),
    #         'past_decoder': OneCycleLR(optimizers['past_decoder'], max_lr=1e-3, div_factor=25.0,
    #                                    total_steps=int(total_steps[0]), pct_start=0.3),
    #         'future_decoder': OneCycleLR(optimizers['future_decoder'], max_lr=1e-3, div_factor=25.0,
    #                                      total_steps=int(total_steps[0]), pct_start=0.3),
    #         'par': OneCycleLR(optimizers['par'], max_lr=1e-3, div_factor=25.0, total_steps=int(total_steps[0]),
    #                           pct_start=0.3),
    #     },
    #     "P2": {
    #         'var': OneCycleLR(optimizers['var'], max_lr=1e-3, div_factor=25.0, total_steps=int(total_steps[1]),
    #                           pct_start=0.3),
    #         'inv': OneCycleLR(optimizers['inv'], max_lr=1e-3, div_factor=25.0, total_steps=int(total_steps[1]),
    #                           pct_start=0.3),
    #         'past_decoder': OneCycleLR(optimizers['past_decoder'], max_lr=1e-3, div_factor=25.0,
    #                                    total_steps=int(total_steps[1]), pct_start=0.3),
    #         'par': OneCycleLR(optimizers['par'], max_lr=1e-3, div_factor=25.0, total_steps=int(total_steps[1]),
    #                           pct_start=0.3),
    #     },
    #     "P3": {
    #         'var': OneCycleLR(optimizers['var'], max_lr=1e-4, div_factor=25.0, total_steps=int(total_steps[2]),
    #                           pct_start=0.3),
    #         'inv': OneCycleLR(optimizers['inv'], max_lr=1e-4, div_factor=25.0, total_steps=int(total_steps[2]),
    #                           pct_start=0.3),
    #         'future_decoder': OneCycleLR(optimizers['future_decoder'], max_lr=1e-4, div_factor=25.0,
    #                                      total_steps=int(total_steps[2]), pct_start=0.3),
    #         'past_decoder': OneCycleLR(optimizers['past_decoder'], max_lr=1e-4, div_factor=25.0,
    #                                    total_steps=int(total_steps[2]), pct_start=0.3),
    #         'par': OneCycleLR(optimizers['par'], max_lr=1e-4, div_factor=25.0, total_steps=int(total_steps[2]),
    #                           pct_start=0.3),
    #     },
    # }

    lr_schedulers = {
        "P1": None,
        "P2": None,
        "P3": None,
        "P4": None,
        "P5": None,
        "P6": None
    }

    # TRAINING HAPPENS IN 4 STEPS:
    assert (len(args.num_epochs) == 6)
    # 1. Train the invariant encoder along with the future decoder to learn z
    # 2. Train everything except invariant encoder to learn the other variant latent variables
    training_steps = {f'P{i}': [sum(args.num_epochs[:i - 1]), sum(args.num_epochs[:i])] for i in range(1, 7)}
    print(training_steps)

    if args.resume:
        load_all_model(args, model, optimizers, lr_schedulers, training_steps, num_batches)
        model.cuda()

    def get_training_step(epoch):
        for step, r in training_steps.items():
            if r[0] < epoch <= r[1]:
                return step

    min_metric = 1e10
    metric = min_metric
    beta_scheduler = get_beta(training_steps["P3"][0], 100, 1e-6)
    for epoch in range(args.start_epoch, sum(args.num_epochs) + 1):

        training_step = get_training_step(epoch)
        if training_step in ["P1", "P2", "P3", "P4", "P5"]:
            continue
        logging.info(f"\n===> EPOCH: {epoch} ({training_step})")

        if training_step in ['P1', 'P2']:
            freeze(True, (model.x_to_z, model.x_to_s, model.past_decoder, model.future_decoder, model.coupling_layers_z))
            freeze(False, (model.variant_encoder, model.invariant_encoder))

        if training_step == 'P3':
            freeze(True, (model.future_decoder, model.coupling_layers_z))
            freeze(False, (model.variant_encoder, model.invariant_encoder, model.x_to_s, model.x_to_z, model.past_decoder))

        if training_step == 'P4':
            pass

        elif training_step == 'P5':
            freeze(True, (model.invariant_encoder, model.x_to_z, model.past_decoder, model.future_decoder, model.coupling_layers_z))
            freeze(False, (model.variant_encoder, model.x_to_s))

        # elif training_step == "P6":
        #     freeze(False, (model.invariant_encoder, model.variant_encoder, model.x_to_s, model.x_to_z, model.past_decoder, model.future_decoder, model.coupling_layers_z))

        if training_step in ["P1", "P2", "P3", "P5", "P6"]:
            train_all(args, model, optimizers, train_dataset, epoch, training_step, train_envs_name, writer,
                      beta_scheduler,
                      lr_schedulers,
                      stage='training')

        elif training_step == "P4":
            with torch.no_grad():
                train_all(args, model, optimizers, train_dataset, epoch, training_step, train_envs_name, writer,
                          beta_scheduler,
                          lr_schedulers,
                          stage='training')

        elif training_step == "P7":
            train_all(args, model, optimizers, finetune_dataset, epoch, training_step, valo_envs_name, writer,
                      beta_scheduler,
                      lr_schedulers,
                      stage='training')

        with torch.no_grad():
            if training_step == "P6":
                validate_ade(args, model, train_dataset, epoch, training_step, writer, stage='training')
                validate_ade(args, model, valid_dataset, epoch, training_step, writer, stage='validation')
                metric = validate_ade(args, model, valido_dataset, epoch, training_step, writer, stage='validation o')

            elif training_step == "P7":
                metric = validate_ade(args, model, valido_dataset, epoch, training_step, writer, stage='validation o')

        if training_step in ["P6", "P7"]:
            if metric < min_metric:
                min_metric = metric
                save_all_model(args, model, model_name, optimizers, metric, epoch, training_step)
                print(f'\n{"_" * 150}\n')
        else:
            save_all_model(args, model, model_name, optimizers, metric, epoch, training_step)

    writer.close()


def train_all(args, model, optimizers, train_dataset, epoch, training_step, train_envs_name, writer, beta_scheduler,
              lr_schedulers,
              stage,
              update=True):
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

    assert (stage in ['training', 'validation'])

    # Homogenous batches
    total_loss_meter = AverageMeter("Total Loss", ":.4f")
    e1_loss_meter = AverageMeter("ELBO Loss", ":.4f")
    e2_loss_meter = AverageMeter("ELBO Loss", ":.4f")
    e3_loss_meter = AverageMeter("ELBO Loss", ":.4f")
    p_loss_meter = AverageMeter("Prediction Loss", ":.4f")
    s_vec = []
    clusters = []
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

            if training_step in ["P1", "P2"]:
                l2_loss_rel1 = []
                l2_loss_rel2 = []
                pred_past_rel1, pred_past_rel2 = model(batch, training_step)

                l2_loss_rel1.append(l2_loss(pred_past_rel1, obs_traj_rel, mode="raw"))
                l2_loss_rel2.append(l2_loss(pred_past_rel2, obs_traj_rel, mode="raw"))
                l2_loss_rel1 = torch.stack(l2_loss_rel1, dim=1)
                l2_loss_rel2 = torch.stack(l2_loss_rel2, dim=1)
                loss = erm_loss(l2_loss_rel1, seq_start_end) + erm_loss(l2_loss_rel2, seq_start_end)

            elif training_step == "P3":
                l2_loss_rel1 = []
                l2_loss_rel2 = []
                pred_past_rel, pred_fut_rel = model(batch, training_step)

                l2_loss_rel1.append(l2_loss(pred_past_rel, obs_traj_rel, mode="raw"))
                l2_loss_rel2.append(l2_loss(pred_fut_rel, fut_traj_rel, mode="raw"))
                l2_loss_rel1 = torch.stack(l2_loss_rel1, dim=1)
                l2_loss_rel2 = torch.stack(l2_loss_rel2, dim=1)
                loss = erm_loss(l2_loss_rel1, seq_start_end) + erm_loss(l2_loss_rel2, seq_start_end)

            elif training_step == "P4":
                s = model(batch, training_step)
                s_vec += [s]
                clusters += [torch.tensor(train_idx).repeat(s.shape[0])]
                continue

            elif training_step == "P5":
                l2_loss_elbo = []
                E = model(batch, training_step)
                l2_loss_elbo.append(E)
                l2_loss_elbo = torch.stack(l2_loss_elbo, dim=1)
                elbo_loss = erm_loss(l2_loss_elbo, seq_start_end)
                loss = - elbo_loss

            else:
                l2_loss_rel = []
                l2_loss_elbo1 = []
                l2_loss_elbo2 = []
                l2_loss_elbo3 = []

                log_py, E1, E2, E3 = model(batch, training_step)

                # for i in range(args.num_samples):
                #     pred_loss = - l2_loss(pred_q_rel[i], fut_traj_rel, mode="raw")
                #     l2_loss_rel.append(pred_loss)

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

            # backpropagate if needed
            if stage == 'training' and update:
                loss.backward()

                lr_scheduler_optims = lr_schedulers[training_step]
                # choose which optimizer to use depending on the training step
                if training_step in ['P1', 'P2', 'P3', 'P6']:
                    if lr_scheduler_optims is not None:
                        lr_scheduler_optims['inv'].step()
                    optimizers['inv'].step()

                if training_step in ['P3', 'P6']:
                    if lr_scheduler_optims is not None:
                        lr_scheduler_optims['future_decoder'].step()
                    optimizers['future_decoder'].step()

                if training_step in ['P3', 'P6']:
                    if lr_scheduler_optims is not None:
                        lr_scheduler_optims['past_decoder'].step()
                    optimizers['past_decoder'].step()

                if training_step in ['P1', 'P2', 'P3', 'P5', 'P6']:
                    if lr_scheduler_optims is not None:
                        lr_scheduler_optims['var'].step()
                    optimizers['var'].step()

                if training_step in ['P6']:
                    if lr_scheduler_optims is not None:
                        lr_scheduler_optims['par'].step()
                    optimizers['par'].step()

            total_loss_meter.update(loss.item(), obs_traj.shape[1])
            loss_meter.update(loss.item(), obs_traj.shape[1])
            progress.display(batch_idx + 1)

    # train GMM
    if training_step == "P4":
        S = torch.cat(s_vec, 0)
        S = torch.nn.functional.normalize(S, dim=1).detach().cpu().numpy()
        Y = torch.cat(clusters, 0).numpy()
        pre = model.gmm.fit_predict(S)
        print('Acc={:.4f}%'.format(cluster_acc(pre, Y)[0] * 100))

    if training_step in ["P1", "P2"]:
        writer.add_scalar(f"STGAT_loss/{stage}", total_loss_meter.avg, epoch)

    elif training_step == "P3":
        writer.add_scalar(f"reconstruction_loss/{stage}", total_loss_meter.avg, epoch)

    elif training_step in ["P5"]:
        writer.add_scalar(f"variational_loss/{stage}", total_loss_meter.avg, epoch)

    elif training_step in ["P6"]:
        writer.add_scalar(f"variational_loss/{stage}", total_loss_meter.avg, epoch)
        writer.add_scalar(f"reconstruction_loss/{stage}", e1_loss_meter.avg, epoch)
        writer.add_scalar(f"sreg/{stage}", e2_loss_meter.avg, epoch)
        writer.add_scalar(f"zreg/{stage}", e3_loss_meter.avg, epoch)
        writer.add_scalar(f"pred_loss/{stage}", p_loss_meter.avg, epoch)


def validate_ade(args, model, valid_dataset, epoch, training_step, writer, stage, write=True):
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
                    if stage == "validation o":
                        pred_fut_traj_rel = model(batch, training_step)
                    else:
                        pred_fut_traj_rel = model(batch, training_step, env_idx=val_idx)

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

    ## SAVE VISUALIZATIONS
    # if epoch % 1 == 0 and stage == 'validation':
    # if (stage == 'validation' and rp != None and epoch % 3 == 0) or force and write:

    #     obs = [b[0] for b in rp]
    #     fut = [b[1] for b in rp]
    #     pred = [relative_to_abs(model(b, ts), b[0][-1, :, :2]) for b in rp]
    #     res = [[obs[i], fut[i], pred[i]] for i in range(len(rp))]
    #     fig, array = draw_image(res)
    #     fig.savefig(f'images/visu/pred{epoch}.png')
    #     writer.add_image("Some paths", array, epoch)

    return ade


def cal_ade_fde(fut_traj, pred_fut_traj):
    """
    Compute the ADE and FDE
    """
    ade = displacement_error(pred_fut_traj, fut_traj, mode="raw")
    fde = final_displacement_error(pred_fut_traj[-1], fut_traj[-1], mode="raw")
    return ade, fde


def cluster_acc(Y_pred, Y):
    assert Y_pred.size == Y.size

    D = max(Y_pred.max(), Y.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(Y_pred.size):
        w[Y_pred[i], Y[i]] += 1
    row_ind, col_ind = linear_assignment(w.max() - w)

    return sum(w[row_ind, col_ind]) * 1.0 / Y_pred.size, w


def plot_grad_flow(named_parameters):
    """
    Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow
    """
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n):
            if p.grad is not None:
                layers.append(n)
                ave_grads.append(p.grad.abs().mean())
                max_grads.append(p.grad.abs().max())

    print("done")
    # plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    # plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    # plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    # plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    # plt.xlim(left=0, right=len(ave_grads))
    # plt.ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
    # plt.xlabel("Layers")
    # plt.ylabel("average gradient")
    # plt.title("Gradient flow")
    # plt.grid(True)
    # plt.legend([Line2D([0], [0], color="c", lw=4),
    #             Line2D([0], [0], color="b", lw=4),
    #             Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    # plt.show()


if __name__ == "__main__":
    print('Using GPU: ' + str(torch.cuda.is_available()))
    input_args = get_training_parser().parse_args()
    print('Arguments for training: ', input_args)
    set_logger(os.path.join(input_args.log_dir, "train.log"))
    main(input_args)
