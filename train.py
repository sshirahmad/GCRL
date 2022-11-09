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

    writer = SummaryWriter(log_dir=args.tfdir + '/' + model_name, flush_secs=10)

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
    valo_envs_path, valo_envs_name = get_envs_path(args.dataset_name, "test",
                                                   args.filter_envs)  # +'-'+args.filter_envs_pretrain)
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

    # create the model
    model = CRMF(args).cuda()

    # style related optimizer
    optimizers = {
        'variational': torch.optim.Adam(
            [
                {"params": model.theta_to_s.parameters(), 'lr': args.lrvariation},
                {"params": model.thetax_to_s.parameters(), 'lr': args.lrvariation},
                {"params": model.theta, 'lr': args.lrvariation},
                {"params": model.sigma, 'lr': args.lrvariation},

            ]
        ),
        'var': torch.optim.Adam(
            model.variant_encoder.parameters(),
            lr=args.lrvar,
        ),
        'map': torch.optim.Adam(
            model.mapping.parameters(),
            lr=args.lrmap,
        ),
        'inv': torch.optim.Adam(
            [
                {"params": model.invariant_encoder.parameters(), 'lr': args.lrinv},
                {"params": [model.mean, model.logvar], 'lr': args.lrinv},
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

    if args.resume:
        load_all_model(args, model, optimizers)
        model.cuda()

    # TRAINING HAPPENS IN 4 STEPS:
    assert (len(args.num_epochs) == 5)
    # 1. Train the invariant encoder along with the future decoder to learn z
    # 2. Train everything except invariant encoder to learn the other variant latent variables
    training_steps = {f'P{i}': [sum(args.num_epochs[:i - 1]), sum(args.num_epochs[:i])] for i in range(1, 7)}
    print(training_steps)

    def get_training_step(epoch):
        for step, r in training_steps.items():
            if r[0] < epoch <= r[1]:
                return step

    # SOME TEST
    if args.testonly == 1:
        print('SIMPLY VALIDATE MODEL:')
        validate_ade(args, model, valid_dataset, 300, 'P3', writer, 'validation', write=False)
    elif args.testonly == 2:
        print('TEST TIME MODIF:')
        validate_ade(args, model, valid_dataset, 500, 'P4', writer, 'training', write=False)

    if args.testonly != 0:
        writer.close()
        return

    min_metric = 1e10
    metric = min_metric
    for epoch in range(args.start_epoch, sum(args.num_epochs) + 1):

        training_step = get_training_step(epoch)
        logging.info(f"\n===> EPOCH: {epoch} ({training_step})")

        if training_step in ["P1", "P2"]:
            freeze(True, (model.theta_to_s, model.thetax_to_s, model.past_decoder, model.future_decoder, model.mapping))
            freeze(False, (model.invariant_encoder, model.variant_encoder))

        elif training_step == 'P3':
            freeze(True, (model.variant_encoder, model.theta_to_s, model.thetax_to_s, model.mapping))
            freeze(False, (model.invariant_encoder, model.future_decoder, model.past_decoder))

        elif training_step == 'P4':
            freeze(True, (model.invariant_encoder, model.mapping))
            freeze(False, (
            model.variant_encoder, model.theta_to_s, model.thetax_to_s, model.past_decoder, model.future_decoder))

        elif training_step == 'P5':
            freeze(True, (
            model.invariant_encoder, model.variant_encoder, model.theta_to_s, model.thetax_to_s, model.past_decoder,
            model.future_decoder))
            freeze(False, (model.mapping,))

        if args.finetune:
            freeze(True, (
            model.invariant_encoder, model.variant_encoder, model.past_decoder, model.future_decoder, model.mapping))
            freeze(False, (model.theta_to_s, model.thetax_to_s))

        train_all(args, model, optimizers, train_dataset, epoch, training_step, train_envs_name, writer,
                  stage='training')

        if training_step not in ["P1", "P2"]:
            with torch.no_grad():
                if training_step == "P4":
                    validate_ade(args, model, valid_dataset, epoch, training_step, writer, stage='validation')
                    validate_ade(args, model, train_dataset, epoch, training_step, writer, stage='training')
                elif training_step == "P3":
                    metric = validate_ade(args, model, valido_dataset, epoch, training_step, writer, stage='validation o')
                    validate_ade(args, model, valid_dataset, epoch, training_step, writer, stage='validation')
                    validate_ade(args, model, train_dataset, epoch, training_step, writer, stage='training')

                if training_step == "P5":
                    train_all(args, model, optimizers, valid_dataset, epoch, training_step, val_envs_name, writer, stage='validation')

        if args.finetune:
            if metric < min_metric:
                min_metric = metric
                save_all_model(args, model, model_name, optimizers, metric, epoch, training_step)
                print(f'\n{"_" * 150}\n')
        else:
            save_all_model(args, model, model_name, optimizers, metric, epoch, training_step)

    writer.close()


def train_all(args, model, optimizers, train_dataset, epoch, training_step, train_envs_name, writer,
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

    if args.batch_method == "het" or args.batch_method == "alt":
        train_iter = [iter(train_loader) for train_loader in train_dataset['loaders']]
        total_loss_meter = AverageMeter("Total Loss", ":.4f")
        progress = ProgressMeter(train_dataset['num_batches'], [total_loss_meter], prefix="")

        e_loss_meter = AverageMeter("ELBO Loss", ":.4f")
        p_loss_meter = AverageMeter("Prediction Loss", ":.4f")
        for batch_idx in range(train_dataset['num_batches']):

            # compute loss (which depends on the training step)
            batch_loss = []
            ped_tot = torch.zeros(1).cuda()
            # COMPUTE LOSS ON EACH OF THE ENVIRONMENTS
            for train_idx, (env_iter, env_name) in enumerate(zip(train_iter, train_dataset['names'])):
                try:
                    batch = next(env_iter)
                except StopIteration:
                    raise RuntimeError()

                # transfer batch
                batch = [tensor.cuda() for tensor in batch]
                (obs_traj, _, obs_traj_rel, fut_traj_rel, seq_start_end) = batch

                # reset gradients
                for opt in optimizers.values():
                    opt.zero_grad()

                ped_tot += fut_traj_rel.shape[1]
                dummy_w = torch.nn.Parameter(torch.Tensor([1.0])).cuda()

                if training_step in ["P1", "P2"]:
                    past_pred_rel_inv, past_pred_rel_var = model(batch, training_step)

                    # compute reconstruction loss between output and past
                    l2_loss_rel_inv = torch.stack([l2_loss(past_pred_rel_inv * dummy_w, obs_traj_rel, mode="raw")],
                                                  dim=1)
                    l2_loss_rel_var = torch.stack([l2_loss(past_pred_rel_var, obs_traj_rel, mode="raw")], dim=1)

                    # empirical risk (ERM classic loss)
                    loss_sum_even, loss_sum_odd = erm_loss(l2_loss_rel_inv, seq_start_end, obs_traj_rel.shape[0])
                    loss_inv = loss_sum_even + loss_sum_odd

                    # invariance constraint (IRM)
                    if args.irm and stage == 'training':
                        loss_inv += irm_loss(loss_sum_even, loss_sum_odd, dummy_w, args)

                    loss_sum_even, loss_sum_odd = erm_loss(l2_loss_rel_var, seq_start_end, obs_traj_rel.shape[0])
                    loss_var = loss_sum_even + loss_sum_odd

                    single_env_loss = loss_var + loss_inv

                elif training_step == "P3":
                    log_q_ygx, E = model(batch, training_step, dummy_w=dummy_w)

                    loss_sum_even_p, loss_sum_odd_p = erm_loss(log_q_ygx, seq_start_end, fut_traj_rel.shape[0])
                    predict_loss = loss_sum_even_p + loss_sum_odd_p

                    loss_sum_even_e, loss_sum_odd_e = erm_loss(torch.divide(E, torch.exp(log_q_ygx)), seq_start_end,
                                                               fut_traj_rel.shape[0])
                    elbo_loss = loss_sum_even_e + loss_sum_odd_e

                    single_env_loss = (- predict_loss) + (- elbo_loss)

                    # invariance constraint (IRM)
                    if args.irm and stage == 'training':
                        single_env_loss += irm_loss(loss_sum_even_e + loss_sum_even_p, loss_sum_odd_e + loss_sum_odd_p,
                                                    dummy_w,
                                                    args)

                elif training_step == "P4":
                    log_q_ygthetax, E = model(batch, training_step, env_idx=train_idx)

                    loss_sum_even_p, loss_sum_odd_p = erm_loss(log_q_ygthetax, seq_start_end, fut_traj_rel.shape[0])
                    predict_loss = loss_sum_even_p + loss_sum_odd_p

                    loss_sum_even_e, loss_sum_odd_e = erm_loss(torch.divide(E, torch.exp(log_q_ygthetax)),
                                                               seq_start_end, fut_traj_rel.shape[0])
                    elbo_loss = loss_sum_even_e + loss_sum_odd_e

                    stacked_loss = torch.cat((-predict_loss, -elbo_loss))
                    single_env_loss = torch.sum(stacked_loss)

                else:
                    pred_theta = model(batch, training_step)

                    l2_loss_pred = l2_loss(pred_theta, model.theta[train_idx], mode="sum") / pred_theta.shape[0]

                    single_env_loss = l2_loss_pred

                batch_loss.append(single_env_loss)

            # COMPUTE THE TOTAL LOSS ON ALL ENVIRONMENTS
            loss = torch.zeros(()).cuda()
            batch_loss = torch.stack(batch_loss)
            loss += batch_loss.sum()

            # backpropagate if needed
            if stage == 'training' and update:
                loss.backward(retain_graph=True)

                # choose which optimizer to use depending on the training step
                if training_step in ['P1', 'P2', 'P3']: optimizers['inv'].step()
                if training_step in ['P3', 'P4']: optimizers['future_decoder'].step()
                if training_step in ['P3', 'P4']: optimizers['past_decoder'].step()
                if training_step in ['P1', 'P2', 'P4']: optimizers['var'].step()
                if training_step in ['P4']: optimizers['variational'].step()
                if training_step in ['P5']: optimizers['map'].step()

            if training_step == "P3":
                e_loss_meter.update(elbo_loss.item(), ped_tot.item())
                p_loss_meter.update(predict_loss.item(), ped_tot.item())
            elif training_step == "P4":
                e_loss_meter.update(elbo_loss.item(), ped_tot.item())
                p_loss_meter.update(predict_loss.item(), ped_tot.item())

            total_loss_meter.update(loss.item(), ped_tot.item())
            progress.display(batch_idx + 1)

        if training_step == "P1":
            writer.add_scalar(f"STGAT_loss_p1/{stage}", total_loss_meter.avg, epoch)
        elif training_step == "P2":
            writer.add_scalar(f"STGAT_loss_p2/{stage}", total_loss_meter.avg, epoch)
        elif training_step == "P3":
            writer.add_scalar(f"pred_loss/{stage}", p_loss_meter.avg, epoch)
            writer.add_scalar(f"elbo_loss/{stage}", e_loss_meter.avg, epoch)
            writer.add_scalar(f"irm_loss/{stage}", total_loss_meter.avg, epoch)
            writer.add_scalar(f"mean/{stage}", torch.norm(model.mean), epoch)
            writer.add_scalar(f"log_var/{stage}", torch.norm(model.logvar), epoch)
        elif training_step == "P4":
            writer.add_scalar(f"variational_loss/{stage}", total_loss_meter.avg, epoch)
            writer.add_scalar(f"elbo_loss/{stage}", e_loss_meter.avg, epoch)
            writer.add_scalar(f"pred_loss/{stage}", p_loss_meter.avg, epoch)
            writer.add_scalar(f"theta_hotel/{stage}", torch.norm(model.theta[0]), epoch)
            writer.add_scalar(f"theta_univ/{stage}", torch.norm(model.theta[1]), epoch)
            writer.add_scalar(f"theta_zara1/{stage}", torch.norm(model.theta[2]), epoch)
            writer.add_scalar(f"theta_zara2/{stage}", torch.norm(model.theta[3]), epoch)
            writer.add_scalar(f"sigma_pred/{stage}", model.sigma[0], epoch)
            writer.add_scalar(f"sigma_reconstruct/{stage}", model.sigma[1], epoch)
        else:
            writer.add_scalar(f"theta_loss/{stage}", total_loss_meter.avg, epoch)

    # Homogenous batches
    else:
        total_loss_meter = AverageMeter("Total Loss", ":.4f")
        e_loss_meter = AverageMeter("ELBO Loss", ":.4f")
        p_loss_meter = AverageMeter("Prediction Loss", ":.4f")
        for train_idx, train_loader in enumerate(train_dataset['loaders']):
            loss_meter = AverageMeter("Loss", ":.4f")
            progress = ProgressMeter(len(train_loader), [loss_meter],
                                     prefix="Dataset: {:<20}".format(train_envs_name[train_idx]))
            for batch_idx, batch in enumerate(train_loader):
                batch = [tensor.cuda() for tensor in batch]
                (
                    obs_traj,
                    fut_traj,
                    obs_traj_rel,
                    fut_traj_rel,
                    seq_start_end,
                ) = batch

                # reset gradients
                for opt in optimizers.values():
                    opt.zero_grad()

                dummy_w = torch.nn.Parameter(torch.Tensor([1.0])).cuda()

                if training_step in ["P1", "P2"]:
                    past_pred_rel_inv, past_pred_rel_var = model(batch, training_step)

                    # compute reconstruction loss between output and past
                    l2_loss_rel_inv = torch.stack([l2_loss(past_pred_rel_inv * dummy_w, obs_traj_rel, mode="raw")],
                                                  dim=1)
                    l2_loss_rel_var = torch.stack([l2_loss(past_pred_rel_var, obs_traj_rel, mode="raw")], dim=1)

                    # empirical risk (ERM classic loss)
                    loss_sum_even, loss_sum_odd = erm_loss(l2_loss_rel_inv, seq_start_end, obs_traj_rel.shape[0])
                    loss_inv = loss_sum_even + loss_sum_odd

                    # invariance constraint (IRM)
                    if args.irm and stage == 'training':
                        loss_inv += irm_loss(loss_sum_even, loss_sum_odd, dummy_w, args)

                    loss_sum_even, loss_sum_odd = erm_loss(l2_loss_rel_var, seq_start_end, obs_traj_rel.shape[0])
                    loss_var = loss_sum_even + loss_sum_odd

                    loss = loss_var + loss_inv

                elif training_step == "P3":
                    q_ygx, E = model(batch, training_step, dummy_w=dummy_w)

                    loss_sum_even_p, loss_sum_odd_p = erm_loss(torch.log(q_ygx), seq_start_end, fut_traj_rel.shape[0])
                    predict_loss = loss_sum_even_p + loss_sum_odd_p

                    loss_sum_even_e, loss_sum_odd_e = erm_loss(torch.divide(E, q_ygx), seq_start_end,
                                                               fut_traj_rel.shape[0])
                    elbo_loss = loss_sum_even_e + loss_sum_odd_e

                    loss = (- predict_loss) + (- elbo_loss)

                    # invariance constraint (IRM)
                    if args.irm and stage == 'training':
                        loss += irm_loss(loss_sum_even_e + loss_sum_even_p, loss_sum_odd_e + loss_sum_odd_p, dummy_w,
                                         args)

                    e_loss_meter.update(elbo_loss.item(), obs_traj.shape[1])
                    p_loss_meter.update(predict_loss.item(), obs_traj.shape[1])

                elif training_step == "P4":
                    log_q_ygthetax, E = model(batch, training_step, env_idx=train_idx)

                    loss_sum_even_p, loss_sum_odd_p = erm_loss(log_q_ygthetax, seq_start_end, fut_traj_rel.shape[0])

                    predict_loss = loss_sum_even_p + loss_sum_odd_p

                    loss_sum_even_e, loss_sum_odd_e = erm_loss(torch.divide(E, torch.exp(log_q_ygthetax)),
                                                               seq_start_end, fut_traj_rel.shape[0])

                    elbo_loss = loss_sum_even_e + loss_sum_odd_e

                    stacked_loss = torch.cat((-predict_loss, -elbo_loss))

                    loss = torch.sum(stacked_loss)

                    e_loss_meter.update(elbo_loss.item(), obs_traj.shape[1])
                    p_loss_meter.update(predict_loss.item(), obs_traj.shape[1])

                else:
                    pred_theta = model(batch, training_step)

                    l2_loss_pred = l2_loss(pred_theta, model.theta[train_idx], mode="sum") / pred_theta.shape[0]

                    loss = l2_loss_pred

                # backpropagate if needed
                if stage == 'training' and update:
                    loss.backward(retain_graph=True)

                    # choose which optimizer to use depending on the training step
                    if training_step in ['P1', 'P2', 'P3']: optimizers['inv'].step()
                    if training_step in ['P3', 'P4']: optimizers['future_decoder'].step()
                    if training_step in ['P3', 'P4']: optimizers['past_decoder'].step()
                    if training_step in ['P1', 'P2', 'P4']: optimizers['var'].step()
                    if training_step in ['P4']: optimizers['variational'].step()
                    if training_step in ['P5']: optimizers['map'].step()

                total_loss_meter.update(loss.item(), obs_traj.shape[1])
                loss_meter.update(loss.item(), obs_traj.shape[1])
                progress.display(batch_idx + 1)

        if training_step in "P1":
            writer.add_scalar(f"STGAT_loss_p1/{stage}", total_loss_meter.avg, epoch)
        elif training_step in "P2":
            writer.add_scalar(f"STGAT_loss_p2/{stage}", total_loss_meter.avg, epoch)
        elif training_step == "P3":
            writer.add_scalar(f"pred_loss/{stage}", p_loss_meter.avg, epoch)
            writer.add_scalar(f"elbo_loss/{stage}", e_loss_meter.avg, epoch)
            writer.add_scalar(f"irm_loss/{stage}", total_loss_meter.avg, epoch)
            writer.add_scalar(f"mean/{stage}", torch.norm(model.mean), epoch)
            writer.add_scalar(f"log_var/{stage}", torch.norm(model.logvar), epoch)
        elif training_step == "P4":
            writer.add_scalar(f"variational_loss/{stage}", total_loss_meter.avg, epoch)
            writer.add_scalar(f"elbo_loss/{stage}", e_loss_meter.avg, epoch)
            writer.add_scalar(f"pred_loss/{stage}", p_loss_meter.avg, epoch)
            writer.add_scalar(f"theta_hotel/{stage}", torch.norm(model.theta[0]), epoch)
            writer.add_scalar(f"theta_univ/{stage}", torch.norm(model.theta[1]), epoch)
            writer.add_scalar(f"theta_zara1/{stage}", torch.norm(model.theta[2]), epoch)
            writer.add_scalar(f"theta_zara2/{stage}", torch.norm(model.theta[3]), epoch)
            writer.add_scalar(f"sigma_pred/{stage}", model.sigma[0], epoch)
            writer.add_scalar(f"sigma_reconstruct/{stage}", model.sigma[1], epoch)
        else:
            writer.add_scalar(f"theta_loss/{stage}", total_loss_meter.avg, epoch)


def validate_ade(args, model, valid_dataset, epoch, training_step, writer, stage, write=True):
    """
    Evaluate the performances on the validation set

    Args:
        - stage (str): either 'validation' or 'training': says on which dataset the metrics are computed
    """
    model.eval()

    assert (stage in ['training', 'validation', 'validation o'])
    ade_tot_meter, fde_tot_meter = AverageMeter("ADE", ":.4f"), AverageMeter("FDE", ":.4f")

    logging.info(f"- Computing ADE ({stage})")
    with torch.no_grad():
        for val_idx, (loader, loader_name) in enumerate(zip(valid_dataset['loaders'], valid_dataset['names'])):
            ade_meter, fde_meter = AverageMeter("ADE", ":.4f"), AverageMeter("FDE", ":.4f")

            for batch_idx, batch in enumerate(loader):
                batch = [tensor.cuda() for tensor in batch]
                (obs_traj, fut_traj, _, _, _) = batch

                if training_step == "P3":
                    pred_fut_traj_rel = model(batch, training_step)
                else:
                    pred_fut_traj_rel = model(batch, training_step, env_idx=val_idx)

                # from relative path to absolute path
                pred_fut_traj = relative_to_abs(pred_fut_traj_rel, obs_traj[-1, :, :2])

                # compute ADE and FDE metrics
                ade_, fde_ = cal_ade_fde(fut_traj, pred_fut_traj)

                ade_, fde_ = ade_ / (obs_traj.shape[1] * fut_traj.shape[0]), fde_ / (obs_traj.shape[1])
                ade_meter.update(ade_, obs_traj.shape[1])
                fde_meter.update(fde_, obs_traj.shape[1])
                ade_tot_meter.update(ade_, obs_traj.shape[1])
                fde_tot_meter.update(fde_, obs_traj.shape[1])

            logging.info(f'\t\t ADE on {loader_name:<25} dataset:\t {ade_meter.avg}')

    logging.info(f"Average {stage}:\tADE  {ade_tot_meter.avg:.4f}\tFDE  {fde_tot_meter.avg:.4f}")
    repoch = epoch
    if write:
        writer.add_scalar(f"ade/{stage}", ade_tot_meter.avg, repoch)
        writer.add_scalar(f"fde/{stage}", fde_tot_meter.avg, repoch)

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

    return ade_tot_meter.avg


def compute_ade_(pred_fut_traj_rel, obs_traj, fut_traj):
    pred_fut_traj = relative_to_abs(pred_fut_traj_rel, obs_traj[-1, :, :2])
    ade_, fde_ = cal_ade_fde(fut_traj, pred_fut_traj)
    ade_ = ade_ / (fut_traj.shape[0] * obs_traj.shape[1])
    return ade_


def compute_ade_single(pred_fut_traj_rel, obs_traj, fut_traj, wto):
    return compute_ade_(pred_fut_traj_rel[:, wto * NUMBER_PERSONS:NUMBER_PERSONS * (wto + 1)],
                        obs_traj[:, wto * NUMBER_PERSONS:NUMBER_PERSONS * (wto + 1)],
                        fut_traj[:, wto * NUMBER_PERSONS:NUMBER_PERSONS * (wto + 1)])


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
