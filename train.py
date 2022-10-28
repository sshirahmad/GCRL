import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from loader import data_loader
from parser_file import get_training_parser
from utils import *
from models import CRMF
from losses import erm_loss, irm_loss
from visualize import draw_image, draw_solo, draw_solo_all


def main(args):
    # Set environment variables
    set_seed_globally(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num
    model_name = get_model_name(args, time=False)
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

    for dataset, ds_name in zip((train_dataset, valid_dataset, valido_dataset), ('Train', 'Validation', 'Validation O')):
        print(ds_name + ' dataset: ', dataset)

    args.n_units = (
            [args.traj_lstm_hidden_size]
            + [int(x) for x in args.hidden_units.strip().split(",")]
            + [args.graph_lstm_hidden_size]
    )
    args.n_heads = [int(x) for x in args.heads.strip().split(",")]

    # create the model
    model = CRMF(args).cuda()
    sigma_elbo = torch.nn.Parameter(torch.tensor([65.0], device="cuda"))
    sigma_pred = torch.nn.Parameter(torch.tensor([0.0], device="cuda"))

    # style related optimizer
    optimizers = {
        'variant': torch.optim.Adam(
            [
                {"params": model.variant_encoder.parameters(), "lr": args.lrvar},
                {"params": model.variational_mapping.parameters(), 'lr': args.lrvar},
                {"params": model.theta_to_c.parameters(), 'lr': args.lrvar},
                {"params": model.theta_to_u.parameters(), 'lr': args.lrvar},
                {"params": [sigma_elbo, sigma_pred], 'lr': args.lrvar}
            ]
        ),
        'inv': torch.optim.Adam(
            model.invariant_encoder.parameters(),
            lr=args.lrinv,
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
        sigma_pred, sigma_elbo = load_all_model(args, model, optimizers)
        model.cuda()

    # TRAINING HAPPENS IN 6 STEPS:
    assert (len(args.num_epochs) == 2)
    # 1. Train the invariant encoder along with the future decoder to learn z
    # 2. Train everything except invariant encoder to learn the other variant latent variables
    training_steps = {f'P{i}': [sum(args.num_epochs[:i - 1]), sum(args.num_epochs[:i])] for i in range(1, 3)}
    print(training_steps)

    def get_training_step(epoch):
        if epoch <= 0:
            return 'P1'
        for step, r in training_steps.items():
            if r[0] < epoch <= r[1]:
                return step
        return 'P3'

    # SOME TEST
    if args.testonly == 1:
        print('SIMPLY VALIDATE MODEL:')
        validate_ade(model, valid_dataset, 300, 'P3', writer, 'validation', write=False)
        validate_ade(model, valid_dataset, 300, 'P6', writer, 'validation', write=False)
    elif args.testonly == 2:
        print('DEPRECATED')
    elif args.testonly == 3:
        print('TEST TIME MODIF:')
        validate_ade(model, valid_dataset, 500, 'P6', writer, 'training', write=False)

    if args.testonly != 0:
        writer.close()
        return

    min_metric = 1e10
    metric = min_metric
    for epoch in range(args.start_epoch, sum(args.num_epochs) + 1):

        training_step = get_training_step(epoch)
        logging.info(f"\n===> EPOCH: {epoch} ({training_step})")

        if training_step == 'P1':
            freeze(True, (model.variant_encoder, model.variational_mapping, model.theta_to_c, model.theta_to_u, model.past_decoder))
            freeze(False, (model.invariant_encoder, model.future_decoder))
        elif training_step == 'P2':
            freeze(True, (model.invariant_encoder,))
            freeze(False, (model.variant_encoder, model.variational_mapping, model.theta_to_c, model.theta_to_u, model.past_decoder,
                           model.future_decoder))

        train_all(args, model, optimizers, train_dataset, epoch, training_step, train_envs_name,
                  writer, stage='training')

        with torch.no_grad():
            metric = validate_ade(model, valido_dataset, epoch, training_step, writer, stage='validation o')
            validate_ade(model, valid_dataset, epoch, training_step, writer, stage='validation')
            validate_ade(model, train_dataset, epoch, training_step, writer, stage='training')

            #### EVALUATE ALSO THE TRAINING ADE and the validation loss
            # if epoch % 2 == 0:
            #     train_all(args, model, optimizers, valid_dataset, epoch, training_step, val_envs_name,
            #               writer, sigma_pred, sigma_elbo, stage='validation')

        if args.finetune:
            if metric < min_metric:
                min_metric = metric
                save_all_model(args, model, optimizers, metric, epoch, training_step)
                print(f'\n{"_" * 150}\n')
        else:
            save_all_model(args, model, optimizers, metric, epoch, training_step)

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
        loss_meter = AverageMeter("Loss", ":.4f")
        progress = ProgressMeter(train_dataset['num_batches'], [loss_meter], prefix="")

        for _ in range(train_dataset['num_batches']):

            # compute loss (which depends on the training step)
            batch_loss = []
            ped_tot = torch.zeros(1).cuda()
            # COMPUTE LOSS ON EACH OF THE ENVIRONMENTS
            for env_iter, env_name in zip(train_iter, train_dataset['names']):
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
                scale = torch.tensor(1.).cuda().requires_grad_()

                if training_step == "P1":
                    fut_pred_rel = []
                    for _ in range(args.best_k):
                        fut_pred_rel += [model(batch, training_step)]

                    # compute variety loss between output and future
                    l2_loss_rel = torch.stack(
                        [l2_loss(fut_pred_rel[i] * scale, fut_traj_rel, mode="raw") for i in range(args.best_k)], dim=1)

                    # empirical risk (ERM classic loss)
                    loss_sum_even, loss_sum_odd = erm_loss(l2_loss_rel, seq_start_end, fut_traj_rel.shape[0])
                    single_env_loss = loss_sum_even + loss_sum_odd

                    # invariance constraint (IRM)
                    if args.irm and stage == 'training':
                        single_env_loss += irm_loss(loss_sum_even, loss_sum_odd, scale, args)

                batch_loss.append(single_env_loss)

            # COMPUTE THE TOTAL LOSS ON ALL ENVIRONMENTS
            loss = torch.zeros(()).cuda()

            # content loss
            if training_step in ['P1', 'P2']:
                batch_loss = torch.stack(batch_loss)
                loss += batch_loss.sum()

            # backpropagate if needed
            if stage == 'training' and update:
                loss.backward()

                # choose which optimizer to use depending on the training step
                if args.finetune and args.finetune != 'all':
                    if training_step in ['P1', 'P2', 'P3', ] and args.finetune == 'stgat_enc': optimizers['inv'].step()
                    if training_step in ['P3', 'P6'] and args.finetune == 'decoder': optimizers['decoder'].step()
                    if training_step in ['P4', 'P6'] and args.finetune in ['style', 'integ+']: optimizers[
                        'style'].step()
                    if training_step in ['P5', 'P6'] and args.finetune in ['integ', 'integ+']: optimizers[
                        'integ'].step()
                else:
                    if training_step in ['P1']: optimizers['inv'].step()
                    if training_step in ['P1', 'P2']: optimizers['future_decoder'].step()
                    if training_step in ['P2']: optimizers['past_decoder'].step()
                    if training_step in ['P2']: optimizers['style'].step()

            loss_meter.update(loss.item(), ped_tot.item())
        progress.display(train_dataset['num_batches'])
        writer.add_scalar(f"{'erm' if training_step != 'P4' else 'style'}_loss/{stage}", loss_meter.avg, epoch)

    # Homogenous batches
    else:
        total_loss_meter = AverageMeter("Total Loss", ":.4f")
        r_loss_meter = AverageMeter("Reconstruction Loss", ":.4f")
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

                scale = torch.tensor(1.).cuda().requires_grad_()

                if training_step == "P1":
                    fut_pred_rel = []
                    for _ in range(args.best_k):
                        fut_pred_rel += [model(batch, training_step)]

                    # compute variety loss between output and future
                    l2_loss_rel = torch.stack(
                        [l2_loss(fut_pred_rel[i] * scale, fut_traj_rel, mode="raw") for i in range(args.best_k)], dim=1)

                    # empirical risk (ERM classic loss)
                    loss_sum_even, loss_sum_odd = erm_loss(l2_loss_rel, seq_start_end, fut_traj_rel.shape[0])
                    loss = loss_sum_even + loss_sum_odd

                    # invariance constraint (IRM)
                    if args.irm and stage == 'training':
                        loss += irm_loss(loss_sum_even, loss_sum_odd, scale, args)

                else:
                    q_ygx, A1, A2, pred_past_rel = model(batch, training_step)
                    l2_loss_reconst = l2_loss(pred_past_rel, obs_traj_rel, mode="raw")
                    loss_sum_even, loss_sum_odd = erm_loss(l2_loss_reconst, seq_start_end, obs_traj_rel.shape[0])
                    r_loss = loss_sum_even + loss_sum_odd

                    loss_sum_even, loss_sum_odd = erm_loss(torch.log(q_ygx + 1e-6), seq_start_end, fut_traj_rel.shape[0])
                    predict_loss = loss_sum_even + loss_sum_odd
                    loss_sum_even, loss_sum_odd = erm_loss(torch.multiply(A1, -l2_loss_reconst) + A2, seq_start_end, obs_traj_rel.shape[0])
                    elbo_loss = loss_sum_even + loss_sum_odd

                    loss = (- predict_loss) + 1e25 * (- elbo_loss)

                    r_loss_meter.update(r_loss.item(), obs_traj.shape[1])
                    e_loss_meter.update(elbo_loss.item(), obs_traj.shape[1])
                    p_loss_meter.update(predict_loss.item(), obs_traj.shape[1])

                # backpropagate if needed
                if stage == 'training' and update:
                    loss.backward()

                # choose which optimizer to use depending on the training step
                if training_step in ['P1']: optimizers['inv'].step()
                if training_step in ['P1', 'P2']: optimizers['future_decoder'].step()
                if training_step in ['P2']: optimizers['past_decoder'].step()
                if training_step in ['P2']: optimizers['variant'].step()

                total_loss_meter.update(loss.item(), obs_traj.shape[1])
                loss_meter.update(loss.item(), obs_traj.shape[1])
                progress.display(batch_idx + 1)

        if training_step == "P1":
            writer.add_scalar(f"irm_loss/{stage}", total_loss_meter.avg, epoch)
        elif training_step == "P2":
            writer.add_scalar(f"Total_loss/{stage}", total_loss_meter.avg, epoch)
            writer.add_scalar(f"Reconstruction_loss/{stage}", r_loss_meter.avg, epoch)
            writer.add_scalar(f"ELBO_loss/{stage}", e_loss_meter.avg, epoch)
            writer.add_scalar(f"Predict_loss/{stage}", p_loss_meter.avg, epoch)


def validate_ade(model, valid_dataset, epoch, training_step, writer, stage, write=True):
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
        for loader, loader_name in zip(valid_dataset['loaders'], valid_dataset['names']):
            ade_meter, fde_meter = AverageMeter("ADE", ":.4f"), AverageMeter("FDE", ":.4f")

            for batch_idx, batch in enumerate(loader):
                batch = [tensor.cuda() for tensor in batch]
                (obs_traj, fut_traj, _, _, _) = batch

                pred_fut_traj_rel = model(batch, training_step)

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


if __name__ == "__main__":
    print('Using GPU: ' + str(torch.cuda.is_available()))
    input_args = get_training_parser().parse_args()
    print('Arguments for training: ', input_args)
    set_logger(os.path.join(input_args.log_dir, "train.log"))
    main(input_args)
