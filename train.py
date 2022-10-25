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

    ref_pictures = [[b.cuda() for b in next(iter(loader))] for loader in val_loaders]

    # If finetuning, we need to load the pretrain datasets for style contrastive loss
    pretrain_loaders = None
    if args.filter_envs_pretrain:
        # do we also reduce the pretrain loaders?? If yes,
        logging.info("Initializing Training Set for contrastive loss")
        pretrain_envs_path, pretrain_envs_name = get_envs_path(args.dataset_name, "train", args.filter_envs_pretrain)
        pretrain_loaders = [data_loader(args, train_env_path, train_env_name, pt=True) for
                            train_env_path, train_env_name in
                            zip(pretrain_envs_path, pretrain_envs_name)]
        print(pretrain_envs_name)

    # training routine length
    num_batches_train = min([len(train_loader) for train_loader in train_loaders])
    if args.filter_envs_pretrain:
        num_batches_pretrain = min([len(train_loader) for train_loader in pretrain_loaders])
    num_batches_val = min([len(val_loader) for val_loader in val_loaders])

    # bring different dataset all together for simplicity of the next functions
    train_dataset = {'loaders': train_loaders, 'names': train_envs_name, 'num_batches': num_batches_train}
    valid_dataset = {'loaders': val_loaders, 'names': val_envs_name, 'num_batches': num_batches_val}
    if args.filter_envs_pretrain:
        pretrain_dataset = {'loaders': pretrain_loaders, 'names': pretrain_envs_name,
                            'num_batches': num_batches_pretrain}
    else:
        pretrain_dataset = None

    for dataset, ds_name in zip((train_dataset, valid_dataset, pretrain_dataset), ('Train', 'Validation', 'Pretrain')):
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
        'style': torch.optim.Adam(
            [
                {"params": model.variant_encoder.parameters(), "lr": args.lrvar},
                {"params": model.variational_mapping.parameters(), 'lr': args.lrvm},
                {"params": model.theta_to_c.parameters(), 'lr': args.lrcmap}
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
        load_all_model(args, model, optimizers)
        model.cuda()

    # TRAINING HAPPENS IN 6 STEPS:
    assert (len(args.num_epochs) == 2)
    # 1. (deprecated, was used for first step of stgat training)
    # 2. (deprecated, was used for second step of stgat training)
    # 3. initial training of the entire model, without any style input
    # 4. train style encoder using classifier, separate from pipeline
    # 5. train the integrator (that joins the style and the invariant features)
    # 6. fine-tune the integrator, decoder, style encoder with everything working
    training_steps = {f'P{i}': [sum(args.num_epochs[:i - 1]), sum(args.num_epochs[:i])] for i in range(1, 3)}
    print(training_steps)

    def get_training_step(epoch):
        if epoch <= 0:
            return 'P1'
        for step, r in training_steps.items():
            if r[0] < epoch <= r[1]:
                return step
        return 'P6'

    training_step = get_training_step(args.start_epoch)
    if args.finetune:
        with torch.no_grad():
            validate_ade(model, train_dataset, args.start_epoch - 1, 'P6', writer, stage='training', args=args)
            metric = validate_ade(model, valid_dataset, args.start_epoch - 1, 'P6', writer, stage='validation',
                                  args=args)
            min_metric = metric
            if args.reduce == 64:
                save_all_model(args, model, optimizers, metric, -1, 'P6')
                return
            print(f'\n{"_" * 150}\n')
            train_all(args, model, optimizers, train_dataset, pretrain_dataset, args.start_epoch - 1, 'P6', writer,
                      stage='training', update=False)
            train_all(args, model, optimizers, valid_dataset, pretrain_dataset, args.start_epoch - 1, 'P6', writer,
                      stage='validation')
    else:
        min_metric = 1e10

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
        if args.ttr > 0:
            train_latent_space(args, model, valid_dataset, pretrain_dataset, writer)

    if args.testonly != 0:
        writer.close()
        return

    min_metric = 1e10
    metric = min_metric
    for epoch in range(args.start_epoch, sum(args.num_epochs) + 1):

        training_step = get_training_step(epoch)
        logging.info(f"\n===> EPOCH: {epoch} ({training_step})")

        if training_step == 'P1':
            freeze(True, (model.variant_encoder, model.variational_mapping, model.theta_to_c, model.past_decoder))
            freeze(False, (model.invariant_encoder, model.future_decoder))
        elif training_step == 'P2':
            freeze(True, (model.invariant_encoder,))
            freeze(False, (model.variant_encoder, model.variational_mapping, model.theta_to_c, model.past_decoder, model.future_decoder))

        train_all(args, model, optimizers, train_dataset, pretrain_dataset, epoch, training_step, train_envs_name,
                  writer, stage='training')

        with torch.no_grad():
            if training_step == 'P2':
                metric = validate_ade(model, valid_dataset, epoch, training_step,  writer, stage='validation', args=args)

            #### EVALUATE ALSO THE TRAINING ADE and the validation loss
            # validate_ade(model, valid_dataset_o, epoch, training_step, writer, stage='validation_o')
            # validate_ade(model, valid_dataseto, 300, 'P6', writer, 'validation', write=False, args=args)
            # if epoch % 2 == 0:
            #     train_all(args, model, optimizers, valid_dataset, pretrain_dataset, epoch, training_step, writer, stage='validation')                
            #     if training_step == 'P4':
            #         validate_er(model, train_dataset, epoch, writer, stage='training')
            #     else:
            #         validate_ade(model, train_dataset, epoch, training_step, writer, stage='training')
            # validate_ade(model, train_dataset, epoch, training_step, writer, stage='training', args=args)

        if args.finetune:
            if metric < min_metric:
                min_metric = metric
                save_all_model(args, model, optimizers, metric, epoch, training_step)
                print(f'\n{"_" * 150}\n')
        else:
            save_all_model(args, model, optimizers, metric, epoch, training_step)

    writer.close()


def train_all(args, model, optimizers, train_dataset, pretrain_dataset, epoch, training_step, train_envs_name, writer, stage,
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
    train_iter = [iter(loader) for loader in train_dataset['loaders']]
    pretrain_iter = [iter(loader) for loader in pretrain_dataset['loaders']] if pretrain_dataset else None
    loss_meter = AverageMeter("Loss", ":.4f")

    if args.batch_hetero:
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

    else:
        for train_idx, train_loader in enumerate(train_dataset['loaders']):
            loss_meter = AverageMeter("Loss", ":.4f")
            progress = ProgressMeter(len(train_loader), [loss_meter], prefix="Dataset: {:<25}".format(train_envs_name[train_idx]))
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

                loss_meter.update(loss.item(), obs_traj.shape[1])

            progress.display(batch_idx+1)
            writer.add_scalar(f"{'erm' if training_step != 'P4' else 'style'}_loss/{stage}", loss_meter.avg, epoch)


def validate_ade(model, valid_dataset, epoch, training_step, writer, stage, write=True, args=None):
    """
    Evaluate the performances on the validation set

    Args:
        - stage (str): either 'validation' or 'training': says on which dataset the metrics are computed
    """
    model.eval()

    assert (stage in ['training', 'validation', 'validation_o'])
    ade_tot_meter, fde_tot_meter = AverageMeter("ADE", ":.4f"), AverageMeter("FDE", ":.4f")

    logging.info(f"- Computing ADE ({stage})")
    with torch.no_grad():
        for loader, loader_name in zip(valid_dataset['loaders'], valid_dataset['names']):
            ade_meter, fde_meter = AverageMeter("ADE", ":.4f"), AverageMeter("FDE", ":.4f")

            for batch_idx, batch in enumerate(loader):
                batch = [tensor.cuda() for tensor in batch]
                (obs_traj, fut_traj, _, _, _, _, _) = batch
                if training_step <= 'P3':
                    ts = 'P3'
                else:
                    ts = 'P6'
                pred_fut_traj_rel = model(batch, ts)

                # from relative path to absolute path
                pred_fut_traj = relative_to_abs(pred_fut_traj_rel, obs_traj[-1, :, :2])

                # compute ADE and FDE metrics
                ade_, fde_ = cal_ade_fde(fut_traj, pred_fut_traj)
                ade_, fde_ = ade_ / (obs_traj.shape[1] * fut_traj.shape[0]), fde_ / (obs_traj.shape[1])
                ade_meter.update(ade_, obs_traj.shape[1]), fde_meter.update(fde_, obs_traj.shape[1])
                ade_tot_meter.update(ade_, obs_traj.shape[1]), fde_tot_meter.update(fde_, obs_traj.shape[1])

            logging.info(f'\t\t ADE on {loader_name:<25} dataset:\t {ade_meter.avg}')

    logging.info(f"Average {stage}:\tADE  {ade_tot_meter.avg:.4f}\tFDE  {fde_tot_meter.avg:.4f}")
    # repoch = epoch if args.num_epochs[4] == 0 else epoch - 370
    repoch = epoch
    if write: writer.add_scalar(f"ade/{stage}", ade_tot_meter.avg, repoch)

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


def train_latent_space(args, model, train_dataset, pretrain_dataset, writer): #TODO train theta_to_c
    freeze(True, (model,))  # freeze all models, it's test time
    model.eval()
    ade_tot_meters = [AverageMeter('error_rate_tot', ":.4f") for _ in range(args.ttr + 1)]
    loss_tot_meters = [AverageMeter('loss_meter', ":.4f") for _ in range(args.ttr + 1)]

    logging.info(f"- Optimizing latent spaces ")

    for loader, loader_name in zip(train_dataset['loaders'], train_dataset['names']):
        label = train_dataset['labels'][loader_name]

        for idx, batch in enumerate(loader):

            # get this batch
            batch = [tensor.cuda() for tensor in batch]
            (obs_traj, fut_traj, _, _, _, style_input, _) = batch

            # fig, array = draw_image([[obs_traj, fut_traj, fut_traj]])
            # writer.add_image("RefImages", array, 0)
            # return

            # encode the latent spaces that we'll optimize, create optimizer
            inv_latent_space = model.inv_encoder(obs_traj)
            inv_latent_space = inv_latent_space.detach()
            inv_latent_space.requires_grad = True
            opt = torch.optim.Adam((inv_latent_space,), lr=args.ttrlr)

            # encode the ground truth style that we'll used as goal
            ref_low_dim, style_encoding = model.style_encoder(style_input, 'both')
            ref_low_dim = ref_low_dim.detach()

            if args.wrongstyle:
                print(pretrain_dataset['names'])
                other_style_input = next(iter(pretrain_dataset['loaders'][2]))[5].cuda()
                other_ref_low_dim, _ = model.style_encoder(other_style_input, 'both')
                other_ref_low_dim = other_ref_low_dim.detach()
                lab_tensor = torch.stack((torch.tensor(label).cuda(), torch.tensor(label + 1).cuda()), dim=0)
            else:
                lab_tensor = torch.tensor(label).cuda().unsqueeze(0)

            wot_num = 64
            evolutions = [[] for _ in range(wot_num)]  # store the predictions at each step for visualization

            for wto in tqdm(range(wot_num)):  # we optimize seq per seq. ID of the seq we'll optimize

                for k in range(args.ttr):  # number of steps of optimization

                    opt.zero_grad()

                    # do the prediction, compute the low dim style space of the prediction
                    traj_pred_rel_k = model.decoder(inv_latent_space, style_encoding)
                    traj_pred_k = relative_to_abs(traj_pred_rel_k, obs_traj[-1, :, :2])
                    pred_full_path = torch.cat((obs_traj, traj_pred_k))
                    pred_style = model.style_encoder(from_abs_to_social(pred_full_path), 'low')

                    if args.wrongstyle:
                        # set label to wrong label to harm the prediction
                        other_style = torch.clone(other_ref_low_dim)
                        other_style[wto] = pred_style[wto]
                        style_tensor = torch.stack((torch.clone(ref_low_dim), other_style),
                                                   dim=0)  # get the batch of social encoding of sequences of ONE ENV

                    else:
                        # replace the first seq style GT by first seq style prediction
                        style_tensor = torch.clone(ref_low_dim).unsqueeze(
                            0)  # get the batch of social encoding of sequences of ONE ENV
                        style_tensor[0][wto] = pred_style[
                            wto]  # replace seq number WTO in the batch of seq social encodings

                    # compute loss           
                    loss = criterion.contrastive_loss(style_tensor, lab_tensor)

                    # update metrics & visualization
                    loss_tot_meters[k].update(loss.item())
                    # ade_list.append(compute_ade_single(traj_pred_rel_k, obs_traj, fut_traj, wto))
                    ade_tot_meters[k].update(compute_ade_single(traj_pred_rel_k, obs_traj, fut_traj,
                                                                wto))  # compute_ade_single() compute ADE just on seq number WTO
                    if k in [0, 1, 5, 9] + [i * 20 - 1 for i in range(20)]:
                        fig, array = draw_image([[obs_traj, fut_traj, traj_pred_k.detach()]])
                        writer.add_image("Some paths", array, k)
                    evolutions[wto].append(traj_pred_k.detach())  # save for visualization

                    # backward and optimize
                    loss.backward()
                    opt.step()

                traj_pred_rel_k = model.decoder(inv_latent_space, style_encoding)
                ade_tot_meters[k + 1].update(compute_ade_single(traj_pred_rel_k, obs_traj, fut_traj, wto))
                evolutions[wto].append(traj_pred_k.detach())

    all_res = []
    for evo in evolutions:
        res = [[obs_traj, fut_traj, pred] for i, pred in enumerate(evo) if i in [0, 1, 3, 5, 10]]
        all_res.append(res)

    fig, array = draw_solo_all(all_res)
    writer.add_image("evol/refinement", array, 0)

    for k in range(args.ttr + 1):
        logging.info(
            f"average ade during refinement [{k}]:\t  {ade_tot_meters[k].avg:.6f}   \t  loss refinement [{k}]:\t  {loss_tot_meters[k].avg:.8f} ")
        writer.add_scalar(f"ade_refine/plot", ade_tot_meters[k].avg, k)
        if k < args.ttr: writer.add_scalar(f"loss_refine/plot", loss_tot_meters[k].avg, k)


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
