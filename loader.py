from torch.utils.data import DataLoader, ConcatDataset

from trajectories import TrajectoryDataset, SynTrajectoryDataset, seq_collate_social, seq_collate
from utils import set_domain_shift, set_batch_size


def data_loader(args, paths, name, finetune=False, test=False, pt=False):
    alpha_e = set_domain_shift(args.domain_shifts, name)

    # ETH-UCY Dataset
    if args.dataset_name in ('eth', 'hotel', 'univ', 'zara1', 'zara2'):
        dsets = []
        for path in paths:
            dsets.append(TrajectoryDataset(
                path,
                alpha_e=alpha_e,
                obs_len=args.obs_len,
                fut_len=args.fut_len,
                skip=args.skip,
                delim=args.delim,
                n_coordinates=args.n_coordinates,
                add_confidence=args.add_confidence,
                finetune_ratio=args.finetune_ratio,
                finetune=finetune,
                test=test
            ))
        dset = ConcatDataset(dsets)

        batch_size = set_batch_size(args.batch_method, args.batch_size, name)

        loader = DataLoader(
            dset,
            batch_size=batch_size,
            shuffle=args.shuffle,
            num_workers=args.loader_num_workers,
            collate_fn=seq_collate,
            pin_memory=False
        )

    # Synthetic Dataset
    elif 'synthetic' in args.dataset_name or args.dataset_name in ['synthetic', 'v2', 'v2full', 'v4']:
        reduce = 0
        if ('train' in paths and not pt):
            reduce = args.reduce
        if args.reduceall != 0:
            reduce = args.reduceall  # used to test quickly
        dset = SynTrajectoryDataset(
            paths,  # path
            obs_len=args.obs_len,
            fut_len=args.fut_len,
            n_coordinates=args.n_coordinates,
            add_confidence=args.add_confidence,
            alpha_e=alpha_e,
            reduce=reduce,
            finetune_ratio=args.finetune_ratio,
            finetune=finetune,
            test=test
        )

        batch_size = set_batch_size(args.batch_method, args.batch_size, name)

        loader = DataLoader(
            dset,
            batch_size=batch_size,
            shuffle=args.shuffle,
            num_workers=args.loader_num_workers,
            collate_fn=seq_collate_social,
            pin_memory=False
        )

    else:
        raise ValueError('Unrecognized dataset name "%s"' % args.dataset_name)

    return loader