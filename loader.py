from torch.utils.data import DataLoader, ConcatDataset

from trajectories import TrajectoryDataset, seq_collate
from utils import set_domain_shift, set_batch_size


def data_loader(args, paths, name, finetune=False, test=False):
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
                finetune_ratio=args.finetune_ratio,
                add_confidence=args.add_confidence,
                finetune=finetune,
                test=test
            ))
        dset = ConcatDataset(dsets)

    else:
        raise ValueError('Unrecognized dataset name "%s"' % args.dataset_name)

    batch_size = set_batch_size(args.batch_method, args.batch_size, name)

    loader = DataLoader(
        dset,
        batch_size=batch_size,
        shuffle=args.shuffle,
        num_workers=args.loader_num_workers,
        collate_fn=seq_collate,
        pin_memory=False
    )
    return loader
