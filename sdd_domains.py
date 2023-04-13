import os
import pandas as pd
import numpy as np
from utils import *
from parser_file import get_training_parser
import math
from sklearn.cluster import KMeans
import cv2


def mask_step(x, step):
    """
    Create a mask to only contain the step-th element starting from the first element. Used to downsample
    """
    mask = np.zeros_like(x)
    mask[::step] = 1
    return mask.astype(bool)


def split_fragmented(df):
    """
    Split trajectories when fragmented (defined as frame_{t+1} - frame_{t} > 1)
    Formally, this is done by changing the metaId at the fragmented frame and below
    :param df: DataFrame containing trajectories
    :return: df: DataFrame containing trajectories without fragments
    """

    def split_at_fragment_lambda(x, frag_idx, gb_frag):
        """ Used only for split_fragmented() """
        metaId = x.metaId.iloc()[0]
        counter = 0
        if metaId in frag_idx:
            split_idx = gb_frag.groups[metaId]
            for split_id in split_idx:
                x.loc[split_id:, 'newMetaId'] = '{}_{}'.format(metaId, counter)
                counter += 1
        return x

    gb = df.groupby('metaId', as_index=False)
    # calculate frame_{t+1} - frame_{t} and fill NaN which occurs for the first frame of each track
    df['frame_diff'] = gb['frame'].diff().fillna(value=1.0).to_numpy()
    fragmented = df[df['frame_diff'] != 1.0]  # df containing all the first frames of fragmentation
    gb_frag = fragmented.groupby('metaId')  # helper for gb.apply
    frag_idx = fragmented.metaId.unique()  # helper for gb.apply
    df['newMetaId'] = df['metaId']  # temporary new metaId

    df = gb.apply(split_at_fragment_lambda, frag_idx, gb_frag)
    df['metaId'] = pd.factorize(df['newMetaId'], sort=False)[0]
    df = df.drop(columns='newMetaId')

    return df


def downsample(df, step):
    """
    Downsample data by the given step. Example, SDD is recorded in 30 fps, with step=30, the fps of the resulting
    df will become 1 fps. With step=12 the result will be 2.5 fps. It will do so individually for each unique
    pedestrian (metaId)
    :param df: pandas DataFrame - necessary to have column 'metaId'
    :param step: int - step size, similar to slicing-step param as in array[start:end:step]
    :return: pd.df - downsampled
    """
    mask = df.groupby(['metaId'])['metaId'].transform(mask_step, step=step)

    return df[mask]


def filter_short_trajectories(df, threshold):
    """
    Filter trajectories that are shorter in timesteps than the threshold
    :param df: pandas df with columns=['x', 'y', 'frame', 'trackId', 'sceneId', 'metaId']
    :param threshold: int - number of timesteps as threshold, only trajectories over threshold are kept
    :return: pd.df with trajectory length over threshold
    """

    len_per_id = df.groupby(by='metaId', as_index=False).count()  # sequence-length for each unique pedestrian
    idx_over_thres = len_per_id[len_per_id['frame'] >= threshold]  # rows which are above threshold
    idx_over_thres = idx_over_thres['metaId'].unique()  # only get metaIdx with sequence-length longer than threshold
    df = df[df['metaId'].isin(idx_over_thres)]  # filter df to only contain long trajectories

    return df


def sliding_window(df, window_size, stride):
    """
    Assumes downsampled df, chunks trajectories into chunks of length window_size. When stride < window_size then
    chunked trajectories are overlapping
    :param df: df
    :param window_size: sequence-length of one trajectory, mostly obs_len + pred_len
    :param stride: timesteps to move from one trajectory to the next one
    :return: df with chunked trajectories
    """

    def groupby_sliding_window(x, window_size, stride):
        x_len = len(x)
        n_chunk = (x_len - window_size) // stride + 1
        idx = []
        metaId = []
        for i in range(n_chunk):
            idx += list(range(i * stride, i * stride + window_size))
            metaId += ['{}_{}'.format(x.metaId.unique()[0], i)] * window_size
        # temp = x.iloc()[(i * stride):(i * stride + window_size)]
        # temp['new_metaId'] = '{}_{}'.format(x.metaId.unique()[0], i)
        # df = df.append(temp, ignore_index=True)
        df = x.iloc()[idx]
        df['newMetaId'] = metaId
        return df

    gb = df.groupby(['metaId'], as_index=False)
    df = gb.apply(groupby_sliding_window, window_size=window_size, stride=stride)
    df['metaId'] = pd.factorize(df['newMetaId'], sort=False)[0]
    df = df.drop(columns='newMetaId')
    df = df.reset_index(drop=True)

    return df


def average_speed(df, seq_len):

    def groupby_average_speed(x, seq_len=None):
        if seq_len is None:
            speed = sum(x.speed) / len(x)
        else:
            dist = math.sqrt((x.iloc[0]['x'] - x.iloc[-1]['x'])**2 + (x.iloc[0]['y'] - x.iloc[-1]['y'])**2)
            speed = dist / seq_len

        return speed

    gb = df.groupby(['envId', 'sceneId', 'metaId'], as_index=False)
    df = gb.apply(groupby_average_speed, seq_len=seq_len)
    df.rename(columns={None: 'speed'}, inplace=True)

    gb = df.groupby(['envId', 'sceneId'], as_index=False)
    df = gb.apply(groupby_average_speed)
    df.rename(columns={None: 'speed'}, inplace=True)

    return df.set_index(['envId', 'sceneId'])


def split_data(df, df_speed):

    def groupby_labels(x, df_speed):
        e = x.loc[:, 'envId'].to_numpy()[0]
        s = x.loc[:, 'sceneId'].to_numpy()[0]
        x['labels'] = df_speed.loc[(e, s), 'labels']

        return x

    gb = df.groupby(['envId', 'sceneId'], as_index=False)
    df = gb.apply(groupby_labels, df_speed=df_speed)

    df1 = df[df.labels == 0].reset_index()
    df2 = df[df.labels == 1].reset_index()
    df3 = df[df.labels == 2].reset_index()
    df4 = df[df.labels == 3].reset_index()

    return df1, df2, df3, df4


def get_image_list(df, data_dir):

    def groupby_images(x, data_dir):
        scene = x.sceneId.unique()
        envId = x.loc[:, 'envId'].to_numpy()[0]

        im_list = []
        for s in scene:
            im_list += [os.path.join(data_dir, envId, s, 'reference.jpg')]

        return np.stack(im_list)

    gb = df.groupby(['envId'], as_index=False)
    df = gb.apply(groupby_images, data_dir)

    images = []
    for env_list in df.values:
        for s in env_list:
            images += [s]

    return images


def main(args):

    _dir = os.path.dirname(__file__)
    data_dir = os.path.join(_dir, "datasets", args.dataset_name)

    envs = os.listdir(data_dir)
    SDD_cols = ['trackId', 'xmin', 'ymin', 'xmax', 'ymax', 'frame', 'lost', 'occluded', 'generated', 'label']
    data = []
    print('loading data')
    for env in envs:
        scenes = os.listdir(data_dir + f'/{env}')
        for scene in scenes:
            scene_path = os.path.join(data_dir, env, scene, 'annotations.txt')
            scene_df = pd.read_csv(scene_path, header=0, names=SDD_cols, delimiter=' ')
            # Calculate center point of bounding box
            scene_df['x'] = (scene_df['xmax'] + scene_df['xmin']) / 2
            scene_df['y'] = (scene_df['ymax'] + scene_df['ymin']) / 2
            scene_df = scene_df[scene_df['label'] == 'Pedestrian']  # drop non-pedestrians
            scene_df = scene_df[scene_df['lost'] == 0]  # drop lost samples
            scene_df = scene_df.drop(columns=['xmin', 'xmax', 'ymin', 'ymax', 'occluded', 'generated', 'label', 'lost'])
            scene_df['envId'] = env
            scene_df['sceneId'] = scene
            # new unique id by combining scene_id and track_id
            scene_df['env&rec&trackId'] = [envId + '_' + recId + '_' + str(trackId).zfill(4) for envId, recId, trackId in
                                           zip(scene_df.envId, scene_df.sceneId, scene_df.trackId)]

            data.append(scene_df)

    data = pd.concat(data, ignore_index=True)
    rec_trackId2metaId = {}
    for i, j in enumerate(data['env&rec&trackId'].unique()):
        rec_trackId2metaId[j] = i
    data['metaId'] = [rec_trackId2metaId[i] for i in data['env&rec&trackId']]
    data = data.drop(columns=['env&rec&trackId'])

    # split fragmented trajectories and assign a new id
    data = split_fragmented(data)

    # downsample the data specified by skip
    data = downsample(data, step=args.skip)

    # filter out short trajectories
    data = filter_short_trajectories(data, threshold=args.fut_len + args.obs_len)
    data = sliding_window(data, window_size=args.fut_len + args.obs_len, stride=args.fut_len + args.obs_len)

    data_speed = average_speed(data, seq_len=args.obs_len + args.fut_len)
    kmeans = KMeans(n_clusters=4).fit(data_speed.loc[:, 'speed'].to_numpy().reshape(-1, 1))
    data_speed['labels'] = kmeans.labels_

    print("Number of samples in each domain:", sum(kmeans.labels_ == 3), sum(kmeans.labels_ == 2), sum(kmeans.labels_ == 1), sum(kmeans.labels_ == 0))

    df1, df2, df3, df4 = split_data(data, data_speed)

    im_list1 = get_image_list(df1, data_dir)
    im_list2 = get_image_list(df2, data_dir)
    im_list3 = get_image_list(df3, data_dir)
    im_list4 = get_image_list(df4, data_dir)

    test_trajs1 = df1
    test_images1 = im_list1
    train_trajs1 = pd.concat([df2, df3, df4])
    train_images1 = np.concatenate([im_list2, im_list3, im_list4])

    test_trajs2 = df2
    test_images2 = im_list2
    train_trajs2 = pd.concat([df1, df3, df4])
    train_images2 = np.concatenate([im_list1, im_list3, im_list4])

    test_trajs3 = df3
    test_images3 = im_list3
    train_trajs3 = pd.concat([df1, df2, df4])
    train_images3 = np.concatenate([im_list1, im_list2, im_list4])

    test_trajs4 = df4
    test_images4 = im_list4
    train_trajs4 = pd.concat([df1, df2, df3])
    train_images4 = np.concatenate([im_list1, im_list2, im_list3])

    train_data = [train_trajs1, train_trajs2, train_trajs3, train_trajs4]
    train_images = [train_images1, train_images2, train_images3, train_images4]
    test_data = [test_trajs1, test_trajs2, test_trajs3, test_trajs4]
    test_images = [test_images1, test_images2, test_images3, test_images4]
    for i in range(4):
        path_train = data_dir + f"/domain{i}" + "/train"
        path_test = data_dir + f"/domain{i}" + "/test"

        if not os.path.exists(path_train):
            os.makedirs(path_train)

        for im in train_images[i]:
            env = im.split('\\')[8]
            scene = im.split('\\')[9]
            image = cv2.imread(im)
            cv2.imwrite(os.path.join(path_train, f"{env}_{scene}_reference.jpg"), image)

        train_data[i].to_pickle(path_train + "/train_trajs.pkl")

        if not os.path.exists(path_test):
            os.makedirs(path_test)

        for im in test_images[i]:
            env = im.split('\\')[8]
            scene = im.split('\\')[9]
            image = cv2.imread(im)
            cv2.imwrite(os.path.join(path_test, f"{env}_{scene}_reference.jpg"), image)

        test_data[i].to_pickle(path_test + "/test_trajs.pkl")


if __name__ == "__main__":
    input_args = get_training_parser().parse_args()
    main(input_args)