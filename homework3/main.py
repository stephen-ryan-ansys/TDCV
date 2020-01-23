import numpy as np
import tensorflow as tf
import pathlib
import re
import warnings

warnings.filterwarnings("error")
data_dir = pathlib.Path.cwd()
labels = ['ape', 'benchvise', 'cam', 'cat', 'duck']

def get_pose_map(path):
    pose_map = {}
    with open(path) as f:
        lines = [line.rstrip().strip() for line in f]

        key = ''
        count = 0
        for line in lines:
            # Empty line
            if len(line) == 0:
                continue

            if count % 2 == 0:
                # Header
                key = line.split()[-1]
            else:
                # Pose
                poses = np.array(line.split(), dtype=np.float64)
                pose_map[key] = poses

            count = count + 1

    return pose_map

def get_image(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    return image

def get_data(folder_path):
    data = {}
    for label in labels:
        path = folder_path + '/' + label + '/*.png'
        image_data = tf.data.Dataset.list_files(str(data_dir/path), shuffle=False)

        pose_map = get_pose_map(str(data_dir/(folder_path + '/' + label + '/poses.txt')))

        label_data = []
        for d in image_data:
            file_name = str(d.numpy().decode('UTF-8')).split('/')[-1]

            image = get_image(d)
            pose = pose_map[file_name]

            label_data.append({
                'image': image,
                'pose': pose,
                'label': label
            })

        data[label] = np.array(label_data)

    return data

def split(s_real):
    with open(str(data_dir/('data/real/training_split.txt'))) as f:
        indices = np.array(f.readline().strip().split(', ')).astype(np.int)
    print(len(indices))

    s_subtrain = {}
    s_test = {}
    for label in s_real:
        s_subtrain[label] = s_real[label][indices]

        mask = np.ones(len(s_real[label]), np.bool)
        mask[indices] = 0
        s_test[label] = s_real[label][mask]

    return s_subtrain, s_test

def combine(a, b):
    for label in labels:
        a[label] = np.append(a[label], b[label])

    return a

def is_equal(a, b):
    eps = 1e-6
    return np.abs(a - b) < eps

def qw_distance(q1, q2):
    dot = np.dot(q1, q2)
    # Sometimes it's 1.0000000000000002 ¯\_(ツ)_/¯
    if (is_equal(dot, 1)):
        dot = 1

    return 2 * np.arccos(np.abs(dot))

def find_puller(anchor, s_db):
    label = anchor['label']
    mini = -1
    for candidate in s_db[label]:
        q1 = anchor['pose']
        q2 = candidate['pose']
        distance = qw_distance(q1, q2)
        if (mini == -1 or mini > distance):
            puller = candidate
            mini = distance

    return puller

def find_pusher(anchor, s_db):
    label = anchor['label']
    for key in s_db:
        if (key != label):
            return s_db[key][np.random.randint(len(s_db))]

def generate_batch(s_train, s_db, batch_size):
    flat = np.array([s_train[key] for key in s_train]).flatten()

    chosen = np.arange(len(flat))
    np.random.shuffle(chosen)
    chosen = chosen[0:batch_size]

    triplets = []

    for anchor in flat[chosen]:
        puller = find_puller(anchor, s_db)
        pusher = find_pusher(anchor, s_db)

        triplets.append({
            'anchor': anchor,
            'puller': puller,
            'pusher': pusher
        })

    return np.array(triplets)


s_db = get_data('data/coarse')
s_fine = get_data('data/fine')
s_subtrain, s_test = split(get_data('data/real'))
s_train = combine(s_fine, s_subtrain)

batch = generate_batch(s_train, s_db, 100)

print(len(batch))