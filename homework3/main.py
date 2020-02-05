#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
import pathlib
import re
import warnings
import time
import cv2 as cv
import matplotlib.pyplot as plt
from pyquaternion import Quaternion

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
    image = tf.cast(image, tf.float32)
    image = tf.image.per_image_standardization(image)
    return image


def get_data(folder_path):
    data = {}
    for label in labels:
        path = folder_path + '/' + label + '/*.png'
        image_data = tf.data.Dataset.list_files(str(data_dir/path), shuffle=False)

        pose_map = get_pose_map(str(data_dir/(folder_path + '/' + label + '/poses.txt')))

        label_data = []
        for d in image_data.take(-1):
            file_name = str(d.numpy().decode('UTF-8')).split('/')[-1]

            image = lambda d=d: get_image(d)
            pose = pose_map[file_name]

            label_data.append({
                'image': image,
                'pose': pose,
                'label': label,
                'name': file_name
            })

        label_data.sort(key=lambda f: int(re.sub('\D', '', f['name'])))
        data[label] = np.array(label_data)

    return data


def split(s_real):
    with open(str(data_dir/('data/real/training_split.txt'))) as f:
        indices = np.array(f.readline().strip().split(', ')).astype(np.int)
    
    # print(s_real)
    # n_s_real = len(next(iter(s_real.values())))
    # n_subtrain = len(indices)
    # print("Train split: {}, Test split: {}".format(n_subtrain, n_s_real - n_subtrain))

    s_subtrain = {}
    s_test = {}
    for label in s_real:
        s_subtrain[label] = s_real[label][indices]

        mask = np.ones(len(s_real[label]), np.bool)
        mask[indices] = 0
        s_test[label] = s_real[label][mask]

    # print(s_subtrain)
    return s_subtrain, s_test


def combine(a, b):
    for label in labels:
        a[label] = np.append(a[label], b[label])

    return a


def qw_distance(q1, q2):
    dot = np.dot(q1, q2)
    # Sometimes it's 1.0000000000000002 ¯\_(ツ)_/¯
    if (np.isclose(dot, 1.)):
        dot = 1.

    return 2 * np.arccos(np.abs(dot))


def find_puller(anchor, s_db):
    label = anchor['label']
    mini = float('inf')
    for candidate in s_db[label]:
        q1 = anchor['pose']
        q2 = candidate['pose']
        distance = qw_distance(q1, q2)
        if (mini > distance):
            puller = candidate
            mini = distance

    return puller


def find_pusher(anchor, s_db):
    label = anchor['label']
    if np.random.rand() < 0.5:
        # choose a pusher with same object, random different pose
        pusher = s_db[label][np.random.randint(len(s_db))]
        # keep choosing again if they are the same pose, just in case
        while(np.all(np.isclose(pusher['pose'], anchor['pose']))):
            pusher = s_db[label][np.random.randint(len(s_db))]
        return pusher
    else:
        # choose a pusher of a random different object
        choices = [key for key in s_db if key != label]
        key = np.random.choice(choices)
        return s_db[key][np.random.randint(len(s_db))]


def generate_batch(s_train, s_db, batch_size):
    flat = np.array([s_train[key] for key in s_train]).flatten()

    chosen = np.random.choice(len(flat), batch_size)

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


def visualize(x_a, x_pull, x_push):
    for i in range(len(x_a)):
        im_a = x_a[i]
        im_pull = x_pull[i]
        im_push = x_push[i]

        fig = plt.figure(figsize=(1, 3))
        fig.add_subplot(1, 3, 1)
        plt.imshow(im_a)

        fig.add_subplot(1, 3, 2)
        plt.imshow(im_pull)

        fig.add_subplot(1, 3, 3)
        plt.imshow(im_push)
        plt.show()


def loss_func(model, x):
    m = 0.01

    x_a = tf.convert_to_tensor([x_i['anchor']['image']() for x_i in x])
    x_pull = tf.convert_to_tensor([x_i['puller']['image']() for x_i in x])
    x_push = tf.convert_to_tensor([x_i['pusher']['image']() for x_i in x])
    # visualize(x_a, x_pull, x_push)

    y_a = model(x_a)
    y_pull = model(x_pull)
    y_push = model(x_push)

    square_diff_pos = tf.math.squared_difference(y_a, y_pull)
    square_diff_neg = tf.math.squared_difference(y_a, y_push)
    # print(square_diff_pos)
    dist_pos = tf.reduce_sum(square_diff_pos, axis=1)
    dist_neg = tf.reduce_sum(square_diff_neg, axis=1)

    loss_triplets = tf.reduce_sum(tf.maximum(0., 1. - dist_neg/(dist_pos + m)))
    loss_pairs = tf.reduce_sum(dist_neg)
    loss = loss_triplets + loss_pairs

    # print("L_t: {}, L_p: {}, L: {}".format(loss_triplets, loss_pairs, loss))
    return loss


def grad(model, x):
    with tf.GradientTape() as tape:
        loss_value = loss_func(model, x)

    return loss_value, tape.gradient(loss_value, model.trainable_variables)


# Load the image into memory
def unpack(s):
    return [{'image': value['image'](), 'label': value['label'], 'pose': value['pose'], 'name': value['name']} for values in s.values() for value in values]


def angle_diff(q1, q2):
    a = Quaternion(q1)
    b = Quaternion(q2)

    q = a * b.conjugate
    deg = abs(q.degrees)

    if deg > 180. and np.isclose(deg, 180.):
        deg = 180.

    if deg > 180:
        print('Angle greater than 180:', deg)

    return deg


def plot_hist(model, test_data, db_data, test_image, db_images, epoch):
    db_feats = model(db_images).numpy().astype(np.float32)
    # print(db_feats.shape)

    db_responses = np.arange(len(db_feats)).astype(np.float32)
    # print(db_responses.shape)

    knn = cv.ml.KNearest_create()
    knn.train(db_feats, cv.ml.ROW_SAMPLE, db_responses)

    test_feats = model(test_images).numpy().astype(np.float32)
    ret, results, neighbours, dist = knn.findNearest(test_feats, 1)

    correct = 0.
    w = [0, 0, 0, 0]
    for i, result in enumerate(results):
        idx = int(result[0])
        label = db_data[idx]['label']
        gt_label = test_data[i]['label']
        if label == gt_label:
            pose1 = db_data[idx]['pose']
            pose2 = test_data[i]['pose']
            dist = angle_diff(pose1, pose2)
            if dist < 10:
                w[0] = w[0] + 1
            if dist < 20:
                w[1] = w[1] + 1
            if dist < 40:
                w[2] = w[2] + 1
            if dist < 180:
                w[3] = w[3] + 1

    w = [[x * 100. / len(results)] for x in w]
    data = [[9], [19], [39], [179]]
    plt.hist(data, weights=w, bins=[0, 10, 20, 40, 180], histtype='stepfilled')
    plt.xlabel('Angle tolerance')
    plt.ylabel('%')
    plt.savefig("hist-{}.png".format(epoch))
    plt.clf()
    # plt.show()


def save_epoch(epoch):
    with open('logs/epoch.txt', 'w') as f:
        f.write('%d' % epoch)


def load_epoch():
    with open('logs/epoch.txt', 'r') as f:
        return int(f.readline().strip())


s_db = get_data('data/coarse')
s_fine = get_data('data/fine')
s_subtrain, s_test = split(get_data('data/real'))
s_train = combine(s_fine, s_subtrain)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters=16, kernel_size=(8,8), activation='relu', input_shape=(64,64,3)),
    tf.keras.layers.MaxPool2D(),

    tf.keras.layers.Conv2D(filters=7, kernel_size=(5,5), activation='relu'),
    tf.keras.layers.MaxPool2D(),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(units=256, activation='relu'),

    tf.keras.layers.Dense(units=16, activation='softmax')
])

# Testing
# batch = generate_batch(s_train, s_db, 5)
# print(batch)
# loss_func(model, batch)
# exit()

# Pre compute test and db features
print('Loading test and db features...')
test_data = unpack(s_test)
db_data = unpack(s_db)
test_images = tf.convert_to_tensor([x['image'] for x in test_data])
db_images = tf.convert_to_tensor([x['image'] for x in db_data])
print('Done loading features.')

tc = tf.keras.callbacks.TensorBoard()
tc.set_model(model)
tm = tf.keras.callbacks.ModelCheckpoint('logs/weights', save_weights_only=True)
tm.set_model(model)

batch_size = 30
num_epochs = 50000
optimizer = tf.keras.optimizers.Adam()

load = 1
init_epoch = 0

if load == 1:
    # Load
    print("Loading weight from last epoch.")
    model.load_weights('logs/weights')
    init_epoch = load_epoch()

# Train
for epoch in range(init_epoch, num_epochs):
    batch = generate_batch(s_train, s_db, batch_size)

    loss_value, grads = grad(model, batch)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    if (epoch) % 10 == 0:
        print("Epoch {:03d}: Loss: {}".format(epoch, loss_value))
        logs = {'loss': loss_value }
        tc.on_epoch_end(epoch, logs)
        tm.on_epoch_end(epoch)
        save_epoch(epoch)

    if (epoch) % 1000 == 0:
        plot_hist(model, test_data, db_data, test_images, db_images, epoch)

tc.on_train_end('_')
