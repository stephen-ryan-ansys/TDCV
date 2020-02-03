#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
import pathlib
import re
import warnings
import time

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

            image = lambda : get_image(d)
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


def loss_func(model, x):
    m = 0.01

    x_a = tf.convert_to_tensor([x_i['anchor']['image']() for x_i in x])
    x_pull = tf.convert_to_tensor([x_i['puller']['image']() for x_i in x])
    x_push = tf.convert_to_tensor([x_i['pusher']['image']() for x_i in x])

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


s_db = get_data('data/coarse')
s_fine = get_data('data/fine')
s_subtrain, s_test = split(get_data('data/real'))
s_train = combine(s_fine, s_subtrain)

batch = generate_batch(s_train, s_db, 100)

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
# batch = generate_batch(s_train, s_db, 10)
# loss(model, batch)
# exit()

# Train
## Note: Rerunning this cell uses the same model variables

# Keep results for plotting
train_loss_results = []
train_accuracy_results = []

batch_size = 30
num_epochs = 100
optimizer = tf.keras.optimizers.Adam()

for epoch in range(num_epochs):
    epoch_loss_avg = tf.keras.metrics.Mean()
    epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

    batch = generate_batch(s_train, s_db, batch_size)
    # loss_value, grads = grad(model, batch)
    # epoch_loss_avg(loss_value)

    for x in batch:
        # Optimize the model
        loss_value, grads = grad(model, np.array([x])) # Make it 1-D tensor
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # Track progress
        epoch_loss_avg(loss_value)  # Add current batch loss

        # TODO define accuracy
        # epoch_accuracy(y, model(x, training=True))

    # End epoch
    train_loss_results.append(epoch_loss_avg.result())
    train_accuracy_results.append(epoch_accuracy.result())

    # if epoch % 50 == 0:
    print("Epoch {:03d}: Loss: {}, Accuracy: {}".format(epoch, epoch_loss_avg.result(), epoch_accuracy.result()))