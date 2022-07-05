import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from constants import BATCHES
from sklearn.preprocessing import StandardScaler

rng = tf.random.Generator.from_seed(123, alg='philox')


def augment(image, label):
    seed = rng.make_seeds(2)[0]
    image = tf.image.stateless_random_flip_left_right(image, seed)
    image = tf.image.stateless_random_flip_up_down(image, seed)
    return image, label


def log10(x):
    numerator = tf.math.log(x)
    denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator


def preprocessing(example):
    image = example['image']
    image = tf.expand_dims(image, axis=-1)
    angular_size = log10(example['angular_size'])
    return image, angular_size


def input_fn(mode='train', batch_size=BATCHES):
    """
    mode: 'train', 'validation' or 'test'
    """
    if mode == 'train':
        dataset = tfds.load(
            "structural_fitting",
            split="train[:75%]"
        )
    elif mode == 'validation':
        dataset = tfds.load(
            "structural_fitting",
            split="train[75%:85%]"
        )
    else:
        dataset = tfds.load(
            "structural_fitting",
            split="train[85%:]",
        )

    if mode in ('train', 'validation'):
        dataset = dataset.repeat()
        dataset = dataset.shuffle(10000)

    # Apply data preprocessing
    dataset = dataset.map(preprocessing, num_parallel_calls=tf.data.AUTOTUNE)

    if mode == 'train':
        dataset = dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.batch(batch_size, drop_remainder=True)

    # fetch next batches while training current one (-1 for autotune)
    dataset = dataset.prefetch(-1)

    return dataset


def get_data(dataset, batches=10):
    data = dataset.take(batches)
    images, y_true = [], []
    for d in list(data):
        images.extend(d[0].numpy())
        y_true.extend(d[1].numpy())

    images = np.stack(images)
    y_true = np.array(y_true)

    return images, y_true


def get_num_examples(mode='train', dataset_str='structural_fitting'):
    builder = tfds.builder(dataset_str)
    splits = builder.info.splits
    num = splits['train'].num_examples
    if mode == 'train':
        num *= 0.75
    elif mode == 'validation':
        num *= 0.1
    else:
        num *= 0.15
    return num


if __name__ == '__main__':
    ds_train = input_fn('train')

    len_ds_train = get_num_examples('train')
    len_ds_val = get_num_examples('validation')
    len_ds_test = get_num_examples('test')

    print(len_ds_test, len_ds_val, len_ds_train)
