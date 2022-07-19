import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from constants import BATCHES
from sklearn.preprocessing import StandardScaler
import numpy as np
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.framework import dtypes

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


def _per_image_standardization(image):
    """ Linearly scales `image` to have zero mean and unit norm.
    This op computes `(x - mean) / adjusted_stddev`, where `mean` is the average
    of all values in image, and
    `adjusted_stddev = max(stddev, 1.0/sqrt(image.NumElements()))`.
    `stddev` is the standard deviation of all values in `image`. It is capped
    away from zero to protect against division by 0 when handling uniform images.
    Args:
    image: 1-D tensor of shape `[height, width]`.
    Returns:
    The standardized image with same shape as `image`.
    Raises:
    ValueError: if the shape of 'image' is incompatible with this function.
    """
    image = ops.convert_to_tensor(image, name='image')
    num_pixels = math_ops.reduce_prod(array_ops.shape(image))

    image = math_ops.cast(image, dtype=dtypes.float32)
    image_mean = math_ops.reduce_mean(image)

    variance = (math_ops.reduce_mean(math_ops.square(image)) -
                math_ops.square(image_mean))
    variance = gen_nn_ops.relu(variance)
    stddev = math_ops.sqrt(variance)

    # Apply a minimum normalization that protects us against uniform images.
    min_stddev = math_ops.rsqrt(math_ops.cast(num_pixels, dtypes.float32))
    pixel_value_scale = math_ops.maximum(stddev, min_stddev) + 1e-10
    pixel_value_offset = image_mean

    image = math_ops.subtract(image, pixel_value_offset)
    image = math_ops.div_no_nan(image, pixel_value_scale)
    return image


def preprocessing(example):
    image = example['image']
    image = tf.expand_dims(image, axis=-1)
    image = _per_image_standardization(image)
    image = tf.where(tf.math.is_nan(image), tf.zeros_like(image), image)
    angular_size = log10(example['angular_size'])
    angular_size = tf.where(tf.math.is_nan(angular_size), tf.zeros_like(angular_size), angular_size)
    return image, angular_size


def preprocessing_test(example):
    image = example['image']
    image = tf.expand_dims(image, axis=-1)
    image = _per_image_standardization(image)
    image = tf.where(tf.math.is_nan(image), tf.zeros_like(image), image)
    angular_size = log10(example['angular_size'])
    angular_size = tf.where(tf.math.is_nan(angular_size), tf.zeros_like(angular_size), angular_size)
    return image, angular_size, example['magnitude']


def input_fn(mode='train', dataset_str='structural_fitting', batch_size=BATCHES):
    """
    mode: 'train', 'validation' or 'test'
    """

    shuffle = mode in ('train', 'validation')
    dataset = tfds.load(
        dataset_str,
        split=mode,
        shuffle_files=shuffle
    )

    if shuffle:
        dataset = dataset.repeat()
        dataset = dataset.shuffle(10000)
        # Apply data preprocessing
        dataset = dataset.map(preprocessing, num_parallel_calls=tf.data.AUTOTUNE)
    else:
        dataset = dataset.map(preprocessing_test, num_parallel_calls=tf.data.AUTOTUNE)

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


def get_data_test(dataset, batches=10):
    data = dataset.take(batches)
    images, y_true, magnitude = [], [], []
    for d in list(data):
        images.extend(d[0].numpy())
        y_true.extend(d[1].numpy())
        magnitude.extend(d[2].numpy())
    images = np.stack(images)
    y_true = np.array(y_true)
    magnitude = np.array(magnitude)
    return images, y_true, magnitude


def get_num_examples(mode='train', dataset_str='structural_fitting'):
    builder = tfds.builder(dataset_str)
    splits = builder.info.splits
    return splits[mode].num_examples


if __name__ == '__main__':
    ds_train = input_fn('train')

    len_ds_train = get_num_examples('train')
    len_ds_val = get_num_examples('validation')
    len_ds_test = get_num_examples('test')

    print(len_ds_test, len_ds_val, len_ds_train)
