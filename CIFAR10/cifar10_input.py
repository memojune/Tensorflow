import tensorflow as tf
import numpy as np
import os

def read_cifar10(data_dir, is_train, batch_size, shuffle):
    img_width = 32
    img_height = 32
    img_depth = 3
    label_bytes = 1
    image_bytes = img_width*img_height*img_depth
    with tf.name_scope('input'):
        if is_train:
            filenames = [os.path.join(data_dir, 'data_batch_%d.bin' %i)
                         for i in np.arange(1, 6)]
        else:
            filenames = os.path.join(data_dir, 'test_batch.bin')

        filename_queue = tf.train.string_input_producer(filenames)

        reader = tf.FixedLengthRecordReader(label_bytes+image_bytes)

        key, value = reader.read(filename_queue)

        record_bytes = tf.decode_raw(value, tf.uint8)

        label = tf.slice(record_bytes, [0], [label_bytes])
        label = tf.cast(label, tf.int32)

        image_raw = tf.slice(record_bytes, [label_bytes], [image_bytes])
        image_raw = tf.reshape(image_raw, [img_depth, img_height, img_depth])
        image = tf.transpose(image_raw, (1, 2, 0))
        image = tf.cast(image, tf.float32)
        image = tf.image.per_image_standardization(image)

        if shuffle:
            image_batch, label_batch = tf.train.shuffle_batch([image, label],
                                                              batch_size=batch_size,
                                                              num_threads=16,
                                                              capacity=2000,
                                                              min_after_dequeue=1500)
        else:
            image_batch, label_batch = tf.train.batch([image, label],
                                                       batch_size=batch_size,
                                                       num_threads=16,
                                                       capacity=2000)

        n_classes = 10
        label_batch = tf.one_hot(label_batch, depth=n_classes)
        return image_batch, tf.reshape(label_batch, [batch_size, n_classes])

