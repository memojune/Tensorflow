import os
import numpy as np
import tensorflow as tf
import skimage.io as io
import matplotlib.pyplot as plt

def get_file(file_dir):
    images = []
    temp = []

    for root, subdirs, files in os.walk(file_dir):
        for name in files:
            images.append(os.path.join(root, name))
        for name in subdirs:
            temp.append(os.path.join(root, name))

    labels = []
    for folder in temp:
        n_img = len(os.listdir(folder))
        letter = folder.split('\\')[-1]
        if len(letter) == 1:
            labels.extend([ord(letter) - ord('A') + 1]*n_img)
    temp = np.array([images, labels]).transpose()
    np.random.shuffle(temp)

    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])

    return image_list, label_list

def int64_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def convert_to_tfrecord(images, labels, save_dir, name):
    filename = os.path.join(save_dir, name+'.tfrecords')
    n_samples = len(labels)

    if len(images) != n_samples:
        raise ValueError('Images size %d does not match label size %d.' % (images.shape[0], n_samples))

    writer = tf.python_io.TFRecordWriter(filename)
    print('\nTransformation start......')
    for i in range(n_samples):
        try:
            image = io.imread(images[i])
            image_raw = image.tostring()
            label = int(labels[i])
            example = tf.train.Example(features=tf.train.Features(feature={
                                                                    'label': int64_feature(label),
                                                                    'image_raw': bytes_feature(image_raw)}))
            writer.write(example.SerializeToString())
        except IOError as e:
            print('Could not read:', images[i])
            print('skip it')
    writer.close()
    print('Transformation done!')

def read_and_decode(tfrecords_file, batch_size):
    filename_queue = tf.train.string_input_producer([tfrecords_file])

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    img_features = tf.parse_single_example(serialized_example,
                                           features={
                                               'label': tf.FixedLenFeature([], tf.int64),
                                               'image_raw': tf.FixedLenFeature([], tf.string)})
    image = tf.decode_raw(img_features['image_raw'], tf.uint8)
    image = tf.reshape(image, [28, 28])
    label = tf.cast(img_features['label'], tf.int32)
    image_batch, label_batch = tf.train.batch([image, label],
                                              batch_size=batch_size,
                                              num_threads=32,
                                              capacity=2000)

    return image_batch, tf.reshape(label_batch, [batch_size])

test_dir = 'data\\notMNist_small'
save_dir = 'data'
BATCH_SIZE = 25

# name_test = 'test'
# images, labels = get_file(test_dir)
# convert_to_tfrecord(images, labels, save_dir, name_test)

def plot_images(images, labels):
    for i in range(BATCH_SIZE):
        plt.subplot(5, 5, i+1)
        plt.axis('off')
        plt.title(chr(ord('A')+labels[i]-1), fontsize=14)
        plt.subplots_adjust(top=1.5)
        plt.imshow(images[i])
    plt.show()

tfrecords_file = 'data\\test.tfrecords'
image_batch, label_batch = read_and_decode(tfrecords_file, batch_size=BATCH_SIZE)

with tf.Session() as sess:
    i = 0
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    try:
        while not coord.should_stop() and i < 1:
            image, label = sess.run([image_batch, label_batch])
            plot_images(image, label)
            i += 1
    except :
        print('done!')
    finally:
        coord.request_stop()
    coord.join(threads)






































































