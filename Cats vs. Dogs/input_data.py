import tensorflow as tf
import numpy as np
import os

# train_dir = 'data/train/'
def get_files(file_dir):
    cats = []; label_cats = []
    dogs = []; label_dogs = []
    for file in os.listdir(file_dir):
        name = file.split('.')
        if name[0] == 'cat':
            cats.append(file_dir+file)
            label_cats.append(0)
        else:
            dogs.append(file_dir+file)
            label_dogs.append(1)
    print('There are %d cats and %d dogs' %(len(cats), len(dogs)))
    image_list = np.hstack((cats, dogs))
    label_list = np.hstack((label_cats, label_dogs))
    temp = np.array([image_list, label_list]) # 这里label_list的数据类型转换成了str
    temp = temp.transpose()
    np.random.shuffle(temp)

    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1].astype(int))
    return image_list, label_list

def get_batch(image, label, image_W, image_H, batch_size, capacity):
    image = tf.cast(image, tf.string)
    label = tf.cast(label, tf.int32)

    input_queue = tf.train.slice_input_producer([image, label])
    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(image_contents, channels=3)

    image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)
    image = tf.image.per_image_standardization(image)

    image_batch, label_batch = tf.train.batch([image, label],
                                              batch_size=batch_size,
                                              num_threads=64,
                                              capacity=capacity)
    image_batch = tf.cast(image_batch, tf.float32)
    return image_batch, label_batch

# test functions above
# import matplotlib.pyplot as plt
#
# train_dir = 'data/train/'
#
# BATCH_SIZE = 2
# CAPACITY = 256
# IMG_W = 208
# IMG_H = 208
#
# image_list, label_list = get_files(train_dir)
# image_batch, label_batch  = get_batch(image_list, label_list,
#                                       IMG_W, IMG_H, BATCH_SIZE, CAPACITY)
#
# with tf.Session() as sess:
#     i = 0
#     coord = tf.train.Coordinator()
#     threads = tf.train.start_queue_runners()
#
#     try:
#         while not coord.should_stop() and i < 1:
#             img, label = sess.run([image_batch, label_batch])
#             print(label)
#             for j in range(BATCH_SIZE):
#                 plt.imshow(img[j, :, :, :])
#                 plt.show()
#             i += 1
#     except tf.errors.OutOfRangeError:
#         print('done!')
#     finally:
#         coord.request_stop()
#     coord.join(threads)



