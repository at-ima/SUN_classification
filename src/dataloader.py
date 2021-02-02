import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import datetime as dt


AUTOTUNE = tf.data.experimental.AUTOTUNE

def rescale(image):
    return image/255#uint8 to float32 0~1

class dataloader():
    def __init__(self, batch_size, img_size):
        self.batch_size = batch_size
        self.img_size = img_size
        
    def preprocess(self, image):
        image = tf.image.resize(image, [int(self.img_size*1.5), int(self.img_size*1.5)])
        image = tf.cast(image, tf.float16)
        image = rescale(image)
        return image

    def load_img(self, path):
        image_raw = tf.io.read_file(path)
        image = tf.io.decode_image(image_raw, channels=3, expand_animations=False)
        image = self.preprocess(image)
        return image

    def test_preprocess(self, image):
        image = tf.image.resize(image, [self.img_size, self.img_size])
        image = tf.cast(image, tf.float16)
        image = rescale(image)
        return image

    def load_test_img(self, path):
        image_raw = tf.io.read_file(path)
        image = tf.io.decode_image(image_raw, channels=3, expand_animations=False)
        image = self.test_preprocess(image)
        return image
    
    def __call__(self, image_path_list, label_list,
                 train=True, shuffle=True, shuffle_buffer=None):
        path_ds = tf.data.Dataset.from_tensor_slices(image_path_list)
        label_ds = tf.data.Dataset.from_tensor_slices(label_list)
        if train:
            image_ds = path_ds.map(self.load_img, num_parallel_calls=AUTOTUNE)
        else:
            image_ds = path_ds.map(self.load_test_img, num_parallel_calls=AUTOTUNE)
        ds = tf.data.Dataset.zip((image_ds, label_ds))

        if shuffle_buffer is None:
            shuffle_buffer=len(image_path_list)
        if shuffle:
            ds = ds.shuffle(buffer_size=shuffle_buffer, reshuffle_each_iteration=True)

        ds = ds.batch(self.batch_size)
        ds = ds.prefetch(buffer_size=8)
        return ds
    
class aug_dataloader():
    def __init__(self, batch_size, img_size):
        self.batch_size = batch_size
        self.img_size = img_size
        
    def preprocess(self, image):
        image = tf.image.resize(image, [int(self.img_size*1.5), int(self.img_size*1.5)])
        image = tf.cast(image, tf.float16)
        image = rescale(image)
        
        image = tfa.image.translate_xy(image, tf.random.uniform([2], minval=-0.2, maxval=0.3), 0)
        image = tfa.image.rotate(image, tf.random.uniform([1], minval=-3.14, maxval=3.14),)
        
        image = tf.image.random_crop(image, [self.img_size, self.img_size, 3], seed=int(dt.datetime.now().timestamp()))
        image = tf.image.random_contrast(image, 0.5, 1.3, seed=int(dt.datetime.now().timestamp()))
        return image

    def load_img(self, path):
        image_raw = tf.io.read_file(path)
        image = tf.io.decode_image(image_raw, channels=3, expand_animations=False)
        image = self.preprocess(image)
        return image

    def test_preprocess(self, image):
        image = tf.image.resize(image, [self.img_size, self.img_size])
        image = tf.cast(image, tf.float16)
        image = rescale(image)
        return image

    def load_test_img(self, path):
        image_raw = tf.io.read_file(path)
        image = tf.io.decode_image(image_raw, channels=3, expand_animations=False)
        image = self.test_preprocess(image)
        return image
    
    def __call__(self, image_path_list, label_list,
                  train=True, shuffle=True, shuffle_buffer=None):
        path_ds = tf.data.Dataset.from_tensor_slices(image_path_list)
        label_ds = tf.data.Dataset.from_tensor_slices(label_list)
        if train:
            image_ds = path_ds.map(self.load_img, num_parallel_calls=AUTOTUNE)
        else:
            image_ds = path_ds.map(self.load_test_img, num_parallel_calls=AUTOTUNE)
        ds = tf.data.Dataset.zip((image_ds, label_ds))

        if shuffle_buffer is None:
            shuffle_buffer=len(image_path_list)
        if shuffle:
            ds = ds.shuffle(buffer_size=shuffle_buffer, reshuffle_each_iteration=True)

        ds = ds.batch(self.batch_size)
        ds = ds.prefetch(buffer_size=8)
        return ds
