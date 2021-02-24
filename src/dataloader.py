import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import datetime as dt


AUTOTUNE = tf.data.experimental.AUTOTUNE

def rescale(image):
    return image/255#uint8 to float32 0~1

class dataloader():
    def __init__(self, batch_size, img_size, is_aug=False, ):
        self.batch_size = batch_size
        self.img_size = img_size
        self.is_aug = is_aug
        
    def aug(self, image):
        image = tfa.image.translate_xy(image, tf.random.uniform([2], minval=-0.2, maxval=0.3), 0)
        image = tfa.image.rotate(image, tf.random.uniform([1], minval=-3.14, maxval=3.14),)
        
        image = tf.image.random_crop(image, [self.img_size, self.img_size, 3], 
                                     seed=int(dt.datetime.now().timestamp()))
        image = tf.image.random_contrast(image, 0.5, 1.3,
                                         seed=int(dt.datetime.now().timestamp()))
        return image
    
    def preprocess(self, image):
        if self.is_train:
            tmp_img_size = int(self.img_size*1.5)
        else:
            tmp_img_size = self.img_size
        image = tf.image.resize(image, [tmp_img_size, tmp_img_size])
        image = tf.cast(image, tf.float16)
        image = rescale(image)
        if self.is_aug and self.is_train:
            image = self.aug(image)
        return image

    def load_img(self, path):
        image_raw = tf.io.read_file(path)
        image = tf.io.decode_image(image_raw, channels=3, expand_animations=False)
        image = self.preprocess(image)
        return image
    
    def to_first_channel(self, image):
        return tf.transpose(image, perm=[2, 0, 1])
    
    def __call__(self, image_path_list, label_list, 
                 is_train=True, shuffle=True, shuffle_buffer=None):
        self.is_train = is_train
        path_ds = tf.data.Dataset.from_tensor_slices(image_path_list)
        label_ds = tf.data.Dataset.from_tensor_slices(label_list)
        image_ds = path_ds.map(self.load_img, num_parallel_calls=AUTOTUNE)
            
        ds = tf.data.Dataset.zip((image_ds, label_ds))

        if shuffle_buffer is None:
            shuffle_buffer=len(image_path_list)
        if shuffle:
            ds = ds.shuffle(buffer_size=shuffle_buffer, reshuffle_each_iteration=True)

        ds = ds.batch(self.batch_size)
        ds = ds.prefetch(buffer_size=8)
        return ds
