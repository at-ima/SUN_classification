import tensorflow as tf

AUTOTUNE = tf.data.experimental.AUTOTUNE

def rescale(image):
    return image/255#uint8 to float32 0~1

class dataloader():
    def __init__(self, batch_size, img_size):
        self.batch_size = batch_size
        self.img_size = img_size
        
    def preprocess(self, image):
        image = tf.image.resize(image, [self.img_size*2, self.img_size*2])
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
        ds = ds.prefetch(buffer_size=AUTOTUNE)
        return ds
