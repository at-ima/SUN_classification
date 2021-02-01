import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing

class aug_process(tf.keras.Model):
    def __init__(self, img_size):
        super(aug_process, self).__init__(name='')
        self.rand_trans = preprocessing.RandomTranslation((-0.2, 0.3), (-0.2, 0.3))
        self.rand_rot = preprocessing.RandomRotation((-1, 1))
        self.rand_crop = preprocessing.RandomCrop(img_size, img_size)
        self.rand_cont = preprocessing.RandomContrast(0.2, 0.3)
        
    def call(self, image):
        image = self.rand_trans(image)
        image = self.rand_rot(image)
        image = self.rand_crop(image)
        image = self.rand_cont(image)
        return image