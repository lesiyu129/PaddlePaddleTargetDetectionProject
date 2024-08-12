import paddle.vision.transforms as tf


class DataAugmentationClass:
    def __init__(self, img, size=(224, 224), randomizer=0.75):
        self.img = img
        self.size = size
        self.randomCrop = randomizer
        self.transform = tf.Compose([
            # 对图像调整大小
            # tf.Resize(size=self.size),
            # 基于概率水平翻转图像
            # tf.RandomHorizontalFlip(prob=0.5),
            # 基于概率垂直翻转图像
            # tf.RandomVerticalFlip(prob=0.5),
            # 对图像随机旋转
            # tf.RandomRotation(degrees=(-30, 30)),
            # 对图像随机裁剪
            # tf.RandomCrop(size=(self.size[0]*self.randomCrop, self.size[1]*self.randomCrop)),
            # 对图像进行归一化
            tf.Normalize(mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5], data_format='HWC')
        ])

    def transform_img(self):
        return self.transform(self.img)
