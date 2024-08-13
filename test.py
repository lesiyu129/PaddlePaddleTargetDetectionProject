from utils.DataAugmentationClass import DataAugmentationClass as dAC
import paddle.vision.image
import matplotlib.pyplot as plt
import numpy as np


def coordinate_transformation(box, img_height, img_width):
    x, y, w, h = box
    x = x * img_width
    y = y * img_height
    w = w * img_width
    h = h * img_height
    return np.array([x, y, w, h])


def get_box(x, y, w, h):
    # 左上角坐标
    l_up = ((x - w / 2), (y - h / 2))
    # 左下角坐标
    l_down = ((x - w / 2), (y + h / 2))
    # 右下角坐标
    r_down = ((x + w / 2), (y + h / 2))
    # 右上角坐标
    r_up = ((x + w / 2), (y - h / 2))
    return np.array([l_up, l_down, r_down, r_up])


if __name__ == '__main__':
    boxs = []
    with open('./data/img/1.txt', 'r') as f:
        for line in f.readlines():
            line = line.replace('\n', '')
            list_data = line.split(" ")
            print(list_data)
            for i in range(len(list_data)):
                list_data[i] = float(list_data[i])
            boxs.append(list_data)
    img = paddle.vision.image.image_load('./data/img/1.jpeg')
    img = np.array(img)

    img_dac, boxs, labels = dAC(img, [box[1:] for box in boxs]).transform_img()

    [xx, yy, ww, hh] = coordinate_transformation(boxs[0], img_dac.shape[0], img_dac.shape[1])
    [xx1, yy1, ww1, hh1] = coordinate_transformation(boxs[1], img_dac.shape[0], img_dac.shape[1])
    boxs1 = get_box(xx, yy, ww, hh)
    boxs2 = get_box(xx1, yy1, ww1, hh1)

    plt.figure('img')
    plt.imshow(img_dac)
    # 中心坐标
    plt.plot(xx, yy, 'r*')
    # 左上角坐标
    plt.plot(boxs1[0][0], boxs1[0][1], 'r*')
    # 左下角坐标
    plt.plot(boxs1[1][0], boxs1[1][1], 'r*')
    # 右下角坐标
    plt.plot(boxs1[2][0], boxs1[2][1], 'r*')
    # 右上角坐标
    plt.plot(boxs1[3][0], boxs1[3][1], 'r*')
    # 中心坐标
    plt.plot(xx1, yy1, 'b*')
    # 左上角坐标
    plt.plot(boxs2[0][0], boxs2[0][1], 'b*')
    # 左下角坐标
    plt.plot(boxs2[1][0], boxs2[1][1], 'b*')
    # 右下角坐标
    plt.plot(boxs2[2][0], boxs2[2][1], 'b*')
    # 右上角坐标
    plt.plot(boxs2[3][0], boxs2[3][1], 'b*')

    plt.axis('off')
    plt.title('img')
    plt.show()
