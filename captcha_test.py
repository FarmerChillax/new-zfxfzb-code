from email import header


# -*- coding: utf-8 -*-
'''
    :file: captcha_test.py
    :author: -Farmer
    :url: https://blog.farmer233.top
    :date: 2022/01/19 15:01:24
'''
import numpy as np
import torch
from torch.autograd import Variable
import setting
import train_model.my_dataset as my_dataset
from train_model.model import CNN
import train_model.one_hot_encoding as one_hot_encoding

def main():
    cnn = CNN()
    cnn.eval()
    cnn.load_state_dict(torch.load('model.pkl'))
    print("load cnn net.")

    test_dataloader = my_dataset.get_test_data_loader()

    correct = 0
    total = 0
    for i, (images, labels) in enumerate(test_dataloader):
        image = images
        vimage = Variable(image)
        predict_label = cnn(vimage)

        c0 = setting.ALL_CHAR_SET[np.argmax(predict_label[0, 0:setting.ALL_CHAR_SET_LEN].data.numpy())]
        c1 = setting.ALL_CHAR_SET[np.argmax(predict_label[0, setting.ALL_CHAR_SET_LEN:2 * setting.ALL_CHAR_SET_LEN].data.numpy())]
        c2 = setting.ALL_CHAR_SET[np.argmax(predict_label[0, 2 * setting.ALL_CHAR_SET_LEN:3 * setting.ALL_CHAR_SET_LEN].data.numpy())]
        c3 = setting.ALL_CHAR_SET[np.argmax(predict_label[0, 3 * setting.ALL_CHAR_SET_LEN:4 * setting.ALL_CHAR_SET_LEN].data.numpy())]

        c4 = setting.ALL_CHAR_SET[np.argmax(predict_label[0, 4 * setting.ALL_CHAR_SET_LEN:5 * setting.ALL_CHAR_SET_LEN].data.numpy())]
        c5 = setting.ALL_CHAR_SET[np.argmax(predict_label[0, 5 * setting.ALL_CHAR_SET_LEN:6 * setting.ALL_CHAR_SET_LEN].data.numpy())]

        predict_label = '%s%s%s%s%s%s' % (c0, c1, c2, c3, c4, c5)
        true_label = one_hot_encoding.decode(labels.numpy()[0])
        total += labels.size(0)
        if(predict_label == true_label):
            correct += 1
        if(total%200==0):
            print('我的垃圾模型在 %d 测试样图中的精度: %f %%' % (total, 100 * correct / total))
    print('Test Accuracy of the model on the %d test images: %f %%' % (total, 100 * correct / total))

if __name__ == '__main__':
    main()


