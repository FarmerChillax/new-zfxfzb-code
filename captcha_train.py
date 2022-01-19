# -*- coding: utf-8 -*-
'''
    :file: captcha_train.py
    :author: -Farmer
    :url: https://blog.farmer233.top
    :date: 2022/01/19 15:01:45
'''
import torch
import torch.nn as nn
from torch.autograd import Variable
import train_model.my_dataset as my_dataset
from train_model.model import CNN

# Hyper Parameters
num_epochs = 30
batch_size = 100
learning_rate = 0.001

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(torch.cuda.is_available())
# device = torch.device("cpu")

def main():
    cnn = CNN().to(device)
    # cnn = CNN()
    cnn.train()
    print('init net')
    criterion = nn.MultiLabelSoftMarginLoss()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)

    # Train the Model
    train_dataloader = my_dataset.get_train_data_loader()
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_dataloader):
            images = Variable(images).to(device)
            labels = Variable(labels.float()).to(device)
            predict_labels = cnn(images)
            loss = criterion(predict_labels, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i+1) % 10 == 0:
                print("epoch:", epoch, "step:", i, "loss:", loss.item())
            if (i+1) % 100 == 0:
                torch.save(cnn.state_dict(), "./model.pkl")   #current is model.pkl
                print("save model")
        print("epoch:", epoch, "step:", i, "loss:", loss.item())
    torch.save(cnn.state_dict(), "./model.pkl", _use_new_zipfile_serialization=False)   #current is model.pkl
    print("save last model")

if __name__ == '__main__':
    main()


