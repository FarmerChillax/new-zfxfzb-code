# 基于机器学习识别正方系统验证码
> 本仓库仅存放训练相关代码，封装好的 SDK 请跳转：https://github.com/FarmerChillax/new-school-sdk

正方教务系统验证码识别

![预测结果](img/predict.png)

## 原理概述

通过对网站分析，发现正方的图形验证码是通过 Google 的验证码包进行生成的，因此本库利用 Google 的验证码包生成了大量的数据供训练。测试得到准确率为 99%。

## 相关资料
- 训练数据量: 10w张样图
- 准确率: 99%

![验证结果](img/test.png)

