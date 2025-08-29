# BIT2025-B

## 训练数据集配置
下载TrainSet.zip, 解压后得到TrainSet文件夹，文件结果为
```
TrainSet
├── images
│   └── train
│       ├── 0.jpg
│       ├── 1.jpg
│       └── ...
└── labels
    ├── train.txt
    ├── val.txt
    └── trainval.txt
```

## 模型训练
执行`python main.py --dataroot TrainSet`即可训练。使用不同的`--modelroot`可保存不同的模型。
训练结果保存在`./model_save`文件夹中。
默认会将测试结果保存在`./result.txt`文件中。

## 测试数据集配置
下载TestSetB.zip, 解压后得到TestSeB文件夹，文件结果为
```
TestSetB
├── images
│   └── test
│       ├── 0.jpg
│       ├── 1.jpg
│       └── ...
```
## 测试集局部均衡化处理
执行`python pre_test.py`即可

## 模型推理
执行`python main.py --dataroot TestSetB_CLAHE --testonly`即可进行模型推理。测试结果保存在`./result.txt`文件中。


