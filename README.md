## DCGAN生成人脸

1. 个人练习DCGAN的一个项目。
2. processed_celeba_small是数据集。
3. train_samples.pkl是训练100轮后的模型。
4. 代码中没有关于实时显示loss的代码，后续应该要加。

### 训练效果如下

**训练十轮**



![epoch_10](G:\PythonProject\DCGAN-test\image_samples\epoch_10.png)

二十轮

![epoch_20](G:\PythonProject\DCGAN-test\image_samples\epoch_20.png)

五十轮

![epcoh_50](G:\PythonProject\DCGAN-test\image_samples\epcoh_50.png)

八十轮

![epoch_80](G:\PythonProject\DCGAN-test\image_samples\epoch_80.png)

一百轮

![epoch_100](G:\PythonProject\DCGAN-test\image_samples\epoch_100.png)

可以看到，五十轮的时候效果最好，后面反而更差了。还不太会调参数，课题组又没GPU给我跑，跑一次要好久，就先这样吧。