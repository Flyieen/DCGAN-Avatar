# helper function for viewing a list of passed in sample images
import numpy as np
from matplotlib import pyplot as plt
import pickle as pkl



def view_samples(epoch, samples):
    fig, axes = plt.subplots(figsize=(16, 4), nrows=2, ncols=8, sharey=True, sharex=True) #nrow表示多少行，ncols表示多少列；sharey和sharex默认为false，表示画布中的所有ax都独立，都为true表示ax共享x轴和y轴
    for ax, img in zip(axes.flatten(), samples[epoch]):  # zip函数将对应的元素打包成一个个元组 参考https://www.runoob.com/python/python-func-zip.html
        img = img.detach().cpu().numpy()
        img = np.transpose(img, (1, 2, 0))  #img此时为float类型
        img = ((img + 1) * 255 / (2)).astype(np.uint8)  #转换成unit8，unit8是专门用于存储各种图像的，范围是0-255
        ax.xaxis.set_visible(False)  # 隐藏x坐标轴
        ax.yaxis.set_visible(False)
        img = img.reshape(32, 32, 3)  # unit8是数组格式，需要reshape成32*32*3的图片然后展示
        im = ax.imshow(img)





# Load samples from generator, taken while training
with open('train_samples.pkl', 'rb') as f:
    samples = pkl.load(f)

epoch = -1
_ = view_samples(epoch, samples)
plt.suptitle("epoch = {}".format(epoch))
plt.show()
