import matplotlib
import numpy as np
import torch
from matplotlib import pyplot as plt
from torchvision import datasets
from torchvision.transforms import transforms


def get_dataloader(batch_size, image_size, data_dir='processed_celeba_small/'):
    """
    Batch the neural network data using DataLoader
    :param batch_size: The size of each batch; the number of images in a batch
    :param img_size: The square size of the image data (x, y)
    :param data_dir: Directory where image data is located
    :return: DataLoader with batched data
    """

    # TODO: Implement function and return a dataloader
    transform = transforms.Compose([transforms.Resize(image_size)
                                       , transforms.ToTensor()])

    imagenet_data = datasets.ImageFolder(data_dir, transform=transform)

    data_loader = torch.utils.data.DataLoader(imagenet_data,
                                              batch_size,
                                              shuffle=True)
    return data_loader


# Define function hyperparameters
batch_size = 32
img_size = 32

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
# Call your function and get a dataloader
celeba_train_loader = get_dataloader(batch_size, img_size)


# helper display function
def imshow(img):
    npimg = img.numpy()  #torch.tensor()类型转换成numpy数组类型
    plt.imshow(np.transpose(npimg, (1, 2, 0)))  # 有三维，0表示x，1表示y，2表示z； （x,y,z）->(y,z,x)

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
# obtain one batch of training images
dataiter = iter(celeba_train_loader)
images, _ = dataiter.next() # _ for no labels


# plot the images in the batch, along with the corresponding labels
fig = plt.figure(figsize=(20, 4))  #fig是图，figsize设置图形的大小，a为宽，b为高
plot_size=20
for idx in np.arange(plot_size):
    # add_subplots(1,3,1) 第一个参数 1 是子图的航叔，第二个参数 3 是子图的列数， 第三个参数 1 代表是第一个子图
    ax = fig.add_subplot(2, plot_size/2, idx+1, xticks=[], yticks=[])  #xtricks为空表示移除所有x轴刻度
    imshow(images[idx])
plt.show()