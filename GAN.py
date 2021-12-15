# import the necessary libraries
import pickle as pkl
import matplotlib.pyplot as plt

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision import datasets
from torchvision import transforms
import torch.optim as optim
from torch.utils.data import DataLoader

'''图像的加载和预处理
1. 调整图像大小 2.将其转换为张量 3.将其加载到PyTorch数据集中 4.将其加载到PyTorch DataLoader
'''

def get_dataloader(batch_size, image_size, data_dir='processed_celeba_small/'):
    # Apply the transformations
    transform = transforms.Compose([transforms.Resize(image_size)
                                       , transforms.ToTensor()])
    # Load the dataset
    # ImageFolder(root, transform=None, target_transform=None, loader=default_loader)
    # root：在root指定路径下找图片； transform：对PIL Image进行的转换操作 target_transfrom:对label的转换 loader：给定路径后如何读取图片，默认读取RGB图片为PIL Image对象
    # 因为我们这里采用无监督学习，所以不需要读入标签
    imagenet_data = datasets.ImageFolder(data_dir, transform=transform)

    # Load the image data into dataloader
    data_loader = DataLoader(imagenet_data,
                                     batch_size,
                                     shuffle=True)  # shuffle为True表示打乱数据
    return data_loader

# Define hyperparameters
batch_size = 32
img_size = 32
data_dir = './processed_celeba_small/'
celeba_train_loader = get_dataloader(batch_size, img_size, data_dir)


# 图像预处理
# 在训练中使用tanh激活器，该生成器的输出在-1到1的范围内，需要在该范围内重新缩放图像
# 此函数将重新缩放放入的图像
def scale(img, feature_range=(-1, 1)):
    """
    Scales the input image into given feature_range
    """
    minoutput, maxoutput = feature_range
    img = img * (maxoutput - minoutput) + minoutput
    return img


'''定义识别器算法
1. 定义一个辅助函数，方便创造卷积网络层
2. 设置判别器
'''
# helper conv function
def conv(in_channels, out_channels, kernel_size, stride=2, padding=1, batch_norm=True):
    layers = []
    conv_layer = nn.Conv2d(in_channels, out_channels,
                           kernel_size, stride, padding, bias=False)

    # Appending the layer
    layers.append(conv_layer)
    # Applying the batch normalization if it's given true
    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    # returning the sequential container
    return nn.Sequential(*layers)  # 列表或元组前面加星号作用是将列表解开成几个独立的参数


class Discriminator(nn.Module):

    def __init__(self, conv_dim):
        super(Discriminator, self).__init__()

        self.conv_dim = conv_dim

        # 32 x 32
        self.cv1 = conv(3, self.conv_dim, 4, batch_norm=False)
        # 16 x 16
        self.cv2 = conv(self.conv_dim, self.conv_dim * 2, 4, batch_norm=True)
        # 4 x 4
        self.cv3 = conv(self.conv_dim * 2, self.conv_dim * 4, 4, batch_norm=True)
        # 2 x 2
        self.cv4 = conv(self.conv_dim * 4, self.conv_dim * 8, 4, batch_norm=True)
        # Fully connected Layer
        self.fc1 = nn.Linear(self.conv_dim * 8 * 2 * 2, 1)

    def forward(self, x):
        # After passing through each layer
        # Applying leaky relu activation function
        x = F.leaky_relu(self.cv1(x), 0.2)
        x = F.leaky_relu(self.cv2(x), 0.2)
        x = F.leaky_relu(self.cv3(x), 0.2)
        x = F.leaky_relu(self.cv4(x), 0.2)
        # To pass throught he fully connected layer
        # We need to flatten the image first
        x = x.view(-1, self.conv_dim * 8 * 2 * 2)
        # Now passing through fully-connected layer
        x = self.fc1(x)
        return x


'''定义生成器算法
1. 创建辅助函数
2. 设置生成器
'''
# helper function
def deconv(in_channels, out_channels, kernel_size, stride=2, padding=1, batch_norm=True):
    layers = []
    convt_layer = nn.ConvTranspose2d(in_channels, out_channels,
                                     kernel_size, stride, padding, bias=False)

    # Appending the above conv layer
    layers.append(convt_layer)

    if batch_norm:
        # Applying the batch normalization if True
        layers.append(nn.BatchNorm2d(out_channels))

    # Returning the sequential container
    return nn.Sequential(*layers)


# 生成器
class Generator(nn.Module):

    def __init__(self, z_size, conv_dim):
        super(Generator, self).__init__()

        self.z_size = z_size

        self.conv_dim = conv_dim

        # fully-connected-layer
        self.fc = nn.Linear(z_size, self.conv_dim * 8 * 2 * 2)
        # 2x2
        self.dcv1 = deconv(self.conv_dim * 8, self.conv_dim * 4, 4, batch_norm=True)
        # 4x4
        self.dcv2 = deconv(self.conv_dim * 4, self.conv_dim * 2, 4, batch_norm=True)
        # 8x8
        self.dcv3 = deconv(self.conv_dim * 2, self.conv_dim, 4, batch_norm=True)
        # 16x16
        self.dcv4 = deconv(self.conv_dim, 3, 4, batch_norm=False)
        # 32 x 32

    def forward(self, x):
        # Passing through fully connected layer
        x = self.fc(x)
        # Changing the dimension
        x = x.view(-1, self.conv_dim * 8, 2, 2)
        # Passing through deconv layers
        # Applying the ReLu activation function
        x = F.relu(self.dcv1(x))
        x = F.relu(self.dcv2(x))
        x = F.relu(self.dcv3(x))
        x = F.tanh(self.dcv4(x))
        # returning the modified image
        return x


# 为了帮助模型更快地收敛，我们将初始化线性和卷积层的权重
def weights_init_normal(m):
    """
       Applies initial weights to certain layers in a model .
       The weights are taken from a normal distribution
       with mean = 0, std dev = 0.02.
       :param m: A module or layer in a network
    """
    classname = m.__class__.__name__  # 获取类名
    # For the linear layers
    if 'Linear' in classname:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        m.bias.data.fill_(0.01)
    # For the convolutional layers
    if 'Conv' in classname or 'BatchNorm2d' in classname:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)



'''构建完整的网络
定义模型的超参数并从上面定义的类中实例化鉴别器和生成器。 确保传入正确的输入参数。
'''
def build_network(d_conv_dim, g_conv_dim, z_size):
    D = Discriminator(d_conv_dim)
    G = Generator(z_size=z_size, conv_dim=g_conv_dim)
    # Applying the weight initialization
    D.apply(weights_init_normal)
    G.apply(weights_init_normal)

    print(D)
    print()
    print(G)

    return D, G

# 定义模型超参数，传入模型初始化网络
# Define model hyperparams
d_conv_dim = 32
g_conv_dim = 32
z_size = 100

D, G = build_network(d_conv_dim, g_conv_dim, z_size)


'''Training in GPU
检查您是否可以在 GPU 上训练。 在这里，我们将其设置为布尔变量 `train_on_gpu`。 稍后，需要确保输入的Models，Models inputs和Loss function arguments是否可以在GPU上运行
'''
# Check for a GPU
train_on_gpu = torch.cuda.is_available()
if not train_on_gpu:
    print('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Training on GPU!')


'''损失
1. 鉴别器：d_loss = d_real_loss + d_fake_loss （鉴别器为真实图像输出1，伪图像输出0）
2. 定义两个损失函数，一个是实际损失，另一个是伪造损失
3. 生成器：仅在标签翻转的情况下，发电机损耗才会看起来相似。 生成器的目标是让鉴别器认为其生成的图像是真实的。
'''
device = "cuda:0" if torch. cuda.is_available() else "cpu"

def real_loss(D_out, smooth=False):
    batch_size = D_out.size(0)
    if smooth:
        labels = torch.ones(batch_size) * 0.9
    else:
        labels = torch.ones(batch_size)

    labels = labels.to(device)
    criterion = nn.BCEWithLogitsLoss()
    loss = criterion(D_out.squeeze(), labels)
    return loss


def fake_loss(D_out):
    batch_size = D_out.size(0)
    labels = torch.zeros(batch_size)
    labels = labels.to(device)
    criterion = nn.BCEWithLogitsLoss()
    loss = criterion(D_out.squeeze(), labels)
    return loss


'''定义优化器'''
# 超参数的设置是根据研究论文设置的，已经通过尝试证明了这是最好的
# 这里使用Adam优化器进行训练，可以选择其他的
lr = 0.0002
beta1 = 0.3
beta2 = 0.999  # default value
# Create optimizers for the discriminator D and generator G
d_optimizer = optim.Adam(D.parameters(), lr, betas=(beta1, beta2))
g_optimizer = optim.Adam(G.parameters(), lr, betas=(beta1, beta2))

'''训练'''


def train(D, G, n_epochs, print_every=100):
    '''Trains adversarial networks for some number of epochs
       param, D: the discriminator network
       param, G: the generator network
       param, n_epochs: number of epochs to train for
       param, print_every: when to print and record the models' losses
       return: D and G losses'''

    # move models to GPU
    if train_on_gpu:
        D.cuda()
        G.cuda()

    # keep track of loss and generated, "fake" samples
    samples = []
    losses = []

    # Get some fixed data for sampling. These are images that are held
    # constant throughout training, and allow us to inspect the model's performance
    sample_size = 16
    fixed_z = np.random.uniform(-1, 1, size=(sample_size, z_size))
    fixed_z = torch.from_numpy(fixed_z).float()
    # move z to GPU if available
    if train_on_gpu:
        fixed_z = fixed_z.cuda()

    # epoch training loop
    for epoch in range(n_epochs):

        # batch training loop
        for batch_i, (real_images, _) in enumerate(celeba_train_loader):

            batch_size = real_images.size(0)
            real_images = scale(real_images)

            # ===============================================
            #         YOUR CODE HERE: TRAIN THE NETWORKS
            # ===============================================

            # 1. Train the discriminator on real and fake images
            d_optimizer.zero_grad()

            # real images
            real_images = real_images.to(device)

            dreal = D(real_images)
            dreal_loss = real_loss(dreal)

            # fake images

            # Generate fake images
            z = np.random.uniform(-1, 1, size=(batch_size, z_size))
            z = torch.from_numpy(z).float()
            # move x to GPU, if available
            z = z.to(device)
            fake_images = G(z)

            # loss of fake images
            dfake = D(fake_images)
            dfake_loss = fake_loss(dfake)

            # Adding both lossess
            d_loss = dreal_loss + dfake_loss
            # Backpropogation step
            d_loss.backward()
            d_optimizer.step()

            # 2. Train the generator with an adversarial loss
            g_optimizer.zero_grad()

            # Generate fake images
            z = np.random.uniform(-1, 1, size=(batch_size, z_size))
            z = torch.from_numpy(z).float()
            z = z.to(device)
            fake_images = G(z)

            # Compute the discriminator losses on fake images
            # using flipped labels!
            D_fake = D(fake_images)
            g_loss = real_loss(D_fake, True)  # use real loss to flip labels

            # perform backprop
            g_loss.backward()
            g_optimizer.step()

            # ===============================================
            #              END OF YOUR CODE
            # ===============================================

            # Print some loss stats
            if batch_i % print_every == 0:
                # append discriminator loss and generator loss
                losses.append((d_loss.item(), g_loss.item()))
                # print discriminator and generator loss
                print('Epoch [{:5d}/{:5d}] | d_loss: {:6.4f} | g_loss: {:6.4f}'.format(
                    epoch + 1, n_epochs, d_loss.item(), g_loss.item()))

        ## AFTER EACH EPOCH##
        # this code assumes your generator is named G, feel free to change the name
        # generate and save sample, fake images
        G.eval()  # for generating samples
        samples_z = G(fixed_z)
        samples.append(samples_z)
        G.train()  # back to training mode

    # Save training generator samples
    with open('train_samples_new.pkl', 'wb') as f:  #pkl常用于保存神经网络的模型或各种需要存储的数据；wb是覆盖写，如果需要追加，则为'ab'
        pkl.dump(samples, f)  # 保存生成的图片样本

    # finally return losses
    return losses


'''设置train epochs,然后训练GAN'''
# set number of epochs
n_epochs = 100

# call training function
losses = train(D, G, n_epochs=n_epochs)

'''画出generator和discriminator的曲线'''
fig, ax = plt.subplots()
losses = np.array(losses)
plt.plot(losses.T[0], label='Discriminator', alpha=0.5)
plt.plot(losses.T[1], label='Generator', alpha=0.5)
plt.title("Training Losses")
plt.legend()
plt.savefig("./loss.jpg")
plt.show()