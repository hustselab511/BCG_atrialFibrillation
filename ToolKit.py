import torch
from matplotlib import pyplot as plt
import numpy as np
import pywt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm


class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        return torch.tensor(sample, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


# return (n 1 d)
def readPthData(path):
    data = torch.load(path, weights_only=True)
    return data


def produceLabel(data, labelVal):
    assert labelVal == 0 or labelVal == 1
    label = torch.zeros(data.shape[0], data.shape[1], dtype=torch.long) + labelVal
    return label


# （n 1 d） -> cwt （n 3 imageSize imageSize）
def transferAllDataToCWT(signals, scale=512, wavelet='morl', fs=125, ylim=[0, 20], imageSize=224):
    # 利用JIT加速numpy计算过程
    images = [transferCWT(signal.squeeze(0).numpy(), scale, wavelet,
                          fs, ylim, imageSize) for signal in tqdm(signals, 'GET CWT Image')]
    images_arrays = np.stack(images)
    result = torch.tensor(images_arrays)
    return result


# 读取单条数据（1 1 d） 转换为cwt频谱图 （1 3 d）
# 原始信号、小波尺度、小波基、频率、纵坐标频率范围
# 代码示例见cwtExample
def transferCWT(signal, scale=512, wavelet='morl', fs=125, ylim=[0, 20], imageSize=224):
    # 计算当前使用小波基的中心频率（不同小波基每个尺度具有不同频率）
    fc = pywt.central_frequency(wavelet)
    # 计算频率尺度参数：
    cparam = 2 * fc * scale
    # 生成频率尺度向量
    scales = cparam / np.arange(scale, 1, -1)
    # 分解得到小波系数cwtmatr（尺度个数 x 信号长度） 表示各尺度下各个信号位置对应的系数大小
    # 频率系数frequencies
    [cwtmatr, frequencies] = pywt.cwt(signal, scales, wavelet, 1.0 / fs)

    fig = Figure(figsize=(8, 8))
    canvas = FigureCanvas(fig)

    # 绘制 CWT 结果(注意，这里为了保存图的时候只保存图本体，所以将所有图注关闭)
    ax = fig.add_subplot(111)
    t = np.arange(len(signal)) / fs  # 时间向量
    ax.contourf(t, frequencies, abs(cwtmatr))

    ax.set_ylim(ylim)
    ax.axis('off')  # 关闭坐标轴
    fig.tight_layout(pad=0)  # 调整布局，去除边距

    # 渲染图形
    canvas.draw()

    # 获取图像数据
    width, height = fig.canvas.get_width_height()
    image = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(height, width, 4)[..., :3]

    # 将图像数据转换为 PIL 图像
    pil_image = Image.fromarray(image)
    # 调整图像大小
    resized_image = pil_image.resize([imageSize, imageSize], Image.Resampling.LANCZOS)
    # 将调整后的图像转换回 numpy 数组
    resized_image = np.array(resized_image)

    # image 维度为（h w c） 转换为（c w h）
    image = np.transpose(resized_image, (2, 0, 1))

    return image


# 将transferCWT得到的image（c w h）进行展示
def showCWTNumpy(image):
    image = np.transpose(image, (1, 2, 0))
    # 绘制图像
    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    plt.title('CWT Image')
    plt.axis('off')  # 关闭坐标轴
    plt.show()


# 转换cwt频谱图示例
def cwtExample(signal):
    image = transferCWT(signal)
    print(image.shape)
    image = np.transpose(image, (1, 2, 0))
    # 绘制图像
    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    plt.axis('off')  # 关闭坐标轴
    plt.show()


def flatten_data(data):
    return data.reshape(data.shape[0] * data.shape[1], 1, data.shape[-1])
