from functools import partial
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data

from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune import CLIReporter

from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from sklearn.metrics import r2_score

from tst.transformer import Transformer

from tst.loss import OZELoss

from src.dataset import OzeDataset, SingleDimensionalDataset
from src.utils import compute_loss, fit, Logger, kfold
from src.benchmark import LSTM, BiGRU, ConvGru, FFN
from src.metrics import MSE
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties as fp  # 1、引入FontProperties
import math


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


# 设置随机数种子
setup_seed(20)

# 数据集路径选择
# path = 'F:/PyChromProjects/mtsdata/ArabicDigits/ArabicDigits.mat'  #lenth=6600  input=93 channel=13 output=10
# path = 'F:/PyChromProjects/mtsdata/AUSLAN/AUSLAN.mat'  # input=136 d_channel = 22  d_output = 95
# path = 'F:/PyChromProjects/mtsdata/CharacterTrajectories/CharacterTrajectories.mat'    # input=205 channel=3  output=20
# path = 'F:/PyChromProjects/mtsdata/CMUsubject16/CMUsubject16.mat'  # lenth=29,29  input=580 channel=62 output=2
# path = 'F:/PyChromProjects/mtsdata/ECG/ECG.mat'  # lenth=100  input=152 channel=2 output=2
# path = 'F:/PyChromProjects/mtsdata/JapaneseVowels/JapaneseVowels.mat'  # lenth=270  input=29 channel=12 output=9
# path = 'F:/PyChromProjects/mtsdata/Libras/Libras.mat'  # lenth=180  input=45 channel=2 output=15
# path = 'F:/PyChromProjects/mtsdata/UWave/UWave.mat'  # lenth=4278  input=315 channel=3 output=8
# path = 'F:/PyChromProjects/mtsdata/Wafer/Wafer.mat'  # lenth=896  input=198 channel=6 output=2  换

# path = 'F:/PyChromProjects/mtsdata/WalkvsRun/WalkvsRun.mat'  # lenth=28  input=1918 channel=62 output=2  换
# path = 'F:/PyChromProjects/mtsdata/KickvsPunch/KickvsPunch.mat'  # lenth=16  input=841 channel=62 output=2
# path = 'F:/PyChromProjects/mtsdata/NetFlow/NetFlow.mat'  # lenth=803  input=997 channel=4 output=只有1和13
# path = 'F:/PyChromProjects/mtsdata/PEMS/PEMS.mat'  # lenth=267  input=144 channel=963 output=7

# 加载 单维数据集
train_path = 'F:/PyChromProjects/data/UCRArchive_2018/UCRArchive_2018/Adiac/Adiac_TRAIN.tsv'  # 数据集路径
test_path = 'F:/PyChromProjects/data/UCRArchive_2018/UCRArchive_2018/Adiac/Adiac_TEST.tsv'  # 数据集路径

# Config
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device {device}")

# Load 多维dataset
# dataset_train = OzeDataset(path, 'train')
# dataset_test = OzeDataset(path, 'test')

# Load 单维dataset
dataset_train = SingleDimensionalDataset(train_path)
dataset_test = SingleDimensionalDataset(test_path)

# 多维获取相关参数可在 test1.py 中进行测试
# data_length_p = dataset_train.train_len  # 测试集数据量
# d_input = dataset_train.input_len  # 时间部数量
# d_channel = dataset_train.channel_len  # 时间序列维度
# d_output = dataset_train.output_len  # 分类类别

# 单维获取相关参数
data_length_p = dataset_train.data_length_p
d_input = dataset_train.column  # 时间部数量
d_channel = 20  # 时间序列维度
d_output = dataset_train.feature  # 分类类别

EPOCHS = 1000
BATCH_SIZE = 10
LR = 1e-4
optimizer_p = 'Adagrad'  # 优化器

draw_key = 1  # 大于等于draw_key才保存结果图
test_interval = 1  # 调用test()函数的epoch间隔
result_figure_path = 'result_view'  # 保存结果图像的路径
lst_url = train_path.split('/')
data_set_name = lst_url.pop(-2)

# # Model parameters
d_model = 512  # Lattent dim
q = 6  # Query size
v = 6  # Value size
h = 8  # Number of heads
N = 8  # Number of encoder and decoder to stack
dropout = 0.2  # Dropout rate
pe = True  # Positional encoding
mask = True

dataloader_train = Data.DataLoader(dataset=dataset_train, batch_size=BATCH_SIZE, shuffle=True)
dataloader_test = Data.DataLoader(dataset=dataset_test, batch_size=BATCH_SIZE, shuffle=False)

# dataloader_train = Data.DataLoader(dataset=dataset_train, batch_size=BATCH_SIZE, shuffle=True)
# dataloader_test = Data.DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=False)
net = Transformer(d_input, d_output, d_channel, d_model, q, v, h, N,
                  dropout=dropout, pe=pe, mask=mask).to(device)

if optimizer_p == 'Adam':
    optimizer = optim.Adam(net.parameters(), lr=LR)
elif optimizer_p == 'SGD':
    optimizer = optim.SGD(net.parameters(), lr=LR)
elif optimizer_p == 'Adamax':
    optimizer = optim.Adamax(net.parameters(), lr=LR)
elif optimizer_p == 'Adagrad':
    optimizer = optim.Adagrad(net.parameters(), lr=LR)

# 创建损失函数
loss_function = OZELoss()

# 记录准确率，作图用
correct_list = []
correct_list_ontrain = []


def test(dataloader_test, flag='test_set'):
    correct = 0.0
    total = 0
    with torch.no_grad():
        for x_test, y_test in dataloader_test:
            enc_inputs, dec_inputs = x_test.to(device), y_test.to(device)

            test_outputs = net(enc_inputs)

            _, predicted = torch.max(test_outputs.data, dim=1)
            total += dec_inputs.size(0)
            correct += (predicted.float() == dec_inputs.float()).sum().item()
        if flag == 'test_set':
            correct_list.append((100 * correct / total))
        elif flag == 'train_set':
            correct_list_ontrain.append((100 * correct / total))
        # tune.track.log(mean_accuracy=correct / total)
        print(f'Accuracy on {flag}: %f %%' % (100 * correct / total))

    # return correct / total


#  记录损失值 作图用
loss_list = []


# Prepare loss history
# def train_model(config, checkpoint_dir=None, data_dir=None):

for idx_epoch in range(EPOCHS):
    # running_loss = 0.0
    # epoch_steps = 0
    for idx_batch, (x, y) in enumerate(dataloader_train):
        optimizer.zero_grad()

        # Propagate input

        netout = net(x.to(device))

        # Comupte loss
        loss = loss_function(y.to(device), netout)
        print('Epoch:', '%04d' % (idx_epoch + 1), 'loss =', '{:.6f}'.format(loss))
        loss_list.append(loss.item())

        # Backpropage loss
        loss.backward()

        # Update weights
        optimizer.step()



    if ((idx_epoch + 1) % test_interval) == 0:
        test(dataloader_test)
        test(dataloader_train, 'train_set')
        print(f'max test accuracy: %f %%' % (max(correct_list)))


# 结果可视化 包括绘图和结果打印
def result_visualization():
    # my_font = fp(fname=r"C:\windows\Fonts\msyh.ttc")  # 2、设置字体路径

    # 设置风格
    # plt.style.use('ggplot')
    plt.style.use('seaborn')

    fig = plt.figure()  # 创建基础图
    ax1 = fig.add_subplot(311)  # 创建两个子图
    ax2 = fig.add_subplot(313)

    ax1.plot(loss_list)  # 添加折线
    ax2.plot(correct_list, color='red', label='on Test Dataset')
    ax2.plot(correct_list_ontrain, color='blue', label='on Train Dataset')

    # 设置坐标轴标签 和 图的标题
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('loss')
    ax2.set_xlabel(f'epoch/{test_interval}')
    ax2.set_ylabel('correct')
    ax1.set_title('LOSS')
    ax2.set_title('CORRECT')

    plt.legend(loc='best')

    # 设置文本
    fig.text(x=0.13, y=0.4, s=f'min_loss: {min(loss_list)}' '    '
                              f'min_loss to epoch :{math.ceil((loss_list.index(min(loss_list)) + 1) / math.ceil((data_length_p / BATCH_SIZE)))}' '    '
                              f'last_loss:{loss_list[-1]}' '\n'
                              f'max_correct: {max(correct_list)}%' '    '
                              f'max_correct to training_epoch_number:{(correct_list.index(max(correct_list)) + 1) * test_interval}' '    '
                              f'last_correct: {correct_list[-1]}%' '\n'
                              f'd_model={d_model} dataset = {data_set_name}  q={q}   v={v}   h={h}   N={N} drop_out={dropout}' '\n'
             )

    # 保存结果图   测试不保存图（epoch少于draw_key）
    if EPOCHS >= draw_key:
        plt.savefig(
            f'{result_figure_path}/{data_set_name}{max(correct_list)}% {optimizer_p} epoch={EPOCHS} batch={BATCH_SIZE} lr={LR} [{d_model},{q},{v},{h},{N},{dropout}].png')

    # 展示图
    plt.show()
    print('正确率列表', correct_list)

    print(f'最小loss：{min(loss_list)}\r\n'
          f'最小loss对应的epoch数:{math.ceil((loss_list.index(min(loss_list)) + 1) / math.ceil((data_length_p / BATCH_SIZE)))}\r\n'
          f'最后一轮loss:{loss_list[-1]}\r\n')

    print(f'最大correct：{max(correct_list)}\r\n'
          f'最correct对应的已训练epoch数:{(correct_list.index(max(correct_list)) + 1) * test_interval}\r\n'
          f'最后一轮correct:{correct_list[-1]}')

    # print(f'共耗时{round(time_cost, 2)}分钟')


# 调用结果可视化
result_visualization()
