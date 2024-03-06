"""
demo.py做了如下工作，将训练好的一个分类器（分辨图像是否合成）加载到模型model(该模型在这里用的是resnet)中，
用此模型将指定文件/文件夹中的图片进行辨别，最终输出该图片是合成的置信度
"""

import argparse
import glob
import os

import torch
import torch.nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm

from utils.utils import get_network, str2bool, to_cuda


#主要是进行一个命令行参数的操作
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "-f", "--file", default="data/test/lsun_adm/1_fake/0.png", type=str, help="path to image file or directory of images"
)
parser.add_argument(
    "-m",
    "--model_path",
    type=str,
    default="data/exp/ckpt/lsun_adm/model_epoch_latest.pth",
)
parser.add_argument("--use_cpu", action="store_true", help="uses gpu by default, turn on to use cpu")
parser.add_argument("--arch", type=str, default="resnet50")
parser.add_argument("--aug_norm", type=str2bool, default=True)

args = parser.parse_args()



#该模块主要是把文件路径放到一个file_list的列表里
if os.path.isfile(args.file):#检查args.file路径是否对应一个文件
    print(f"Testing on image '{args.file}'")
    file_list = [args.file]
elif os.path.isdir(args.file):#检查args.file指定的路径是否是一个目录（文件夹）
    file_list = sorted(glob.glob(os.path.join(args.file, "*.jpg")) + glob.glob(os.path.join(args.file, "*.png"))+glob.glob(os.path.join(args.file, "*.JPEG")))
    #获取args.file目录中所有jpg png JPEG文件
    print(f"Testing images from '{args.file}'")
else:
    raise FileNotFoundError(f"Invalid file path: '{args.file}'")


#加载已经训练好的参数权重到model中
model = get_network(args.arch)
state_dict = torch.load(args.model_path, map_location="cpu")#从指定路径加载模型，state_dict将包含加载的模型的权重和参数
if "model" in state_dict:
    state_dict = state_dict["model"]
model.load_state_dict(state_dict)#model必须与state_dict中的参数兼容，以便正确加载参数。
#模型 model 的参数将被更新为 state_dict 中包含的参数值。这通常用于加载预训练模型权重，以便在后续任务中使用这些权重进行微调或进行推理。
model.eval()
if not args.use_cpu:
    model.cuda()

print("*" * 50)

trans = transforms.Compose(
    (
        transforms.Resize(256),# 将输入图像的大小调整为256x256像素
        transforms.CenterCrop(224),# 在256x256的图像上进行中心裁剪，将图像裁剪为224x224像素大小
        transforms.ToTensor(),# 将图像从PIL图像格式转换为PyTorch张量（tensor）格式
    )
)
#transforms.Compose函数来创建一个图像预处理的管道。这个管道将一系列图像变换操作按顺序应用于输入图像。

for img_path in tqdm(file_list, dynamic_ncols=True, disable=len(file_list) <= 1):
#tqdm显示遍历的进度条，dynamic_ncols表示进度条的列数是否自动调整以适应终端窗口的大小，disable这个表示如果file_list中的元素数量不大于1，那么就禁用进度条，不显示进度信息。
    img = Image.open(img_path).convert("RGB")
    img = trans(img)
    if args.aug_norm:
        img = TF.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    in_tens = img.unsqueeze(0)
    #img.unsqueeze(0): 这是 img 张量上的操作，使用了 PyTorch 的 unsqueeze 方法。
    #在括号中的参数 0 表示在索引 0 的位置（即最前面）添加一个新的维度。这将使原始的三维图
    #像数据变成一个四维张量，其中第一个维度用于表示批处理中的样本数量。
    if not args.use_cpu:
        in_tens = in_tens.cuda()

    with torch.no_grad():
        prob = model(in_tens).sigmoid().item()
    print(f"Prob of being synthetic: {prob:.4f}")
