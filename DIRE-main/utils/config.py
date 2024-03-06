import argparse
import os
import sys
from abc import ABC
from typing import Type


class DefaultConfigs(ABC):
    ####### base setting ######
    gpus = [0]
    seed = 3407
    arch = "resnet50"
    datasets = ["zhaolian_train"]
    datasets_test = ["adm_res_abs_ddim20s"]
    mode = "binary"
    class_bal = False
    batch_size = 64
    loadSize = 256
    cropSize = 224
    epoch = "latest"
    num_workers = 20
    serial_batches = False
    isTrain = True

    # data augmentation
    rz_interp = ["bilinear"]
    blur_prob = 0.0
    blur_sig = [0.5]
    jpg_prob = 0.0
    jpg_method = ["cv2"]
    jpg_qual = [75]
    gray_prob = 0.0
    aug_resize = True
    aug_crop = True
    aug_flip = True
    aug_norm = True

    ####### train setting ######
    warmup = False
    warmup_epoch = 3
    earlystop = True
    earlystop_epoch = 5
    optim = "adam"
    new_optim = False
    loss_freq = 400
    save_latest_freq = 2000
    save_epoch_freq = 20
    continue_train = False
    epoch_count = 1
    last_epoch = -1
    nepoch = 400
    beta1 = 0.9
    lr = 0.0001
    init_type = "normal"
    init_gain = 0.02
    pretrained = True

    # paths information
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))#D:\github_local_project\DIRE-main
    #os.path.dirname(__file__)表示当前文件所处的目录，“.."表示上级目录
    #os.path.join(os.path.dirname(__file__), "..")表示当前文件所处目录的上级目录
    #os.path.abspath将相对路径转为绝对路径
    dataset_root = os.path.join(root_dir, "data")#D:\github_local_project\DIRE-main\data
    exp_root = os.path.join(root_dir, "data", "exp")#D:\github_local_project\DIRE-main\data\exp
    _exp_name = ""
    exp_dir = ""
    ckpt_dir = ""
    logs_path = ""
    ckpt_path = ""

    @property
    def exp_name(self):
        return self._exp_name

    @exp_name.setter#可以设置_exp_name的值
    def exp_name(self, value: str):
        self._exp_name = value
        self.exp_dir: str = os.path.join(self.exp_root, self.exp_name)
        #D:\github_local_project\DIRE-main\data\exp\self.exp_name(这里self.exp_name就是self._exp_name)
        self.ckpt_dir: str = os.path.join(self.exp_dir, "ckpt")
        self.logs_path: str = os.path.join(self.exp_dir, "logs.txt")

        os.makedirs(self.exp_dir, exist_ok=True)
        os.makedirs(self.ckpt_dir, exist_ok=True)

    def to_dict(self):
        """
        获取self的属性及其对应的值，并将符合条件的保存到dic字典中
        """
        dic = {}
        for fieldkey in dir(self):
            fieldvalue = getattr(self, fieldkey)
            if not fieldkey.startswith("__") and not callable(fieldvalue) and not fieldkey.startswith("_"):
                dic[fieldkey] = fieldvalue
        return dic


def args_list2dict(arg_list: list):
    """
    将arg_list中偶数和奇数元素成对，并返回一个字典
    如a=[1,'a',2,'b'],则返回{1:'a',2:'b'}
    """
    assert len(arg_list) % 2 == 0, f"Override list has odd length: {arg_list}; it must be a list of pairs"
    #assert在代码中进行断言检查，确保特定条件为真。如果条件为假，assert语句会引发 AssertionError 异常
    #异常会显示Override list has ...语句
    return dict(zip(arg_list[::2], arg_list[1::2]))
    #将arg_list里的偶数元素和奇数元素成对


def str2bool(v: str) -> bool:
    """
    将字符串返回为对应的bool变量
    """
    if isinstance(v, bool):
        return v
    elif v.lower() in ("true", "yes", "on", "y", "t", "1"):
        return True
    elif v.lower() in ("false", "no", "off", "n", "f", "0"):
        return False
    else:
        return bool(v)


def str2list(v: str, element_type=None) -> list:
    """
    如果v不是list,tuple,set,则进行处理返回list
    否则直接返回v
    """
    if not isinstance(v, (list, tuple, set)):
        v = v.lstrip("[").rstrip("]")
        v = v.split(",")
        v = list(map(str.strip, v))
        if element_type is not None:
            v = list(map(element_type, v))
    return v


CONFIGCLASS = Type[DefaultConfigs]


parser = argparse.ArgumentParser()
parser.add_argument("--gpus", default=[0], type=int, nargs="+")
parser.add_argument("--exp_name", default="", type=str)
parser.add_argument("--ckpt", default="model_epoch_latest.pth", type=str)
parser.add_argument("opts", default=[], nargs=argparse.REMAINDER)
args = parser.parse_args()

if os.path.exists(os.path.join(DefaultConfigs.exp_root, args.exp_name, "config.py")):
#检查是否存在指定路径下的配置文件config.py
    sys.path.insert(0, os.path.join(DefaultConfigs.exp_root, args.exp_name))
    #将特定目录添加到模块搜索路径中(并且放置在首位)以确保 Python能够找到并导入该目录中的模块。
    from config import cfg

    cfg: CONFIGCLASS
else:
    cfg = DefaultConfigs()

if args.opts:
    opts = args_list2dict(args.opts)
    for k, v in opts.items():
        if not hasattr(cfg, k):
            raise ValueError(f"Unrecognized option: {k}")
        original_type = type(getattr(cfg, k))
        if original_type == bool:
            setattr(cfg, k, str2bool(v))
        elif original_type in (list, tuple, set):
            setattr(cfg, k, str2list(v, type(getattr(cfg, k)[0])))
        else:
            setattr(cfg, k, original_type(v))

cfg.gpus: list = args.gpus
os.environ["CUDA_VISIBLE_DEVICES"] = ", ".join([str(gpu) for gpu in cfg.gpus])
cfg.exp_name = args.exp_name
cfg.ckpt_path: str = os.path.join(cfg.ckpt_dir, args.ckpt)

if isinstance(cfg.datasets, str):
    cfg.datasets = cfg.datasets.split(",")
