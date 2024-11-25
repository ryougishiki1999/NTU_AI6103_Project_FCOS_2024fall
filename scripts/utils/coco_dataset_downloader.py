import os
import sys
sys.path.append(os.path.normpath(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..', 'src')))

from config import DefaultConfig

import fiftyone as fo
import fiftyone.zoo as foz

dataset_name = "coco-2017"  # 替换为你想要下载的数据集名称
#target_dir = "/root/autodl-tmp/coco_2017"  # 替换为你要下载数据集的目标文件夹路径
target_dir = DefaultConfig.coco_data_dir_path

train_dataset = foz.load_zoo_dataset(
    dataset_name,
    split="train",  # 指定要下载的数据集拆分（如 "train"、"validation" 或 "test"）
    dataset_dir=target_dir,  # 指定下载数据集的目标文件夹
    max_samples=None,  # 指定要下载的最大样本数，设置为 None 表示下载整个数据集
    download_if_necessary=True,  # 如果数据集不存在，则下载它
)

val_dataset = foz.load_zoo_dataset(
    dataset_name,
    split="validation",  # 指定要下载的数据集拆分（如 "train"、"validation" 或 "test"）
    dataset_dir=target_dir,  # 指定下载数据集的目标文件夹
    max_samples=None,  # 指定要下载的最大样本数，设置为 None 表示下载整个数据集
    download_if_necessary=True,  # 如果数据集不存在，则下载它
)

# test_dataset = foz.load_zoo_dataset(
#     dataset_name,
#     split="test",  # 指定要下载的数据集拆分（如 "train"、"validation" 或 "test"）
#     dataset_dir=target_dir,  # 指定下载数据集的目标文件夹
#     max_samples=None,  # 指定要下载的最大样本数，设置为 None 表示下载整个数据集
#     download_if_necessary=True,  # 如果数据集不存在，则下载它
# )