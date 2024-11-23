import fiftyone as fo
import fiftyone.zoo as foz
from config import DefaultConfig
dataset_name = "coco-2017"  # 替换为你想要下载的数据集名称
#target_dir = "/root/autodl-tmp/coco_2017"  # 替换为你要下载数据集的目标文件夹路径
target_dir = DefaultConfig.coco_path

dataset = foz.load_zoo_dataset(
    dataset_name,
    split="train",  # 指定要下载的数据集拆分（如 "train"、"validation" 或 "test"）
    dataset_dir=target_dir,  # 指定下载数据集的目标文件夹
    max_samples=None,  # 指定要下载的最大样本数，设置为 None 表示下载整个数据集
    download_if_necessary=True,  # 如果数据集不存在，则下载它
)

dataset = foz.load_zoo_dataset(
    dataset_name,
    split="validation",  # 指定要下载的数据集拆分（如 "train"、"validation" 或 "test"）
    dataset_dir=target_dir,  # 指定下载数据集的目标文件夹
    max_samples=None,  # 指定要下载的最大样本数，设置为 None 表示下载整个数据集
    download_if_necessary=True,  # 如果数据集不存在，则下载它
)

dataset = foz.load_zoo_dataset(
    dataset_name,
    split="test",  # 指定要下载的数据集拆分（如 "train"、"validation" 或 "test"）
    dataset_dir=target_dir,  # 指定下载数据集的目标文件夹
    max_samples=None,  # 指定要下载的最大样本数，设置为 None 表示下载整个数据集
    download_if_necessary=True,  # 如果数据集不存在，则下载它
)

