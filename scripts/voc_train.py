import os
import sys

sys.path.append(os.path.normpath(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'src')))
print(sys.path)

from config import DefaultConfig

if DefaultConfig.choosen_backbone is DefaultConfig.Backbone.ResNet50:
    print("current backbone:", DefaultConfig.choosen_backbone)
    raise ValueError("VOC Datset Training only supports MobileNetV2 or VovNet27Slim or ResNet18 as backbone now!")


from model.fcos import FCOSDetector
import torch
from dataset.VOC_dataset import VOCDataset
import math, time
from dataset.augment import Transforms
import os
import numpy as np
import random
import torch.backends.cudnn as cudnn
import argparse
from torch.utils.tensorboard import SummaryWriter

#writer=SummaryWriter(log_dir="/content/drive/MyDrive/FCOS/logs/resnet2")
epochs = 30
batch_size = 8
n_cpu = 4
n_gpu = '0'

os.environ["CUDA_VISIBLE_DEVICES"] = n_gpu
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
cudnn.benchmark = False
cudnn.deterministic = True
random.seed(0)
transform = Transforms()
# train_dataset = VOCDataset(root_dir='/content/VOC2012_train_val',resize_size=[512,800],
#                            split='trainval',use_difficult=False,is_train=True,augment=transform)
train_dataset = VOCDataset(root_dir=DefaultConfig.voc_2012_data_dir_path,resize_size=[512,800],
                           split='trainval',use_difficult=False,is_train=True,augment=transform)

model = FCOSDetector(mode="training").cuda()
model = torch.nn.DataParallel(model)
# model.load_state_dict(torch.load('/mnt/cephfs_new_wj/vc/zhangzhenghao/FCOS.Pytorch/output1/model_6.pth'))

BATCH_SIZE = batch_size
EPOCHS = epochs
#WARMPUP_STEPS_RATIO = 0.12
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                           collate_fn=train_dataset.collate_fn,
                                           num_workers=n_cpu, worker_init_fn=np.random.seed(0))
print("total_images : {}".format(len(train_dataset)))
steps_per_epoch = len(train_dataset) // BATCH_SIZE
TOTAL_STEPS = steps_per_epoch * EPOCHS
WARMPUP_STEPS = 501


LR_INIT = 1e-2
LR_END = 1e-5
optimizer = torch.optim.SGD(model.parameters(),lr =LR_INIT,momentum=0.9,weight_decay=0.0001)

# def lr_func():
#      if GLOBAL_STEPS < WARMPUP_STEPS:
#          lr = GLOBAL_STEPS / WARMPUP_STEPS * LR_INIT
#      else:
#          lr = LR_END + 0.5 * (LR_INIT - LR_END) * (
#              (1 + math.cos((GLOBAL_STEPS - WARMPUP_STEPS) / (TOTAL_STEPS - WARMPUP_STEPS) * math.pi))
#          )
#      return float(lr)


def train_model():
    GLOBAL_STEPS = 1
    model.train()

    for epoch in range(EPOCHS):
        for epoch_step, data in enumerate(train_loader):

            batch_imgs, batch_boxes, batch_classes = data
            batch_imgs = batch_imgs.cuda()
            batch_boxes = batch_boxes.cuda()
            batch_classes = batch_classes.cuda()

            #lr = lr_func()
            if GLOBAL_STEPS < WARMPUP_STEPS:
              lr = float(GLOBAL_STEPS / WARMPUP_STEPS * LR_INIT)
              for param in optimizer.param_groups:
                  param['lr'] = lr
            # if GLOBAL_STEPS == 20001:
            #   lr = LR_INIT * 0.1
            #   for param in optimizer.param_groups:
            #       param['lr'] = lr
            # if GLOBAL_STEPS == 27001:
            #   lr = LR_INIT * 0.01
            #   for param in optimizer.param_groups:
            #       param['lr'] = lr
            else:
              lr=LR_END+0.5*(LR_INIT-LR_END)*(
              (1+math.cos((GLOBAL_STEPS-WARMPUP_STEPS)/(TOTAL_STEPS-WARMPUP_STEPS)*math.pi))
              )
              for param in optimizer.param_groups:
                  param['lr'] = lr

            start_time = time.time()

            optimizer.zero_grad()
            losses = model([batch_imgs, batch_boxes, batch_classes])
            loss = losses[-1]
            loss.mean().backward()
            optimizer.step()

            end_time = time.time()
            cost_time = int((end_time - start_time) * 1000)
            print(
                "global_steps:%d epoch:%d steps:%d/%d cls_loss:%.4f cnt_loss:%.4f reg_loss:%.4f cost_time:%dms lr=%.4e total_loss:%.4f" % \
                (GLOBAL_STEPS, epoch + 1, epoch_step + 1, steps_per_epoch, losses[0].mean(), losses[1].mean(),
                losses[2].mean(), cost_time, lr, loss.mean()))

            # writer.add_scalar("cls_loss",losses[0].mean(),global_step=GLOBAL_STEPS)
            # writer.add_scalar("cnt_loss",losses[1].mean(),global_step=GLOBAL_STEPS)
            # writer.add_scalar("reg_loss",losses[2].mean(),global_step=GLOBAL_STEPS)
            # writer.add_scalar("total_loss",loss.mean(),global_step=GLOBAL_STEPS)
            # writer.add_scalar("lr",lr,global_step=GLOBAL_STEPS)

            GLOBAL_STEPS += 1

        torch.save(model.state_dict(),
                   os.path.join(DefaultConfig.voc_check_points_training_dir_path, "model_{}.pth".format(epoch + 1)))
        # torch.save(model.state_dict(),
        #           "/content/drive/MyDrive/FCOS/checkpoint/resnet2/model_{}.pth".format(epoch + 1))
    # writer.close()
train_model()