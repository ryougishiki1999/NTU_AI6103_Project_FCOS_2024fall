import os
import sys
sys.path.append(os.path.normpath(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'src')))

from config import DefaultConfig

if DefaultConfig.choosen_backbone is not DefaultConfig.Backbone.ResNet50:
    print("current backbone:", DefaultConfig.choosen_backbone)
    raise ValueError("CoCo Datset Training only supports RestNet50 as backbone now!")

from tabnanny import check
from model.fcos import FCOSDetector
import torch
from dataset.COCO_dataset import COCODataset
import math,time
from dataset.augment import Transforms
import numpy as np
import random
import torch.backends.cudnn as cudnn
import argparse
from torch.utils.tensorboard import SummaryWriter
from config import DefaultConfig

os.environ["CUDA_VISIBLE_DEVICES"]=DefaultConfig.gpu_id

train_dataset = COCODataset(
    DefaultConfig.coco_train_data_path,
    DefaultConfig.coco_train_label_path,
    transform=Transforms()
)

model=FCOSDetector(mode="training").cuda()
model = torch.nn.DataParallel(model)
batch_size=DefaultConfig.batch_size
epochs=DefaultConfig.epochs
train_loader=torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True,collate_fn=train_dataset.collate_fn,
                                         num_workers=DefaultConfig.num_workers,worker_init_fn = np.random.seed(0))
steps_per_epoch=len(train_dataset)//batch_size
total_steps=steps_per_epoch*epochs
warmup_stps=500
warmup_factor = 1.0 / 3.0
steps=0
init_lr=0.01
optimizer = torch.optim.SGD(model.parameters(),lr =init_lr,momentum=0.9,weight_decay=0.0001)
lr_schedule = [120000, 160000]
def lr_decrease(step):
    lr = init_lr
    if step < warmup_stps:
        alpha = float(step) / warmup_stps
        warmup_factor = warmup_factor * (1.0 - alpha) + alpha
        lr = lr*warmup_factor
    else:
        for i in range(len(lr_schedule)):
            if step < lr_schedule[i]:
                break
            lr *= 0.1
    return float(lr)

checkpoint_path = DefaultConfig.coco_train_eval_check_point_path


if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    total_cls_losses = checkpoint['total_cls_losses']
    total_cnt_losses = checkpoint['total_cnt_losses']
    total_reg_losses = checkpoint['total_reg_losses']
    total_losses_avg = checkpoint['total_losses_avg']

    steps = checkpoint['global_steps']
    print(f"{'-'*30}\nReveal from checkpoint: epoch={start_epoch}, global_steps={steps}{'-'*30}\n")
else:
    start_epoch = 0
    total_cls_losses = []
    total_cnt_losses = []
    total_reg_losses = []
    total_losses_avg = []
    print(f"{'-'*30}\nNo checkpint found, training from scratch{'-'*30}")

model.train()

for epoch in range(start_epoch, epochs): 
    cls_losses = []
    cnt_losses = []
    reg_losses = []
    losses_avg = []
    for epoch_step,data in enumerate(train_loader):

        imgs_per_batch,boxes_per_batch,classes_per_batch=data
        imgs_per_batch=imgs_per_batch.cuda()
        boxes_per_batch=boxes_per_batch.cuda()
        classes_per_batch=classes_per_batch.cuda()

        lr = lr_decrease(steps)
        for param in optimizer.param_groups:
            param['lr']=lr
        
        start_time=time.time()

        optimizer.zero_grad()
        losses=model([imgs_per_batch,boxes_per_batch,classes_per_batch])
        loss=losses[-1]
        loss.mean().backward()
        torch.nn.utils.clip_grad_norm(model.parameters(),3)
        optimizer.step()

        end_time=time.time()
        cost_time=int((end_time-start_time)*1000)
        cls_losses.append(losses[0].mean().item())
        cnt_losses.append(losses[1].mean().item())
        reg_losses.append(losses[2].mean().item())
        losses_avg.append(loss.mean().item())
        print("global_steps:%d epoch:%d steps:%d/%d cls_loss:%.4f cnt_loss:%.4f reg_loss:%.4f cost_time:%dms lr=%.4e total_loss:%.4f"%\
            (steps,epoch+1,epoch_step+1,steps_per_epoch,losses[0].mean(),losses[1].mean(),losses[2].mean(),cost_time,lr, loss.mean()))
        # save loss
        steps+=1

    total_cls_losses.extend(cls_losses)
    total_cnt_losses.extend(cnt_losses)
    total_reg_losses.extend(reg_losses)
    total_losses_avg.extend(losses_avg)
    checkpoint = {
        'epoch': epoch+1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'global_steps': steps,
        'total_cls_losses': total_cls_losses,
        'total_cnt_losses': total_cnt_losses,
        'total_reg_losses': total_reg_losses,
        'total_losses_avg': total_losses_avg
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"{'-'*30}\nRunning successful! Save checkpoint: epoch={epoch}, global_steps={steps}{'-'*30}")
    import numpy as np
import matplotlib.pyplot as plt

def downsample_then_smooth(data, sample_factor=100, smooth_factor=0.9):

    downsampled = data[::sample_factor]
    
    smoothed = []
    for point in downsampled:
        if smoothed:
            previous = smoothed[-1]
            smoothed.append(previous * smooth_factor + point * (1 - smooth_factor))
        else:
            smoothed.append(point)
            
    return np.array(smoothed)


smooth_classification = downsample_then_smooth(total_cls_losses, sample_factor=100, smooth_factor=0.9)
smooth_centerness = downsample_then_smooth(total_cnt_losses, sample_factor=100, smooth_factor=0.9)
smooth_regression = downsample_then_smooth(total_reg_losses, sample_factor=100, smooth_factor=0.9)
smooth_total = downsample_then_smooth(total_losses_avg, sample_factor=100, smooth_factor=0.9)

plt.figure(figsize=(15, 10))

# Classification Loss
plt.subplot(2, 2, 1)
plt.plot(smooth_classification, 'b-', label='Smoothed')
plt.title('Classification Loss')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()

# Centerness Loss
plt.subplot(2, 2, 2)
plt.plot(smooth_centerness, 'b-', label='Smoothed')
plt.title('Centerness Loss')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()

# Regression Loss
plt.subplot(2, 2, 3)
plt.plot(smooth_regression, 'b-', label='Smoothed')
plt.title('Regression Loss')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()

# Total Loss
plt.subplot(2, 2, 4)
plt.plot(smooth_total, 'b-', label='Smoothed')
plt.title('Total Loss (Avg)')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig(DefaultConfig.coco_out_loss_path)