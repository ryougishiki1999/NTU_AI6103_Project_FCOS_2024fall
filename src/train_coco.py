from tabnanny import check
from model.fcos import FCOSDetector
import torch
from dataset.COCO_dataset import COCODataset
import math,time
from dataset.augment import Transforms
import os
import numpy as np
import random
import torch.backends.cudnn as cudnn
import argparse
from torch.utils.tensorboard import SummaryWriter
from config import DefaultConfig

writer = SummaryWriter('/root/tf-logs')

parser = argparse.ArgumentParser()
# parser.add_argument("--epochs", type=int, default=24, help="number of epochs")
parser.add_argument("--epochs", type=int, default=24, help="number of epochs")

parser.add_argument("--batch_size", type=int, default=4, help="size of each image batch")
parser.add_argument("--n_cpu", type=int, default=4, help="number of cpu threads to use during batch generation")
parser.add_argument("--n_gpu", type=str, default='0', help="number of cpu threads to use during batch generation")
opt = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"]=opt.n_gpu
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
cudnn.benchmark = False
cudnn.deterministic = True

random.seed(0)
transform = Transforms()

# train_dataset=COCODataset("/root/autodl-tmp/coco_2017/train/data",
#                           '/root/autodl-tmp/coco_2017/train/labels.json',transform=transform)

train_dataset = COCODataset(
    DefaultConfig.coco_train_data_path,
    DefaultConfig.coco_train_label_path,
    transform=transform
)

model=FCOSDetector(mode="training").cuda()
model = torch.nn.DataParallel(model)
BATCH_SIZE=opt.batch_size
EPOCHS=opt.epochs
train_loader=torch.utils.data.DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True,collate_fn=train_dataset.collate_fn,
                                         num_workers=opt.n_cpu,worker_init_fn = np.random.seed(0))
steps_per_epoch=len(train_dataset)//BATCH_SIZE
TOTAL_STEPS=steps_per_epoch*EPOCHS
WARMUP_STEPS=500
WARMUP_FACTOR = 1.0 / 3.0
GLOBAL_STEPS=0
LR_INIT=0.01
optimizer = torch.optim.SGD(model.parameters(),lr =LR_INIT,momentum=0.9,weight_decay=0.0001)
lr_schedule = [120000, 160000]
def lr_func(step):
    lr = LR_INIT
    if step < WARMUP_STEPS:
        alpha = float(step) / WARMUP_STEPS
        warmup_factor = WARMUP_FACTOR * (1.0 - alpha) + alpha
        lr = lr*warmup_factor
    else:
        for i in range(len(lr_schedule)):
            if step < lr_schedule[i]:
                break
            lr *= 0.1
    return float(lr)
# 检查点文件的路径
#checkpoint_path = "/root/autodl-tmp/res/checkpoints/coco_model.pth"
#checkpoint_path = DefaultConfig.self_check_point_path
checkpoint_path = DefaultConfig.tmp_check_point_path


# 加载检查点
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    total_cls_losses = checkpoint['total_cls_losses']
    total_cnt_losses = checkpoint['total_cnt_losses']
    total_reg_losses = checkpoint['total_reg_losses']
    total_losses_avg = checkpoint['total_losses_avg']

    GLOBAL_STEPS = checkpoint['global_steps']
    print(f"从检查点恢复训练: epoch={start_epoch}, global_steps={GLOBAL_STEPS}")
else:
    start_epoch = 0
    total_cls_losses = []
    total_cnt_losses = []
    total_reg_losses = []
    total_losses_avg = []
    print("未找到检查点,从头开始训练")

model.train()

for epoch in range(start_epoch, EPOCHS):
    # counter = 0 
    # create three list for losses 
    cls_losses = []
    cnt_losses = []
    reg_losses = []
    losses_avg = []
    for epoch_step,data in enumerate(train_loader):
        # counter += 1
        # if counter > 10:
        #     break
        batch_imgs,batch_boxes,batch_classes=data
        batch_imgs=batch_imgs.cuda()
        batch_boxes=batch_boxes.cuda()
        batch_classes=batch_classes.cuda()

        lr = lr_func(GLOBAL_STEPS)
        for param in optimizer.param_groups:
            param['lr']=lr
        
        start_time=time.time()

        optimizer.zero_grad()
        losses=model([batch_imgs,batch_boxes,batch_classes])
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
            (GLOBAL_STEPS,epoch+1,epoch_step+1,steps_per_epoch,losses[0].mean(),losses[1].mean(),losses[2].mean(),cost_time,lr, loss.mean()))
        # save loss

        
        # if GLOBAL_STEPS % 1000 == 0:
        # writer.add_scalar('Loss/train', loss.mean(), GLOBAL_STEPS)
        # writer.add_scalar('Loss/cls_loss', losses[0].mean(), GLOBAL_STEPS)
        # writer.add_scalar('Loss/cnt_loss', losses[1].mean(), GLOBAL_STEPS)
        # writer.add_scalar('Loss/reg_loss', losses[2].mean(), GLOBAL_STEPS)
        # writer.add_scalar('LearningRate', lr, GLOBAL_STEPS)
        GLOBAL_STEPS+=1
    total_cls_losses.extend(cls_losses)

    total_cnt_losses.extend(cnt_losses)
    total_reg_losses.extend(reg_losses)
    total_losses_avg.extend(losses_avg)
    checkpoint = {
        'epoch': epoch+1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'global_steps': GLOBAL_STEPS,
        'total_cls_losses': total_cls_losses,
        'total_cnt_losses': total_cnt_losses,
        'total_reg_losses': total_reg_losses,
        'total_losses_avg': total_losses_avg
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"已保存检查点: epoch={epoch}, global_steps={GLOBAL_STEPS}")
        # torch.save(model.state_dict(),"/root/autodl-tmp/res/checkpoints/coco_model.pth".format(epoch+1))
            
    
    
    
writer.close()







