'''
@Author: xxxmy
@Github: github.com/VectXmy
@Date: 2019-09-26
@Email: xxxmy@foxmail.com
'''

import cv2
from model.fcos import FCOSDetector
import torch
from torchvision import transforms
import numpy as np
# from dataloader.VOC_dataset import VOCDataset
# from dataloader.COCO_dataset import COCODataset
import time
CLASSES_NAME = (
    '__back_ground__', 'person', 'bicycle', 'car', 'motorcycle',
    'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant',
    'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
    'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle',
    'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli',
    'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet',
    'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
    'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush')
def preprocess_img(image,input_ksize):
    '''
    resize image and bboxes 
    Returns
    image_paded: input_ksize  
    bboxes: [None,4]
    '''
    min_side, max_side    = input_ksize
    h,  w, _  = image.shape

    smallest_side = min(w,h)
    largest_side=max(w,h)
    scale=min_side/smallest_side
    if largest_side*scale>max_side:
        scale=max_side/largest_side
    nw, nh  = int(scale * w), int(scale * h)
    image_resized = cv2.resize(image, (nw, nh))

    pad_w=32-nw%32
    pad_h=32-nh%32

    image_paded = np.zeros(shape=[nh+pad_h, nw+pad_w, 3],dtype=np.float32)
    image_paded[:nh, :nw, :] = image_resized
    return image_paded
    
def convertSyncBNtoBN(module):
    module_output = module
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        module_output = torch.nn.BatchNorm2d(module.num_features,
                                               module.eps, module.momentum,
                                               module.affine,
                                               module.track_running_stats)
        if module.affine:
            module_output.weight.data = module.weight.data.clone().detach()
            module_output.bias.data = module.bias.data.clone().detach()
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
    for name, child in module.named_children():
        module_output.add_module(name,convertSyncBNtoBN(child))
    del module
    return module_output
if __name__=="__main__":
    class Config():
        #backbone
        pretrained=False
        freeze_stage_1=True
        freeze_bn=True

        #fpn
        fpn_out_channels=256
        use_p5=True
        
        #head
        class_num=80
        use_GN_head=True
        prior=0.01
        add_centerness=True
        cnt_on_reg=False

        #training
        strides=[8,16,32,64,128]
        limit_range=[[-1,64],[64,128],[128,256],[256,512],[512,999999]]

        #inference
        score_threshold=0.5
        nms_iou_threshold=0.4
        max_detection_boxes_num=50

    model=FCOSDetector(mode="inference",config=Config)
    # model=torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    # print("INFO===>success convert BN to SyncBN")
    # model = torch.nn.DataParallel(model)
    model=model.cuda().eval()

    model.load_state_dict(torch.load(f"/root/autodl-tmp/res/checkpoints/FCOS_R_50_FPN_1x_my.pth", map_location=torch.device('cuda')))
    # model=convertSyncBNtoBN(model)
    # print("INFO===>success convert SyncBN to BN")
    print("===>success loading model")

    import os

    from torch import nn

    a=torch.rand((3, 832, 992)).unsqueeze_(dim=0).cuda()
    with torch.no_grad():
        out=model(a.unsqueeze_(dim=0))
    
    # #我们枚举整个网络的所有层
    # for i,m in enumerate(model.modules()):
    #     #让网络依次经过和原先结构相同的层，我们就可以获取想要的层的输出
    #     if isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d) or\
    #             isinstance(m, nn.ReLU) or isinstance(m, nn.MaxPool2d) or isinstance(m, nn.AdaptiveAvgPool2d):
    #         print(m)
    #         a= m(a)
    #     #我只想要第一个全连接层的输出
    #     elif isinstance(m, nn.Linear):
    #         print(m)	
    #         #和源代码一样，将他展平成一维的向量
    #         a = torch.flatten(a, 1)
    #         #获取第一个全连接层的输出
    #         a= m(a)
    #         break
    # print(a)
    root="/root/autodl-tmp/fcos/test_images/"
    names=os.listdir(root)
    for name in names:
        img_bgr=cv2.imread(root+name)
        img_pad=preprocess_img(img_bgr,[800,1024])
        # img=cv2.cvtColor(img_pad.copy(),cv2.COLOR_BGR2RGB)
        img=img_pad.copy()
        img_t=torch.from_numpy(img).float().permute(2,0,1)
        img1= transforms.Normalize([102.9801, 115.9465, 122.7717],[1.,1.,1.])(img_t)
        # img1=transforms.ToTensor()(img1)
        # img1= transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225),inplace=True)(img1)
        img1=img1.cuda()
        

        start_t=time.time()
        with torch.no_grad():
            out=model(img1.unsqueeze_(dim=0))
        end_t=time.time()
        cost_t=1000*(end_t-start_t)
        print("===>success processing img, cost time %.2f ms"%cost_t)
        # print(out)
        scores,classes,boxes=out

        boxes=boxes[0].cpu().numpy().tolist()
        classes=classes[0].cpu().numpy().tolist()
        scores=scores[0].cpu().numpy().tolist()

        # scores = scores[scores > score_threshold]
        # classes = classes[:len(scores)]
        # boxes = boxes[:len(scores)]
        import  matplotlib.pyplot as plt
        import matplotlib.patches as patches
        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i) for i in np.linspace(0, 1, 20)]

        for i,box in enumerate(boxes):
            pt1=(int(box[0]),int(box[1]))
            pt2=(int(box[2]),int(box[3]))
            img_pad=cv2.rectangle(img_pad,pt1,pt2,(0,255,0))
            img_pad=cv2.putText(img_pad,"%s %.3f"%(CLASSES_NAME[int(classes[i])],scores[i]),(int(box[0]),int(box[1])+10),cv2.FONT_HERSHEY_SIMPLEX,0.5,[0,200,20],2)
            b_color = colors[int(classes[i]) - 1]
            bbox = patches.Rectangle((box[0],box[1]),width=box[2]-box[0],height=box[3]-box[1],linewidth=1,facecolor='none',edgecolor=b_color)


        cv2.imwrite("/root/autodl-tmp/res/out_images/"+name,img_pad)
    
for i,box in enumerate(boxes):
            pt1=(int(box[0]),int(box[1]))
            pt2=(int(box[2]),int(box[3]))
            img_pad=cv2.rectangle(img_pad,pt1,pt2,(0,255,0))
            b_color = colors[int(classes[i]) - 1]
            bbox = patches.Rectangle((box[0],box[1]),width=box[2]-box[0],height=box[3]-box[1],linewidth=1,facecolor='none',edgecolor=b_color)
            ax.add_patch(bbox)
            plt.text(box[0], box[1], s="%s %.3f"%(VOCDataset.CLASSES_NAME[int(classes[i])],scores[i]), color='white',
                     verticalalignment='top',
                     bbox={'color': b_color, 'pad': 0})


