from matplotlib.ticker import NullLocator
import cv2
from model.fcos import FCOSDetector
import torch
from torchvision import transforms
import numpy as np
# from dataloader.VOC_dataset import VOCDataset
# from dataloader.COCO_dataset import COCODataset
import time
from config import DefaultConfig
import glob
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
        score_threshold=0.3
        nms_iou_threshold=0.4
        max_detection_boxes_num=50

    model=FCOSDetector(mode="inference",config=Config)
    # model=torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    # print("INFO===>success convert BN to SyncBN")
    #model = torch.nn.DataParallel(model)
    model=model.cuda().eval()
        
    model.load_state_dict(torch.load(f"/root/autodl-tmp/res/checkpoints/FCOS_R_50_FPN_1x_my.pth", map_location=torch.device('cuda')))
    #model.load_state_dict(torch.load(DefaultConfig.self_check_point_path, map_location=torch.device('cuda')) ['model_state_dict'])
    # model=convertSyncBNtoBN(model)
    # print("INFO===>success convert SyncBN to BN")
    print("===>success loading model")

    import os
    #root="/root/autodl-tmp/fcos/test_images/"
    root = DefaultConfig.test_images_path
    print(root)
    #names=os.listdir(root)
    names=glob.glob(os.path.join(root, "*.jpg"))
    names = [os.path.basename(name) for name in names]
    print(names)
    for name in names:
        #img_bgr=cv2.imread(root+name)
        img_bgr=cv2.imread(os.path.join(root, name))
        img_pad=preprocess_img(img_bgr,[800,1024])
        # img=cv2.cvtColor(img_pad.copy(),cv2.COLOR_BGR2RGB)
        img=img_pad.copy()
        img_t=torch.from_numpy(img).float().permute(2,0,1)
        img1= transforms.Normalize([102.9801, 115.9465, 122.7717],[1.,1.,1.])(img_t)
        # img1=transforms.ToTensor()(img1)
        # img1= transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225),inplace=True)(img1)
        img1=img1.cuda()
        

        start_t=time.time()
        # print(f"img1: {img1}\n img1 shape: {img1.shape}")
        # assert False
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
        num_classes = len(CLASSES_NAME)
        colors = [(np.random.randint(255), np.random.randint(255), np.random.randint(255)) for _ in range(num_classes)]


        # for i,box in enumerate(boxes):
        #     pt1=(int(box[0]),int(box[1]))
        #     pt2=(int(box[2]),int(box[3]))
        #     img_pad=cv2.rectangle(img_pad,pt1,pt2,(0,255,0))
        #     img_pad=cv2.putText(img_pad,"%s %.3f"%(CLASSES_NAME[int(classes[i])],scores[i]),(int(box[0]),int(box[1])+10),cv2.FONT_HERSHEY_SIMPLEX,0.5,[0,200,20],2)

        # cv2.imwrite("/root/autodl-tmp/res/out_images/"+name,img_pad)


        for i, box in enumerate(boxes):
            pt1 = (int(box[0]), int(box[1]))
            pt2 = (int(box[2]), int(box[3]))
            
            # 根据类别索引获取对应的颜色
            color = colors[int(classes[i]) - 1]
            
            # 绘制检测框
            img_pad = cv2.rectangle(img_pad, pt1, pt2, color, 2)
            
            # 绘制半透明的填充框
            overlay = img_pad.copy()
            cv2.rectangle(overlay, pt1, pt2, color, -1)
            img_pad = cv2.addWeighted(overlay, 0.3, img_pad, 0.7, 0)
            
            # 绘制类别标签和置信度
            label = f"{CLASSES_NAME[int(classes[i])]}: {scores[i]:.3f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            label_bg_pt1 = (pt1[0], pt1[1] - label_size[1] - 5)
            label_bg_pt2 = (pt1[0] + label_size[0] + 5, pt1[1])
            cv2.rectangle(img_pad, label_bg_pt1, label_bg_pt2, color, -1)
            cv2.putText(img_pad, label, (pt1[0]+3, pt1[1]-3), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if not os.path.exists(DefaultConfig.out_images_path):
            os.makedirs(DefaultConfig.out_images_path)
        
        cv2.imwrite(os.path.join(DefaultConfig.out_images_path, name), img_pad)
        #cv2.imwrite("/root/autodl-tmp/res/out_images/"+name, img_pad)

# for i,box in enumerate(boxes):
#             pt1=(int(box[0]),int(box[1]))
#             pt2=(int(box[2]),int(box[3]))
#             img_pad=cv2.rectangle(img_pad,pt1,pt2,(0,255,0))
#             b_color = colors[int(classes[i]) - 1]
#             bbox = patches.Rectangle((box[0],box[1]),width=box[2]-box[0],height=box[3]-box[1],linewidth=1,facecolor='none',edgecolor=b_color)
#             ax.add_patch(bbox)
#             plt.text(box[0], box[1], s="%s %.3f"%(VOCDataset.CLASSES_NAME[int(classes[i])],scores[i]), color='white',
#                      verticalalignment='top',
#                      bbox={'color': b_color, 'pad': 0})
#         plt.axis('off')
#         plt.gca().xaxis.set_major_locator(NullLocator())
#         plt.gca().yaxis.set_major_locator(NullLocator())
#         plt.savefig('out_images/{}'.format(name), bbox_inches='tight', pad_inches=0.0)
#         plt.close()


