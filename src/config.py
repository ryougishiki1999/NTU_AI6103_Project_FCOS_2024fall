import os
from enum import Enum
from enum import auto

class DefaultConfig():
    #backbone
    class Backbone(Enum):
        ResNet50 = auto()
        ResNet18 = auto()
        MobileNetV2 = auto()
        VovNet27Slim = auto()
    
    choosen_backbone = Backbone.ResNet18 # change backbone here
    pretrained=True
    freeze_stage_1=True
    freeze_bn=True

    #fpn
    fpn_out_channels=256
    use_p5=True
    
    #head
    if choosen_backbone == Backbone.ResNet50:
        class_num=80 # use coco dataset
    else:
        class_num=20 # use voc dataset
    
    use_GN_head=True
    prior=0.01
    add_centerness=True
    cnt_on_reg=True

    #training
    strides=[8,16,32,64,128]
    limit_range=[[-1,64],[64,128],[128,256],[256,512],[512,999999]]
    gpu_id='0'
    batch_size = 4
    epochs = 25
    num_workers = 4

    #inference
    if choosen_backbone == Backbone.ResNet50:
        score_threshold=0.05 # for coco dataset
    else:
        score_threshold=0.5 # for voc dataset

    nms_iou_threshold=0.6
    max_detection_boxes_num=1000
    
    # project path
    project_root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    assets_path = os.path.join(project_root_path, "assets")
    
    data_path = os.path.join(project_root_path, "data")
    out_path = os.path.join(project_root_path, "out")
    out_featuremaps_dir_path = os.path.join(out_path, "featuremaps")
    for path in [data_path, out_path, out_featuremaps_dir_path]:
        if not os.path.exists(path):
            os.makedirs(path)
    
    # coco dataset
    #coco_data_dir_path = "/root/autodl-tmp/coco_2017/"'
    coco_data_dir_path = os.path.join(data_path, "coco_2017")
    if not os.path.exists(coco_data_dir_path):
        os.makedirs(coco_data_dir_path)
    coco_train_data_path = os.path.join(coco_data_dir_path, "train", "data")
    coco_train_label_path = os.path.join(coco_data_dir_path, "train", "labels.json")
    coco_val_data_path = os.path.join(coco_data_dir_path, "validation", "data")
    coco_val_label_path = os.path.join(coco_data_dir_path, "validation", "labels.json")
    
    coco_backbone_resnet50_path  = os.path.join(assets_path, "resnet50-19c8e357.pth")
    coco_check_points_dir_path = os.path.join(assets_path, "coco_checkpoints")
    coco_check_point_path = os.path.join(coco_check_points_dir_path, "coco_37.2.pth")
    coco_train_eval_check_point_path = os.path.join(coco_check_points_dir_path, "coco_model.pth")
    coco_detection_check_point_path = os.path.join(coco_check_points_dir_path, "FCOS_R_50_FPN_1x_my.pth")
    
    coco_detect_images_dir_path = os.path.join(assets_path, "coco_detect_images")
    coco_detect_out_images_dir_path = os.path.join(out_path, "coco_detect_out_images")
    coco_out_dir_path = os.path.join(out_path, "coco_out")
    coco_out_loss_path = os.path.join(coco_out_dir_path, "loss.png")
    
    for path in [coco_detect_out_images_dir_path, coco_out_dir_path]:
        if not os.path.exists(path):
            os.makedirs(path)
    
    #voc dataset
    voc_2012_data_dir_path = os.path.join(data_path, "VOCdevkit", "VOC2012")
    voc_2007_data_dir_path = os.path.join(data_path, "VOCdevkit", "VOC2007")
    voc_check_points_dir_path = os.path.join(assets_path, "voc_checkpoints")
    voc_check_points_training_dir_path = os.path.join(voc_check_points_dir_path, "training")
    
    voc_mobileNet_check_point_path = os.path.join(voc_check_points_dir_path, "mobileNet_model.pth")
    voc_vovNet_check_point_path = os.path.join(voc_check_points_dir_path, "vovNet_model.pth")
    voc_resNet18_check_point_path = os.path.join(voc_check_points_dir_path, "resNet18_model.pth")
    
    voc_detect_images_dir_path = os.path.join(assets_path, "voc_detect_images")
    voc_detect_out_images_dir_path = os.path.join(out_path, "voc_detect_out_images")
    
    for path in [voc_check_points_training_dir_path, voc_detect_out_images_dir_path]:
        if not os.path.exists(path):
            os.makedirs(path)
    

    