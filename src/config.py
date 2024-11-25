import os
class DefaultConfig():
    #backbone
    pretrained=True
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
    cnt_on_reg=True

    #training
    strides=[8,16,32,64,128]
    limit_range=[[-1,64],[64,128],[128,256],[256,512],[512,999999]]
    gpu_id='0'
    batch_size = 4
    epochs = 25
    num_workers = 4

    #inference
    score_threshold=0.05
    nms_iou_threshold=0.6
    max_detection_boxes_num=1000
    
    # project path
    
    project_root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    assets_path = os.path.join(project_root_path, "assets")
    data_path = os.path.join(project_root_path, "data")
    out_path = os.path.join(project_root_path, "out")
    
    #
    coco_path = "/root/autodl-tmp/coco_2017/"
    coco_train_data_path = os.path.join(coco_path, "train", "data")
    coco_train_label_path = os.path.join(coco_path, "train", "labels.json")
    coco_val_data_path = os.path.join(coco_path, "validation", "data")
    coco_val_label_path = os.path.join(coco_path, "validation", "labels.json")
    
    check_points_dir_path = os.path.join(assets_path, "checkpoints")
    check_point_path = os.path.join(check_points_dir_path, "coco_37.2.pth")
    self_check_point_path = os.path.join(check_points_dir_path, "coco_model.pth")
    # tmp_check_point_path = os.path.join(check_points_dir_path, "coco_tmp.pth")
    
    test_images_path = os.path.join(assets_path, "test_images")
    out_images_path = os.path.join(out_path, "images")
    out_loss_path = os.path.join(out_images_path, "loss.png")