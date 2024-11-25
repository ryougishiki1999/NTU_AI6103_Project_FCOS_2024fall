# import torch
# import math, random
# from PIL import Image
# import random
# import torchvision.transforms as transforms
# class Transforms(object):
#     def __init__(self):
#         pass

#     def __call__(self, img, boxes):
#         if random.random() < 0.3:
#             img, boxes = colorJitter(img, boxes)
#         if random.random() < 0.5:
#             img, boxes = random_rotation(img, boxes)
#         if random.random() < 0.5:
#             img, boxes = random_crop_resize(img, boxes)
#         return img, boxes

# def colorJitter(img, boxes, brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1):
#     img = transforms.ColorJitter(brightness=brightness,
#                 contrast=contrast, saturation=saturation, hue=hue)(img)
#     return img, boxes



# def random_rotation(img, boxes, degree=10):
#     d = random.uniform(-degree, degree)
#     w, h = img.size
#     rx0, ry0 = w / 2.0, h / 2.0
#     img = img.rotate(d)
#     a = -d / 180.0 * math.pi
#     boxes = torch.from_numpy(boxes)
#     new_boxes = torch.zeros_like(boxes)
#     new_boxes[:, 0] = boxes[:, 1]
#     new_boxes[:, 1] = boxes[:, 0]
#     new_boxes[:, 2] = boxes[:, 3]
#     new_boxes[:, 3] = boxes[:, 2]
#     for i in range(boxes.shape[0]):
#         ymin, xmin, ymax, xmax = new_boxes[i, :]
#         xmin, ymin, xmax, ymax = float(xmin), float(ymin), float(xmax), float(ymax)
#         x0, y0 = xmin, ymin
#         x1, y1 = xmin, ymax
#         x2, y2 = xmax, ymin
#         x3, y3 = xmax, ymax
#         z = torch.FloatTensor([[y0, x0], [y1, x1], [y2, x2], [y3, x3]])
#         tp = torch.zeros_like(z)
#         tp[:, 1] = (z[:, 1] - rx0) * math.cos(a) - (z[:, 0] - ry0) * math.sin(a) + rx0
#         tp[:, 0] = (z[:, 1] - rx0) * math.sin(a) + (z[:, 0] - ry0) * math.cos(a) + ry0
#         ymax, xmax = torch.max(tp, dim=0)[0]
#         ymin, xmin = torch.min(tp, dim=0)[0]
#         new_boxes[i] = torch.stack([ymin, xmin, ymax, xmax])
#     new_boxes[:, 1::2].clamp_(min=0, max=w - 1)
#     new_boxes[:, 0::2].clamp_(min=0, max=h - 1)
#     boxes[:, 0] = new_boxes[:, 1]
#     boxes[:, 1] = new_boxes[:, 0]
#     boxes[:, 2] = new_boxes[:, 3]
#     boxes[:, 3] = new_boxes[:, 2]
#     boxes = boxes.numpy()
#     return img, boxes



# def _box_inter(box1, box2):
#     tl = torch.max(box1[:,None,:2], box2[:,:2])  # [n,m,2]
#     br = torch.min(box1[:,None,2:], box2[:,2:])  # [n,m,2]
#     hw = (br-tl).clamp(min=0)  # [n,m,2]
#     inter = hw[:,:,0] * hw[:,:,1]  # [n,m]
#     return inter



# def random_crop_resize(img, boxes, crop_scale_min=0.2, aspect_ratio=[3./4, 4./3], remain_min=0.7, attempt_max=10):
#     success = False
#     boxes = torch.from_numpy(boxes)
#     for attempt in range(attempt_max):
#         # choose crop size
#         area = img.size[0] * img.size[1]
#         target_area = random.uniform(crop_scale_min, 1.0) * area
#         aspect_ratio_ = random.uniform(aspect_ratio[0], aspect_ratio[1])
#         w = int(round(math.sqrt(target_area * aspect_ratio_)))
#         h = int(round(math.sqrt(target_area / aspect_ratio_)))
#         if random.random() < 0.5:
#             w, h = h, w
#         # if size is right then random crop
#         if w <= img.size[0] and h <= img.size[1]:
#             x = random.randint(0, img.size[0] - w)
#             y = random.randint(0, img.size[1] - h)
#             # check
#             crop_box = torch.FloatTensor([[x, y, x + w, y + h]])
#             inter = _box_inter(crop_box, boxes) # [1,N] N can be zero
#             box_area = (boxes[:, 2]-boxes[:, 0])*(boxes[:, 3]-boxes[:, 1]) # [N]
#             mask = inter>0.0001 # [1,N] N can be zero
#             inter = inter[mask] # [1,S] S can be zero
#             box_area = box_area[mask.view(-1)] # [S]
#             box_remain = inter.view(-1) / box_area # [S]
#             if box_remain.shape[0] != 0:
#                 if bool(torch.min(box_remain > remain_min)):
#                     success = True
#                     break
#             else:
#                 success = True
#                 break
#     if success:
#         img = img.crop((x, y, x+w, y+h))
#         boxes -= torch.Tensor([x,y,x,y])
#         boxes[:,1::2].clamp_(min=0, max=h-1)
#         boxes[:,0::2].clamp_(min=0, max=w-1)
#         # ow, oh = (size, size)
#         # sw = float(ow) / img.size[0]
#         # sh = float(oh) / img.size[1]
#         # img = img.resize((ow,oh), Image.BILINEAR)
#         # boxes *= torch.FloatTensor([sw,sh,sw,sh])
#     boxes = boxes.numpy()
#     return img, boxes
import torch
import math, random
from PIL import Image
import random
import torchvision.transforms as transforms
class Transforms(object):
    def __init__(self):
        pass

    def __call__(self, img, boxes):
        if random.random() < 0.3:
            img, boxes = adjust_color(img, boxes)
        if random.random() < 0.3:
            img, boxes = random_flip(img, boxes)
        if random.random() < 0.3:
            img, boxes = crop_resize(img, boxes)
        return img, boxes

def adjust_color(img, boxes):
    """Adjust the brightness, contrast, saturation, and hue of the image without relying on any pre-existing methods."""
    import numpy as np

    # Convert image to numpy array
    img_np = np.array(img).astype(np.float32)

    # Adjust brightness
    brightness_factor = random.uniform(0.8, 1.2)
    img_np = img_np * brightness_factor

    # Adjust contrast
    contrast_factor = random.uniform(0.8, 1.2)
    mean = img_np.mean(axis=(0, 1), keepdims=True)
    img_np = (img_np - mean) * contrast_factor + mean

    # Clip the values to be in valid range
    img_np = np.clip(img_np, 0, 255)

    # Convert to HSV color space
    img_uint8 = img_np.astype(np.uint8)
    img_hsv = Image.fromarray(img_uint8, mode='RGB').convert('HSV')
    img_hsv_np = np.array(img_hsv).astype(np.float32)

    # Adjust saturation
    saturation_factor = random.uniform(0.8, 1.2)
    img_hsv_np[:, :, 1] *= saturation_factor

    # Adjust hue
    hue_factor = random.uniform(-18, 18)  # Hue shift in degrees (-18, 18)
    img_hsv_np[:, :, 0] = (img_hsv_np[:, :, 0] + hue_factor) % 256  # PIL uses 0-255 for hue

    # Clip saturation and hue to valid ranges
    img_hsv_np[:, :, 0] = np.clip(img_hsv_np[:, :, 0], 0, 255)
    img_hsv_np[:, :, 1] = np.clip(img_hsv_np[:, :, 1], 0, 255)
    img_hsv_np[:, :, 2] = np.clip(img_hsv_np[:, :, 2], 0, 255)

    # Convert back to RGB
    img_hsv_uint8 = img_hsv_np.astype(np.uint8)
    img_hsv = Image.fromarray(img_hsv_uint8, mode='HSV')
    img_rgb = img_hsv.convert('RGB')

    return img_rgb, boxes

def random_flip(img, boxes):
    """Randomly flip the image horizontally and/or vertically and adjust bounding boxes."""
    w, h = img.size
    flipped = False

    # Horizontal flip
    if random.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        boxes = boxes.copy()
        boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
        flipped = True

    # Vertical flip
    if random.random() < 0.5:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
        boxes = boxes.copy()
        boxes[:, [1, 3]] = h - boxes[:, [3, 1]]
        flipped = True

    return img, boxes


def crop_resize(img, boxes, scale_range=(0.8, 1.0), ratio_range=(0.9, 1.1)):
    """Randomly crop the image and adjust bounding boxes."""
    w, h = img.size
    aspect_ratio = w / h
    scale = random.uniform(*scale_range)
    ratio = random.uniform(*ratio_range)

    # Compute new dimensions
    new_w = int(w * scale * math.sqrt(ratio / aspect_ratio))
    new_h = int(h * scale / math.sqrt(ratio / aspect_ratio))

    # Ensure new dimensions are within image bounds
    new_w = min(new_w, w)
    new_h = min(new_h, h)

    # Randomly select top-left corner for cropping
    left = random.randint(0, w - new_w)
    top = random.randint(0, h - new_h)

    # Crop the image
    img = img.crop((left, top, left + new_w, top + new_h))

    # Adjust bounding boxes
    boxes = boxes.copy()
    boxes[:, [0, 2]] -= left
    boxes[:, [1, 3]] -= top

    # Remove boxes that are completely outside the cropped image
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, new_w)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, new_h)
    keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    boxes = boxes[keep]

    # Resize back to original size
    img = img.resize((w, h), Image.BILINEAR)
    scale_w = w / new_w
    scale_h = h / new_h
    boxes[:, [0, 2]] *= scale_w
    boxes[:, [1, 3]] *= scale_h

    return img, boxes

#import os
#import sys
#sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
#from config import DefaultConfig
#import glob
#import numpy as np
#import matplotlib.pyplot as plt
#from PIL import ImageDraw

