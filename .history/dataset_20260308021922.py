"""
YOLO格式数据集加载器
支持多类别、多目标检测任务
"""
import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import random

class YOLODataset(Dataset):
    def __init__(self, image_dir, label_dir, input_size=(640, 640), 
                 augment=False, augment_params=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.input_size = input_size
        self.augment = augment
        self.augment_params = augment_params or {}
        
        self.image_files = []
        if os.path.exists(image_dir):
            for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                self.image_files.extend(
                    [f for f in os.listdir(image_dir) if f.lower().endswith(ext)]
                )
        
        self.image_files.sort()
        print(f"Found {len(self.image_files)} images in {image_dir}")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label_name = os.path.splitext(img_name)[0] + '.txt'
        label_path = os.path.join(self.label_dir, label_name)
        
        boxes = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    data = line.strip().split()
                    if len(data) == 5:
                        class_id = int(data[0])
                        cx, cy, w, h = map(float, data[1:5])
                        boxes.append([class_id, cx, cy, w, h])
        
        boxes = np.array(boxes, dtype=np.float32)
        if len(boxes) == 0:
            boxes = np.zeros((0, 5), dtype=np.float32)
        
        if self.augment:
            image, boxes = self.augment_data(image, boxes)

        h, w = image.shape[:2]
        image_resized, scale, pad = self.letterbox_resize(image, self.input_size)
        
        if len(boxes) > 0:
            boxes = self.adjust_boxes(boxes, (w, h), scale, pad)

        image_tensor = torch.from_numpy(image_resized).permute(2, 0, 1).float() / 255.0
        
        target = {
            'boxes': torch.from_numpy(boxes),
            'image_id': idx,
            'orig_size': torch.tensor([h, w])
        }
        
        return image_tensor, target
    
    def letterbox_resize(self, image, target_size):
        h, w = image.shape[:2]
        target_h, target_w = target_size
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        padded = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
        pad_w = (target_w - new_w) // 2
        pad_h = (target_h - new_h) // 2
        padded[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = resized
        return padded, scale, (pad_w, pad_h)
    
    def adjust_boxes(self, boxes, orig_size, scale, pad):
        if len(boxes) == 0:
            return boxes
        
        orig_w, orig_h = orig_size
        pad_w, pad_h = pad
        target_h, target_w = self.input_size

        boxes_pixel = boxes.copy()
        boxes_pixel[:, 1] *= orig_w  # cx
        boxes_pixel[:, 2] *= orig_h  # cy
        boxes_pixel[:, 3] *= orig_w  # w
        boxes_pixel[:, 4] *= orig_h  # h
        boxes_pixel[:, 1] = boxes_pixel[:, 1] * scale + pad_w
        boxes_pixel[:, 2] = boxes_pixel[:, 2] * scale + pad_h
        boxes_pixel[:, 3] *= scale
        boxes_pixel[:, 4] *= scale
        
        boxes_pixel[:, 1] /= target_w
        boxes_pixel[:, 2] /= target_h
        boxes_pixel[:, 3] /= target_w
        boxes_pixel[:, 4] /= target_h
        boxes_pixel[:, 1:5] = np.clip(boxes_pixel[:, 1:5], 0, 1)
        
        return boxes_pixel
    
    def augment_data(self, image, boxes):
        if random.random() < 0.5:
            image = self.augment_hsv(image)
        
        flip_lr = self.augment_params.get('flip_lr', 0.5)
        if random.random() < flip_lr:
            image = np.fliplr(image).copy()
            if len(boxes) > 0:
                boxes[:, 1] = 1.0 - boxes[:, 1]  # 翻转cx
        
        return image, boxes
    
    def augment_hsv(self, image):
        hsv_h = self.augment_params.get('hsv_h', 0.015)
        hsv_s = self.augment_params.get('hsv_s', 0.7)
        hsv_v = self.augment_params.get('hsv_v', 0.4)
        
        r = np.random.uniform(-1, 1, 3) * [hsv_h, hsv_s, hsv_v] + 1
        hue, sat, val = cv2.split(cv2.cvtColor(image, cv2.COLOR_RGB2HSV))
        
        dtype = image.dtype
        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)
        
        im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        image = cv2.cvtColor(im_hsv, cv2.COLOR_HSV2RGB)
        
        return image


def collate_fn(batch):
    images = []
    targets = []
    
    for img, target in batch:
        images.append(img)
        targets.append(target)
    
    images = torch.stack(images, 0)
    
    return images, targets


if __name__ == "__main__":
    # 测试数据加载
    dataset = YOLODataset(
        image_dir="data/train/images",
        label_dir="data/train/labels",
        input_size=(640, 640),
        augment=True
    )
    
    if len(dataset) > 0:
        img, target = dataset[0]
        print(f"Image shape: {img.shape}")
        print(f"Boxes shape: {target['boxes'].shape}")
        print(f"Boxes: {target['boxes']}")
    else:
        print("No data found. Please add images and labels to data/train/")

