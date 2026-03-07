"""
数据可视化脚本 - 查看数据集和标注
"""
import os
import cv2
import argparse
import numpy as np
from pathlib import Path


def visualize_annotations(image_path, label_path, class_names=None):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Cannot read image: {image_path}")
        return None
    
    h, w = image.shape[:2]
    
    if not os.path.exists(label_path):
        print(f"Label file not found: {label_path}")
        return image
    
    with open(label_path, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        data = line.strip().split()
        if len(data) != 5:
            continue
        
        class_id = int(data[0])
        cx, cy, bw, bh = map(float, data[1:5])
        
        cx_px = int(cx * w)
        cy_px = int(cy * h)
        bw_px = int(bw * w)
        bh_px = int(bh * h)
        
        x1 = int(cx_px - bw_px / 2)
        y1 = int(cy_px - bh_px / 2)
        x2 = int(cx_px + bw_px / 2)
        y2 = int(cy_px + bh_px / 2)
        
        np.random.seed(class_id)
        color = tuple(np.random.randint(0, 255, 3).tolist())
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.circle(image, (cx_px, cy_px), 3, color, -1)
        
        if class_names and class_id < len(class_names):
            label = f"{class_names[class_id]}"
        else:
            label = f"Class {class_id}"
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(image, (x1, y1 - text_h - 5), (x1 + text_w, y1), color, -1)
        cv2.putText(image, label, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return image


def dataset_statistics(image_dir, label_dir):
    print("\n" + "="*60)
    print("数据集统计")
    print("="*60)
    
    image_files = []
    for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
        image_files.extend(Path(image_dir).glob(f'*{ext}'))
    
    num_images = len(image_files)
    print(f"图像数量: {num_images}")
    
    if num_images == 0:
        return
    

    class_counts = {}
    total_objects = 0
    images_with_labels = 0
    
    for img_path in image_files:
        label_name = img_path.stem + '.txt'
        label_path = os.path.join(label_dir, label_name)
        
        if not os.path.exists(label_path):
            continue
        
        images_with_labels += 1
        
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            data = line.strip().split()
            if len(data) == 5:
                class_id = int(data[0])
                class_counts[class_id] = class_counts.get(class_id, 0) + 1
                total_objects += 1
    
    print(f"有标签的图像数量: {images_with_labels}")
    print(f"总目标数量: {total_objects}")
    
    if images_with_labels > 0:
        print(f"平均每张图像目标数: {total_objects / images_with_labels:.2f}")
    
    print(f"\n类别分布:")
    for class_id in sorted(class_counts.keys()):
        count = class_counts[class_id]
        percentage = 100 * count / total_objects
        print(f"  类别 {class_id}: {count} ({percentage:.2f}%)")
    
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Visualize YOLO dataset')
    parser.add_argument('--image_dir', type=str, default='E:/dataset/vedai_visible/images',
                       help='Path to image directory')
    parser.add_argument('--label_dir', type=str, default='E:/dataset/vedai_visible/labels',
                       help='Path to label directory')
    parser.add_argument('--output_dir', type=str, default='visualizations',
                       help='Path to save visualizations')
    parser.add_argument('--num_samples', type=int, default=10,
                       help='Number of samples to visualize')
    parser.add_argument('--stats_only', default=True, action='store_true',
                       help='Only show statistics')
    args = parser.parse_args()

    dataset_statistics(args.image_dir, args.label_dir)
    
    if args.stats_only:
        return

    image_files = []
    for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
        image_files.extend(Path(args.image_dir).glob(f'*{ext}'))
    
    if len(image_files) == 0:
        print("No images found!")
        return
    os.makedirs(args.output_dir, exist_ok=True)
    num_to_visualize = min(args.num_samples, len(image_files))
    print(f"Visualizing {num_to_visualize} samples...")
    
    for i, img_path in enumerate(image_files[:num_to_visualize]):
        label_name = img_path.stem + '.txt'
        label_path = os.path.join(args.label_dir, label_name)
        vis_image = visualize_annotations(str(img_path), label_path)
        
        if vis_image is not None:
            output_path = os.path.join(args.output_dir, f'vis_{img_path.name}')
            cv2.imwrite(output_path, vis_image)
            print(f"Saved: {output_path}")
    print(f"\nVisualization complete! Check {args.output_dir}/ directory.")

if __name__ == "__main__":
    main()

