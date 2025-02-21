import os
from pycocotools.coco import COCO
import numpy as np
import cv2


def generate_masks(data_path):
    """
    Tạo ảnh mask từ file COCO annotation và lưu vào thư mục mới.

    Args:
        coco_json_path (str): Đường dẫn đến file COCO JSON.
        output_folder (str): Thư mục để lưu ảnh mask.
    """
    # Load COCO dataset

    coco_json_path = f"{data_path}/_annotations.coco.json"
    output_folder = f"{data_path}/masks"
    coco = COCO(coco_json_path)

    # Tạo thư mục output nếu chưa có
    os.makedirs(output_folder, exist_ok=True)

    # Lấy tất cả ID ảnh trong dataset
    image_ids = coco.getImgIds()

    for image_id in image_ids:
        # Lấy thông tin ảnh (kích thước ảnh gốc)
        image_info = coco.loadImgs(image_id)[0]
        height, width = image_info["height"], image_info["width"]
        image_filename = image_info["file_name"]

        # Lấy danh sách annotation của ảnh đó
        ann_ids = coco.getAnnIds(imgIds=image_id)
        annotations = coco.loadAnns(ann_ids)

        # Tạo mask với kích thước ảnh gốc
        mask = np.zeros((height, width), dtype=np.uint8)

        for ann in annotations:
            rle = coco.annToMask(ann) * 255  # Chuyển thành mask
            if rle.shape != (height, width):
                rle = cv2.resize(rle, (width, height), interpolation=cv2.INTER_NEAREST)  # Resize nếu cần

            mask = np.maximum(mask, rle)  # Hợp nhất mask

        # Lưu mask ra ảnh PNG
        mask_filename = os.path.join(output_folder, image_filename)
        cv2.imwrite(mask_filename, mask)

    print(f"✅ Đã tạo {len(image_ids)} ảnh mask trong thư mục: {output_folder}")

import os
import numpy as np
import cv2

def compute_iou(gt_mask, pred_mask):
    """Compute IoU between ground truth mask and predicted mask"""
    intersection = np.logical_and(gt_mask, pred_mask).sum()
    union = np.logical_or(gt_mask, pred_mask).sum()

    if union == 0:  # Tránh chia cho 0
        return 1.0 if intersection == 0 else 0.0
    
    return intersection / union

def compute_mean_iou(gt_folder, pred_folder):
    """Compute mean IoU for all images in gt_folder and pred_folder"""
    iou_scores = []
    
    gt_files = sorted(os.listdir(gt_folder))
    pred_files = sorted(os.listdir(pred_folder))

    for gt_file, pred_file in zip(gt_files, pred_files):
        gt_path = os.path.join(gt_folder, gt_file)
        pred_path = os.path.join(pred_folder, pred_file)

        # Load mask images as grayscale (0-255)
        gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        pred_mask = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)

        # Convert to binary (0 for background, 1 for object)
        gt_mask = (gt_mask > 127).astype(np.uint8)
        pred_mask = (pred_mask > 127).astype(np.uint8)

        # Compute IoU
        iou = compute_iou(gt_mask, pred_mask)
        iou_scores.append(iou)
    
    # Compute mean IoU
    mean_iou = np.mean(iou_scores)
    return mean_iou

if __name__ =="__main__":

    dataset_path = 'assets3'
    gt_folder = f"{dataset_path}/masks"
    pred_folder = f"{dataset_path}/predict_masks"

    # Tính mIoU
    mean_iou = compute_mean_iou(gt_folder, pred_folder)
    print(f"Mean IoU: {mean_iou:.4f}")


