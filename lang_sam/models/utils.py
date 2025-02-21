import logging
import cv2
import torch
import numpy as np


def get_device_type() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        logging.warning("No GPU found, using CPU instead")
        return "cpu"


device_type = get_device_type()
DEVICE = torch.device(device_type)

if torch.cuda.is_available():
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True



def draw_bboxes_cv2(image_pil, output):
    # Chuyển đổi ảnh PIL sang định dạng numpy cho OpenCV
    image_np = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

    # Lấy thông tin từ output
    boxes = output['boxes']
    labels = output['labels']
    scores = output['scores']

    # Duyệt qua từng box
    for i, box in enumerate(boxes):
        # Lấy toạ độ box
        x_min, y_min, x_max, y_max = map(int, box)  # Chuyển sang int cho cv2

        # Vẽ bbox bằng cv2.rectangle
        cv2.rectangle(
            image_np,
            (x_min, y_min),
            (x_max, y_max),
            color=(0, 0, 255),  # Màu đỏ (BGR)
            thickness=2
        )

        # Tạo label với score
        label = f"{labels[i]}: {scores[i]:.2f}"

        # Vẽ label bằng cv2.putText
        cv2.putText(
            image_np,
            label,
            (x_min, y_min - 10),  # Góc trên bên trái của bbox
            cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.6,
            color=(0, 0, 255),  # Màu đỏ (BGR)
            thickness=2
        )

    # Hiển thị ảnh với OpenCV
    width, height = image_pil.size

    image_np = cv2.resize(image_np, (800, height * 800 // width))
    cv2.imshow('Image with BBoxes', image_np)
    cv2.waitKey(0)
    cv2.destroyAllWindows()