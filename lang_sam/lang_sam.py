import numpy as np
from PIL import Image
import cv2

from lang_sam.models.gdino import GDINO
from lang_sam.models.sam import SAM
from lang_sam.models.utils import DEVICE
from lang_sam.models.clip import Clip


class LangSAM:
    def __init__(self, sam_type="sam2.1_hiera_small", ckpt_path: str | None = None, device=DEVICE):
        self.sam_type = sam_type

        self.sam = SAM()
        self.sam.build_model(sam_type, ckpt_path, device=device)
        self.gdino = GDINO()
        self.gdino.build_model(device=device)
        self.clip = Clip()

    def predict(
        self,
        images_pil: list[Image.Image],
        texts_prompt: list[str],
        box_threshold: float = 0.3,
        text_threshold: float = 0.25,
        check_photo: bool = False,
    ):
        """Predicts masks for given images and text prompts using GDINO and SAM models.

        Parameters:
            images_pil (list[Image.Image]): List of input images.
            texts_prompt (list[str]): List of text prompts corresponding to the images.
            box_threshold (float): Threshold for box predictions.
            text_threshold (float): Threshold for text predictions.

        Returns:
            list[dict]: List of results containing masks and other outputs for each image.
            Output format:
            [{
                "boxes": np.ndarray,
                "scores": np.ndarray,
                "masks": np.ndarray,
                "mask_scores": np.ndarray,
            }, ...]
        """

        gdino_results = self.gdino.predict(images_pil, texts_prompt, box_threshold, text_threshold)
        all_results = []
        sam_images = []
        sam_boxes = []
        sam_indices = []
        for idx, result in enumerate(gdino_results):
            result = {k: (v.cpu().numpy() if hasattr(v, "numpy") else v) for k, v in result.items()}

            if check_photo:
                # Duyệt từ cuối về đầu để tránh thay đổi index khi xóa phần tử
                for i in reversed(range(len(result["boxes"]))):
                    box = result["boxes"][i].clip(0, None)
                    cropped_image = images_pil[idx].crop(box)
                    if self.clip.predict(cropped_image) == 1:
                        result["boxes"] = np.delete(result["boxes"], i, axis=0)
                        result["scores"] = np.delete(result["scores"], i, axis=0)
                        result["labels"].pop(i)
                                    
            
            processed_result = {
                **result,
                "masks": [],
                "mask_scores": [],
            }

            if result["labels"]:
                sam_images.append(np.asarray(images_pil[idx]))
                sam_boxes.append(processed_result["boxes"])
                sam_indices.append(idx)

            all_results.append(processed_result)
        if sam_images:
            print(f"Predicting {len(sam_boxes)} masks")
            masks, mask_scores, _ = self.sam.predict_batch(sam_images, xyxy=sam_boxes)
            for idx, mask, score in zip(sam_indices, masks, mask_scores):
                all_results[idx].update(
                    {
                        "masks": mask,
                        "mask_scores": score,
                    }
                )
            print(f"Predicted {len(all_results)} masks")
        return all_results


if __name__ == "__main__":
    model = LangSAM()
    image_path = "./assets/26.jpg"
    image_path2 = "./assets/3.jpg"
    out = model.predict(
        [Image.open(image_path).convert("RGB"), Image.open(image_path2).convert("RGB")],
        ["real product.", "real product."],
    )

    image_pil = Image.open(image_path).convert("RGB")
    image_array = np.asarray(image_pil)
    
    from lang_sam.utils import draw_image

    results = out[0]
    output_image = draw_image(
            image_array,
            results["masks"],
            results["boxes"],
            results["scores"],
            results["labels"],
        )
    output_image = Image.fromarray(np.uint8(output_image)).convert("RGB")

    output_image.show()

    

 
