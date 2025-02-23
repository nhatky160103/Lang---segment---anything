from io import BytesIO

import numpy as np
from PIL import Image
import os
import yaml
from pathlib import Path
from itertools import islice


from lang_sam import LangSAM
from lang_sam.utils import draw_image



def load_config(file_path="lang_sam/config.yaml"):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)

config = load_config()



class Pipeline():
    def __init__(self):
        """Initialize or load the LangSAM model."""
        self.model = LangSAM(sam_type=config["sam_type"])
        print("LangSAM model initialized.")

    def predict(self, text_prompt: str, dataset_path: Path, is_preview: bool = False, 
                is_save: bool = False, check_photo: bool = False, parallel_size: int = 4) -> dict:
        
        """Perform prediction using the LangSAM model.

        Args:
            text_prompt (str): Text description of the object.
            dataset_path (Path): Path to the dataset.
            is_preview (bool): Whether to display the results.
            is_save (bool): Whether to save the results.
            check_photo (bool): Whether to check the input images.
            parallel_size (int): Number of images to process in parallel.

        Returns:
            dict: Results including processed images and model predictions.
        """
        image_paths = []
        image_pil = []

        # Read images from the dataset folder
        for image_name in os.listdir(f"{dataset_path}/images"):
            image_path = os.path.join(f"{dataset_path}/images", image_name)
            image = Image.open(image_path).convert("RGB")
            image_pil.append(image)
            image_paths.append(Path(image_path))

        # Function to split the image list into batches of parallel_size
        def batch_iter(iterable, batch_size):
            """Splits a list into smaller batches of size batch_size."""
            it = iter(iterable)
            while batch := list(islice(it, batch_size)):
                yield batch


        results = []
        output_images = []
        
        for batch_images, batch_paths in zip(batch_iter(image_pil, parallel_size), batch_iter(image_paths, parallel_size)):
            text_prompts = [text_prompt] * len(batch_images)

            batch_results = self.model.predict(
                images_pil=batch_images,
                texts_prompt=text_prompts,
                box_threshold=config["box_threshold"],
                text_threshold=config["text_threshold"],
                check_photo=check_photo
            )
            results.extend(batch_results)

            if is_save:
                Path(f"{dataset_path}/predicts").mkdir(exist_ok=True)
                Path(f"{dataset_path}/predict_masks").mkdir(exist_ok=True)

            for i, result in enumerate(batch_results):
                if not len(result["masks"]):
                    print(f"No masks detected in {batch_paths[i].name}. Returning original image.")
                    continue

        
                image_array = np.asarray(batch_images[i])
                output_image = draw_image(
                    image_array,
                    result["masks"],
                    result["boxes"],
                    result["scores"],
                    result["labels"],
                )

                output_image = Image.fromarray(np.uint8(output_image)).convert("RGB")

                combined_mask = np.any(result["masks"].astype(bool), axis=0).astype(np.uint8) * 255  
                combined_mask = Image.fromarray(combined_mask)

                if is_preview:
                    output_image.show()
                    combined_mask.show()

                if is_save:
                    output_image.save(f"{dataset_path}/predicts/{batch_paths[i].name}")
                    combined_mask.save(f"{dataset_path}/predict_masks/{batch_paths[i].name}")

                output_images.append(output_image)

        return {"results": results, 'output_images': output_images}


if __name__ == "__main__":
    
    dataset_path = "./assets"
    pipeline = Pipeline()
    text_promt = "product."
    pipeline.predict(text_promt, dataset_path, is_preview= True, check_photo= False, is_save= True)
