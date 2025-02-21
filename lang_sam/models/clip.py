import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

class Clip():
    def __init__(self, model_id: str = "openai/clip-vit-base-patch32"):
        self.model_id = model_id
        self.model = CLIPModel.from_pretrained(self.model_id)
        self.processor = CLIPProcessor.from_pretrained(self.model_id)

    def predict(self, image: Image):
        text_variants = [
            ["real product", "drawing product"],
            ["a realistic photo of a product", "a hand-drawn illustration of a product"],
            ["a high-quality real-world photograph", "a sketch, painting, or digital artwork"],
            ["a real-world product photo with natural lighting", "a cartoon, anime-style, or digital painting"],
            ["a realistic photo captured by a camera", "a hand-drawn or computer-generated artwork"]
        ]
        
        best_result = None
        best_prob = 0

        for texts in text_variants:
            inputs = self.processor(text=texts, images=image, return_tensors="pt", padding=True)
            outputs = self.model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)
            class_id = torch.argmax(probs, dim=1)

            if probs[0, class_id] > best_prob:
                best_prob = probs[0, class_id]
                best_result = class_id

        return best_result
    

if __name__ == "__main__":
        
    class_id = Clip().predict(Image.open('assets/24.jpg'))

    print(class_id)


