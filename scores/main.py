# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import os
from brisque import BRISQUE
import PIL.Image
import torch
import clip

def compute_brisque_scores(images):
    scores = []
    for image_path in images:
        print(f"Processing image: {image_path}")
        image = PIL.Image.open(image_path)
        # Initialize BRISQUE and compute the score
        brisque = BRISQUE()
        score = brisque.score(image)
        print(f"Score: {score}")
        scores.append(score)
    return scores

def compute_clip_scores(images, prompts):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    for image_path, prompt in zip(images, prompts):
        image = preprocess(PIL.Image.open(image_path)).unsqueeze(0).to(device)
        text = clip.tokenize([prompt]).to(device)
        with torch.no_grad():
            image_features = model.encode_image(image)
            text_features = model.encode_text(text)
        similarity = torch.cosine_similarity(text_features, image_features, dim=1)
        print(f"CLIP score: {similarity.item()}")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    image_path = "./images"
    files = os.listdir(image_path)
    images = [file for file in files if file.lower().endswith(('.png', '.jpg', '.jpeg'))]
    images = [os.path.join(image_path, image_name) for image_name in images]
    compute_brisque_scores(images)
    compute_clip_scores(images, ["Pikachu drinking coffee", "Pikachu drinking coffee", "Pikachu drinking coffee"])

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
