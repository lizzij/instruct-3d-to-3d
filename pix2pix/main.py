# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import replicate
import os
import time


def process_images(folder_path, prompts, replicate_model):
    # List all files in the given folder
    files = os.listdir(folder_path)
    images = [file for file in files if file.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # Process each image
    results = []
    for image_name, prompt in zip(images, prompts):
        start_time = time.time()
        image_path = os.path.join(folder_path, image_name)
        print(f"Processing image: {image_path}")
        with open(image_path, "rb") as image:
            output = replicate.run(
                replicate_model,
                input={
                    "image": image,
                    "prompt": prompt,
                    "scheduler": "K_EULER_ANCESTRAL",
                    "num_outputs": 1,
                    "guidance_scale": 7.5,
                    "num_inference_steps": 100,
                    "image_guidance_scale": 1.5
                }
            )
            print(f"Generate image successfully: {output}. Time spent: {time.time() - start_time} seconds")
            results.append(output)

    return results


def count_images_in_folder(folder_path):
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
    files = os.listdir(folder_path)
    image_count = sum(file.lower().endswith(tuple(image_extensions)) for file in files)
    return image_count

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    image_path = "./images"
    img_count = count_images_in_folder(image_path)
    print(f"{img_count} images under folder: {image_path}")
    prompts = ["make the image black and whilte" for _ in range(img_count)]
    replicate_model = "timothybrooks/instruct-pix2pix:30c1d0b916a6f8efce20493f5d61ee27491ab2a60437c13c588468b9810ec23f"
    outputs = process_images(image_path, prompts, replicate_model)
    for output in outputs:
        print(output)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
