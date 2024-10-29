import os
from PIL import Image
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def load_json_data(json_file_path):
    with open(json_file_path, "r") as f:
        json_data = json.load(f)
    return json_data

def load_image(image_path):
    """Loads an image from a local file path."""
    return Image.open(image_path)

def crop_image(img, points):
    """Crops the image based on the polygon points."""
    x_min = min([p[0] for p in points])
    y_min = min([p[1] for p in points])
    x_max = max([p[0] for p in points])
    y_max = max([p[1] for p in points])
    return img.crop((x_min, y_min, x_max, y_max))

def visualize_polygon(img, points):
    """Visualizes the polygon (bounding box) on the image."""
    img_np = np.array(img)
    fig, ax = plt.subplots(1)
    ax.imshow(img_np)

    # Plot polygon
    poly = patches.Polygon(points, linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(poly)
    plt.show()

def process_and_save_images(image_dir, json_dir, output_dir):
    """Processes each image with its respective JSON annotation and saves cropped images by class."""
    for json_file in os.listdir(json_dir):
        if json_file.endswith(".json"):
            json_path = os.path.join(json_dir, json_file)
            json_data = load_json_data(json_path)
            
            # Load image
            image_path = os.path.join(image_dir, json_data['imagePath'])
            if not os.path.exists(image_path):
                print(f"Image {image_path} not found, skipping.")
                continue
            
            img = load_image(image_path)
            image_id = os.path.splitext(json_data['imagePath'])[0]  # Use the image filename without extension as ID

            # Process each shape (polygon) in the JSON data
            for idx, shape in enumerate(json_data['shapes']):
                label = shape['label']
                points = shape['points']
                
                # Crop the image based on the points
                cropped_img = crop_image(img, points)
                
                # Save cropped image based on label
                save_cropped_image(cropped_img, output_dir, label, image_id, idx)

def save_cropped_image(cropped_img, output_dir, label, image_id, bbox_idx):
    """Saves the cropped image in the specified directory by label."""
    label_dir = os.path.join(output_dir, label)  # Save based on the disease class (label)
    if not os.path.exists(label_dir):
        os.makedirs(label_dir)
    
    cropped_img_path = os.path.join(label_dir, f"{image_id}_bbox_{bbox_idx}.jpg")
    cropped_img.save(cropped_img_path)
    print(f"Saved: {cropped_img_path}")

if __name__ == "__main__":
    root = "/home/youmin/workspace/VFMs-Adapters-Ensemble/dental-disease-classification"
    json_dir = os.path.join(root, "dentex2023/test/label")  # Directory containing individual JSON files
    image_dir = os.path.join(root, "dentex2023/test/input")  # Local directory containing images
    
    # Setting the directory to save cropped images
    output_dir = "test_cropped_images"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Process and save images by class without data splitting
    process_and_save_images(image_dir, json_dir, output_dir)
