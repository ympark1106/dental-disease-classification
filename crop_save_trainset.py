import os
from PIL import Image
import json
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.patches as patches



'''
In the diagnosis,
there are four specific classes corresponding to four different diagnoses: 
caries, deep caries, periapical lesions, and impacted teeth. 
'''



def load_json_data(json_file_path):
    with open(json_file_path, "r") as f:
        json_data = json.load(f)
    return json_data


def load_image(image_path):
    """
    Loads an image from a local file path.
    """
    return Image.open(image_path)

def crop_image(img, bbox):
    """
    Crops the image according to the given bbox.
    The bbox format is [x_min, y_min, width, height], so x_max and y_max need to be calculated for cropping.
    """
    x_min, y_min, width, height = bbox
    x_max = x_min + width
    y_max = y_min + height
    cropped_img = img.crop((x_min, y_min, x_max, y_max))
    return cropped_img

    
def visualize_bbox(img, bbox):
    """
    Visualizes the bounding box on the image.
    """
    # Convert PIL image to numpy array for plotting
    img_np = np.array(img)

    # Create a matplotlib figure
    fig, ax = plt.subplots(1)
    ax.imshow(img_np)

    # Get bbox coordinates
    x_min, y_min, width, height = bbox

    # Create a rectangle patch for the bbox
    rect = patches.Rectangle((x_min, y_min), width, height, linewidth=2, edgecolor='r', facecolor='none')

    # Add the patch to the Axes
    ax.add_patch(rect)

    plt.show()


     
def process_and_save_images(image_dir, dataset, json_data, output_dir, split):
    """
    Crops and saves images based on the local image dataset and json_data.
    Crops all bboxes in a single image and saves them with the original image ID and bbox index.
    """
    for idx, (example, json_info, _) in enumerate(dataset):
        
        # Load the image from the local path
        image_path = os.path.join(image_dir, example['file_name'])  # Assuming file_name is in the dataset
        img = load_image(image_path)

        # print to check the structure of json_info and ensure bbox exists
        print(f"Processing image: {example['file_name']}")

        # Check the structure of json_info
        # print(f"json_info structure: {json_info}")

        bboxes = json_info.get('bboxes', [json_info.get('bbox', [0, 0, 0, 0])])  # Ensure it defaults to [0, 0, 0, 0] if no bbox

        # Print bbox information to debug why it's all zero
        print(f"Bboxes for image {example['file_name']}: {bboxes}")

        category_id_3 = json_info.get('category_id_3')  
        image_id = example.get('id', idx)

        for bbox_idx, bbox in enumerate(bboxes):
            cropped_img = crop_image(img, bbox)
            if category_id_3 is not None:
                save_cropped_image(cropped_img, output_dir, split, category_id_3, image_id, bbox_idx)



                
def save_cropped_image(cropped_img, output_dir, split_dir, category_id_3, image_id, bbox_idx):
    """
    Saves the cropped image in the specified directory.
    The images are stored in separate directories by disease category (category_id_3),
    and file names are generated with original image id and bbox index.
    """
    label_dir = os.path.join(output_dir, split_dir, str(category_id_3))  # category_id_3 기준으로 저장
    if not os.path.exists(label_dir):
        os.makedirs(label_dir)
    
    cropped_img_path = os.path.join(label_dir, f"cropped_image_{image_id}_bbox_{bbox_idx}.jpg")  
    cropped_img.save(cropped_img_path)
    print(f"Saved: {cropped_img_path}")


def split_and_save_data(image_dir, all_annotations, output_dir):
    """
    Splits the dataset into train, val, and test sets based on category_id_3 (disease labels) 
    and processes and saves images for each split.
    """
    # Splitting the dataset based on category_id_3 (disease classification)
    train_data, val_data = train_test_split(all_annotations, test_size=0.2, stratify=[ann[1].get('category_id_3') for ann in all_annotations], random_state=42)
    # train_data, val_data = train_test_split(train_data, test_size=0.2, stratify=[ann[1].get('category_id_3') for ann in train_data], random_state=42)

    # Processing and saving images for train, val, and test splits
    for split, data in zip(['train', 'val'], [train_data, val_data]):
        process_and_save_images(image_dir, data, json_data, output_dir, split)

if __name__ == "__main__":
    root = "/home/youmin/workspace/VFMs-Adapters-Ensemble/dental-disease-classification"
    json_file_path = os.path.join(root, "dentex2023/train/train_quadrant_enumeration_disease.json")
    image_dir = os.path.join(root, "dentex2023/train/xrays")  
    
    # Loading the JSON data
    json_data = load_json_data(json_file_path)

    # Setting the directory to save cropped images
    output_dir = "cropped_images"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    all_annotations = []
    for idx, example in enumerate(json_data['images']):
        json_info = json_data['annotations'][idx]
        all_annotations.append((example, json_info, idx))


    first_example = all_annotations[0]
    first_image_info = first_example[0]  
    # {'height': 1316, 'width': 2744, 'id': 1, 'file_name': 'train_673.png'}
    first_bbox = first_example[1]['bbox']
    # [542.0, 698.0, 220.0, 271.0]
    
    # 이미지 로드
    first_image_path = os.path.join(image_dir, first_image_info['file_name'])
    first_image = load_image(first_image_path)
    
    # 시각화
    # visualize_bbox(first_image, first_bbox)
    
    test_img = load_image('/home/youmin/workspace/VFMs-Adapters-Ensemble/dental-disease-classification/dentex2023/train/xrays/train_199.png')
    test_bbox = [782.0, 715.0, 232.0, 289.0]
    
    visualize_bbox(test_img, test_bbox)

    # Splitting the dataset and cropping and saving images based on category_id_3
    split_and_save_data(image_dir, all_annotations, output_dir)
