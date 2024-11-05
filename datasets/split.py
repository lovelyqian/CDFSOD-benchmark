import json
import random

def split_dataset(input_file, train_output_file, test_output_file, train_size=0.8):
    # Load the input JSON data (which is in COCO-like format)
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    images = data['images']
    annotations = data['annotations']
    categories = data['categories']
    
    random.shuffle(images)
    
    num_train = int(len(images) * train_size)
    
    # Split images into train and test sets
    train_images = images[:num_train]   
    test_images = images[num_train:]    
    
    # Get image IDs for train and test sets
    train_image_ids = [image['id'] for image in train_images]
    test_image_ids = [image['id'] for image in test_images]
    
    # Filter annotations based on the image IDs
    train_annotations = [annotation for annotation in annotations if annotation['image_id'] in train_image_ids]
    test_annotations = [annotation for annotation in annotations if annotation['image_id'] in test_image_ids]
    
    # Construct the train and test datasets
    train_data = {
        "images": train_images,
        "annotations": train_annotations,
        "categories": categories  
    }
    
    test_data = {
        "images": test_images,
        "annotations": test_annotations,
        "categories": categories 
    }
    
    # Save the train and test datasets into separate JSON files
    with open(train_output_file, 'w') as f:
        json.dump(train_data, f, indent=4)
    
    with open(test_output_file, 'w') as f:
        json.dump(test_data, f, indent=4)
    
    print(f"Train dataset saved to {train_output_file}")
    print(f"Test dataset saved to {test_output_file}")

split_dataset('data.json', 'train.json', 'test.json', train_size=0.8)
