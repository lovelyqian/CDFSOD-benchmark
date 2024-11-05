import json
import random

def filter_k_shot_json(input_file, output_file, k_shot):
    # Read the COCO FORMAT JSON data
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    images = data['images']
    annotations = data['annotations']
    categories = data['categories']

    # Build a mapping from image_id to image information
    image_id_to_image = {image['id']: image for image in images}

    category_to_image_annotations = {}
    
    # Group annotations by category and image
    for annotation in annotations:
        category_id = annotation['category_id']
        image_id = annotation['image_id']
        
        if category_id not in category_to_image_annotations:
            category_to_image_annotations[category_id] = {}
        
        if image_id not in category_to_image_annotations[category_id]:
            category_to_image_annotations[category_id][image_id] = annotation
    
    selected_annotations = []
    selected_image_ids = set()
    
    # For each category, randomly select k images and ensure 1 annotation per image
    for category_id, image_annotations in category_to_image_annotations.items():
        image_ids = list(image_annotations.keys())
        
        if len(image_ids) <= k_shot:
            selected_image_ids.update(image_ids)
            selected_annotations.extend(image_annotations[image_id] for image_id in image_ids)
        else:
            selected_image_ids_k = random.sample(image_ids, k_shot)
            selected_image_ids.update(selected_image_ids_k)
            selected_annotations.extend(image_annotations[image_id] for image_id in selected_image_ids_k)
    
    # Filter the images based on the selected image IDs
    selected_images = [image_id_to_image[image_id] for image_id in selected_image_ids]

    # Construct the new JSON data structure
    kshot_data = {
        "images": selected_images,
        "annotations": selected_annotations,
        "categories": categories  # Keep all categories unchanged
    }
    
    # Write the filtered data to the new JSON file
    with open(output_file, 'w') as f:
        json.dump(kshot_data, f, indent=4)
    
    print(f"{k_shot}-shot JSON data has been saved to {output_file}")

# Example usage
filter_k_shot_json('/path/to/file.json]', '1_shot.json', k_shot=1)
