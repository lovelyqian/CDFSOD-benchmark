# 1. Convert Datasets to COCO Format
Before starting, ensure that your datasets are converted into the COCO format (JSON file). The COCO format is widely used for object detection tasks and consists of three main components:

**images**: Contains the metadata of the images (ID, file name, dimensions, etc.).

**annotations**: Contains the annotations for each image, including the bounding box coordinates, segmentation masks, and category IDs.

**categories**: Contains the list of object categories (classes).

You can use custom scripts or available tools to convert datasets from various formats (Pascal VOC, YOLO, etc.) into the COCO format. Once the datasets are in COCO format, proceed to the next step.

# 2. Split the Dataset into Training and Testing Sets
After converting your dataset into COCO format, use the provided [split.py](https://github.com/lovelyqian/CDFSOD-benchmark/blob/main/datasets/split.py) script to split the dataset into training and testing sets. The script randomly divides the dataset into the specified proportions (e.g., 80% for training and 20% for testing).

Here’s how to use the split.py:
### 1.Place your COCO format dataset (JSON file) in the appropriate directory.
### 2.Run the split.py command to split the dataset

This script will output two new JSON files:

1.train.json: Contains the training data.

2.test.json: Contains the testing data.

(If the dataset has its own partitions, we follow their original partitions)

# 3. Select k-shot Samples from the Training Set
After splitting the dataset, you can select a suitable number of samples from the training set for k-shot learning. k-shot refers to selecting a small number (k) of examples for each category in the dataset.

To generate a k-shot dataset from the training set, follow these steps:

### 1.Use the training dataset generated from the split (e.g., train.json).
### 2.Run a custom script to extract k-shot samples for each category from the training dataset. You can set k to 1, 5 or 10 as needed for your k-shot learning task.（make sure always to be one annotation in a picture）

Once this process is complete, the kshot_train.json will contain the selected k-shot data, which can be used for training k-shot models.

# 4. Generate Prototypes
Edit and run [build_prototypes.sh](https://github.com/lovelyqian/CDFSOD-benchmark/blob/main/build_prototypes.sh).
```
bash build_prototypes.sh
```

# 5. Adding Custom Datasets

Before training, update these two files: [lib/categories.py](https://github.com/lovelyqian/CDFSOD-benchmark/blob/main/lib/categories.py#L73), [detectron2/data/datasets/build.py](https://github.com/lovelyqian/CDFSOD-benchmark/blob/main/detectron2/data/datasets/builtin.py#L320). Then adding your datasets name in 'datasets_name'.
