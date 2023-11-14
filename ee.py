import cv2
import tensorflow as tf
# import tensorflow_datasets as tfds
from tensorflow import keras
from keras import optimizers
import keras_cv
import numpy as np
from keras_cv import bounding_box
import os
# import resource

from keras_cv import visualization
import tqdm

pretrained_model = keras_cv.models.RetinaNet.from_preset(
    "retinanet_resnet50_pascalvoc", bounding_box_format="xywh"
)
"""
To use the `RetinaNet` architecture with a ResNet50 backbone, you'll need to
    resize your image to a size that is divisible by 64.  This is to ensure
    compatibility with the number of downscaling operations done by the convolution
    layers in the ResNet.
"""
inference_resizing = keras_cv.layers.Resizing(
    640, 640, pad_to_aspect_ratio=True, bounding_box_format="xywh"
)

class_ids = [
    "Aeroplane",
    "Bicycle",
    "Bird",
    "Boat",
    "Bottle",
    "Bus",
    "Car",
    "Cat",
    "Chair",
    "Cow",
    "Dining Table",
    "Dog",
    "Horse",
    "Motorbike",
    "Person",
    "Potted Plant",
    "Sheep",
    "Sofa",
    "Train",
    "Tvmonitor",
    "Total",
]

class_mapping = dict(zip(range(len(class_ids)), class_ids))

prediction_decoder = keras_cv.layers.MultiClassNonMaxSuppression(
    bounding_box_format="xywh",
    from_logits=True,
    # Decrease the required threshold to make predictions get pruned out
    iou_threshold=0.2,
    # Tune confidence threshold for predictions to pass NMS
    confidence_threshold=0.7,
)

pretrained_model.prediction_decoder = prediction_decoder

BATCH_SIZE = 4


def visualize_dataset(inputs, value_range, rows, cols, bounding_box_format):
    inputs = next(iter(inputs.take(1)))
    images, bounding_boxes = inputs["images"], inputs["bounding_boxes"]
    visualization.plot_bounding_box_gallery(
        images,
        value_range=value_range,
        rows=rows,
        cols=cols,
        y_true=bounding_boxes,
        scale=5,
        font_scale=0.7,
        bounding_box_format=bounding_box_format,
        class_mapping=class_mapping,
    )


def unpackage_raw_tfds_inputs(inputs, bounding_box_format):
    image = inputs["image"]
    boxes = keras_cv.bounding_box.convert_format(
        inputs["objects"]["bbox"],
        images=image,
        source="rel_yxyx",
        target=bounding_box_format,
    )
    bounding_boxes = {
        "classes": tf.cast(inputs["objects"]["label"], dtype=tf.float32),
        "boxes": tf.cast(boxes, dtype=tf.float32),
    }
    return {"images": tf.cast(image, tf.float32), "bounding_boxes": bounding_boxes}


# def load_pascale_voc_dataset(split, path, bounding_box_format):
#     """
#     split: train, valid, test is a string that checks if i have train, valid, test folders in my path or train, valid or just train
#     path: the path to the dataset
#     bounding_box_format: xywh or yxyx
#     the expected folder structure is:
#     path
#         train
#             images
#                 1.jpg
#                 2.jpg
#                 ...
#             annotations
#                 1.xml
#                 2.xml
#                 ...
#         valid
#             images
#                 1.jpg
#                 2.jpg
#                 ...
#             annotations
#                 1.xml
#                 2.xml
#                 ...
#         test
#             images
#                 1.jpg
#                 2.jpg
#                 ...
#             annotations
#                 1.xml
#                 2.xml
#                 ...
#     """
#     import xml.etree.ElementTree as ET
#     images_path = os.path.join(path, split, "images")
#     annotations_path = os.path.join(path, split, "annotations")
#
#     image_files = [file for file in os.listdir(images_path) if file.endswith('.jpg')]
#
#     data = []
#
#     for img_file in image_files:
#         image_path = os.path.join(images_path, img_file)
#         annotation_path = os.path.join(annotations_path, os.path.splitext(img_file)[0] + ".xml")
#
#         image = cv2.imread(image_path)
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         height, width, _ = image.shape
#
#         tree = ET.parse(annotation_path)
#         root = tree.getroot()
#
#         boxes = []
#         classes = []
#
#         for obj in root.findall('object'):
#             class_name = obj.find('name').text
#             xmin = int(obj.find('bndbox').find('xmin').text)
#             ymin = int(obj.find('bndbox').find('ymin').text)
#             xmax = int(obj.find('bndbox').find('xmax').text)
#             ymax = int(obj.find('bndbox').find('ymax').text)
#
#             # Convert to the desired bounding box format (e.g., "xywh")
#             if bounding_box_format == "xywh":
#                 x = xmin
#                 y = ymin
#                 w = xmax - xmin
#                 h = ymax - ymin
#                 boxes.append([x, y, w, h])
#
#             # Append the class label
#             classes.append(class_name)
#
#         data.append({"image": image, "height": height, "width": width, "boxes": boxes, "classes": classes})
#
#     return data
import os
import tensorflow as tf
from keras.preprocessing.image import load_img
import xml.etree.ElementTree as ET


def load_custom_dataset(split, path, bounding_box_format):
    if split not in ['train', 'valid', 'test']:
        raise ValueError("Split should be 'train', 'valid', or 'test'.")

    images_path = os.path.join(path, split, 'images')
    annotations_path = os.path.join(path, split, 'annotations')

    dataset = []

    for filename in os.listdir(annotations_path):
        if filename.endswith('.xml'):
            annotation_path = os.path.join(annotations_path, filename)
            image_filename = filename.replace('.xml', '.jpg')
            image_path = os.path.join(images_path, image_filename)

            if os.path.exists(image_path):
                image = load_img(image_path)
                bounding_boxes = parse_annotation(annotation_path, bounding_box_format)
                if bounding_boxes:
                    dataset.append({"image": image, "bounding_boxes": bounding_boxes})

    return tf.data.Dataset.from_tensor_slices(dataset)


def parse_annotation(annotation_path, bounding_box_format):
    tree = ET.parse(annotation_path)
    root = tree.getroot()

    boxes = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        box = obj.find('bndbox')
        xmin = int(box.find('xmin').text)
        ymin = int(box.find('ymin').text)
        xmax = int(box.find('xmax').text)
        ymax = int(box.find('ymax').text)

        if bounding_box_format == 'xywh':
            x = (xmin + xmax) / 2.0
            y = (ymin + ymax) / 2.0
            w = xmax - xmin
            h = ymax - ymin
            boxes.append({"class": name, "x": x, "y": y, "w": w, "h": h})
        elif bounding_box_format == 'yxyx':
            boxes.append({"class": name, "ymin": ymin, "xmin": xmin, "ymax": ymax, "xmax": xmax})

    return boxes


train_ds = load_custom_dataset(split='train', path='ds', bounding_box_format='xywh')
valid_ds = load_custom_dataset(split='valid', path='ds', bounding_box_format='xywh')
train_ds = train_ds.shuffle(BATCH_SIZE * 4)

train_ds = train_ds.ragged_batch(BATCH_SIZE, drop_remainder=True)
valid_ds = valid_ds.ragged_batch(BATCH_SIZE, drop_remainder=True)

visualize_dataset(
    train_ds, bounding_box_format="xywh", value_range=(0, 255), rows=2, cols=2
)

visualize_dataset(
    valid_ds,
    bounding_box_format="xywh",
    value_range=(0, 255),
    rows=2,
    cols=2,
    # If you are not running your experiment on a local machine, you can also
    # make `visualize_dataset()` dump the plot to a file using `path`:
    # path="eval.png"
)

augmenter = keras.Sequential(
    layers=[
        keras_cv.layers.RandomFlip(mode="horizontal", bounding_box_format="xywh"),
        keras_cv.layers.JitteredResize(
            target_size=(640, 640), scale_factor=(0.75, 1.3), bounding_box_format="xywh"
        ),
    ]
)

train_ds = train_ds.map(augmenter, num_parallel_calls=tf.data.AUTOTUNE)
visualize_dataset(
    train_ds, bounding_box_format="xywh", value_range=(0, 255), rows=2, cols=2
)

inference_resizing = keras_cv.layers.Resizing(
    640, 640, bounding_box_format="xywh", pad_to_aspect_ratio=True
)
valid_ds = valid_ds.map(inference_resizing, num_parallel_calls=tf.data.AUTOTUNE)


visualize_dataset(
    valid_ds, bounding_box_format="xywh", value_range=(0, 255), rows=2, cols=2
)


def dict_to_tuple(inputs):
    return inputs["images"], bounding_box.to_dense(
        inputs["bounding_boxes"], max_boxes=32
    )


base_lr = 0.005
# including a global_clipnorm is extremely important in object detection tasks
optimizer = tf.keras.optimizers.SGD(
    learning_rate=base_lr, momentum=0.9, global_clipnorm=10.0
)

pretrained_model.compile(
    classification_loss="focal",
    box_loss="smoothl1",
)

result = pretrained_model.evaluate(valid_ds.take(1), verbose=1)

model = keras_cv.models.RetinaNet.from_preset(
    "resnet50_imagenet",
    num_classes=len(class_mapping),
    # For more info on supported bounding box formats, visit
    # https://keras.io/api/keras_cv/bounding_box/
    bounding_box_format="xywh",
)

model.compile(
    classification_loss="focal",
    box_loss="smoothl1",
    optimizer=optimizer,
    # We will use our custom callback to evaluate COCO metrics
    metrics=None,
)

model.fit(
    train_ds.take(20),
    validation_data=valid_ds.take(20),
    # Run for 10-35~ epochs to achieve good scores.
    epochs=1
    # callbacks=[EvaluateCOCOMetricsCallback(eval_ds.take(20))],
)

model = keras_cv.models.RetinaNet.from_preset(
    "retinanet_resnet50_pascalvoc", bounding_box_format="xywh"
)

visualization_ds = valid_ds.unbatch()
visualization_ds = visualization_ds.ragged_batch(16)
visualization_ds = visualization_ds.shuffle(8)


def visualize_detections(model, dataset, bounding_box_format):
    images, y_true = next(iter(dataset.take(1)))
    y_pred = model.predict(images)
    y_pred = bounding_box.to_ragged(y_pred)
    visualization.plot_bounding_box_gallery(
        images,
        value_range=(0, 255),
        bounding_box_format=bounding_box_format,
        y_true=y_true,
        y_pred=y_pred,
        scale=4,
        rows=2,
        cols=4,
        show=True,
        font_scale=0.7,
        class_mapping=class_mapping,
    )


model.prediction_decoder = keras_cv.layers.MultiClassNonMaxSuppression(
    bounding_box_format="xywh",
    from_logits=True,
    iou_threshold=0.5,
    confidence_threshold=0.75,
)

visualize_detections(model, dataset=visualization_ds, bounding_box_format="xywh")

stable_diffusion = keras_cv.models.StableDiffusionV2(512, 512)
images = stable_diffusion.text_to_image(
    prompt="A zoomed out photograph of a cool looking cat.  The cat stands in a beautiful forest",
    negative_prompt="unrealistic, bad looking, malformed",
    batch_size=4,
    seed=1231,
)
y_pred = model.predict(images)
visualization.plot_bounding_box_gallery(
    images,
    value_range=(0, 255),
    y_pred=y_pred,
    rows=2,
    cols=2,
    scale=5,
    font_scale=0.7,
    bounding_box_format="xywh",
    class_mapping=class_mapping,
)
