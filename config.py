import os
from object_detection.utils import config_util
from google.protobuf import text_format
cwd = os.getcwd()

if not os.path.exists('model'):
    os.mkdir('model')

if not os.path.exists(f'{cwd}/result'):
    os.mkdir(f'{cwd}/result')

TRAIN_DATASET_PATH = f'{cwd}/dataset/train'
VALID_DATASET_PATH = f'{cwd}/dataset/valid'
TEST_DATASET_PATH = f'{cwd}/dataset/test'
MODEL_PATH = f'{cwd}/model'

MODEL = 'efficientdet_lite0'
MODEL_NAME = 'model.tflite'
CLASSES = ['BlackBall', 'BlueBall', 'BrownBall', 'GreenBall', 'Pile', 'PinkBall', 'RedBall', 'Table', 'WhiteBall',
           'YellowBall']
EPOCHS = 20
BATCH_SIZE = 4
