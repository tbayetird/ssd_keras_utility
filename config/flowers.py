import os

ROOT_FOLDER=os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
DATA_DIR = 'D:\\datas\\test_perso\\'
IMG_SHAPE=[300,300,3]
CLASSES=['background','car']
CHECKPOINT_NAME='ssd300_flowers_checkpoint.h5'
MODEL_NAME = 'ssd300_flowers.h5'
EPOCHS=25
BATCH_SIZE=8
