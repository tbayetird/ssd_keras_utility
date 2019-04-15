import os

ROOT_FOLDER=os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
DATA_DIR = 'D:\\datas\\pirogues-plage\\'
IMG_SHAPE=[300,300,3]
CLASSES=['background','pirogue']
CHECKPOINT_NAME='ssd300_pirogues_plage_checkpoint.h5'
MODEL_NAME = 'ssd300_pirogues_plage.h5'
EPOCHS=25
BATCH_SIZE=32
PREDICT_BATCH_SIZE=1
