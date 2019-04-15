import os

ROOT_FOLDER=os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

PATH = os.path.join(ROOT_FOLDER,'models','ssd300_pirogues.h5')
IMG_SHAPE=[300,300,3]
CLASSES=['background','pirogue']
