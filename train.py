from config import flowers,pirogues_mer,pirogues_plage
from ssd_train import train_VOC
from utils import generateset as gs


# train_VOC(pirogues_plage)

gs.PascalVOC_generate(pirogues_mer.IM_DIR,pirogues_mer.SETS_DIR,20)
train_VOC(pirogues_mer)
