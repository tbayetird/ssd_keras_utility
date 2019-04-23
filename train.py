from config import flowers,pirogues_mer,pirogues_plage
from ssd_train import train_VOC
from utils import generateset as gs

gs.PascalVOC_set_generate(pirogues_plage.IM_DIR,pirogues_plage.SETS_DIR,20)
train_VOC(pirogues_plage)

# gs.PascalVOC_set_generate(pirogues_mer.IM_DIR,pirogues_mer.SETS_DIR,20)
# train_VOC(pirogues_mer)
