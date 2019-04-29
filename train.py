import argparse
ap = argparse.ArgumentParser()
ap.add_argument("-n", "--name", required=True,
	help="name of the configuration file related to the dataset to be trained")
args = vars(ap.parse_args())

from utils import generateconfig as gc
from ssd_train import train_VOC
train_VOC(gc.get_config_from_name(args['name']))
