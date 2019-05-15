import argparse
ap = argparse.ArgumentParser()
ap.add_argument("-n", "--name", required=True,
	help="name of the configuration file related to the dataset to be tested")
args = vars(ap.parse_args())

from utils import generateconfig as gc
# from utils.generateconfig import get_config_from_name
from ssd_test import test_config
test_config(gc.get_config_from_name(args['name']))
