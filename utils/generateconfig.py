import glob, os
import shutil
import os

def setup_config(data_location,model_name):
    # To be used only in the ssd/ level
    from utils import generateset as gens
    from utils import generateconfig as genc
    from config.models import ssd300
    #TODO : add more models and give the user a choice 
    ### Generating the database architecture and moving data accordingly
    gens.PascalVOC_generate(data_location)

    ### Generating the config
    conf=genc.generate_data_config(data_location,model_name,ssd300)
    new_file_config = open(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))),'config',model_name+'.py'),'w')
    new_file_config.write(conf)
    new_file_config.close()

    ##add the new config in the __init__ file of the config
    #TODO : only add it if it doesn't exist
    file_config = open(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))),'config','__init__.py'),'a')
    file_config.write('from . import {} \n'.format(model_name))
    file_config.close()


    ### Launching data labelling tool
    #TODO : adjust classes of the tool to the model classes
    label_tool_path=os.path.join(os.path.dirname(os.path.realpath(__file__)),'labelimg','labelImg.exe')
    os.system('{}'.format(label_tool_path))

def generate_data_config(datapath,
                    modelName,
                    modelConfig,
                    epochs=40,
                    batch_size=16,
                    predict_batch_size=1):
    print('Generating new configuration ')
    config='import os \n \n'
    config += 'ROOT_FOLDER=os.path.dirname(os.path.dirname(os.path.realpath(__file__)))' + '\n'
    config += 'DATA_DIR = \'{}\''.format(datapath.replace('\\','\\\\')) + '\n'
    config += 'IM_DIR = os.path.join(DATA_DIR,\'Images\')' +'\n'
    config += 'SETS_DIR = os.path.join(DATA_DIR,\'ImageSets\')' + '\n'
    config += 'IMG_SHAPE = {}'.format(modelConfig.IMG_SHAPE) +'\n'
    config += 'CLASSES = {}'.format(modelConfig.CLASSES) + '\n'
    config += 'CHECKPOINT_NAME= \'{}'.format(modelName+'_checkpoint.h5\' \n')
    config += 'MODEL_NAME = \'{}'.format(modelName +'.h5\' \n')
    config += 'EPOCHS= {}'.format(epochs) + '\n'
    config += 'BATCH_SIZE={}'.format(batch_size) + '\n'
    config += 'PREDICT_BATCH_SIZE = {}'.format(predict_batch_size) + '\n'
    return(config)

def get_config_from_name(config_name):
    #To be used only in the ssd/ level
    import imp
    import importlib
    import config
    # config = importlib.import_module('..config')
    imp.reload(config)
    configuration= importlib.import_module('config.{}'.format(config_name))
    return configuration
