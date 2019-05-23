from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler, TerminateOnNaN, CSVLogger
from keras.optimizers import Adam, SGD
from matplotlib import pyplot as plt
from keras.models import load_model
from keras import backend as K
from math import ceil
import numpy as np
import os

from keras_layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast
from keras_layers.keras_layer_DecodeDetections import DecodeDetections
from keras_layers.keras_layer_L2Normalization import L2Normalization
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_loss_function.keras_ssd_loss import SSDLoss
from models.keras_ssd300 import ssd_300

from ssd_encoder_decoder.ssd_output_decoder import decode_detections, decode_detections_fast
from data_generator.data_augmentation_chain_original_ssd import SSDDataAugmentation
from data_generator.object_detection_2d_misc_utils import apply_inverse_transforms
from data_generator.object_detection_2d_photometric_ops import ConvertTo3Channels
from data_generator.object_detection_2d_data_generator import DataGenerator
from data_generator.object_detection_2d_geometric_ops import Resize
from ssd_encoder_decoder.ssd_input_encoder import SSDInputEncoder


def train_VOC(config) :
    '''
    Train the given configuration ; the configuration must be constructed
    according to the utility script found in utils/generateconfig.py. 

    Arguments:
        config : the configuration of the model to use ; should already be
            loaded

    '''
    ###################################
    ### PATHS AND PARAMETERS
    ##################################
    datadir=config.DATA_DIR
    local_dir=config.ROOT_FOLDER
    img_shape=config.IMG_SHAPE
    classes = config.CLASSES
    checkpoint_output=os.path.join(local_dir,'models',config.CHECKPOINT_NAME)
    model_output=os.path.join(local_dir,'models',config.MODEL_NAME)
    img_height = img_shape[0] # Height of the model input images
    img_width = img_shape[1] # Width of the model input images
    img_channels = img_shape[2] # Number of color channels of the model input images
    mean_color = [123, 117, 104] # The per-channel mean of the images in the dataset. Do not change this value if you're using any of the pre-trained weights.
    swap_channels = [2, 1, 0] # The color channel order in the original SSD is BGR, so we'll have the model reverse the color channel order of the input images.
    n_classes = 20 # Number of positive classes, e.g. 20 for Pascal VOC, 80 for MS COCO
    scales_pascal = [0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05] # The anchor box scaling factors used in the original SSD300 for the Pascal VOC datasets
    scales = scales_pascal
    aspect_ratios = [[1.0, 2.0, 0.5],
                     [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                     [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                     [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                     [1.0, 2.0, 0.5],
                     [1.0, 2.0, 0.5]] # The anchor box aspect ratios used in the original SSD300; the order matters
    two_boxes_for_ar1 = True
    steps = [8, 16, 32, 64, 100, 300] # The space between two adjacent anchor box center points for each predictor layer.
    offsets = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5] # The offsets of the first anchor box center points from the top and left borders of the image as a fraction of the step size for each predictor layer.
    clip_boxes = False # Whether or not to clip the anchor boxes to lie entirely within the image boundaries
    variances = [0.1, 0.1, 0.2, 0.2] # The variances by which the encoded target coordinates are divided as in the original implementation
    normalize_coords = True
    batch_size = config.BATCH_SIZE # Change the batch size if you like, or if you run into GPU memory issues.

    ###################################
    ### BUILDING MODEL
    ##################################
    K.clear_session() # Clear previous models from memory.

    model = ssd_300(image_size=(img_height, img_width, img_channels),
                    n_classes=n_classes,
                    mode='training',
                    l2_regularization=0.0005,
                    scales=scales,
                    aspect_ratios_per_layer=aspect_ratios,
                    two_boxes_for_ar1=two_boxes_for_ar1,
                    steps=steps,
                    offsets=offsets,
                    clip_boxes=clip_boxes,
                    variances=variances,
                    normalize_coords=normalize_coords,
                    subtract_mean=mean_color,
                    swap_channels=swap_channels)

    weights_path = os.path.join(local_dir,'weights','VGG_VOC0712_SSD_300x300_iter_120000.h5')
    model.load_weights(weights_path, by_name=True)

    #adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    sgd = SGD(lr=0.001, momentum=0.9, decay=0.0, nesterov=False)
    ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)
    model.compile(optimizer=sgd, loss=ssd_loss.compute_loss)

    ###################################
    ### LOADING DATAS
    ##################################
    train_dataset = DataGenerator(load_images_into_memory=False, hdf5_dataset_path=None)
    val_dataset = DataGenerator(load_images_into_memory=False, hdf5_dataset_path=None)
    images_dir                   = os.path.join(datadir,'Images')
    annotations_dir              = os.path.join(datadir,'Annotations')
    trainval_image_set_filename  = os.path.join(datadir,'ImageSets','train.txt')
    test_image_set_filename      = os.path.join(datadir,'ImageSets','val.txt')

    # The XML parser needs to now what object class names to look for and in which order to map them to integers.
    #

    train_dataset.parse_xml(images_dirs=[images_dir],
                        image_set_filenames=[trainval_image_set_filename],
                        annotations_dirs=[annotations_dir],
                        classes=classes,
                        include_classes='all',
                        exclude_truncated=False,
                        exclude_difficult=False,
                        ret=False)

    val_dataset.parse_xml(images_dirs=[images_dir],
                          image_set_filenames=[test_image_set_filename],
                          annotations_dirs=[annotations_dir],
                          classes=classes,
                          include_classes='all',
                          exclude_truncated=False,
                          exclude_difficult=True,
                          ret=False)

    train_dataset.create_hdf5_dataset(file_path='flowers_train.h5',
                                  resize=False,
                                  variable_image_size=True,
                                  verbose=True)

    val_dataset.create_hdf5_dataset(file_path='flowers_val.h5',
                                    resize=False,
                                    variable_image_size=True,
                                    verbose=True)


    ssd_data_augmentation = SSDDataAugmentation(img_height=img_height,
                                                img_width=img_width,
                                                background=mean_color)
    convert_to_3_channels = ConvertTo3Channels()
    resize = Resize(height=img_height, width=img_width)

    # The encoder constructor needs the spatial dimensions of the model's predictor layers to create the anchor boxes.
    predictor_sizes = [model.get_layer('conv4_3_norm_mbox_conf').output_shape[1:3],
                       model.get_layer('fc7_mbox_conf').output_shape[1:3],
                       model.get_layer('conv6_2_mbox_conf').output_shape[1:3],
                       model.get_layer('conv7_2_mbox_conf').output_shape[1:3],
                       model.get_layer('conv8_2_mbox_conf').output_shape[1:3],
                       model.get_layer('conv9_2_mbox_conf').output_shape[1:3]]

    ssd_input_encoder = SSDInputEncoder(img_height=img_height,
                                        img_width=img_width,
                                        n_classes=n_classes,
                                        predictor_sizes=predictor_sizes,
                                        scales=scales,
                                        aspect_ratios_per_layer=aspect_ratios,
                                        two_boxes_for_ar1=two_boxes_for_ar1,
                                        steps=steps,
                                        offsets=offsets,
                                        clip_boxes=clip_boxes,
                                        variances=variances,
                                        matching_type='multi',
                                        pos_iou_threshold=0.5,
                                        neg_iou_limit=0.5,
                                        normalize_coords=normalize_coords)

    train_generator = train_dataset.generate(batch_size=batch_size,
                                             shuffle=True,
                                             transformations=[ssd_data_augmentation],
                                             label_encoder=ssd_input_encoder,
                                             returns={'processed_images',
                                                      'encoded_labels'},
                                             keep_images_without_gt=False)

    val_generator = val_dataset.generate(batch_size=batch_size,
                                         shuffle=False,
                                         transformations=[convert_to_3_channels,
                                                          resize],
                                         label_encoder=ssd_input_encoder,
                                         returns={'processed_images',
                                                  'encoded_labels'},
                                         keep_images_without_gt=False)

    # Get the number of samples in the training and validations datasets.
    train_dataset_size = train_dataset.get_dataset_size()
    val_dataset_size   = val_dataset.get_dataset_size()

    print("Number of images in the training dataset:\t{:>6}".format(train_dataset_size))
    print("Number of images in the validation dataset:\t{:>6}".format(val_dataset_size))

    ###################################
    ### PREPARE TRAINING
    ##################################

    def lr_schedule(epoch):
        if epoch < 80:
            return 0.001
        elif epoch < 100:
            return 0.0001
        else:
            return 0.00001

    model_checkpoint = ModelCheckpoint(filepath=checkpoint_output,
                                       monitor='val_loss',
                                       verbose=1,
                                       save_best_only=True,
                                       save_weights_only=False,
                                       mode='auto',
                                       period=1)

    early_stopping = EarlyStopping(monitor='val_loss',
                                       min_delta=0.0,
                                       patience=10,
                                       verbose=1)

    learning_rate_scheduler = LearningRateScheduler(schedule=lr_schedule,
                                                    verbose=1)

    terminate_on_nan = TerminateOnNaN()

    callbacks = [model_checkpoint,
                 learning_rate_scheduler,
                 terminate_on_nan,
                 early_stopping
                 ]

    ###################################
    ### TRAINING
    ##################################
    epochs=config.EPOCHS
    steps_per_epoch = ceil(train_dataset_size/batch_size)
    model.summary()
    history = model.fit_generator(
                                  generator=train_generator,
                                  steps_per_epoch=steps_per_epoch,
                                  epochs=epochs,
                                  callbacks=callbacks,
                                  validation_data=val_generator,
                                  validation_steps=ceil(val_dataset_size/batch_size)
                                  )

    model.save(model_output)

# train_VOC(flowers)
