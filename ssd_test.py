from keras import backend as K
from keras.models import load_model
import numpy as np
import os, cv2
from matplotlib import pyplot as plt

from keras_loss_function.keras_ssd_loss import SSDLoss
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_layers.keras_layer_DecodeDetections import DecodeDetections
from keras_layers.keras_layer_L2Normalization import L2Normalization
from ssd_encoder_decoder.ssd_output_decoder import decode_detections, decode_detections_fast

from data_generator.object_detection_2d_data_generator import DataGenerator
from data_generator.object_detection_2d_geometric_ops import Resize
from data_generator.object_detection_2d_photometric_ops import ConvertTo3Channels
from data_generator.object_detection_2d_misc_utils import apply_inverse_transforms

def test_config(config):
    '''
    Test the given configuration ; the configuration should already have been
    used for training purposes, or this will return an error (see ssd_train.py)

    Arguments:
        config : the configuration of the model to use ; should already be
            loaded.

    '''
    local_dir = config.ROOT_FOLDER
    data_dir = config.DATA_DIR
    img_shape=config.IMG_SHAPE
    img_height = img_shape[0] # Height of the model input images
    img_width = img_shape[1] # Width of the model input images
    img_channels = img_shape[2] # Number of color channels of the model input images
    n_classes = 20 # Number of positive classes, e.g. 20 for Pascal VOC, 80 for MS COCO
    normalize_coords = True

    K.clear_session() # Clear previous models from memory.
    print("[INFO] loading model...")
    model_path = os.path.join(local_dir,'models',config.MODEL_NAME)

    # We need to create an SSDLoss object in order to pass that to the model loader.
    ssd_loss = SSDLoss(neg_pos_ratio=3, n_neg_min=0, alpha=1.0)
    model = load_model(model_path, custom_objects={'AnchorBoxes': AnchorBoxes,
                                                   'L2Normalization': L2Normalization,
                                                   'DecodeDetections': DecodeDetections,
                                                   'compute_loss': ssd_loss.compute_loss})
    classes=config.CLASSES
    dataset = DataGenerator(load_images_into_memory=False, hdf5_dataset_path=None)
    dataset_images_dir      = os.path.join(data_dir,'Images')
    dataset_annotations_dir      = os.path.join(data_dir,'Annotations/')
    dataset_test_image_set_filename     = os.path.join(data_dir,'ImageSets\\test.txt')

    dataset.parse_xml(images_dirs=[dataset_images_dir],
                          image_set_filenames=[dataset_test_image_set_filename],
                          annotations_dirs=[dataset_annotations_dir],
                          classes=classes,
                          include_classes='all',
                          exclude_truncated=False,
                          exclude_difficult=True,
                          ret=False)
    dataset.create_hdf5_dataset(file_path=config.MODEL_NAME,
                                    resize=False,
                                    variable_image_size=True,
                                    verbose=True)

    convert_to_3_channels = ConvertTo3Channels()
    resize = Resize(height=img_height, width=img_width)
    dataset_size   = dataset.get_dataset_size()

    print("Number of images in the dataset:\t{:>6}".format(dataset_size))

    predict_generator = dataset.generate(batch_size=config.PREDICT_BATCH_SIZE,
                                             shuffle=True,
                                             transformations=[convert_to_3_channels,
                                                              resize],
                                             label_encoder=None,
                                             returns={'processed_images',
                                                      'filenames',
                                                      'inverse_transform',
                                                      'original_images',
                                                      'original_labels'},
                                             keep_images_without_gt=False)

    count=0
    while True and count <dataset_size:
        batch_images, batch_filenames, batch_inverse_transforms, batch_original_images, batch_original_labels = next(predict_generator)
        i=0
        print("Image:", batch_filenames[i])
        print()
        print("Ground truth boxes:\n")
        print(np.array(batch_original_labels[i]))

        y_pred = model.predict(batch_images)
        y_pred_decoded = decode_detections(y_pred,
                                       confidence_thresh=0.5,
                                       iou_threshold=0.4,
                                       top_k=200,
                                       normalize_coords=normalize_coords,
                                       img_height=img_height,
                                       img_width=img_width)
        y_pred_decoded_inv = apply_inverse_transforms(y_pred_decoded, batch_inverse_transforms)

        np.set_printoptions(precision=2, suppress=True, linewidth=90)
        print("Predicted boxes:\n")
        print('   class   conf xmin   ymin   xmax   ymax')
        print(y_pred_decoded_inv[i])
        # cv2.imshow('original image',batch_original_images[i])
        # cv2.waitKey(800)
        # cv2.destroyAllWindows()
        colors = plt.cm.hsv(np.linspace(0, 1, n_classes+1)).tolist()
        plt.figure(figsize=(15,8))
        plt.imshow(batch_original_images[i])

        current_axis = plt.gca()
        len_orig=0
        for box in batch_original_labels[i]:
            len_orig+=1
            xmin = box[1]
            ymin = box[2]
            xmax = box[3]
            ymax = box[4]
            label = '{}'.format(classes[int(box[0])])
            current_axis.add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, color='green', fill=False, linewidth=2))
            current_axis.text(xmin, ymin, label, size='x-large', color='white', bbox={'facecolor':'green', 'alpha':1.0})

        len_found=0
        for box in y_pred_decoded_inv[i]:
            len_found+=1
            xmin = box[2]
            ymin = box[3]
            xmax = box[4]
            ymax = box[5]
            color = colors[int(box[0])]
            label = '{}: {:.2f}'.format(classes[int(box[0])], box[1])
            current_axis.add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, color=color, fill=False, linewidth=2))
            current_axis.text(xmin, ymin, label, size='x-large', color='white', bbox={'facecolor':color, 'alpha':1.0})

        print('Number of original boxes : {}'.format(len_orig))
        print('Number of found boxes : {}'.format(len_found))
        plt.show()
        count+=1
