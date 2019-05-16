from keras import backend as K
from keras.models import load_model
from keras.preprocessing import image
from keras.optimizers import Adam
from imageio import imread
import numpy as np
import os, cv2
import gc
from matplotlib import pyplot as plt

from models.keras_ssd300 import ssd_300
from keras_loss_function.keras_ssd_loss import SSDLoss
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_layers.keras_layer_DecodeDetections import DecodeDetections
from keras_layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast
from keras_layers.keras_layer_L2Normalization import L2Normalization

from ssd_encoder_decoder.ssd_output_decoder import decode_detections, decode_detections_fast

from data_generator.object_detection_2d_data_generator import DataGenerator
from data_generator.object_detection_2d_photometric_ops import ConvertTo3Channels
from data_generator.object_detection_2d_geometric_ops import Resize
from data_generator.object_detection_2d_misc_utils import apply_inverse_transforms

from config.models import ssd300_pirogues_mer ,ssd300_road
from utils import filenav as fn
from utils import vidutils as vd
from utils.objecttracker import ObjectTracker



def inference_on_folder(model_config,
                        folder_path,
                        SHOW_ALL=False,
                        confidence_threshold=0.5,
                        DISPLAY=True
                        ):
        '''
        Launches inference from a keras neural network

        Arguments:
            weight_matrix (array): A 2D Numpy array that represents the weight matrix
                for the matching process. If `(m,n)` is the shape of the weight matrix,
                it must be `m <= n`. The weights can be integers or floating point
                numbers. The matching process will maximize, i.e. larger weights are
                preferred over smaller weights.
            threshold (float): A float that represents the threshold (i.e. lower bound)
                that must be met by a pair of elements to produce a match.

        Returns:
            Two 1D Numpy arrays of equal length that represent the matched indices. The first
            array contains the indices along the first axis of `weight_matrix`, the second array
            contains the indices along the second axis.
        '''
    #TODO : check parameters
    img_shape=model_config.IMG_SHAPE
    img_height = img_shape[0] # Height of the model input images
    img_width = img_shape[1] # Width of the model input images
    img_channels = img_shape[2] # Number of color channels of the model input images
    normalize_coords = True

    K.clear_session() # Clear previous models from memory.
    print("[INFO] loading model...")

    # We need to create an SSDLoss object in order to pass that to the model loader.
    ssd_loss = SSDLoss(neg_pos_ratio=3, n_neg_min=0, alpha=1.0)
    model = load_model(model_config.PATH, custom_objects={'AnchorBoxes': AnchorBoxes,
                                                   'L2Normalization': L2Normalization,
                                                   'DecodeDetections': DecodeDetections,
                                                   'compute_loss': ssd_loss.compute_loss})

    print("[INFO] loading datas ... ")
    orig_images = [] # Store the images here.
    input_images = [] # Store resized versions of the images here.
    image_paths= fn.findAllIn(folder_path)
    for img_path in image_paths :
        img_path=os.path.join(folder_path,img_path)
        orig_images.append(imread(img_path))
        img = image.load_img(img_path, target_size=(img_height, img_width))
        img = image.img_to_array(img)
        input_images.append(img)
    input_images = np.array(input_images)
    print("[INFO] making predictions ...  ")
    y_pred = model.predict(input_images)
    y_pred_thresh = [y_pred[k][y_pred[k,:,1] > confidence_threshold] for k in range(y_pred.shape[0])]

    y_pred_decoded=decode_detections(y_pred,
                                   confidence_thresh=confidence_threshold,
                                   iou_threshold=0.4,
                                   top_k=200,
                                   normalize_coords=normalize_coords,
                                   img_height=img_height,
                                   img_width=img_width)

    print("[INFO] add predictions and display ...")
    np.set_printoptions(precision=2, suppress=True, linewidth=90)

    classes = model_config.CLASSES
    colors = plt.cm.hsv(np.linspace(0, 1, len(classes))).tolist()

    for i in range(len(orig_images)):
        if(len(y_pred_decoded[i]) != 0 or SHOW_ALL ):
            # print(len(y_pred_decoded[i]))
            plt.figure(figsize=(10,8))
            plt.imshow(orig_images[i])
            current_axis = plt.gca()
            for box in y_pred_decoded[i]:
                # Transform the predicted bounding boxes for the 300x300 image to the original image dimensions.
                xmin = box[2] * orig_images[i].shape[1] / img_width
                ymin = box[3] * orig_images[i].shape[0] / img_height
                xmax = box[4] * orig_images[i].shape[1] / img_width
                ymax = box[5] * orig_images[i].shape[0] / img_height
                color = colors[int(box[0])]
                label = '{}: {:.2f}'.format(classes[int(box[0])], box[1])
                current_axis.add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, color=color, fill=False, linewidth=2))
                current_axis.text(xmin, ymin, label, size='x-large', color='white', bbox={'facecolor':color, 'alpha':1.0})
            if (DISPLAY):
                plt.show()

def inference_on_image(model_config,image_path):
    #TODO : check parameters
    img_shape=model_config.IMG_SHAPE
    img_height = img_shape[0] # Height of the model input images
    img_width = img_shape[1] # Width of the model input images
    img_channels = img_shape[2] # Number of color channels of the model input images
    normalize_coords = True

    K.clear_session() # Clear previous models from memory.

    ssd_loss = SSDLoss(neg_pos_ratio=3, n_neg_min=0, alpha=1.0)
    model_path = model_config.PATH
    model = load_model(model_path, custom_objects={'AnchorBoxes': AnchorBoxes,
                                                   'L2Normalization': L2Normalization,
                                                   'DecodeDetections': DecodeDetections,
                                                   'compute_loss': ssd_loss.compute_loss})


    orig_images = [] # Store the images here.
    input_images = [] # Store resized versions of the images here.

    # We'll only load one image in this example.
    img_path = image_path
    orig_images.append(imread(img_path))
    img = image.load_img(img_path, target_size=(img_height, img_width))
    img = image.img_to_array(img)
    input_images.append(img)
    input_images = np.array(input_images)
    y_pred = model.predict(input_images)
    confidence_threshold = 0.5
    y_pred_thresh = [y_pred[k][y_pred[k,:,1] > confidence_threshold] for k in range(y_pred.shape[0])]

    y_pred_decoded=decode_detections(y_pred,
                                   confidence_thresh=0.5,
                                   iou_threshold=0.4,
                                   top_k=200,
                                   normalize_coords=normalize_coords,
                                   img_height=img_height,
                                   img_width=img_width)

    # Set the colors for the bounding boxes
    classes = model_config.CLASSES
    colors = plt.cm.hsv(np.linspace(0, 1, len(classes))).tolist()

    plt.figure(figsize=(20,12))
    plt.imshow(orig_images[0])

    current_axis = plt.gca()

    for box in y_pred_decoded[0]:
        # Transform the predicted bounding boxes for the 300x300 image to the original image dimensions.
        xmin = box[2] * orig_images[0].shape[1] / img_width
        ymin = box[3] * orig_images[0].shape[0] / img_height
        xmax = box[4] * orig_images[0].shape[1] / img_width
        ymax = box[5] * orig_images[0].shape[0] / img_height
        color = colors[int(box[0])]
        label = '{}: {:.2f}'.format(classes[int(box[0])], box[1])
        current_axis.add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, color=color, fill=False, linewidth=2))
        current_axis.text(xmin, ymin, label, size='x-large', color='white', bbox={'facecolor':color, 'alpha':1.0})
    plt.show()

#TODO : check parameters
def inference_on_video(model_config,
                       video_path,
                       Display=True,
                       change_save_dir=False,
                       confidence_threshold=0.5,
                       tracking=False,
                       ot = ObjectTracker()):
    img_shape=model_config.IMG_SHAPE
    img_height = img_shape[0] # Height of the model input images
    img_width = img_shape[1] # Width of the model input images
    img_channels = img_shape[2] # Number of color channels of the model input images
    normalize_coords = True
    save_dir = os.path.join(model_config.ROOT_FOLDER,'outputs')
    if(change_save_dir):
        save_dir=os.path.join(os.path.dirname(video_path),'outputs')
    video_name=video_path.split('\\')
    video_name=video_name[-1]
    ###################################
    ### LOADING MODEL
    ##################################
    K.clear_session() # Clear previous models from memory.

    ssd_loss = SSDLoss(neg_pos_ratio=3, n_neg_min=0, alpha=1.0)
    model_path = model_config.PATH
    model = load_model(model_path, custom_objects={'AnchorBoxes': AnchorBoxes,
                                                   'L2Normalization': L2Normalization,
                                                   'DecodeDetections': DecodeDetections,
                                                   'compute_loss': ssd_loss.compute_loss})

    ###################################
    ### PROCESSING VIDEO STREAM
    ##################################
    print("[INFO] Processing video stream...")

    vs = cv2.VideoCapture(video_path)
    # (orig_images,input_images,out)=vd.handleVideoStreams(vs,save_dir,video_name,img_width,img_height)
    (orig_images,input_images,out)=vd.handleVideoStreams(vs,save_dir,video_name,img_width,img_height)
    # print('[DEBUG] Number of images : {}'.format(len(orig_images)))
    ###################################
    ### PREDICTING
    ##################################
    predName='pred_'+video_name
    print("[INFO] Predicting results on video stream  ")
    if(fn.exist(save_dir,predName)):
        print("[INFO] Loading pre-existing predictions")
        y_pred=np.load(os.path.join(save_dir,predName))
    else:
        y_pred = model.predict(input_images)
        np.save(os.path.join(save_dir,predName),y_pred)
    y_pred_thresh = [y_pred[k][y_pred[k,:,1] > confidence_threshold] for k in range(y_pred.shape[0])]

    # print('[DEBUG] Number of predictions : ', len(y_pred))
    # print('[DEBUG] Predictions : ', y_pred)
    ###################################
    ### APPLYING PREDICTIONS ON IMAGES
    ##################################
    print("[INFO] Applying predictions ")
    try :
        y_pred_decoded=decode_detections(y_pred,
                                       confidence_thresh=confidence_threshold,
                                       iou_threshold=0.4,
                                       top_k=200,
                                       normalize_coords=normalize_coords,
                                       img_height=img_height,
                                       img_width=img_width)
    except IndexError:
        y_pred_decoded = y_pred_thresh
        pass

    # print('[DEBUG] Number of kept predictions : ', len(y_pred_decoded))
    # print('[DEBUG] Predictions : ', y_pred_decoded)
    # Set the colors for the bounding boxes
    classes = model_config.CLASSES
    colors = plt.cm.hsv(np.linspace(0, 1, len(classes))).tolist()
    for i in range(len(y_pred_decoded)):
        img=orig_images[i]
        rects=[]
        for box in y_pred_decoded[i]:
            # Transform the predicted bounding boxes for the 300x300 image to the original image dimensions.
            xmin = int(box[2] * orig_images[i].shape[1] / img_width)
            ymin = int(box[3] * orig_images[i].shape[0] / img_height)
            xmax = int(box[4] * orig_images[i].shape[1] / img_width)
            ymax = int(box[5] * orig_images[i].shape[0] / img_height)
            color = colors[int(box[0])]
            label = '{}: {:.2f}'.format(classes[int(box[0])], box[1])
            cv2.putText(img,label,(xmin,ymin),cv2.FONT_HERSHEY_SIMPLEX,0.5,color,2)
            cv2.rectangle(img,(xmin,ymin),(xmax,ymax),color,2)
            rects.append((xmin,ymin,xmax,ymax))

        if (tracking):
            objects= ot.update(rects)
            numObj=0
            for (objectID,object) in objects.items():
                numObj+=1
                text = "ID {}".format(objectID)
                (xmin,ymin,xmax,ymax)=object.getRect()
                centroid = object.getCentroid()
                predictedCentroid = object.getPredictedCentroid()
                cv2.putText(img,text,(xmax-30,ymin),
                cv2.FONT_HERSHEY_SIMPLEX,0.5,(40,40,255),2)
                cv2.circle(img,(centroid[0],centroid[1]),4,(0,255,0),-1)
                cv2.circle(img,(predictedCentroid[0],predictedCentroid[1]),4,(0,0,255),-1)
                cv2.line(img,(centroid[0],centroid[1]),
                        (predictedCentroid[0],predictedCentroid[1]),(0,0,255))

        out.write(img)

    ###################################
    ### DISPLAYING RESULTS
    ##################################
    if(Display):
        print("[INFO] Displaying results ")
        for i in range(len(y_pred_decoded)):
            img=orig_images[i]
            cv2.imshow("video",img)
            key = cv2.waitKey(50) & 0xFF

    vs.release()
    out.release()
    cv2.destroyAllWindows()
    del(orig_images)
    del(input_images)
    gc.collect()

def inference_on_big_video(model_config,video_path,output_name,batch_size,conf_thresh,tracking=False):
    #TODO : check parameters
    #Divide original videos into batches
    folder_path=vd.divideVideo(video_path,batch_size)
    if not (fn.exist(folder_path,'outputs')):
        os.mkdir(os.path.join(folder_path,'outputs'))
    #Treat all batches
    ot = ObjectTracker()
    for (i,batch) in enumerate(fn.findAllIn(folder_path)):
        print('[INFO] : Processing Batch number {}'.format(i))
        vid_path=os.path.join(folder_path,batch)
        inference_on_video(model_config,vid_path,Display=False,change_save_dir=True,confidence_threshold=conf_thresh,tracking=tracking,ot=ot)
    vd.stitch_videos(os.path.join(folder_path,'outputs'),output_name)


# inference_on_image(ssd300_pirogues_mer,'D:\\datas\\pirogues-mer\\DJI_0090.JPG')
# inference_on_folder(ssd300,'D:\\datas\\pirogues-mer\\test\\')
# inference_on_folder(ssd300_pirogues_mer,'D:\\datas\\pirogues-mer\\Images')
# inference_on_video(ssd300_pirogues_mer,'D:\\datas\\pirogues-mer\\videos_parrot\\Disco_0\\Disco_0_2.mp4',change_save_dir=True,tracking=True)
# inference_on_video(ssd300_road,'D:\\workspace\\Keras\\ssd_keras\\data\\Frogger3.mp4',tracking=True)
inference_on_big_video(ssd300_pirogues_mer,'D:\\datas\\pirogues-mer\\videos_parrot\\Disco_0.mp4','full_stitched_video.mp4',2000,0.3,tracking=True)
# vd.stitch_videos('D:\\datas\\pirogues-mer\\videos_parrot\\Disco_0','full_stitched_vid.mp4')
