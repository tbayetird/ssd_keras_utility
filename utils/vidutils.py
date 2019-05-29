import os,cv2
import numpy as np
from . import filenav as fn

def handleVideoStreams(vs,save_dir,vidName,img_width,img_height):
    print('     [Debug] Starting video handling ')
    orig_images = [] # Store the images here.
    input_images = [] # Store resized versions of the images here.
    ex=False
    images_stock_name='imstock_'+vidName
    if(fn.exist(save_dir,images_stock_name)):
        print("[INFO] Loading existing video datas")
        input_images = np.load(os.path.join(save_dir,images_stock_name))
        ex=True
    print('     [Debug] Starting video process ')
    vid_count=0
    while(vs.isOpened()):
    # while(vs.isOpened()):
        ok,frame=vs.read()
        vid_count+=1
        if not ok:
            break
        orig_images.append(frame)
        vidShape=frame.shape[:2]
        if not ex:
            frame = cv2.resize(frame,(img_width,img_height),interpolation=cv2.INTER_AREA)
            input_images.append(frame)
    print('     [Debug] Starting array construction ')
    input_images = np.array(input_images)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(os.path.join(save_dir,vidName),
                                fourcc,24.0,(vidShape[1],vidShape[0]))
    if not ex:
        np.save(os.path.join(save_dir,images_stock_name),input_images)
    return [orig_images,input_images,out]

def handleTruncatedVideoStreams(vs,save_dir,vidName,img_width,img_height,truncs):
    print('     [Debug] Starting video handling ')
    orig_images = [] # Store the images here.
    input_images = [] # Store resized versions of the images here.
    images_stock_name='imstock_'+vidName
    print('     [Debug] Starting video process ')
    vid_count=0
    while(vs.isOpened()):
    # while(vs.isOpened()):
        ok,frame=vs.read()
        vid_count+=1
        if not ok:
            break
        if (vid_count < truncs[0] or vid_count > truncs[1]):
            continue
        orig_images.append(frame)
        vidShape=frame.shape[:2]
        frame = cv2.resize(frame,(img_width,img_height),interpolation=cv2.INTER_AREA)
        input_images.append(frame)
        del frame 
    print('     [Debug] Starting array construction ')
    input_images = np.array(input_images)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(os.path.join(save_dir,vidName),
                                fourcc,24.0,(vidShape[1],vidShape[0]))
    return [orig_images,input_images,out]

# def handleBigVideoStreams(vs, save_dir,vid_name,img_width,img_height,batch_size):
#     print('     [DEBUG] Starting video handling')
#     orig_images = [[]] # Store the images here.
#     input_images = [[]] # Store resized versions of the images here.
#     ind = 0
#     vid_count=0
#     while(vs.isOpened()):
#     # while(vs.isOpened()):
#         ok,frame=vs.read()
#         vid_count+=1
#         if not ok:
#             break
#         if(vid_count==2000):
#             vid_count=0
#             ind+=1
#             print('[DEBUG] Batch number {}'.format(ind))
#             orig_images.append([])
#             input_images.append([])
#         orig_images[ind].append(frame)
#         vidShape=frame.shape[:2]
#         frame = cv2.resize(frame,(img_width,img_height),interpolation=cv2.INTER_AREA)
#         input_images[ind].append(frame)
#
#     print('     [Debug] Starting array construction ')
#     input_images = np.array(input_images)
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     out = cv2.VideoWriter(os.path.join(save_dir,vidName),
#                                 fourcc,20.0,(vidShape[1],vidShape[0]))
#     np.save(os.path.join(save_dir,images_stock_name),input_images)
#     return [orig_images,input_images,out]

def divideVideo(videoPath,batchSize):
    #Creating a folder with the video name
    folderPath= videoPath[:(videoPath.rfind('.'))]
    vidName= (videoPath.split('\\'))[-1]
    if fn.exist(os.path.dirname(folderPath),folderPath.split('\\')[-1]):
        print('[INFO] Folder for video batching already exists')
        return folderPath
    print('[INFO] Creating folder for video batching')
    os.mkdir(folderPath)

    #reading the video and dividing it in batches
    vs=cv2.VideoCapture(videoPath)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    im_count=0
    batch_count=0
    batch_name=vidName[:(vidName.rfind('.'))]+'_{}'.format(batch_count)+'.mp4'
    while(vs.isOpened()):
        ok,frame = vs.read()
        if not ok:
            break
        if(im_count==0):
            vidShape=frame.shape[:2]
            out = cv2.VideoWriter(os.path.join(folderPath,batch_name),
                                    fourcc,24.0,(vidShape[1],vidShape[0]))

        if(im_count==batchSize):
            im_count=0
            batch_count+=1
            batch_name=vidName[:(vidName.rfind('.'))]+'_{}'.format(batch_count)+'.mp4'
            out.release()
            continue
        im_count+=1
        out.write(frame)
    out.release()
    vs.release()
    return folderPath

def stitch_videos(videoFolder,videoName):
    first=True
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    for (i,vid) in enumerate(fn.findAllIn(videoFolder)):
        if not (fn.ext(vid)=='mp4'):
            continue
        print('[INFO] Stitching {}'.format(vid))
        vs=cv2.VideoCapture(os.path.join(videoFolder,vid))
        while(vs.isOpened()):
            ok,frame=vs.read()
            if not ok:
                break
            vidShape=frame.shape[:2]
            if (first):
                out = cv2.VideoWriter(os.path.join(videoFolder,videoName),
                                        fourcc,24.0,(vidShape[1],vidShape[0]))
                first=False
            out.write(frame)
        vs.release()
    out.release()

def screenshotVideos(videoPath,outputFolder,screenshotNames,freq,max=100000):
    vs = cv2.VideoCapture(videoPath)
    im_count=0
    save_count=0
    while(vs.isOpened()):
        ok,frame = vs.read()
        if not ok:
            break
        if (im_count==freq):
            im_name = os.path.join(outputFolder,screenshotNames+'_{}'.format(save_count)+'.jpg')
            print('[INFO] Savinf image {}'.format(im_name))
            im_count=0
            save_count+=1
            cv2.imwrite(im_name,frame)
        if(save_count==max):
            break
        im_count +=1
    vs.release()

# screenshotVideos('D:\\datas\\pirogues-mer\\videos_parrot\\Disco_0.mp4','D:\\datas\\pirogues-mer\\videos_parrot\\Images','Disco_0_scr',50)
