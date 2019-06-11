from imageio import imread,imwrite
from matplotlib import pyplot as plt
import cv2,os

def slice_image_horizontal(image_path,size):
    im = imread(image_path)
    im_height = im.shape[0]
    tmp_height = 0
    plt.imshow(im)
    while tmp_height < im_height:
        tmp_im = im[tmp_height : tmp_height+size,2500:]
        tmp_height += size
        imwrite('D:\\datas\\pirogues-plage\\slice_im\\slice_0050_{}.jpg'.format(tmp_height),
                tmp_im)

#TODO : include strides
def decompose(im,im_path,height,width):
    h,w = im.shape[0],im.shape[1]
    for i in range(int(h/height)):
        for j in range(int(w/width)):
            hmin = i*height
            hmax = (i+1)*height
            wmin = j*width
            wmax = (j+1)*width
            img_name=im_path +'_{}_{}.jpg'.format(i,j)
            cv2.imwrite(img_name,im[hmin:hmax,wmin:wmax,:])
            # cv2.imshow('decomposed image ({},{})'.format(i,j),im[hmin:hmax,wmin:wmax,:])
            # cv2.waitKey(850)
            # cv2.destroyAllWindows()

# Preparing the dataset for the SSD use
# Images coming from the Mavic Pro 2 drone are in 3648*5472 pixels format
# We need to format them in lower images for memory reasons
# New size will be 556 * 684

def subdivise_image_folder(im_dir,height,width):
    imgs = []
    imgs_name=[]
    for root,dirs,files in os.walk(im_dir):
        for file in files :
            imgs.append(cv2.imread(os.path.join(im_dir,file)))
            imgs_name.append(os.path.join(im_dir,file))

    for (i,im) in enumerate(imgs):
        decompose(im,imgs_name[i],height,width)
        os.system("del {}".format(imgs_name[i]))

subdivise_image_folder('D:\\datas\\pirogues-plage\\Images',512,512)
