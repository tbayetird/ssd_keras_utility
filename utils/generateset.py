import glob, os
import shutil
from . import filenav as fn

def YOLO_generate(dataset_path,destination_dir,percentage_test):
    # Create and/or truncate train.txt and test.txt
    file_train = open(os.path.join(destination_dir,'train.txt'), 'w')
    file_test = open(os.path.join(destination_dir,'test.txt'), 'w')

    # Populate train.txt and test.txt
    counter = 1
    index_test = round(100 / percentage_test)
    for pathAndFilename in glob.iglob(os.path.join(dataset_path, "*.jpg")):
        title, ext = os.path.splitext(os.path.basename(pathAndFilename))

        if counter == index_test+1:
            counter = 1
            file_test.write(dataset_path + "/" + title + '.jpg' + "\n")
        else:
            file_train.write(dataset_path + "/" + title + '.jpg' + "\n")
            counter = counter + 1
    close(file_train)
    close(file_test)

## Generating a PascalVOC-like dataset from an image folder
# image_folder : folder containing the images to be used
# destination_foler : where to generate the dataset
def PascalVOC_generate(image_folder,destination_folder='',percentage_test=20):
    if(destination_folder==''):
        destination_folder=image_folder
    PascalVOC_architecture_generate(image_folder,destination_folder)
    PascalVOC_set_generate(os.path.join(destination_folder,'Images'),
                            os.path.join(destination_folder,'ImageSets'),
                            percentage_test)


## Generating a PascalVOC_like architecture from an image folder
# image_folder : where to find the images to set up the architecture. Can be empty
# dest_folder :  where to build the architecture. At default, image_folder.
def PascalVOC_architecture_generate(image_folder,dest_folder=''):
    if(dest_folder==''):
        dest_folder=image_folder
    if not (fn.exist(dest_folder,'Images')):
        os.mkdir(os.path.join(dest_folder,'Images'))
    if not (fn.exist(dest_folder,'ImageSets')):
        os.mkdir(os.path.join(dest_folder,'ImageSets'))
    if not (fn.exist(dest_folder,'Annotations')):
        os.mkdir(os.path.join(dest_folder,'Annotations'))
    for im in fn.findAllIn(image_folder):
        im_path=os.path.join(image_folder,im)
        new_path=os.path.join(dest_folder,'Images',im)
        shutil.move(im_path,new_path)



## Generating a PascalVOC-like sets for train, test and validation
# dataset_path : path to the folder containing the images
# destination : path to the folder that is to contain the sets
# percentage_test : percentage of the data that is to be used as test set
def PascalVOC_set_generate(dataset_path,destination_dir,percentage_test):
    file_train = open(os.path.join(destination_dir,'train.txt'), 'w')
    file_test = open(os.path.join(destination_dir,'test.txt'), 'w')
    file_val =  open(os.path.join(destination_dir,'val.txt'), 'w')

    # Populate train.txt and test.txt
    counter = 1
    counter_2=1
    index_test = round(100 / int(percentage_test))
    for pathAndFilename in glob.iglob(os.path.join(dataset_path, "*.jpg")):
        title, ext = os.path.splitext(os.path.basename(pathAndFilename))

        if counter == index_test+1:
            counter = 1
            file_test.write(title+'\n')
        else:
            if (counter_2==5):
                counter_2=1
                file_val.write(title+'\n')
            else:
                file_train.write(title+'\n')
                counter_2=counter_2+1
            counter = counter + 1

    for pathAndFilename in glob.iglob(os.path.join(dataset_path, "*.PNG")):
        title, ext = os.path.splitext(os.path.basename(pathAndFilename))

        if counter == index_test+1:
            counter = 1
            file_test.write(title+'\n')
        else:
            if (counter_2==5):
                counter_2=1
                file_val.write(title+'\n')
            else:
                file_train.write(title+'\n')
                counter_2=counter_2+1
            counter = counter + 1

# PascalVOC_architecture_generate('D:\\datas\\pirogues-mer\\test')
# import argparse
# ap = argparse.ArgumentParser()
# ap.add_argument("-p", "--dataset_path", required=True,
# 	help="path to dataset")
# ap.add_argument("-d", "--destination_dir", required=True,
# 	help="destination directory")
# ap.add_argument("-i", "--percentage_test", required=True,
# 	help="percentage going into test (10 for 10 per cent)")
# args = vars(ap.parse_args())
# PascalVOC_generate(args["dataset_path"],args["destination_dir"],args["percentage_test"])
