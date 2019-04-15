import glob, os

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

def PascalVOC_generate(dataset_path,destination_dir,percentage_test):
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
