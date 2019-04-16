import os

def exist(path,name):
    for root, dirs, files in os.walk(path):
        if name in files:
            return True
    return False

def find(name,path):
    for root,dirs,files in os.walk(path):
        if name in files:
            return os.path.join(root,name)

def findAllIn(path):
    for root,dirs,files in os.walk(path):
        return files
