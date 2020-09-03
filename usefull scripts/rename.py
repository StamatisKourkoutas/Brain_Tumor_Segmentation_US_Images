'''
    Code used to rename multiple images and their masks in a directory
    and produce a txt with the list of the images' names
'''

import os 
import re

def main():
    path = "/home/stamatis/Desktop/Imperial Thesis/Thesis code/trainingnew/"
    imgs = path+"image/"
    masks = path+"mask/"
    type = []
    
    files = os.listdir(imgs)
    files.sort()
    
    #Create train.txt
    outF = open(path+"train.txt", "w")
    
    for count, oldname in enumerate(files):
        if oldname[-4::]==".png":
            type = ".png"
        elif oldname[-4::]==".jpg":
            type = ".jpg"
        
        src = imgs + oldname
        newname = "image" + str(count+1) + type
        dst = imgs + newname
        
        os.rename(src, dst)

        # Used to get a list of the image names in a txt file.
        print(newname, file=outF)
    
    outF.close()
    
    files = os.listdir(masks)
    files.sort()
    for count, oldname in enumerate(files):
        if oldname[-4::]==".png":
            type = ".png"
        elif oldname[-4::]==".jpg":
            type = ".jpg"
        
        src = masks + oldname
        newname = "image" + str(count+1) + type
        dst = masks + newname
        
        #os.rename(src, dst)

if __name__ == '__main__':
    main()
