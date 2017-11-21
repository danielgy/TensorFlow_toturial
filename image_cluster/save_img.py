import h5py
import cv2
import shutil
import os

h5f=h5py.File('./continue/days_3/predict.h5','r')
predict=h5f['predict']

image_list=[]
root="E:\\image_cluster\\continue\\days_3"

image_root=root+"_resize"
for file in os.listdir(image_root):
    filename=os.path.join(image_root,file)
    im = cv2.imread(filename,0)
    image_list.append(im)

predict_root=root+'_predict\\'
if os.path.exists(predict_root):
    shutil.rmtree(predict_root)
os.mkdir(predict_root)
for dirs in set(predict):
    path=predict_root+str(dirs)
    if not os.path.exists(path):
        os.mkdir(path)

dirs=os.listdir(root)
for i in range(len(predict)):  
    saveroot=predict_root+str(predict[i])
    name=os.path.join(saveroot,dirs[i])
    cv2.imwrite(name,image_list[i])
