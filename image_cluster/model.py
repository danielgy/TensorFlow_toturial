
import tensorflow as tf
import numpy as np
import os
import cv2
import vgg
import shutil
from sklearn.cluster import KMeans
#import h5py

#TODO: read images from h5 file
#h5f=h5py.File('./data/image.h5','r')
#image=h5f['image']
image_list=[]
root="E:\\image_cluster\\continue\\days_2_resize\\"
for file in os.listdir(root):
    filename=os.path.join(root,file)
    im = cv2.imread(filename,0)
    image_list.append(im)

#immatrix = np.array(image_list)


#TODO: load VGG and extract feature

VGG_PATH = 'imagenet-vgg-verydeep-19.mat'
vgg_weights, vgg_mean_pixel = vgg.load_net(VGG_PATH)
CONTENT_LAYERS = ('relu3_1','relu3_2', 'relu4_1', 'relu5_1','relu5_2')
layer='relu5_2'
input_image=cv2.imread("E:\\image_cluster\\continue\\days_2_resize\\2_300001_2016-02-02_2016-02-03_1.png")
feature_data=[]
shape = (1,) + input_image.shape
g = tf.Graph()
with g.as_default(),  tf.Session() as sess:
    image = tf.placeholder('float', shape=shape)
    net = vgg.net_preloaded(vgg_weights, image, 'avg')
    for name in os.listdir(root):
        filename=os.path.join(root,name)
        input_image=cv2.imread(filename)
    #    content_features = {}
#    for i in range(len(image_list)):
#        input_image=image_list[i]
        content_pre = np.array([vgg.preprocess(input_image, vgg_mean_pixel)])
        content_features = net[layer].eval(feed_dict={image: content_pre})[0]
        feature_data.append(content_features.flatten())


#TODO: kmeans and save images with predict folder

kmeans = KMeans(n_clusters=10, random_state=0).fit(feature_data)
predict=kmeans.predict(feature_data)



predict_root='E:\\image_cluster\\continue_day_2_predict\\'
shutil.rmtree(predict_root)
os.mkdir(predict_root)
for dirs in set(predict):
    path=predict_root+str(dirs)
    if not os.path.exists(path):
        os.mkdir(path)

root="E:\\image_cluster\\continue\\days_2\\"
dirs=os.listdir(root)
for i in range(len(predict)):  
    saveroot=predict_root+str(predict[i])
    name=os.path.join(saveroot,dirs[i])
    cv2.imwrite(name,image_list[i])
