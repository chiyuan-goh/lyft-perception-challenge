import numpy as np
import cv2 
import os
import matplotlib.pyplot as plt

from keras.models import Sequential, load_model
from keras.layers.convolutional import Conv2D, Deconvolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Activation
from keras.activations import softmax
from keras.layers.core import Reshape

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

burl = '/ext/data/lyft-challenge/Train'
img_url = os.path.join(burl, "CameraRGB")
label_url = os.path.join(burl, "CameraSeg")

def process_label(yimg):
    #replace lane markings 
    label_ch = yimg[:,:,2]
    veh_pixels = (label_ch == 10).nonzero()
    hood_indices = (veh_pixels[0] >= 496).nonzero()[0]
    hood_pixels = (veh_pixels[0][hood_indices], \
                   veh_pixels[1][hood_indices])
    
    label_ch[hood_pixels] = 0
    label_ch[label_ch == 6] = 7
    
    h, w = label_ch.shape
    cls_img = np.zeros((h, w, 3))
    
    #channel 1: car, channel 2: road, channel 3: others
    cls_img[:,:, 0] = label_ch == 10
    cls_img[:,:, 1] = label_ch == 7
    cls_img[:,:, 2] = (label_ch != 10) & (label_ch != 7)
        
    return cls_img

def generate_samples(xpaths, ypaths, batch_size):
    batch_size = batch_size

    while 1:
        xpaths, ypaths = shuffle(xpaths, ypaths)
        nbatches =  len(xpaths) // batch_size
        
        for i in range(nbatches):
            xs = xpaths[i * batch_size: i*batch_size + batch_size]
            ys = ypaths[i * batch_size: i*batch_size + batch_size]
            
            xdata = []
            ydata = []
            for n in range(len(xs)):
                ximg = cv2.imread(os.path.join(img_url, xs[n]))
                yimg = cv2.imread((os.path.join(label_url, ys[n])))
                
                ydata.append(process_label(yimg))
                xdata.append(ximg)


            #reflect 1
            ridx = np.random.choice(range(batch_size))
            reflect_img = cv2.flip(xdata[ridx], 1)
            reflect_label = cv2.flip(ydata[ridx], 1)
            xdata.append(reflect_img)
            ydata.append(reflect_label)

            xdata = np.array(xdata).astype("float")
            xdata = xdata/255. - 0.5
            ydata = np.array(ydata).reshape( (batch_size+1, 600*800, 3))
            yield xdata, ydata

def segnet(num_classes, img_shape):
    model = Sequential()
    num_features = 64
    
    #cropping and pre-process
    #model.add()
    #model.add()
    
    #encoders
    model.add(Conv2D(num_features, nb_row=7, nb_col=7, subsample=(1,1), activation='relu', border_mode='same', input_shape=img_shape))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='same'))
    
    model.add(Conv2D(num_features, nb_row=7, nb_col=7, subsample=(1,1), activation='relu', border_mode='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='same'))
    
    model.add(Conv2D(num_features, nb_row=7, nb_col=7, subsample=(1,1), activation='relu', border_mode='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='same'))
    
    model.add(Conv2D(num_features, nb_row=7, nb_col=7, subsample=(1,1), activation='relu', border_mode='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='same'))
    
    #decoders
    model.add(Deconvolution2D(num_features, nb_row=8, nb_col=8, 
              subsample=(2,2), 
              border_mode='same',
              output_shape=(None, img_shape[0]//(2**3), img_shape[1]//(2**3), num_features),
              activation='relu'))
    
    model.add(Deconvolution2D(num_features, nb_row=8, nb_col=8, 
          subsample=(2,2), 
          border_mode='same',
          output_shape=(None, img_shape[0]//(2**2), img_shape[1]//(2**2), num_features),
          activation='relu'))
    
    model.add(Deconvolution2D(num_features, nb_row=8, nb_col=8, 
          subsample=(2,2), 
          border_mode='same',
          output_shape=(None, img_shape[0]//(2**1), img_shape[1]//(2**1), num_features),
          activation='relu'))
    
    model.add(Deconvolution2D(num_classes, nb_row=8, nb_col=8, 
          subsample=(2,2), 
          border_mode='same',
          output_shape=(None, img_shape[0], img_shape[1], num_classes),
          activation='relu'))
    
    print("nclass", num_classes)
    model.add(Reshape((800*600, num_classes)))
#     prediction layer
    model.add(Activation("softmax"))
    
    model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])
    
    return model

def run():
    batch_size=4
    #img_size = 
    
    xdata =  os.listdir(img_url)
    ylabel = os.listdir(label_url)
    
    xtrain, xtest, ytrain, ytest = train_test_split(xdata, ylabel, test_size=0.2)
    train_gen = generate_samples(xtrain, ytrain, batch_size=batch_size)
    test_gen = generate_samples(xtest, ytest, batch_size=batch_size)
    
    imgsize = (600, 800, 3)
    model = segnet(3, imgsize)
    print(len(xtrain))
    runobj = model.fit_generator(train_gen,
                                samples_per_epoch= len(xtrain)//batch_size * batch_size,
                                nb_epoch=4,
                                validation_data=test_gen, 
                                nb_val_samples = (len(xtest)//batch_size) * batch_size,
                                verbose=1)
        
    model.save("m1")
