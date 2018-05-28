import numpy as np
import cv2 
import os
import matplotlib.pyplot as plt

from keras.models import Sequential, load_model
from keras.layers.convolutional import Conv2D, Deconvolution2D, UpSampling2D, ZeroPadding2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers import Activation
from keras.activations import softmax
from keras.layers.core import Reshape
from keras.applications.vgg16 import VGG16
from keras.optimizers import SGD

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

#burl = '/ext/data/lyft-challenge/Train'
burl = '/home/cy/Desktop/Train'
img_url = os.path.join(burl, "CameraRGB")
label_url = os.path.join(burl, "CameraSeg")

test_img_gl = os.listdir(label_url)[0]
label_gl = cv2.imread(os.path.join(label_url, test_img_gl) )[:,:,2]
veh_pixels_gl = (label_gl == 10).nonzero()
hood_indices_gl = (veh_pixels_gl[0] >= 496).nonzero()[0]
hood_pixels_gl = (veh_pixels_gl[0][hood_indices_gl], \
               veh_pixels_gl[1][hood_indices_gl])

def process_label(yimg):
    #replace lane markings 
    label_ch = yimg[:,:,2]

    label_ch[hood_pixels_gl] = 0
    label_ch[label_ch == 6] = 7
    
    h, w = label_ch.shape
    cls_img = np.zeros((h, w, 3))
    
    #channel 1: car, channel 2: road, channel 3: others
    cls_img[:,:, 0] = label_ch == 10
    cls_img[:,:, 1] = label_ch == 7
    cls_img[:,:, 2] = (label_ch != 10) & (label_ch != 7)

    weights = np.zeros_like(label_ch, dtype='float')
    weights[label_ch == 10] = 5.
    weights[label_ch == 7] = 2.
    weights[(label_ch != 10) & (label_ch != 7)] = .25
        
    return cls_img, weights

def generate_samples(xpaths, ypaths, batch_size, train=False):
    batch_size = batch_size

    #cars, road, none

    cweights = [14.259816, 1., 0.25325792]

    while 1:
        xpaths, ypaths = shuffle(xpaths, ypaths)
        nbatches =  len(xpaths) // batch_size
        
        for i in range(nbatches):
            sample_weights = []
            
            xs = xpaths[i * batch_size: i*batch_size + batch_size]
            ys = ypaths[i * batch_size: i*batch_size + batch_size]
            
            xdata = []
            ydata = []
            for n in range(len(xs)):
                ximg = cv2.imread(os.path.join(img_url, xs[n]))
                yimg = cv2.imread((os.path.join(label_url, ys[n])))

                ximg[hood_pixels_gl[0],hood_pixels_gl[1], :] = 0.
                hot_label, weight = process_label(yimg)
                
                ydata.append(hot_label)
                xdata.append(ximg)
                sample_weights.append(weight)

                #reflect 1
                ridx = np.random.choice(range(batch_size))
                reflect_img = cv2.flip(ximg, 1)
                reflect_label = cv2.flip(hot_label, 1)
                reflect_weight = cv2.flip(weight, 1)
                xdata.append(reflect_img)
                ydata.append(reflect_label)
                sample_weights.append(reflect_weight)

            xdata = np.array(xdata).astype("float")
            xdata = xdata/255.# - 0.5
            #ydata = np.array(ydata).reshape( (batch_size, 600*800, 3))
            ydata = np.array(ydata).reshape( (batch_size * 2, 600*800, 3))
            sample_weights = np.array(sample_weights).reshape( (batch_size * 2, 800*600))

            if not train:
                yield xdata, ydata
            else:
                yield xdata, ydata, np.array(sample_weights)


def vgg_segnet(nclass, img_shape):
    base_model  = VGG16(include_top=False, weights='imagenet', input_shape=img_shape)

    kernel_size = 3

    upsample_filters = [512, 512, 256, 128, nclass]

    model = Sequential()
    model.add(base_model)

    for i in range(4):
        model.add(UpSampling2D(size=(2,2)))
        model.add(Conv2D(upsample_filters[i], nb_row=kernel_size, nb_col=kernel_size, subsample=(1,1), border_mode='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2D(upsample_filters[i], nb_row=kernel_size, nb_col=kernel_size, subsample=(1,1), border_mode='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2D(upsample_filters[i], nb_row=kernel_size, nb_col=kernel_size, subsample=(1,1), border_mode='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(upsample_filters[-1], nb_row=kernel_size, nb_col=kernel_size, subsample=(1, 1), border_mode='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(upsample_filters[-1], nb_row=kernel_size, nb_col=kernel_size, subsample=(1, 1), border_mode='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(upsample_filters[-1], nb_row=kernel_size, nb_col=kernel_size, subsample=(1, 1), border_mode='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))


    model.add(Reshape((img_shape[0]*img_shape[1], nclass)))
    model.add(Activation("softmax"))

    print("running coinsatnt")
    sgd = SGD(lr=0.01, momentum=0.9, decay=0.)
    model.compile(sgd, 'categorical_crossentropy', metrics=['categorical_accuracy'], sample_weight_mode="temporal")

    return model

def segnet(num_classes, img_shape):
    model = Sequential()
    num_features = 64
    
    #cropping and pre-process
    #model.add()
    #model.add()

    #model.add(ZeroPadding2D((4, 0), input_shape=img_shape))

    #encoders
    model.add(Conv2D(num_features, nb_row=7, nb_col=7, subsample=(1,1), border_mode='same', input_shape=img_shape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='same'))
    
    model.add(Conv2D(num_features, nb_row=7, nb_col=7, subsample=(1,1), border_mode='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='same'))
    
    model.add(Conv2D(num_features, nb_row=7, nb_col=7, subsample=(1,1),  border_mode='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='same'))
    
    model.add(Conv2D(num_features, nb_row=7, nb_col=7, subsample=(1,1),  border_mode='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='same'))

    #decoders
    # model.add(UpSampling2D(size=(2,2)))
    # model.add(Conv2D(num_features, nb_row=7, nb_col=7, subsample=(1,1), border_mode='same'))
    # model.add(BatchNormalization())
    # model.add(Activation('relu'))
    #
    # model.add(UpSampling2D(size=(2, 2)))
    # model.add(Conv2D(num_features, nb_row=7, nb_col=7, subsample=(1, 1), border_mode='same'))
    # model.add(BatchNormalization())
    # model.add(Activation('relu'))
    #
    #
    # model.add(UpSampling2D(size=(2, 2)))
    # model.add(Conv2D(num_features, nb_row=7, nb_col=7, subsample=(1, 1), border_mode='same'))
    # model.add(BatchNormalization())
    # model.add(Activation('relu'))
    #
    # model.add(UpSampling2D(size=(2, 2)))
    # model.add(Conv2D(num_classes, nb_row=7, nb_col=7, subsample=(1, 1), border_mode='same'))
    # model.add(BatchNormalization())
    # model.add(Activation('relu'))
    #
    #
    # model.add(Cropping2D(cropping=((4,4), (0,0))) )

    #decoders
    model.add(Deconvolution2D(num_features, nb_row=8, nb_col=8,
              subsample=(2,2),
              border_mode='same',
              output_shape=(None, img_shape[0]//(2**3), img_shape[1]//(2**3), num_features)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Deconvolution2D(num_features, nb_row=8, nb_col=8,
          subsample=(2,2),
          border_mode='same',
          output_shape=(None, img_shape[0]//(2**2), img_shape[1]//(2**2), num_features)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Deconvolution2D(num_features, nb_row=8, nb_col=8,
          subsample=(2,2),
          border_mode='same',
          output_shape=(None, img_shape[0]//(2**1), img_shape[1]//(2**1), num_features)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Deconvolution2D(num_classes, nb_row=8, nb_col=8,
          subsample=(2,2),
          border_mode='same',
          output_shape=(None, img_shape[0], img_shape[1], num_classes)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    
    model.add(Reshape((800*600, num_classes)))
#     prediction layer
    model.add(Activation("softmax"))
    
    model.compile('adam', 'categorical_crossentropy', metrics=['categorical_accuracy'], sample_weight_mode="temporal")
    
    return model

# def run():
#     batch_size=4
#     #img_size =
#
#     xdata =  os.listdir(img_url)
#     ylabel = os.listdir(label_url)
#
#     xtrain, xtest, ytrain, ytest = train_test_split(xdata, ylabel, test_size=0.2)
#     train_gen = generate_samples(xtrain, ytrain, batch_size=batch_size, train=True)
#     test_gen = generate_samples(xtest, ytest, batch_size=batch_size)
#
#     imgsize = (600, 800, 3)
#     model = segnet(3, imgsize)
#     #print(len(xtrain))
#     runobj = model.fit_generator(train_gen,
#                                 samples_per_epoch= len(xtrain)//batch_size * batch_size * 1.25,
#                                 nb_epoch=7,
#                                 validation_data=test_gen,
#                                 nb_val_samples = (len(xtest)//batch_size) * batch_size,
#                                 nb_worker=2,
#                                 nb_val_worker=2,
#                                 verbose=1)
#
#     model.save("m2")
    
if __name__ == '__main__':
    run()
