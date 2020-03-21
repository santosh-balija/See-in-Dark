#from __future__ import division
import os,time,scipy.io
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers.python.layers import initializers
import numpy as np
import pdb
import rawpy, imageio
import glob
import sys
import warnings
import cv2
warnings.filterwarnings("ignore")



def lrelu(x):
    return tf.maximum(x*0.2,x)

def upsample_and_concat(x1, x2, output_channels, in_channels):

    pool_size = 2
    deconv_filter = tf.Variable(tf.truncated_normal( [pool_size, pool_size, output_channels, in_channels], stddev=0.02))
    deconv = tf.nn.conv2d_transpose(x1, deconv_filter, tf.shape(x2) , strides=[1, pool_size, pool_size, 1] )

    deconv_output =  tf.concat([deconv, x2],3)
    deconv_output.set_shape([None, None, None, output_channels*2])

    return deconv_output

def network(input):
    conv1=slim.conv2d(input,32,[3,3], rate=1, activation_fn=lrelu,scope='g_conv1_1')
    conv1=slim.conv2d(conv1,32,[3,3], rate=1, activation_fn=lrelu,scope='g_conv1_2')
    pool1=slim.max_pool2d(conv1, [2, 2], padding='SAME' )

    conv2=slim.conv2d(pool1,64,[3,3], rate=1, activation_fn=lrelu,scope='g_conv2_1')
    conv2=slim.conv2d(conv2,64,[3,3], rate=1, activation_fn=lrelu,scope='g_conv2_2')
    pool2=slim.max_pool2d(conv2, [2, 2], padding='SAME' )

    conv3=slim.conv2d(pool2,128,[3,3], rate=1, activation_fn=lrelu,scope='g_conv3_1')
    conv3=slim.conv2d(conv3,128,[3,3], rate=1, activation_fn=lrelu,scope='g_conv3_2')
    pool3=slim.max_pool2d(conv3, [2, 2], padding='SAME' )

    conv4=slim.conv2d(pool3,256,[3,3], rate=1, activation_fn=lrelu,scope='g_conv4_1')
    conv4=slim.conv2d(conv4,256,[3,3], rate=1, activation_fn=lrelu,scope='g_conv4_2')
    pool4=slim.max_pool2d(conv4, [2, 2], padding='SAME' )

    conv5=slim.conv2d(pool4,512,[3,3], rate=1, activation_fn=lrelu,scope='g_conv5_1')
    conv5=slim.conv2d(conv5,512,[3,3], rate=1, activation_fn=lrelu,scope='g_conv5_2')

    up6 =  upsample_and_concat( conv5, conv4, 256, 512  )
    conv6=slim.conv2d(up6,  256,[3,3], rate=1, activation_fn=lrelu,scope='g_conv6_1')
    conv6=slim.conv2d(conv6,256,[3,3], rate=1, activation_fn=lrelu,scope='g_conv6_2')

    up7 =  upsample_and_concat( conv6, conv3, 128, 256  )
    conv7=slim.conv2d(up7,  128,[3,3], rate=1, activation_fn=lrelu,scope='g_conv7_1')
    conv7=slim.conv2d(conv7,128,[3,3], rate=1, activation_fn=lrelu,scope='g_conv7_2')

    up8 =  upsample_and_concat( conv7, conv2, 64, 128 )
    conv8=slim.conv2d(up8,  64,[3,3], rate=1, activation_fn=lrelu,scope='g_conv8_1')
    conv8=slim.conv2d(conv8,64,[3,3], rate=1, activation_fn=lrelu,scope='g_conv8_2')

    up9 =  upsample_and_concat( conv8, conv1, 32, 64 )
    conv9=slim.conv2d(up9,  32,[3,3], rate=1, activation_fn=lrelu,scope='g_conv9_1')
    conv9=slim.conv2d(conv9,32,[3,3], rate=1, activation_fn=lrelu,scope='g_conv9_2')

    conv10=slim.conv2d(conv9,12,[1,1], rate=1, activation_fn=None, scope='g_conv10')
    out = tf.depth_to_space(conv10,2)
    return out


def pack_raw(raw):
    #pack Bayer image to 4 channels
    im = raw.raw_image_visible.astype(np.float32) 
    im = np.maximum(im - 512,0)/ (16383 - 512) #subtract the black level

    im = np.expand_dims(im,axis=2) 
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    out = np.concatenate((im[0:H:2,0:W:2,:], 
                       im[0:H:2,1:W:2,:],
                       im[1:H:2,1:W:2,:],
                       im[1:H:2,0:W:2,:]), axis=2)
    return out


if __name__ == '__main__':

    test_id = sys.argv[1]
    ratio = float(sys.argv[2])
    org_id = sys.argv[3]
		
	
    gt_dir = './dataset/Sony/long/'
    input_dir = './dataset/Sony/short/'
    checkpoint_dir = './checkpoint/Sony/'
    result_dir = './result_Sony/'

    #get train and test IDs
    train_fns = glob.glob(gt_dir + '0*.ARW')
    train_ids = []
    for i in range(len(train_fns)):
        _, train_fn = os.path.split(train_fns[i])
        train_ids.append(int(train_fn[0:5]))



    ps = 512 #patch size for training
    save_freq = 500

    DEBUG = 0
    if DEBUG == 1:
      save_freq = 2
      train_ids = train_ids[0:5]
      test_ids = test_ids[0:5]



    sess=tf.Session()
    in_image=tf.placeholder(tf.float32,[None,None,None,4])
    gt_image=tf.placeholder(tf.float32,[None,None,None,3])
    out_image=network(in_image)
    #G_loss = tf.reduce_mean(tf.abs(out_image - gt_image))

    saver=tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    ckpt=tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt:
        print('loaded '+ckpt.model_checkpoint_path)
        saver.restore(sess,ckpt.model_checkpoint_path)
        
        
  
   
    in_files = glob.glob(test_id)
    #print(in_files)
    in_path = in_files[0]
    in_fn = os.path.split(in_path)
    #print(in_path)
    
    out_files = glob.glob(org_id)
    out_path = out_files[0]
    #_, out_fn = os.path.split(out_path)
    
    raw = rawpy.imread(in_path)
    input_full = np.expand_dims(pack_raw(raw),axis=0) *ratio
            

    input_full = np.minimum(input_full,1.0)
	
    raw1 = rawpy.imread(out_path)
    raw1 = raw1.postprocess() #---------->numpy array
    raw1 = raw1/255
	#imageio.imsave('out.tiff',raw)
    print("Work in progress, Please wait")         
    output =sess.run(out_image,feed_dict={in_image: input_full})#numpy array
    #print(type(output))-------> numpy array
    #print(output.shape)--->(1 , 2848 , 4256 , 3 )
   
    output = output[0,:,:,:]
    #print(output.shape)-----> (2848 , 4256 , 3 )
    #print(output[0,0,0])
  
    loss = np.mean(np.absolute( raw1 - output ))
    outName = test_id[:1]+'_see_in_the_dark.png'
    out_img = scipy.misc.toimage(output*255,  high=255, low=0, cmin=0, cmax=255).save('./' + outName)
    #print(type(out_img))
    #G_loss = tf.reduce_mean(tf.abs(out_img - gv_img))
    #print(G_loss)
   
    print("The results are out :) !!!!!")
    print(loss)
