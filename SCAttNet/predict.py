# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 19:14:27 2018

@author: lhf
"""

import tensorflow as tf
from scattnet import inference
import os
import matplotlib.pyplot as plt
import numpy as np
import glob
import scipy
batch_size=1
# placeholder is depricated in TF2
# placeholder is different from tf.variable
# variables are learnable parameters
img=tf.placeholder(tf.float32,[batch_size,256,256,3])
test_img=sorted(glob.glob('./dataset/test/37/*.png'))
phase_train = tf.placeholder(tf.bool, name='phase_train')
# Inference is imported from scattnet
pred = inference(img,phase_train)

saver=tf.train.Saver()
def save():
    # In TF2 there is no need to use initializer. In TF2, variables are initialized immediately when they are created. There is no longer a need to run variable initializers before using them. 
    tf.global_variables_initializer().run()
    checkpoint_dir = './checkpoint/'
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    # Returns CheckpointState proto from the "checkpoint" file.
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
    for j in range(0,len(test_img)):
                x_batch=test_img[j]
                i=x_batch.split('/')[-1]
                x_batch=scipy.misc.imread(x_batch)/255.0
                x_batch=np.expand_dims(x_batch,axis=0)
                feed_dict = {   img: x_batch,
                              
                                phase_train: False   
                            }
                # Inference session is called after setting up with the test images and the phase_train is set to False.
                pred1=sess.run(pred,feed_dict=feed_dict)
            
                # np.argmax: Returns the indices of the maximum values along an axis.
                predict=np.argmax(pred1,axis=3)
                predict=np.squeeze(predict).astype(np.uint8)
                scipy.misc.imsave('./37/'+i,predict)

def vis():
    
    tf.global_variables_initializer().run()
    checkpoint_dir = './checkpoint/'
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
    for j in range(0,len(test_img)):             
                x_batch=test_img[j]
                x_batch1=scipy.misc.imread(x_batch)/255.0
                x_batch=np.expand_dims(x_batch1,axis=0)
                feed_dict = {   img: x_batch,

                                phase_train: False   
                                
                            }
                pred1=sess.run(pred,feed_dict=feed_dict)
            
                predict=np.argmax(pred1,axis=3)
                predict=np.squeeze(predict)

                # Visualization is performed from below!
                plt.imshow(x_batch1)
                plt.show()

                plt.imshow(predict)
                plt.show()
with tf.Session() as sess:
    save()
   # vis()
