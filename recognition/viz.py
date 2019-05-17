"""Visualization modules"""

import cv2
import numpy as np
import itertools
import os
import mxnet as mx

def _fill_buf(buf, i, img, shape):
    n = buf.shape[0]/shape[0]
    m = buf.shape[1]/shape[1] # n*h/h

    sx = (i%m)*shape[1]
    sy = (i//m)*shape[0]
    buf[sy:sy+shape[0], sx:sx+shape[1], :] = img


def layout(X, flip=False):
    assert len(X.shape) == 4
    #X = X.transpose((0, 2, 3, 1)) ##batch x h x w x c
    #X = np.clip(X * 255.0, 0, 255).astype(np.uint8)
    n = int(np.ceil(np.sqrt(X.shape[0])))
    buff = np.zeros((n*X.shape[1], n*X.shape[2], X.shape[3]), dtype=np.uint8)
    for i, img in enumerate(X):
        img = np.flipud(img) if flip else img
        _fill_buf(buff, i, img, X.shape[1:3])
    if buff.shape[-1] == 1:
        return buff.reshape(buff.shape[0], buff.shape[1])
    #if X.shape[-1] != 1:
    #    buff = cv2.cvtColor(buff, cv2.COLOR_BGR2RGB)
    return buff


def imshow(title, X, waitsec=1, flip=False):
    """Show images in X and wait for wait sec.
    """
    buff = layout(X, flip=flip)
    cv2.imshow(title, buff)
    cv2.waitKey(waitsec)

def imsave(path, X, flip=False, quality=85):
    """save images
    """
    buff = layout(X, flip=flip)
    cv2.imwrite(path, buff, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    #print 'images are saving to disk %s'%path


save_img_batches = 2000
save_img_prefix = "/job_data/image_vis_output/"
color_range = []
for i in itertools.product([255,0], repeat = 3):
    color_range.append(i)

def save_img(ndarray, label, path):
    if not os.path.exists(save_img_prefix):
        try:
            os.mkdir(save_img_prefix)
        except Exception:
            return
    img = mx.nd.abs((ndarray * 128.0 + 128)).asnumpy().astype(np.uint8) #N,C,H,W
    img = img.transpose(0, 2, 3, 1)
    save_img = img.copy()
    batch_label = label.asnumpy()
    for i in range(img.shape[0]):
        if img[i].shape[2] == 3:
            save_img[i] = cv2.cvtColor(img[i], cv2.COLOR_RGB2BGR)
        else:
            save_img[i] = img[i]
        curLabel = int(batch_label[i])
        cv2.putText(save_img[i], '%d'%(int(curLabel)), (save_img[i].shape[0]/8, save_img[i].shape[1]/4), 1, 1.2, color_range[curLabel % len(color_range)], thickness=2,lineType=8)
    label_index = np.argsort(batch_label)
    save_img = save_img[label_index]
    imsave(save_img_prefix + path, save_img, flip=False, quality=100)
