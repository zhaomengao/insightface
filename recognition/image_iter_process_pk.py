from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
import logging
import sys
import numbers
import math
import time
import datetime
import numpy as np
import cv2
import copy
import time
import mxnet as mx
from mxnet import ndarray as nd
from mxnet import io
from mxnet import recordio
from mxnet import context
from mxnet.ndarray._internal import _cvimresize as imresize
import mxnet.gluon.data.dataloader as dataloader
sys.path.append(os.path.join(os.path.dirname(__file__), 'common'))
import multiprocessing
import threading
import Queue
import indexed_recordio
import viz

logger = logging.getLogger()

def genaratePsf(length,angle):
    EPS=np.finfo(float).eps
    alpha = (angle-math.floor(angle/ 180) *180) /180* math.pi
    cosalpha = math.cos(alpha)
    sinalpha = math.sin(alpha)
    if cosalpha < 0:
        xsign = -1
    elif angle == 90:
        xsign = 0
    else:
        xsign = 1
    psfwdt = 1;
    sx = int(math.fabs(length*cosalpha + psfwdt*xsign - length*EPS))
    sy = int(math.fabs(length*sinalpha + psfwdt - length*EPS))
    psf1=np.zeros((sy,sx))

    half = length/2
    for i in range(0,sy):
        for j in range(0,sx):
            psf1[i][j] = i*math.fabs(cosalpha) - j*sinalpha
            rad = math.sqrt(i*i + j*j)
            if  rad >= half and math.fabs(psf1[i][j]) <= psfwdt:
                temp = half - math.fabs((j + psf1[i][j] * sinalpha) / cosalpha)
                psf1[i][j] = math.sqrt(psf1[i][j] * psf1[i][j] + temp*temp)
            psf1[i][j] = psfwdt + EPS - math.fabs(psf1[i][j])
            if psf1[i][j] < 0:
                psf1[i][j] = 0
    anchor=(0,0)

    if angle<90 and angle>0:
        psf1=np.fliplr(psf1)
        anchor=(psf1.shape[1]-1,0)
    elif angle>-90 and angle<0:
        psf1=np.flipud(psf1)
        psf1=np.fliplr(psf1)
        anchor=(psf1.shape[1]-1,psf1.shape[0]-1)
    elif anchor<-90:
        psf1=np.flipud(psf1)
        anchor=(0,psf1.shape[0]-1)
    psf1=psf1/psf1.sum()
    return psf1,anchor

def generate_template(img_shape):
    """Helper function for spatial brightness augmentation"""
    template_h = nd.ones(img_shape)
    template_w = nd.ones(img_shape)
    for i in range(template_h.shape[1]):
        template_h[:,i,:] = i * 1.0 / template_h.shape[1]
    for i in range(template_h.shape[0]):
        template_w[i,:,:] = i * 1.0 / template_w.shape[0]
    return template_h, template_w

def get_inter_method(inter_method, sizes=()):
    if inter_method == 9:
        if sizes:
            assert len(sizes) == 4
            oh, ow, nh, nw = sizes
            if nh > oh and nw > ow:
                return 2
            elif nh < oh and nw < ow:
                return 3
            else:
                return 1
        else:
            return 2
    if inter_method == 10:
        return random.randint(0, 4)
    return inter_method


class MultiContextConstArray(object):
    """Multi-context array, the array will be buffered in multiple contexts"""
    def __init__(self, arr):
        self.ori_arr = arr
        self.ctx_dict = {arr.context: arr}

    def get(self, ctx):
        ret_arr = self.ctx_dict.get(ctx)
        if ret_arr is None:
            ret_arr = self.ori_arr.as_in_context(ctx)
            self.ctx_dict[ctx] = ret_arr
        return ret_arr


class FaceImageIter(io.DataIter):

    def __init__(self, batch_size, data_shape,
                 num_samples_per_class='1,1,1',
                 max_samples_per_class=1,
                 pk_replace = 0,
                 unpack64=0,
                 sample_type=0,
                 repet_flag=0, repet_num=1,
                 path_imgrec=None,
                 shuffle=False,
                 databatch_process_callback=None,
                 rand_mirror=False, cutoff=0,
                 num_parts=1, part_index=0,
                 bright_jitter_prob=0.0,
                 bright_jitter_range=0.0,
                 gaussblur_prob=0.0,
                 gaussblur_kernelsz_max=9,
                 gaussblur_sigma=0.0,
                 motionblur_prob=0.0,
                 spatial_bright_prob=0.0,
                 spatial_bright_range=0.0,
                 jpeg_compress_prob=0.0,
                 jpeg_compress_quality_max=95,
                 jpeg_compress_quality_min=90,
                 rand_gray_prob=0.0,
                 downsample_prob=0.0,
                 downsample_min_width=8,
                 inter_method=1,
                 contrast_prob=0.0,
                 contrast_range=0.0,
                 offline_feature=0,
                 data_name='data', label_name='softmax_label',
                 single_channel=0,
                 **kwargs):
        super(FaceImageIter, self).__init__()
        self.num_parts = num_parts
        assert path_imgrec
        if sample_type == 1 :
            num_samples_per_class=[int(i.strip()) for i in num_samples_per_class.split(',')]
            assert len(num_samples_per_class) == 3
        if sample_type == 2 or sample_type == 3:
            num_samples_per_class=int(num_samples_per_class.split(',').strip()[0])
            assert int(max_samples_per_class / num_samples_per_class) * num_samples_per_class == max_samples_per_class
        if repet_flag:
           assert repet_num > 1

        self.databatch_process_callback = databatch_process_callback
        assert path_imgrec
        if path_imgrec:
            logging.info('loading recordio %s...',
                         path_imgrec)
            path_imgidx = path_imgrec[0:-4]+".idx"
            self.imgrec = recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')  # pylint: disable=redefined-variable-type
            s = self.imgrec.read_idx(0)
            if unpack64 == 0:
               logging.info('unpack32 used!')
               header, _ = recordio.unpack(s)
            else:
               logging.info('unpack64 used!')
               header, _ = indexed_recordio.unpack64(s)
            if header.flag>0:
              logging.info('header0 label %s'%(str(header.label)))
              self.header0 = (int(header.label[0]), int(header.label[1]))
              #assert(header.flag==1)
              self.imgidx = range(1, int(header.label[0]))
              self.id2range = {}
              self.id_num = {}
              self.seq_identity = range(int(header.label[0]), int(header.label[1]))
              for identity in self.seq_identity:
                s = self.imgrec.read_idx(identity)
                if unpack64 == 0:
                    header, _ = recordio.unpack(s)
                else:
                    header, _ = indexed_recordio.unpack64(s)
                a,b = int(header.label[0]), int(header.label[1])
                self.id2range[identity] = (a,b)
                count = b-a

                self.id_num[identity] = b - a
              logging.info('id2range %d'%(len(self.id2range)))
            else:
              self.imgidx = list(self.imgrec.keys)
            if shuffle or num_parts > 1:
              if sample_type == 0:
                self.seq = self.imgidx
              elif sample_type == 1 or sample_type == 2:
                self.seq = None
              self.len_seq = len(self.imgidx)#for local test
              self.oseq = self.imgidx
            else:
              self.seq = None

            if num_parts > 1:
                assert part_index < num_parts
                if sample_type == 0: #part_type ==0, sample shuffle;part_type==1, id shuffle;
                  random.seed(10)
                  random.shuffle(self.seq)
                  logging.info('[Init]Now rank: %d, and random seq is: %s'%(part_index,str(self.seq[:100])))
                  epc_size = len(self.seq)
                  epc_size_part = epc_size // num_parts
                  if part_index == num_parts-1:
                     self.seq = self.seq[part_index*epc_size_part : ]
                  else:
                     self.seq = self.seq[part_index*epc_size_part : (part_index+1)*epc_size_part]
                else:
                  seq_identity = copy.deepcopy(self.seq_identity)
                  random.seed(10)
                  random.shuffle(seq_identity)
                  #logging.info('[Init]Now rank: %d, and random seq_identity is: %s'%(part_index,str(seq_identity[:100])))
                  epc_size = len(seq_identity)
                  epc_size_part = epc_size // num_parts
                  self.len_seq = len(self.imgidx)

                  #find the minimum of number of samples in machines
                  for index in range(num_parts):
                     if index == num_parts-1:
                        count_seqid = seq_identity[index*epc_size_part : ]
                     else:
                        count_seqid = seq_identity[index*epc_size_part : (index+1)*epc_size_part]

                     if sample_type == 1:
                        count_seq = []
                        for identity in count_seqid:
                           id2range = range(self.id2range[identity][0], self.id2range[identity][1])
                           pk_num = num_samples_per_class[0] if self.id_num[identity]<num_samples_per_class[1] else num_samples_per_class[2]
                           num_lack=pk_num-self.id_num[identity]
                           if num_lack>0 and pk_replace:
                             pad_sample = np.random.choice(id2range,size=num_lack,replace=True)
                             id2range.extend(pad_sample)
                           if num_lack<0:
                             id2range = np.random.choice(id2range, size=pk_num, replace=False)
                           count_seq.extend(id2range)
                     elif sample_type == 2 or sample_type == 4:
                        count_seq = []
                        for identity in count_seqid:
                            id2range = range(self.id2range[identity][0], self.id2range[identity][1])
                            count_seq.extend(id2range)
                     elif sample_type == 3:
                        count_seq = []
                        for identity in count_seqid:
                            id2range = range(self.id2range[identity][0], self.id2range[identity][1])
                            if self.id_num[identity] > max_samples_per_class:
                               id2range = np.random.choice(id2range, size=max_samples_per_class, replace=False)
                            count_seq.extend(id2range)

                     logging.info('Seq lenth on %d:%d'%(index,len(count_seq)))
                     if self.len_seq > len(count_seq):
                        self.len_seq = len(count_seq)

                  if part_index == num_parts - 1:
                     self.seq_identity = seq_identity[part_index*epc_size_part : ]
                  else:
                     self.seq_identity = seq_identity[part_index*epc_size_part : (part_index+1)*epc_size_part]
                  self.seq = []

        self.check_data_shape(data_shape)
        if single_channel == 0:
            self.provide_data = [(data_name, (batch_size,) + data_shape)]
        else:
            new_data_shape = (1,data_shape[1],data_shape[2])
            self.provide_data = [(data_name, (batch_size,) + new_data_shape)]
        self.provide_label = [(label_name, (batch_size,))]

        self.use_KD = offline_feature > 0
        if self.use_KD:
            assert all(aug==0 for aug in [bright_jitter_prob, gaussblur_prob,
                motionblur_prob, spatial_bright_prob, jpeg_compress_prob,
                downsample_prob, contrast_prob]),\
                        'data augmentation is not allowed when using offline KD'
            self.emb_size = offline_feature
            self.provide_data.append(mx.io.DataDesc(
                'kd_weight_data', (batch_size, offline_feature)))
            self.provide_label.append(mx.io.DataDesc(
                'offline_feature_label', (batch_size, offline_feature)))
        self.batch_size = batch_size
        self.data_shape = data_shape
        self.shuffle = shuffle
        self.image_size = '%d,%d'%(data_shape[1],data_shape[2])
        self.rand_mirror = rand_mirror
        print('rand_mirror', rand_mirror)
        self.cutoff = cutoff

        self.bright_jitter_prob = bright_jitter_prob
        self.bright_jitter_range = bright_jitter_range
        self.gaussblur_prob = gaussblur_prob
        self.gaussblur_kernelsz_max = gaussblur_kernelsz_max
        self.gaussblur_sigma = gaussblur_sigma
        self.motionblur_prob = motionblur_prob
        self.spatial_bright_prob = spatial_bright_prob
        self.spatial_bright_range = spatial_bright_range
        self.jpeg_compress_prob = jpeg_compress_prob
        self.jpeg_compress_quality_max = jpeg_compress_quality_max
        self.jpeg_compress_quality_min = jpeg_compress_quality_min
        self.rand_gray_prob = rand_gray_prob
        self.downsample_prob = downsample_prob
        self.downsample_min_width = downsample_min_width
        self.contrast_prob = contrast_prob
        self.contrast_range = contrast_range
        assert self.bright_jitter_range >= 0 and self.bright_jitter_prob >= 0 and \
                self.gaussblur_prob >= 0 and \
                self.gaussblur_kernelsz_max >= 0 and self.gaussblur_kernelsz_max % 2 == 1 and \
                self.motionblur_prob >= 0 and \
                self.spatial_bright_range >= 0 and self.spatial_bright_prob >= 0 and \
                self.jpeg_compress_prob >= 0 and \
                self.jpeg_compress_quality_min > 0 and \
                self.jpeg_compress_quality_min < jpeg_compress_quality_max and \
                self.jpeg_compress_quality_max <= 100 and \
                self.rand_gray_prob >= 0 and \
                self.downsample_prob >= 0 and \
                self.downsample_min_width > 0  and self.downsample_min_width < data_shape[2] and \
                self.contrast_range >= 0 and self.contrast_prob >= 0

        if self.bright_jitter_prob > 0:
            logging.info(
                    'Using brightness jitter augmentation prob={}. range={}'.
                    format(self.bright_jitter_prob, self.bright_jitter_range)
                    )
        if self.spatial_bright_prob > 0:
            logging.info(
                    'Using spatial brightness augmentation prob={}, range={}'.
                    format(self.spatial_bright_prob, self.spatial_bright_range)
                    )
            template_h, template_w = generate_template((self.data_shape[1],
                    self.data_shape[2], self.data_shape[0]))
            self.template_h = MultiContextConstArray(template_h)
            self.template_w = MultiContextConstArray(template_w)
        if self.gaussblur_prob > 0:
            logging.info(
                    'Using Gaussian blur augmentation with probability={}, max kernel size={}, sigma={}'.
                    format(self.gaussblur_prob,
                        self.gaussblur_kernelsz_max,
                        self.gaussblur_sigma)
                        )
        if self.motionblur_prob > 0:
            logging.info(
                    'Using motion blur augmentation with probability={}'.
                    format(self.motionblur_prob)
                    )
        if self.jpeg_compress_prob > 0:
            logging.info(
                    'Using JPEG compression with probability={}, max quality={}, min quality={}'.
                    format(self.jpeg_compress_prob,
                        self.jpeg_compress_quality_max,
                        self.jpeg_compress_quality_min)
                    )
        if self.rand_gray_prob > 0:
            logging.info(
                    'Using random gray augmentation with probability={}'.
                    format(self.rand_gray_prob)
                    )
            gray_mult = nd.array([[0.299, 0.299, 0.299],
                [0.587, 0.587, 0.587],
                [0.114, 0.114, 0.114]])
            self.gray_mult = MultiContextConstArray(gray_mult)
        if self.downsample_prob > 0:
            self.inter_method = inter_method
            assert (inter_method >= 1 and inter_method <= 4) or \
                    inter_method == 9 or inter_method == 10, \
                    'invalid inter_method: choose from (0,1,2,3,4,9,10)'
            logging.info(
                    'Using downsampling augmentation with probability={}, min downsampling width={}'.
                    format(self.downsample_prob, self.downsample_min_width)
                    )
        if self.contrast_prob > 0:
            logging.info(
                    'Using contrast jitter augmentation prob={}, range={}'.
                    format(self.contrast_prob, self.contrast_range)
                    )
            contrast_coef = nd.array([[[0.299, 0.587, 0.114]]])
            self.contrast_coef = MultiContextConstArray(contrast_coef)

        self.cur = 0
        self.is_init = False
        self.sample_type = sample_type
        self.num_samples_per_class = num_samples_per_class
        self.pk_replace = pk_replace
        self.repet_flag = repet_flag
        self.repet_num = repet_num
        self.cur_repet = 0
        self.last_data = None
        self.max_samples_per_class = max_samples_per_class
        self.unpack64 = unpack64
        self.reset()
        self.single_channel=single_channel

    def reset(self):
        """Resets the iterator to the beginning of the data."""
        print('call reset()')
        self.cur = 0

        if self.repet_flag:
           self.cur_repet = 0
           self.last_data = None

        if self.shuffle:
          # random sampling
          if self.sample_type == 0:
            random.seed(time.time())
            random.shuffle(self.seq)

          # PK sampling, partial samples are used in each epoch
          elif self.sample_type == 1:
            seq_identity = copy.deepcopy(self.seq_identity)
            random.seed(time.time())
            random.shuffle(seq_identity)

            self.seq = []
            for identity in seq_identity:
              id2range = range(self.id2range[identity][0], self.id2range[identity][1])
              pk_num = self.num_samples_per_class[0] if self.id_num[identity]<self.num_samples_per_class[1] else self.num_samples_per_class[2]
              num_lack=pk_num-self.id_num[identity]
              if num_lack>0 and self.pk_replace:
                pad_sample = np.random.choice(id2range,size=num_lack,replace=True)
                id2range.extend(pad_sample)
              if num_lack<0:
                id2range = np.random.choice(id2range, size=pk_num, replace=False)
              self.seq.extend(id2range)
           #random.shuffle(self.seq)

          # PK sampling, all samples are used in each epoch
          elif self.sample_type == 2:

            id_seq = []
            img_seq = []
            self.seq = []
            seq_identity = copy.deepcopy(self.seq_identity)

            id2range = {}
            for identity in seq_identity:
              id2range[identity] = range(self.id2range[identity][0], self.id2range[identity][1])

            while(len(seq_identity) > 0):
              random.seed(time.time())
              random.shuffle(seq_identity)
              for identity in seq_identity:
                id_seq.append(len(id_seq))

                random.shuffle(id2range[identity])
                if len(id2range[identity]) < self.num_samples_per_class:
                  #id_samples = list(np.random.choice(id2range[identity], size=self.num_samples_per_class, replace=True))
                  id_samples = list(np.random.choice(id2range[identity], size=len(id2range[identity]), replace=False))
                  id2range[identity] = []
                else:
                  id_samples = id2range[identity][: self.num_samples_per_class]
                  id2range[identity] = id2range[identity][self.num_samples_per_class :]
                if id2range[identity] == []:
                  seq_identity.remove(identity)
                  del id2range[identity]

                img_seq.append(id_samples)
            random.seed(time.time())
            random.shuffle(id_seq)
            for i in id_seq:
              self.seq.extend(img_seq[i])

          # PK sampling, partial samples are used in each epoch
          elif self.sample_type == 3:

            id_seq = []
            img_seq = []
            self.seq = []
            seq_identity = copy.deepcopy(self.seq_identity)

            id2range = {}
            for identity in seq_identity:
              id2range[identity] = range(self.id2range[identity][0], self.id2range[identity][1])
              if len(id2range[identity]) > self.max_samples_per_class:
                id2range[identity] = list(np.random.choice(id2range[identity], size=self.max_samples_per_class, replace=False))

            while(len(seq_identity) > 0):
              random.seed(time.time())
              random.shuffle(seq_identity)
              for identity in seq_identity:
                id_seq.append(len(id_seq))

                random.shuffle(id2range[identity])
                if len(id2range[identity]) < self.num_samples_per_class:
                  #id_samples = list(np.random.choice(id2range[identity], size=self.num_samples_per_class, replace=True))
                  id_samples = list(np.random.choice(id2range[identity], size=len(id2range[identity]), replace=False))
                  id2range[identity] = []
                else:
                  id_samples = id2range[identity][: self.num_samples_per_class]
                  id2range[identity] = id2range[identity][self.num_samples_per_class :]
                if id2range[identity] == []:
                  seq_identity.remove(identity)
                  del id2range[identity]

                img_seq.append(id_samples)

            random.seed(time.time())
            random.shuffle(id_seq)
            for i in id_seq:
              self.seq.extend(img_seq[i])

          #id shuffle only
          elif self.sample_type == 4:
            self.seq = []
            seq_identity = copy.deepcopy(self.seq_identity)
            random.seed(time.time())
            random.shuffle(seq_identity)
            for identity in seq_identity:
              id2range = range(self.id2range[identity][0], self.id2range[identity][1])
              self.seq.extend(id2range)
        if not self.sample_type ==  0 and self.num_parts > 1:
           self.seq = self.seq[:self.len_seq]
        if self.seq is None and self.imgrec is not None:
            self.imgrec.reset()

    @property
    def num_samples(self):
      return len(self.seq)

    def next_sample(self, index=None, lock=None, imgrec=None):
        """Helper function for reading in next sample."""
        #set total batch size, for example, 1800, and maximum size for each people, for example 45
        if self.seq is not None:
          while True:
            idx = self.seq[index]
            if imgrec is not None:
              lock.acquire()
              s = imgrec.read_idx(idx)
              lock.release()
              if self.unpack64 == 0:
                header, img = recordio.unpack(s)
              else:
                header, img = indexed_recordio.unpack64(s)
              label = header.label
              return label, img, None, None
            elif self.imgrec is not None:
              if self.cur >= len(self.seq):
                  raise StopIteration
              idx = self.seq[self.cur]
              self.cur += 1

              s = self.imgrec.read_idx(idx)
              if self.unpack64 == 0:
                header, img = recordio.unpack(s)
              else:
                header, img = indexed_recordio.unpack64(s)
              label = header.label
              return label, img, None, None
            else:
              label, fname, bbox, landmark = self.imglist[idx]
              return label, self.read_image(fname), bbox, landmark
        else:
            s = self.imgrec.read()
            if s is None:
                raise StopIteration
            if self.unpack64 == 0:
                header, img = recordio.unpack(s)
            else:
                header, img = indexed_recordio.unpack64(s)
            return header.label, img, None, None

    def brightness_aug(self, src, x):
      alpha = 1.0 + random.uniform(-x, x)
      src *= alpha
      src = nd.clip(src, 0, 255)
      # src[np.where(src > 255)] = 255
      return src

    def contrast_aug(self, src, prob, contrast_range):
        if random.random() < prob:
            src = src.astype('float32')
            alpha = 1.0 + random.uniform(-contrast_range, contrast_range)
            gray = src * self.contrast_coef.get(ctx=src.context)
            gray = (3.0 * (1.0 - alpha) / gray.size) * nd.sum(gray)
            src *= alpha
            src += gray
            src = nd.clip(src, 0, 255)
        return src.astype('uint8')

    def saturation_aug(self, src, x):
      alpha = 1.0 + random.uniform(-x, x)
      coef = np.array([[[0.299, 0.587, 0.114]]])
      gray = src * coef
      gray = np.sum(gray, axis=2, keepdims=True)
      gray *= (1.0 - alpha)
      src *= alpha
      src += gray
      return src

    def color_aug(self, img, prob, jitter_range):
        if random.random() < prob:
            img = img.astype('float32')
            augs = [self.brightness_aug]
            # augs = [self.brightness_aug, self.contrast_aug, self.saturation_aug]
            random.shuffle(augs)
            for aug in augs:
                #print(img.shape)
                img = aug(img, jitter_range)
                #print(img.shape)
        return img.astype('uint8')

    def gauss_blur_aug(self, src, prob, kernelsz, sigma):
        if random.random() < prob:
            kernel_size = random.randint(2, kernelsz//2)*2+1
            src = cv2.GaussianBlur(src.asnumpy().astype('float32'), (kernel_size,kernel_size), sigma)
            src = nd.array(src)
        return src.astype('uint8')

    def motion_blur_aug(self, src, prob):
        if random.random() < prob:
            length = random.randint(9, 18)
            angle = random.randint(-180, 180)
            if angle % 90 == 0:
                return src
            kernel, anchor = genaratePsf(length, angle)
            src = cv2.filter2D(src.asnumpy().astype('float32'), -1, kernel, anchor=anchor)
            src = nd.array(src)
        return src.astype('uint8')

    def spatial_bright_aug(self, img, template_h, template_w, prob, jitter_range):
        # img = np.copy(img)
        if random.random() < prob:
            template_h = template_h.get(ctx=img.context)
            template_w = template_w.get(ctx=img.context)
            img = img.astype('float32')

            angle = random.randint(0, 360)
            magnitude = random.uniform(-jitter_range, jitter_range)

            h_rand = math.cos(angle * 3.14159 / 180.0)
            w_rand = math.sin(angle * 3.14159 / 180.0)
            new_template_h = (1 - template_h) * h_rand * h_rand if h_rand < 0 else template_h * h_rand * h_rand
            new_template_w = (1 - template_w) * w_rand * w_rand if w_rand < 0 else template_w * w_rand * w_rand
            img = img * (1 + (new_template_h + new_template_w) * magnitude)
            img = nd.clip(img, 0, 255)
        return img.astype('uint8')

    def jpeg_compress_aug(self, img, prob, max_quality, min_quality):
        if random.random() < prob:
            quality = random.random() * (max_quality - min_quality) + min_quality
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
            res, encimg = cv2.imencode('.jpg', img.asnumpy(), encode_param)
            decimg = cv2.imdecode(encimg, -1)
            decimg = nd.array(decimg)
            return decimg.astype('uint8')
        else:
            return img

    def rand_gray_aug(self, img, prob):
        if random.random() < prob:
            # img = img.astype(np.float32)
            img = nd.dot(img.astype('float32'), self.gray_mult.get(ctx=img.context))
        return img.astype('uint8')

    def rand_downsample_aug(self, img, prob, min_downsample_width):
        if random.random() < prob:
            new_w = int(random.random() * (self.data_shape[2] - min_downsample_width) \
                    + min_downsample_width)
            new_h = int(new_w * self.data_shape[2] / self.data_shape[1])
            org_w = int(self.data_shape[2])
            org_h = int(self.data_shape[1])
            interpolation_method = get_inter_method(
                    self.inter_method,
                    (self.data_shape[1], self.data_shape[2], new_h, new_w)
                    )
            img = imresize(img, new_w, new_h, interp=interpolation_method)
            img = imresize(img, org_w, org_h, interp=interpolation_method)
            return img.astype('uint8')
        else:
            return img

    def mirror_aug(self, img):
      _rd = random.randint(0,1)
      if _rd==1:
        for c in xrange(img.shape[2]):
          img[:,:,c] = np.fliplr(img[:,:,c])
      return img

    def next(self, lock=None, imgrec=None, index=None):
        """Returns the next batch of data."""
        batch_size = self.batch_size
        if self.single_channel == 0:
           c, h, w = self.data_shape
           batch_data = nd.empty((batch_size, c, h, w))
        else:
           _, h, w = self.data_shape
           batch_data = nd.empty((batch_size, 1, h, w))
        if self.provide_label is not None:
            batch_label = nd.empty(self.provide_label[0][1])
        if self.use_KD:
            batch_data = [batch_data]
            kd_weight = mx.nd.ones((batch_size, self.emb_size))
            batch_data.append(kd_weight)
            batch_label = [batch_label]
            batch_teacher_feat = nd.empty((batch_size, self.emb_size))
            batch_label.append(batch_teacher_feat)
        if index is None:
            index = random.sample(range(0, len(self.seq)), batch_size)
        i = 0
        try:
            while i < batch_size:
                label, s, bbox, landmark = self.next_sample(index[i], lock, imgrec)
                _data = self.imdecode(s)
                if self.rand_mirror:
                  _rd = random.randint(0,1)
                  if _rd==1:
                    _data = mx.ndarray.flip(data=_data, axis=1)
                if self.cutoff>0:
                  centerh = random.randint(0, _data.shape[0]-1)
                  centerw = random.randint(0, _data.shape[1]-1)
                  half = self.cutoff//2
                  starth = max(0, centerh-half)
                  endh = min(_data.shape[0], centerh+half)
                  startw = max(0, centerw-half)
                  endw = min(_data.shape[1], centerw+half)
                  _data = _data.astype('float32')
                  _data[starth:endh, startw:endw, :] = 127.5
                _data = self.augmentation_transform(_data)
                if not self.single_channel == 0:
                   _data = _data.astype('float32')
                   _gray = (_data[:,:,0]*0.299+_data[:,:,1]*0.587+_data[:,:,2]*0.114).reshape(h,w,1)
                   _data = _gray.astype('uint8')
                   _data = _data.astype('float32')
                data = [_data]
                try:
                    self.check_valid_image(data)
                except RuntimeError as e:
                    logging.debug('Invalid image, skipping:  %s', str(e))
                    continue
                for datum in data:
                    assert i < batch_size, 'Batch size must be multiples of augmenter output length'
                    #print(datum.shape)
                    if not self.use_KD:
                        if not isinstance(label, numbers.Number):
                            label = label[0]
                        batch_data[i][:] = self.postprocess_data(datum)
                        batch_label[i][:] = label
                    else:
                        batch_data[0][i][:] = self.postprocess_data(datum)
                        batch_label[0][i][:] = label[0]
                        batch_label[1][i][:] = label[2:]
                    i += 1
        except StopIteration:
            if i < batch_size: # same as last_batch_handle='discard'
                raise StopIteration

        if self.databatch_process_callback is not None:
            ## Subtract Mean, scale Or Trans RGB2YUV
            if not self.use_KD:
                batch_data = self.databatch_process_callback(batch_data)
                return io.DataBatch([batch_data], [batch_label], batch_size-i)
            else:
                batch_data[0] = self.databatch_process_callback(batch_data[0])
                return io.DataBatch(batch_data, batch_label, batch_size-i)

    def check_data_shape(self, data_shape):
        """Checks if the input data shape is valid"""
        if not len(data_shape) == 3:
            raise ValueError('data_shape should have length 3, with dimensions CxHxW')
        if not data_shape[0] == 3:
            raise ValueError('This iterator expects inputs to have 3 channels.')

    def check_valid_image(self, data):
        """Checks if the input data is valid"""
        if len(data[0].shape) == 0:
            raise RuntimeError('Data shape is wrong')

    def imdecode(self, s):
        """Decodes a string or byte string to an NDArray.
        See mx.img.imdecode for more details."""
        img = mx.image.imdecode(s) #mx.ndarray
        return img

    def read_image(self, fname):
        """Reads an input image `fname` and returns the decoded raw bytes.

        Example usage:
        ----------
        >>> dataIter.read_image('Face.jpg') # returns decoded raw bytes.
        """
        with open(os.path.join(self.path_root, fname), 'rb') as fin:
            img = fin.read()
        return img

    def augmentation_transform(self, data):
        """Transforms input data with specified augmentation."""
        if self.bright_jitter_prob > 0 and self.spatial_bright_prob > 0:
            bright_aug = np.random.choice(['bright_jitter', 'spatial_bright'],
                    1, p=[0.3, 0.7])
            if bright_aug[0] == 'bright_jitter':
                data = self.color_aug(data, self.bright_jitter_prob, self.bright_jitter_range)
            else:
                data = self.spatial_bright_aug(data,
                    self.template_h, self.template_w,
                    self.spatial_bright_prob,
                    self.spatial_bright_range)
        else:
            if self.bright_jitter_prob > 0:
                data = self.color_aug(data, self.bright_jitter_prob, self.bright_jitter_range)
            elif self.spatial_bright_prob > 0:
                data = self.spatial_bright_aug(data,
                    self.template_h, self.template_w,
                    self.spatial_bright_prob,
                    self.spatial_bright_range)
        if self.gaussblur_prob > 0:
            data = self.gauss_blur_aug(data,
                self.gaussblur_prob,
                self.gaussblur_kernelsz_max,
                self.gaussblur_sigma)
        if self.motionblur_prob > 0:
            data = self.motion_blur_aug(data,
                self.motionblur_prob)
        if self.jpeg_compress_prob > 0:
            data = self.jpeg_compress_aug(data,
                self.jpeg_compress_prob,
                self.jpeg_compress_quality_max,
                self.jpeg_compress_quality_min)
        if self.rand_gray_prob > 0:
            data = self.rand_gray_aug(data,
                self.rand_gray_prob)
        if self.downsample_prob > 0:
            data = self.rand_downsample_aug(data,
                self.downsample_prob, self.downsample_min_width)
        if self.contrast_prob > 0:
            data = self.contrast_aug(data, self.contrast_prob, self.contrast_range)
        return data

    def postprocess_data(self, datum):
        """Final postprocessing step before image is loaded into the batch."""
        return nd.transpose(datum, axes=(2, 0, 1))

class FaceImageIterList(io.DataIter):
  def __init__(self, iter_list):
    assert len(iter_list)>0
    self.provide_data = iter_list[0].provide_data
    self.provide_label = iter_list[0].provide_label
    self.iter_list = iter_list
    self.cur_iter = None

  def reset(self):
    self.cur_iter.reset()

  def next(self):
    self.cur_iter = random.choice(self.iter_list)
    while True:
      try:
        ret = self.cur_iter.next()
      except StopIteration:
        self.cur_iter.reset()
        continue
      return ret

class PrefetchProcessIter(io.DataIter):
    def __init__(self, data_iter, prefetch_process=4, capacity=4, part_index=0):
        super(PrefetchProcessIter, self).__init__()
        assert data_iter is not None
        self.data_iter = data_iter
        self.batch_size = self.provide_data[0][1][0]
        if hasattr(self.data_iter, 'use_KD'):
            self.use_KD = data_iter.use_KD
        else:
            self.use_KD = False
        self.rank = part_index
        self.batch_counter = 0

        if hasattr(self.data_iter, 'epoch_size'):
            self.epoch_size = self.data_iter.epoch_size
            if self.data_iter.epoch_size is None:
                self.epoch_size = int(self.data_iter.num_samples/self.batch_size)
        else:
            self.epoch_size = int(self.data_iter.num_samples/self.batch_size)

        self.next_iter = 0
        self.prefetch_process = prefetch_process

        self._data_queue = dataloader.Queue(maxsize=capacity)
        self._data_buffer = Queue.Queue(maxsize=capacity*2)
        self._index_queue = multiprocessing.Queue()

        self.prefetch_reset_event = multiprocessing.Event()
        self.epoch_end_event = multiprocessing.Event()
        self.next_reset_event = threading.Event()

        self.lock = multiprocessing.Lock()
        self.imgrec = self.data_iter.imgrec

        def prefetch_func(data_queue, event, end_event):
            while True:
                if event.is_set() and (not end_event.is_set()):
                    index = []
                    i = 0
                    while i < self.batch_size:
                        try:
                            index.append(self._index_queue.get())
                            i += 1
                        except:
                            end_event.set()
                    if i == self.batch_size:
                        next_data = self.data_iter.next(self.lock, self.imgrec, index)
                        if self.use_KD:
                            data_queue.put((
                                (dataloader.default_mp_batchify_fn(next_data.data[0]),
                                    dataloader.default_mp_batchify_fn(next_data.data[1])),
                                (dataloader.default_mp_batchify_fn(next_data.label[0]),
                                    dataloader.default_mp_batchify_fn(next_data.label[1]))
                                ))
                        else:
                            data_queue.put((dataloader.default_mp_batchify_fn(next_data.data[0]),
                                        dataloader.default_mp_batchify_fn(next_data.label[0])))

        def next_func(data_queue, event):
            while True:
                if event.is_set():
                    batch, label = data_queue.get(block=True)
                    if self.use_KD:
                        batch_data = dataloader._as_in_context(batch[0],
                                context.cpu())
                        batch_weight = dataloader._as_in_context(batch[1],
                                context.cpu())
                        batch = (batch_data, batch_weight)
                        label_gt = dataloader._as_in_context(label[0],
                                context.cpu())
                        label_gt = label_gt.reshape((label_gt.shape[0],))
                        label_teacher = dataloader._as_in_context(label[1],
                                context.cpu())
                        label = (label_gt, label_teacher)
                    else:
                        batch = dataloader._as_in_context(batch, context.cpu())
                        label = dataloader._as_in_context(label, context.cpu())
                        label = label.reshape((label.shape[0],))
                    self._data_buffer.put((batch, label))

        # producer next
        self.produce_lst = []
        for ith in range(prefetch_process):
            p_process = multiprocessing.Process(target=prefetch_func,
                                                args=(self._data_queue, self.prefetch_reset_event,
                                                      self.epoch_end_event))
            p_process.daemon = True
            p_process.start()
            self.produce_lst.append(p_process)

        # consumer get
        self.data_buffer = {}
        self.prefetch_thread = threading.Thread(target=next_func,
                                                args=(self._data_queue, self.next_reset_event))
        self.prefetch_thread.daemon = True
        self.prefetch_thread.start()

        # first epoch
        self.reset()

    def __del__(self):
        self.__clear_queue()

        for i_process in self.produce_lst:
            i_process.join()
        self.prefetch_thread.join()

    def __clear_queue(self):
        """ clear the queue"""
        while True:
            try:
                self._data_queue.get_nowait()
            except:
                break
        while True:
            try:
                self._data_buffer.get_nowait()
            except:
                break
        while True:
            try:
                self._index_queue.get_nowait()
            except:
                break

    @property
    def provide_data(self):
        return self.data_iter.provide_data

    @property
    def provide_label(self):
        return self.data_iter.provide_label

    def reset(self):
        self.epoch_end_event.set()
        self.next_iter = 0
        self.data_iter.reset()
        self.__clear_queue()

        assert self._index_queue.empty()
        seq_index = range(0, len(self.data_iter.seq))
        random.shuffle(seq_index)
        for index in range(0, len(self.data_iter.seq)):
            self._index_queue.put(seq_index[index])

        self.prefetch_reset_event.set()
        self.next_reset_event.set()
        self.epoch_end_event.clear()

    def iter_next(self):
        self.next_iter += 1
        if self.next_iter > self.epoch_size:
            self.prefetch_reset_event.clear()
            self.next_reset_event.clear()
            return False
        else:
            return True

    def next(self):
        if self.iter_next():
            self.batch_counter += 1
            batch, label = self._data_buffer.get(block=True)
            if self.use_KD:
                if self.batch_counter % viz.save_img_batches == 0:
                    viz.save_img(batch[0], label[0],
                            "%d_%d.jpg"%(self.rank, self.batch_counter))
                return io.DataBatch(data=[batch[0],batch[1]],
                        label=[label[0],label[1]], pad=0)
            else:
                if self.batch_counter % viz.save_img_batches == 0:
                    viz.save_img(batch, label, "%d_%d.jpg"%(self.rank, self.batch_counter))
                return io.DataBatch(data=[batch], label=[label], pad=0)
        else:
            raise StopIteration

def PrefetchFaceIter(prefetch_process=8, prefetch_process_keep=16, local_run = False, **kwargs):
    use_prefetch = True if local_run == False else False
    if use_prefetch:
      data_iter = PrefetchProcessIter(
              FaceImageIter(**kwargs),
              prefetch_process, prefetch_process_keep,
              part_index=kwargs['part_index'],
              )
      import atexit
      atexit.register(lambda a : a.__del__(), data_iter)
    else:
      data_iter = FaceImageIter(**kwargs)
      data_iter.epoch_size=int(data_iter.num_samples/data_iter.provide_data[0][1][0])
    return data_iter


