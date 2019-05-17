import mxnet as mx
import numpy as np
np_yuv_array = (np.array([76, 150, 29, -43, -84, 127, 127,-106, -21]) * 1.0 / 256).reshape((3,3))
yuv_matrix = mx.nd.array(np_yuv_array)
yuv_bias = mx.nd.array(np.array([128.0/256 - 128,  128.0/256, 128.0/256]))
yuv_matrix_inv = mx.nd.array(np.linalg.inv(np_yuv_array))

_trans_g_mean = 0
_trans_g_scale = 0

def _trans_mean_scale_callback(ndarray):
    return (ndarray - _trans_g_mean) * _trans_g_scale

def _trans_rgb2yuv_callback(ndarray):
    batch_rgb = ndarray
    N, C, H, W = batch_rgb.shape
    assert(C == 3)
    batch_trans_flatten = batch_rgb.reshape((0,0,-1))  # N, C, HW
    batch_trans = (batch_trans_flatten.transpose((0,2,1))).reshape((-1,3)) # N, HW, C --> N*HW, C
    final_yuv = mx.nd.FullyConnected(data=batch_trans, num_hidden=3, weight=yuv_matrix, bias=yuv_bias) # N*HW, C
    final_yuv = (final_yuv.reshape((N, H, W, C))).transpose((0,3,1,2)) / 128.0
    return final_yuv

def _trans_yuv2rgb_callback(ndarray):
    batch_yuv = ndarray * 128.0
    N, C, H, W = batch_yuv.shape
    assert(C == 3)
    batch_trans_flatten = batch_yuv.reshape((0,0,-1))  # N, C, HW
    batch_trans = (batch_trans_flatten.transpose((0,2,1))).reshape((-1,3)) # N, HW, C --> N*HW, C
    batch_trans = batch_trans - yuv_bias
    final_rgb = mx.nd.FullyConnected(data=batch_trans, num_hidden=3, weight=yuv_matrix_inv, no_bias=True) # N*HW, C
    final_rgb = ((final_rgb.reshape((N, H, W, C))).transpose((0,3,1,2)) - 128.0) / 128.0
    return final_rgb

def TransMeanScale(mean, scale):
    global _trans_g_mean, _trans_g_scale
    _trans_g_mean = mean
    _trans_g_scale = scale
    return _trans_mean_scale_callback

# N,C,H,W
def TransRGB2YUV():
    '''
        Input Format : batch_rgb, NCHW with C == 3, and Value Range is [0, 255], type is NDArray
        Output Format : final_rgb, NCHW with C == 3, and Value Range is [-1, 1], type is NDArray
    '''
    return _trans_rgb2yuv_callback

# N,C,H,W
def TransYUV2RGB():
    '''
        Input Format : batch_yuv, NCHW with C == 3, and Value Range is [-1, 1], type is NDArray
        Output Format : final_rgb, NCHW with C == 3, and Value Range is [-1, 1], type is NDArray
    '''
    return _trans_yuv2rgb_callback

if __name__ == "__main__":
    import cv2
    import numpy as np
    img = cv2.imread("/data-sdc/xin.wang/face_data/CASIA-CLEAN/1408274/016.jpg")
    img_yuv_opencv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    A = np.zeros((2,) + img.shape)
    A[0:] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    A[1:] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    A = mx.nd.array(A).transpose((0,3,1,2)) # N, C, H, W
    img_yuv_nd = _trans_rgb2yuv_callback(mx.nd.array(A)) #-->[-1,1] yuv
    img_rgb_nd = _trans_yuv2rgb_callback(img_yuv_nd) #[-1,1] rgb
    img_yuv_nd = img_yuv_nd.transpose((0,2,3,1))*128.0 + 128 # N, H, W, C
    img_rgb_nd = img_rgb_nd.transpose((0,2,3,1))*128.0 + 128

    img_yuv_np = img_yuv_nd.asnumpy()[0].astype(np.uint8)
    img_rgb_np = img_rgb_nd.asnumpy()[0].astype(np.uint8)
    cv2.imwrite("1_opencv.jpg", img_yuv_opencv)
    cv2.imwrite("1_nd.jpg", img_yuv_np)
    img_rgb_np = cv2.cvtColor(img_rgb_np, cv2.COLOR_RGB2BGR)
    cv2.imwrite("1_nd_rgb.jpg", img_rgb_np)
