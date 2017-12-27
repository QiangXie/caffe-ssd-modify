import os
import cv2
import sys
sys.path.insert(0,'../python')
import caffe
import numpy as np
from google.protobuf import text_format
from caffe.proto import caffe_pb2
from timer import Timer

#set args
batch_size = 1
images_path = "../../../../Data/plate_preprocess/raw_data/double_line/"
GPU_ID = 0 
image_resize = 300
model_def = "./deploy.prototxt"
model_weights = './ssd300x300_v1.caffemodel'

SAVE_FLAG = True
save_path = "../../../../Data/plate_preprocess/car_single/2_line_/"


caffe.set_device(GPU_ID)
caffe.set_mode_gpu()

net = caffe.Net(model_def,model_weights,caffe.TEST)

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2, 0, 1))
# mean pixel
transformer.set_mean('data', np.array([104,117,123])) 
# the reference model operates on images in [0,255] range instead of [0,1]
transformer.set_raw_scale('data', 255)  
# the reference model has channels in BGR order instead of RGB
transformer.set_channel_swap('data', (2,1,0))  

def detect_batch_images(net, image_pathes,image_resize,transformer):
   batch_size_ = len(image_pathes)
   net.blobs['data'].reshape(batch_size_,3,image_resize,image_resize)
   for i,image_path in enumerate(image_pathes):
       im_cv2 = cv2.imread(image_path)
       im_resize = cv2.resize(im_cv2,(image_resize,image_resize),interpolation = cv2.INTER_CUBIC)
       im_resize = cv2.cvtColor(im_resize,cv2.COLOR_BGR2RGB)
       image = im_resize/255.0
       
       transformed_image = transformer.preprocess('data',image)
       net.blobs['data'].data[i] = transformed_image
   timer_ = Timer()
   timer_.tic()
   detections = net.forward()['detection_out']
   timer_.toc()

   # Parse the outputs.
   image_index = detections[0,0,:,0] 
   det_label = detections[0,0,:,1]
   det_conf = detections[0,0,:,2]
   det_xmin = detections[0,0,:,3]
   det_ymin = detections[0,0,:,4]
   det_xmax = detections[0,0,:,5]
   det_ymax = detections[0,0,:,6]

   # Get detections with confidence higher than 0.6.
   top_indices = list()
   for i in range(len(det_conf)):
       if det_conf[i] >= 0.6 and int(det_label[i]) == 7:
           top_indices.append(i)
   #top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.6]

   top_img_idx = image_index[top_indices]
   top_conf = det_conf[top_indices]
   top_xmin = det_xmin[top_indices]
   top_ymin = det_ymin[top_indices]
   top_xmax = det_xmax[top_indices]
   top_ymax = det_ymax[top_indices]
   for i in xrange(top_conf.shape[0]):
       xmin = int(round(top_xmin[i] * im_cv2.shape[1]))
       ymin = int(round(top_ymin[i] * im_cv2.shape[0]))
       xmax = int(round(top_xmax[i] * im_cv2.shape[1]))
       ymax = int(round(top_ymax[i] * im_cv2.shape[0]))
       score = top_conf[i]
       print "Vehicle was detected in ",image_pathes[int(top_img_idx[i])],"at:",xmin,ymin,xmax,ymax
       if SAVE_FLAG:
           if top_conf.shape[0] == 1:
               img2save = cv2.imread(image_pathes[int(top_img_idx[i])])[ymin:ymax, xmin:xmax]
               save_img_path = os.path.join(save_path, os.path.split(image_path)[1])
               cv2.imwrite(save_img_path, img2save)
           else:
               img2save = cv2.imread(image_pathes[int(top_img_idx[i])])[ymin:ymax, xmin:xmax]
               save_img_path = os.path.join(save_path, os.path.splitext(os.path.split(image_path)[1])[0] + "_" + str(i) + ".jpg")
               cv2.imwrite(save_img_path, img2save)




images = os.listdir(images_path)
total_timer = Timer()
total_timer.tic()
pathes = list()
for item in images:
    image_path = os.path.join(images_path,item)
    pathes.append(image_path)
    if len(pathes) == batch_size:
        detect_batch_images(net, pathes,image_resize,transformer)
        pathes = []
total_timer.toc()
fps_value = float(len(images)) / total_timer.total_time
print "FPS: {}".format(fps_value)
