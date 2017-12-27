cd /home/swli/Documents/Code/caffe-ssd
./build/tools/caffe train \
--solver="models/VGGNet/VOC0712/SSD_300x300/solver.prototxt" \
--snapshot="models/VGGNet/VOC0712/SSD_300x300/VGG_VOC0712_SSD_300x300_iter_138.solverstate" \
--gpu 1,3,4,8 2>&1 | tee jobs/VGGNet/VOC0712/SSD_300x300/VGG_VOC0712_SSD_300x300.log
