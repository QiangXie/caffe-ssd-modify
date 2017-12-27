import os
import cv2
import subprocess
REDO_LIST = True 



dataPath = "/home/swli/Data/key_plate_pascal/"

testDataPath = os.path.join(dataPath, "test")
trainvalDataPath = os.path.join(dataPath, "trainval")

if REDO_LIST:
    print "Generate test.txt and trainval.txt..."


    testList = open("./test.txt",'w')
    trainvalList = open("./trainval.txt",'w')
    testSizeList = open("./test_name_size.txt",'w')

    for item in os.listdir(os.path.join(testDataPath,"JPEGImages")):
        singleXmlPath = os.path.join(testDataPath,"Annotations",os.path.splitext(item)[0]+".xml")
        singleJpgPath = os.path.join(testDataPath,"JPEGImages",item)
        print "Process {} ...".format(singleJpgPath)
        if not os.path.exists(singleXmlPath):
            print "{} not exists.".format(singleXmlPath)
        else:
            im = cv2.imread(singleJpgPath)
            testList.write(os.path.join("test/JPEGImages",item))
            testList.write(' ')
            testList.write(os.path.join("test/Annotations",os.path.splitext(item)[0]+".xml"))
            testList.write('\n')
            testSizeList.write(os.path.splitext(item)[0])
            testSizeList.write(' ')
            testSizeList.write(str(im.shape[0]))
            testSizeList.write(' ')
            testSizeList.write(str(im.shape[1]))
            testSizeList.write('\n')
            

    for item in os.listdir(os.path.join(trainvalDataPath,"JPEGImages")):
        singleXmlPath = os.path.join(trainvalDataPath,"Annotations",os.path.splitext(item)[0]+".xml")
        singleJpgPath = os.path.join(trainvalDataPath,"JPEGImages",item)
        print "Process {} ...".format(singleJpgPath)
        if not os.path.exists(singleXmlPath):
            print "{} not exists.".format(singleXmlPath)
        else:
            im = cv2.imread(singleJpgPath)
            trainvalList.write(os.path.join("trainval/JPEGImages",item))
            trainvalList.write(' ')
            trainvalList.write(os.path.join("trainval/Annotations",os.path.splitext(item)[0]+".xml"))
            trainvalList.write('\n')

    trainvalList.close()
    testSizeList.close()
    testList.close()
    print "Done."

if os.path.lexists("../../examples/VOC0712/VOC0712_test_lmdb"):
    print "Remove Soft connection..."
    os.remove("../../examples/VOC0712/VOC0712_test_lmdb")
if os.path.lexists("../../examples/VOC0712/VOC0712_trainval_lmdb"):
    os.remove("../../examples/VOC0712/VOC0712_trainval_lmdb")

subprocess.call("bash ./create_data.sh", shell = True)
