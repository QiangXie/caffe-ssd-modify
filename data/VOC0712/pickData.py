import os
import shutil

testDataPath = "/media/mmr6-raid5/xieqiang/DetectionData/peopleAndVehicle/test/"
trainvalDataPath = "/media/mmr6-raid5/xieqiang/DetectionData/peopleAndVehicle/trainval/"
 
testJpgPath = os.path.join(testDataPath,"JPEGImages")
counter = 0
for index,item in enumerate(os.listdir(testJpgPath)):
    srcJpgPath = os.path.join(testJpgPath,item)
    srcXmlPath = os.path.join(testDataPath,"Annotations",os.path.splitext()[0] + ".xml")
    if (index%10) == 0:
    else:
        os.remove(srcXmlPath)
