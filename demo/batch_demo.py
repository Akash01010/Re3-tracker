import cv2
import glob
import numpy as np
import sys
import os.path

basedir = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(basedir, os.path.pardir)))
from tracker import re3_tracker

if not os.path.exists(os.path.join(basedir, 'data')):
    import tarfile
    tar = tarfile.open(os.path.join(basedir, 'data.tar.gz'))
    tar.extractall(path=basedir)

cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Image', 640, 480)
tracker = re3_tracker.Re3Tracker()
# image_paths = sorted(glob.glob(os.path.join(os.path.dirname(__file__), 'data', '*.jpg')))

examples_dir=os.path.dirname(__file__)

images =next(os.walk(os.path.join(examples_dir,"data")))[2]
print(images[0])
efn = [int(f.split('.')[0]) for f in images]
# image_no=sorted(efn)
# image_names_sorted=[image_number+".jpg" for image_number in image_no]
# image_paths=[os.path.join(examples_dir,images_sorted) for images_sorted in image_names_sorted]

#initial_bbox = [190, 158, 249, 215]
initial_bbox = [294,69,452,312]
# Provide a unique id, an image/path, and a bounding box.
tracker.track('ball', os.path.join(examples_dir,"data",images[0]), initial_bbox)
print('ball track started')
count=-1
for img in sorted(efn):
    # print(img)
    count+=1
    image_path=os.path.join("data",str(img)+".jpg")
    # print(image_path)
    image = cv2.imread(image_path)
    # Tracker expects RGB, but opencv loads BGR.
    imageRGB = image[:,:,::-1]
    if count < 1:
        # The track alread exists, so all that is needed is the unique id and the image.
        bbox = tracker.track('ball', imageRGB)
        color = cv2.cvtColor(np.uint8([[[0, 128, 200]]]),
            cv2.COLOR_HSV2RGB).squeeze().tolist()
        cv2.rectangle(image,
                (int(bbox[0]), int(bbox[1])),
                (int(bbox[2]), int(bbox[3])),
                color, 2)
    elif count == 1:
        # Start a new track, but continue the first as well. Only the new track needs an initial bounding box.
        bboxes = tracker.multi_track(['ball', 'logo','mytrack'], imageRGB, {'logo' : [452, 89, 600, 300],'mytrack' : [500, 100, 600, 300]})
        print('logo track started')
    else:
        # Both tracks are started, neither needs bounding boxes.
        bboxes = tracker.multi_track(['ball', 'logo', 'mytrack'], imageRGB)
    if count >= 1:
        f = open(os.path.join(examples_dir,"output_data", str(img)+".txt"), 'w')
        for bb,bbox in enumerate(bboxes):
            color = cv2.cvtColor(np.uint8([[[bb * 255 / len(bboxes), 128, 200]]]),
                cv2.COLOR_HSV2RGB).squeeze().tolist()
            cv2.rectangle(image,
                    (int(bbox[0]), int(bbox[1])),
                    (int(bbox[2]), int(bbox[3])),
                    color, 2)
            print(int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3]))
            f.write(str(bbox[0])+" "+str(bbox[1])+" "+str(bbox[2])+" "+str(bbox[3])+"\n")
        print("\n")
        f.close()
    cv2.imshow('Image', image)
    cv2.waitKey(1)
