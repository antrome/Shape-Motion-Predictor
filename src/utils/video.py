import cv2
import random

vidcap = cv2.VideoCapture('../../videos/yoga.mp4')
length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
i=random.randint(0,length-100)
vidcap.set(cv2.CAP_PROP_POS_FRAMES, i)
success,image = vidcap.read()
count = 0
print(type(count))
print(type(i))

while success:
  f=count+i
  cv2.imwrite("../../videos/frames/frame%d.jpg" % f, image)     # save frame as JPEG file
  success,image = vidcap.read()
  #print('Read a new frame: ', success)
  count += 1

  if count == 100:
    break


