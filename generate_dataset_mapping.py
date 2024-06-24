
import torchvision
import numpy as np
from PIL import Image
import torch

train_1 = [a[0] for a in torchvision.datasets.CIFAR10('./tmp', train=True, download=True)][:1]
print("shape of train 1: " + str(len(train_1)))

inputs = np.load( "exp/cifar10/x_train.npy")
print("shape of inputs: " + str(inputs.shape))
def get_matching_image_index(pil_img):
  img = np.asarray(pil_img)
  minDist = 1000000000
  minDistIndex = -1
  minDistImg = None
  for i,o in enumerate(inputs):
    o = ((o+1)*127.5).astype(np.uint8)
    
    dist = np.sum(np.absolute(o-img))
    if dist < minDist:
        minDist = dist
        minDistIndex = i
        minDistImg = o

    if (o==img).all():
      print("found exact match!")
      return i
  print("minDist: " + str(minDist/(32*32*3)) + ", minDistIndex: " + str(minDistIndex))
  #print("minDistImage: " + str(minDistImg))  
  thisImg = Image.fromarray(minDistImg)
  print("writing new image 1")
  thisImg.save("image_1.jpeg")
  return minDistIndex
mappings = []
for count,newImg in enumerate(train_1):
    #print("new img: " + str(newImg))
    #thatImg = Image.fromarray(newImg)
    #print("that img: " + str(newImg))
    newImg.save("image_2.jpeg","JPEG")
    print("count: " + str(count))
    ind=get_matching_image_index(newImg)
    mappings.append(ind)
    print("ind: " +str(ind))
    print("unique matches so far: " + str(len(set(mappings))) + " out of "+str(count+1))
    #print("new img: " + str(np.array(newImg)))
    
np.save("mappings.npy",np.array(mappings))
print("unique mathces: " + str(len(set(mappings))) + ", out of "+str(len(train_1)))
