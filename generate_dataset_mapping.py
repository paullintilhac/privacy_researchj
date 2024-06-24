
import torchvision
import numpy as np
from PIL import Image

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
    #print("count: " + str(i))
    #o = ((np.transpose(o,(1,0,2))+1)*127.5).astype(np.uint8)
    o = ((o+1)*127.5).astype(np.uint8)
    o_max = np.max(o)
    img_max = np.max(img)
    o_min = np.min(o)
    img_min = np.min(img)
    
    if i==0:
        print("o range: " + str(o_min)+","+str(o_max)+ ", img range: " + str(img_min)+","+str(img_max))
        #print("o shape: " + str(o.shape) + ", img shapeL " +str(img.shape))
        #print("o type: " + str(type(o[0][0][0])) + ", img type: " + str(type(img[0][0][0])))
        #print("o: " + str(o))
        #print("img: " + str(img))
    dist = np.sum(np.absolute(o-img))
    if dist < minDist:
        minDist = dist
        minDistIndex = i
        minDistImg = o

    if (o==img).all():
      return i
  print("minDist: " + str(minDist) + ", minDistIndex: " + str(minDistIndex))
  print("minDistImage: " + str(minDistImg))  
  thisImg = Image.fromarray(minDistImg)
  print("thisImg: " + str(thisImg))
  thisImg.save("image_1.jpeg","JPEG")
for count,newImg in enumerate(train_1):
    print("new img: " + str(newImg))
    #thatImg = Image.fromarray(newImg)
    print("that img: " + str(newImg))
    newImg.save("image_2.jpeg","JPEG")
    print("count: " + str(count))
    ind=get_matching_image_index(newImg)
    print("ind: " +str(ind))
    print("new img: " + str(np.array(newImg)))
