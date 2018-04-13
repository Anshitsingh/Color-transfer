import cv2
import numpy as np
import scipy.misc
from scipy.misc import imread,imresize


target_WIDTH = 800
target_HEIGHT = 600


target = cv2.imread('C:/Users/user/Desktop/SEM 8/ADIPCV/Assignments/assignment 5/Flowerbed.jpg')
target = imresize(target,(target_HEIGHT,target_WIDTH))

painting=target
source = cv2.imread('C:/Users/user/Desktop/SEM 8/ADIPCV/Assignments/assignment 5/Painting_flowers.jpg')
source = imresize(source,(target_HEIGHT,target_WIDTH))
height,width,channel = target.shape

def show_images(source,target,tranformedImage):
    
    draw_image("Source", source)
    draw_image("Target", target)
    draw_image("Output", tranformedImage)
    
    cv2.imwrite("C:/Users/user/Desktop/SEM 8/ADIPCV/Assignments/assignment 5/Pai",tranformedImage)
    cv2.waitKey(0) 
    cv2.destroyAllWindows()



def draw_target(title, target):   
    
    #change the target size 
    height = 700
    resize = cv2.resize(target,(height,int((height/float(target.shape[1]))*target.shape[0])))
    cv2.imshow(title, resize)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def RGB2lab(image):
    r = np.zeros(image.shape,dtype=float)
    temp = np.dot([[1/np.sqrt(3),0,0],[0,1/np.sqrt(6),0],[0,0,1/np.sqrt(2)]],[[1,1,1],[1,1,-2],[1,-1,0]])
    for i in range(0,height):
        for j in range(0,width):
            lms = np.dot([[0.3811,0.5783,0.0402],[0.1967,0.7244,0.0782],[0.0241,0.1288,0.8444]],image[i,j,:3])
            for k in range(channel):
                if lms[k]>0:
                    lms[k]=np.log10(lms[k])
            lab = np.dot(temp,lms)
            r[i,j] = lab
    return r

def lab2RGB(target):
    r = np.zeros(target.shape,dtype=int)
    temp = np.dot([[1,1,1],[1,1,-1],[1,-2,0]],[[np.sqrt(3)/3,0,0],[0,np.sqrt(6)/6,0],[0,0,np.sqrt(2)/2]])
    for i in range(0,height):
        for j in range(0,width):
            lms = 10 ** np.dot(temp,target[i,j,:3])
            rgb = np.dot([[4.4679,-3.5873,0.1193],[-1.2186,2.3809,-0.1624],[0.0479,-0.2439,1.2045]],lms)
            r[i,j] = rgb
            for k in range(channel):
                if r[i,j,k] >= 255:
                    r[i,j,k] = 255
                elif r[i,j,k] <=0:
                    r[i,j,k] = 0
    return r

target = RGB2lab(target)
source = RGB2lab(source)

def getavgstd(target):
    avg = []
    std = []
    target_avg_l = np.mean(target[:,:,0])
    target_std_l = np.std(target[:,:,0])
    target_avg_a = np.mean(target[:,:,1])
    target_std_a = np.std(target[:,:,1])
    target_avg_b = np.mean(target[:,:,2])
    target_std_b = np.std(target[:,:,2])
    avg.append(target_avg_l)
    avg.append(target_avg_a)
    avg.append(target_avg_b)
    std.append(target_std_l)
    std.append(target_std_a)
    std.append(target_std_b)
    return (avg,std)

target_avg,target_std = getavgstd(target)
source_avg,source_std = getavgstd(source)

for i in range(0,height):
    for j in range(0,width):
        for k in range(0,channel):
            t = target[i,j,k]
            t = (t-target_avg[k])*(source_std[k]/target_std[k]) + source_avg[k]
            target[i,j,k] = t
target = lab2RGB(target)

cv2.imwrite('C:/Users/user/Desktop/SEM 8/ADIPCV/Assignments/assignment 5/out1.jpg',target)
