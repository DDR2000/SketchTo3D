import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt

def conditionA(img, y, x):
    x0, y0, x1, y1 = x-1, y-1, x+1, y+1
    #[p2,p3,p4,p5,p6,p7,p8,p9]
    neighbours=[img[y0,x],img[y0,x1],img[y,x1],img[y1,x1],img[y1,x],img[y1,x0],img[y,x0],img[y0,x0]]
    transitions=0
    for i in range(1, len(neighbours)):
        transitions += (neighbours[i]>neighbours[i-1])*(neighbours[i]-neighbours[i-1])
    transitions += (neighbours[0]>neighbours[-1])*(neighbours[0]-neighbours[-1])
    return transitions

def conditionB(img, y, x):
    x0, y0, x1, y1 = x-1, y-1, x+1, y+1
    return img[y0,x] + img[y0,x1] + img[y,x1] + img[y1,x1] + img[y1,x] + img[y1,x0] + img[y,x0] + img[y0,x0]

def thinning(img):
    img = np.where(img>127, 0, 1)
    cycle=0
    while(True):
        note1=[]
        for i in range(len(img[0])-1):
            for j in range(len(img)-1):
                A = conditionA(img, j, i)
                B = conditionB(img, j, i)
                c = img[j,i] and (B >= 2) and (B <= 6) and (A == 1) and ((img[j-1, i]==0) or (img[j, i+1]==0) or (img[j+1, i]==0)) and ((img[j, i+1]==0) or (img[j+1, i]==0) or (img[j, i-1])==0)
                if c:
                    note1.append([j,i])
        for p in note1:
            img[p[0],p[1]]=0
        note2 = []
        for i in range(len(img[0])-1):
            for j in range(len(img)-1):
                A = conditionA(img, j, i)
                B = conditionB(img, j, i)
                c = img[j,i] and (B >= 2) and (B <= 6) and (A == 1) and ((img[j-1, i]==0) or (img[j, i+1]==0) or (img[j, i-1]==0)) and ((img[j-1, i]==0) or (img[j+1, i]==0) or (img[j, i-1])==0)
                if c:
                    note2.append([j,i])
        for p in note2:
            img[p[0],p[1]]=0
        if(len(note1)==0 or len(note2)==0):
            break
        plt.imshow(img,cmap = 'binary')
        plt.savefig("thinned"+str(cycle)+".png")
        cycle+=1

    return img

img = cv2.imread('source.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

kernel_size = 5
blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)

low_threshold = 50
high_threshold = 150
edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

rho = 1  # distance resolution in pixels of the Hough grid
theta = np.pi / 180  # angular resolution in radians of the Hough grid
threshold = 15  # minimum number of votes (intersections in Hough grid cell)
min_line_length = 50  # minimum number of pixels making up a line
max_line_gap = 50  # maximum gap in pixels between connectable line segments
line_image = np.copy(img) * 255  # creating a blank to draw lines on

# Run Hough on edge detected image
# Output "lines" is an array containing endpoints of detected line segments
lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                    min_line_length, max_line_gap)

for line in lines:
    for x1,y1,x2,y2 in line:
        cv2.line(line_image,(x1,y1),(x2,y2),(0,0,0),15)

gray = cv2.cvtColor(line_image, cv2.COLOR_BGR2GRAY)
ret,thresh1 = cv2.threshold(gray,70,255,0)
thresh1 = thinning(thresh1)
cv2.imwrite('res.jpg', line_image)
