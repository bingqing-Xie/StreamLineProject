from centerline.geometry import Centerline
from shapely.geometry import Polygon
import cv2
from SubFun import SubFun
import numpy as np
from config import *
font = cv2.FONT_HERSHEY_DUPLEX
bottomLeftCornerOfText = (10, 500)
fontScale = 0.4
fontColor =COLOR_RED
lineType = 1

folder="ReynoldTest/"
outline = cv2.imread(folder+"P104_12mo_2.5mLmin_Bend_5umbeads_exp500-1.tif")
xmax=outline.shape[0]-1
ymax=outline.shape[1]-1
cv2.imwrite(folder+"image2.png",outline)
wall,edge=SubFun.Get2WallFromImage(outline[:,:,0],True,folder,2)

wall0=wall[0]
wall0=np.array(wall0)
wall0=wall0[np.where((wall0[:,0]!=0) & (wall0[:,0]!=xmax)& (wall0[:,1]!=0) &(wall0[:,1]!=ymax))]
wall0=list(wall0)
if wall0[0][0]>wall0[50][0]:
    wall0.reverse()
wall0=np.array(wall0)
wallxmin=np.argmin(np.array(wall0)[:, 0])
wall0=wall0[np.r_[wallxmin:len(wall0),0:wallxmin]]

wall1=wall[1]
wall1=np.array(wall1)
wall1=wall1[np.where((wall1[:,0]!=0) & (wall1[:,0]!=xmax)& (wall1[:,1]!=0) &(wall1[:,1]!=ymax))]
wall1=list(wall1)
if wall1[0][0]>wall1[50][0]:
    wall1.reverse()
wall1=np.array(wall1)
wallxmin=np.argmin(np.array(wall1)[:, 0])
wall1=wall1[np.r_[wallxmin:len(wall1),0:wallxmin]]


midPoints=list()
mappedJ=dict()
#for i in range(len(wall1)):
for i in [0,len(wall1)-1]:
    dist=cv2.norm(outline.shape)
    short=0
    for j in range(short,len(wall0)):
        dist1=cv2.norm(np.array(wall1[i])-np.array(wall0[j]))
        if j not in mappedJ:
            if dist>dist1:
                dist=dist1
                short=j
    mappedJ[short]=i
    #cv2.line(outline,tuple(wall1[i]),tuple(wall0[short]),COLOR_CYAN)
longline=list(mappedJ.keys())
shortline=list(mappedJ.values())
ratio=(longline[0]-longline[1])/(shortline[0]-shortline[1])
outline_color=outline.copy()
diameter=list()
for i in range(len(wall1)):
    j=int(i*ratio)
    if i%20==0:
        cv2.line(outline_color, tuple(wall1[i]), tuple(wall0[j]), COLOR_CYAN)
    midpoint=((np.array(wall1[i])+np.array(wall0[j]))/2).astype('int')
    cv2.circle(outline_color,tuple(midpoint),1,COLOR_RED)
    diameter.append(cv2.norm(wall1[i]- wall0[j]))
cv2.imwrite(folder+"centerline_sampled.png",outline_color)
meandiameter=np.mean(diameter)

outline_edge=outline.copy()
ret,thresh1 = cv2.threshold(outline,127,255,cv2.THRESH_BINARY)
ret2, labels2,stats2,centroids2 = cv2.connectedComponentsWithStats(255-thresh1[:,:,0])
##Filter CCs if too small < 75 pixels or too big > 500 pixels
##OPTIONS: change 75 and 500 to other numbers
validlabel = np.where((stats2[:,4] >= 25)&(stats2[:,4]<500))
labels2_filt=np.zeros((labels2.shape[0],labels2.shape[1]))
for cc in range(len(validlabel[0])):
    labels2_filt[labels2==validlabel[0][cc]]=cc+1
##Get lines from the CCs
##OPTIONS:
#box ratio control: ratioUB=10, ratioLB=3,
#box area control:  areaUB=2000, areaLB=80,
#Angle control mean+/-std:        angleM=None,angleStd=20,
#Line length control:  pixelLB=75,pixelUB=150
filteredImage2, unfiltered_lines2, filteredLines2, unfilteredImage2, allline_stat2, filline_stat2 = SubFun.detectLineFromCCs(
    outline_edge, labels2_filt.astype('int'), font, fontScale,
    fontColor, lineType=1,ratioLB=2,areaLB=0,ratioUB=100,lineWidthUB=10,printText=False)
cv2.imwrite(folder+"Fil_lines.png",filteredImage2)
#cv2.imwrite(folder+"unFil_lines.png",unfilteredImage2)
filterLength_ave=np.mean(np.array(filteredLines2)[:,2])

print([meandiameter,filterLength_ave])

import matplotlib.pyplot as plt
fig=plt.hist(np.array(filteredLines2)[:,2],100)
plt.xlabel("Streamline length (pixel)")
plt.ylabel("Frequency")
plt.savefig(folder +"Line_length_distribution.png")
plt.close()

fig=plt.hist(diameter,100)
plt.xlabel("Diameter (pixel)")
plt.ylabel("Frequency")
plt.savefig(folder +"Diameter_length_distribution.png")
plt.close()


print("Average streamline length in pixels: "+str(np.array(filteredLines2)[:,2].mean()))
print("Average diameter length in pixels:"+str(diameter.mean()))
