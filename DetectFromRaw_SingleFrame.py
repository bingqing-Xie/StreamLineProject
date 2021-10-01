import cv2
import numpy as np
import tifffile
from config import *
from SubFun import SubFun
import csv
from cv2_rolling_ball import subtract_background_rolling_ball
####Setting default properties for text to be printed on the image
font = cv2.FONT_HERSHEY_DUPLEX
bottomLeftCornerOfText = (10, 500)
fontScale = 0.8
fontColor =COLOR_GREEN
lineType = 3



########Image 2.5ml/min
folder = "Processed87_2/"
contourfile = "P104_12mo_6.4X_2.5mLmin_InnerWall.tif"
fname='P104_12mo_6.4X_2.5mLmin_InnerWall.tif'
#######Image 5ml/min
#folder = "Processed_step78/"
#contourfile = "P104_12mo_6.4X_5mLmin_InnerWall.tif"
#fname='P104_12mo_6.4X_5mLmin_InnerWall.tif'


###########Get wall pixels
outline = cv2.imread(folder + contourfile)
wall,edge=SubFun.GetWallFromImage(outline,False,folder)
edgeColor_lines=cv2.cvtColor(edge,cv2.COLOR_GRAY2BGR)
a = tifffile.imread(folder + fname)
wall2wss = dict()
time = 0.5
viscosity = 3.5
###Loop from all frames
filImage=list()
projImage=list()
for framei in range(len(a)):
#for framei in range(1):
    print(framei)
    gray=a[framei]
    ret2, labels2,stats2,centroids2 = cv2.connectedComponentsWithStats(gray)
    ##Filter CCs if too small < 75 pixels or too big > 500 pixels
    ##OPTIONS: change 75 and 500 to other numbers
    validlabel = np.where((stats2[:,4] >= 75)&(stats2[:,4]<500))
    labels2_filt=np.zeros((labels2.shape[0],labels2.shape[1]))
    for cc in range(len(validlabel[0])):
        labels2_filt[labels2==validlabel[0][cc]]=cc+1
    ##Get lines from the CCs
    ##OPTIONS:
    #box ratio control: ratioUB=10, ratioLB=3,
    #box area control:  areaUB=2000, areaLB=80,
    #Angle control mean+/-std:        angleM=None,angleStd=20,
    #Line length control:  pixelLB=75,pixelUB=150
    #allline_stat2: [startpoint[0], startpoint[1], endpoint[0], endpoint[1], length_rect, angle,bwidth,bheight,  labelindex]
    filteredImage2, unfiltered_lines2, filteredLines2, unfilteredImage2, allline_stat2, filline_stat2 = SubFun.detectLineFromCCs(
        edgeColor_lines, labels2_filt.astype('int'), font, fontScale,
        fontColor, lineType, ratioLB=3, ratioUB=100, areaLB=0, pixelLB=15, lineWidthUB=10)
    cv2.imwrite(folder+"FilteredLines/Fil_lines"+str(framei)+".png",filteredImage2)
    filImage.append(filteredImage2.copy())
    ##Filtered lines renamed as linesFrame
    linesFrame=filteredLines2
    maxD=cv2.norm(edge.shape)
    edgeColor=cv2.cvtColor(edge,cv2.COLOR_GRAY2BGR)
    ##Loop from each line, project it to the wall contour
    ##For each pixel in the wall save the list of lines projected from all frames
    for i in range(len(linesFrame)):
        [startpoint,endpoint,length,angle]=linesFrame[i]
        boxmin,boxmax,min_d=SubFun.FindContourProjectionBox([startpoint, endpoint],[edge.shape[0]-1,edge.shape[1]-1],wall)
        if min_d==0 & boxmin[0]==0 & boxmax[0]==0:
            continue
        #cv2.rectangle(edgeColor,boxmin,boxmax,COLOR_GREEN)
        midpoint=(int((boxmin[0]+boxmax[0])/2),int((boxmin[1]+boxmax[1])/2))
        cv2.line(edgeColor,(int(startpoint[0]),int(startpoint[1])),midpoint,COLOR_CYAN)
        cv2.line(edgeColor,(int(endpoint[0]),int(endpoint[1])),midpoint,COLOR_CYAN)
        cv2.line(edgeColor,(int(startpoint[0]),int(startpoint[1])),(int(endpoint[0]),int(endpoint[1])),COLOR_YELLOW)
        for w in wall:
            if w[0] <= boxmax[0] and w[0] >=boxmin[0] and w[1] <= boxmax[1] and w[1] >=boxmin[1]:
                cv2.circle(edgeColor, w, 2,COLOR_OLIVE)
                if tuple(w) not in wall2wss:
                    wall2wss[tuple(w)]=list()
                wss=(length / time) / min_d * viscosity #mPa
                wall2wss[tuple(w)].append([ framei,i, startpoint[0],startpoint[1], endpoint[0],endpoint[1], length, angle,min_d, wss])
    #cv2.imwrite(folder +str(framei)+ "ProjectLines/edgeProjectedLines.png", edgeColor)
    cv2.imwrite(folder +"ProjectLines/"+str(framei)+ "edgeProjectedLines.png", edgeColor)
    projImage.append(edgeColor.copy())
##Calculated the WSS per point on the wall
##Averaging from all projected lines
##Within the frame, or from all frames
minwss_per_frame=100
maxwss_per_frame=-100
minwss_collapsed_global=100
maxwss_collapsed_global=-100
wssByEdgePoint_split=dict()
wssByEdgePointAll=dict()
for w in wall2wss.keys():
    listwss=np.array(wall2wss[w])
    wssMean=np.mean(listwss[:,9])
    wssByEdgePointAll[w]=wssMean
    if minwss_collapsed_global > wssMean:
        minwss_collapsed_global = wssMean
    if maxwss_collapsed_global < wssMean:
        maxwss_collapsed_global = wssMean
    for framei in range(len(a)):
        #temp mean wss per framei
        wssMean=np.mean(listwss[listwss[:,0]==framei,9])
        if minwss_per_frame > wssMean:
            minwss_per_frame = np.min(listwss[:, 9])
        if maxwss_per_frame < wssMean:
            maxwss_per_frame = np.max(listwss[:, 9])
        if not np.isnan(wssMean):
            #print([framei,wssMean])
            if framei not in wssByEdgePoint_split:
                wssByEdgePoint_split[framei]=dict()
            wssByEdgePoint_split[framei][w]=wssMean
##Plot the wss value in a colormap for all frames
edgeColorWss=cv2.cvtColor(edge,cv2.COLOR_GRAY2BGR)
imgray=np.array(range(256)).astype('uint8')
##OPTIONS: Change color map
imcolor=np.flip(cv2.applyColorMap(imgray,cv2.COLORMAP_AUTUMN),axis=0 )##yellow to red scale
from math import log
for e in wall:
    if tuple(e) in wssByEdgePointAll:
        current_wss=wssByEdgePointAll[tuple(e)]
        wall_color = np.clip((log(current_wss) - log(minwss_collapsed_global)) * 255 / (log(maxwss_collapsed_global) - log(minwss_collapsed_global)), 0, 255).astype('uint8')
        color= ( int (imcolor[wall_color][0] [ 0 ]), int (imcolor[wall_color][0] [ 1 ]), int (imcolor[wall_color][0] [ 2 ]))
        cv2.circle(edgeColorWss,tuple(e),2,color)
cv2.imwrite(folder+"colorEdges_results_thick_all.png",edgeColorWss)
##Plot the wss value in a colormap for each frame
coverage_all=list()
for i in range(len(a)):
    #edgeColorWss = cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)
    edgeColorWss = cv2.cvtColor(a[i], cv2.COLOR_GRAY2BGR)
    edgeColorWss = np.clip(edgeColorWss*0.5 ,0,255)
    imgray = np.array(range(256)).astype('uint8')
    imcolor = np.flip(cv2.applyColorMap(imgray, cv2.COLORMAP_AUTUMN), axis=0)  ##blue to purple scale
    num_wall=len(wall)
    coverage=0
    for e in wall:
        if tuple(e) in wssByEdgePoint_split[i]:
            coverage=coverage+1
            current_wss = wssByEdgePoint_split[i][tuple(e)]
            wall_color = np.clip((log(current_wss) - log(minwss_per_frame)) * 255 / (log(maxwss_per_frame) - log(minwss_per_frame)), 0,
                                 255).astype('uint8')
            color = (int(imcolor[wall_color][0][0]), int(imcolor[wall_color][0][1]), int(imcolor[wall_color][0][2]))
            cv2.circle(edgeColorWss, tuple(e), 2, color)
            cv2.circle(filImage[i], tuple(e), 2, color)
            cv2.circle(projImage[i], tuple(e), 2, color)
    coverage_all.append(coverage/num_wall)
    cv2.imwrite(folder + "individual/colorEdges_origImage_thick_"+str(i)+".png", edgeColorWss)
    cv2.imwrite(folder + "individual/colorEdges_FilLinesImage_thick_"+str(i)+".png", filImage[i])
    cv2.imwrite(folder + "individual/colorEdges_ProjLinesImage_thick_"+str(i)+".png", projImage[i])

##Print the min and max for wss for all frames and individual frame
print([minwss_collapsed_global, maxwss_collapsed_global, minwss_per_frame, maxwss_per_frame])
##Plot the distribution of wss for all frames
import matplotlib.pyplot as plt
fig=plt.hist(np.array(list(wssByEdgePointAll.items()))[:,1].astype('float'),100)
plt.xlabel("WSS value")
plt.ylabel("Frequency")
plt.savefig(folder +fname.split(".tif")[0]+"WSS_distribution.png")
plt.close()
fig=plt.hist(np.array(list(wssByEdgePointAll.items()))[:,1].astype('float'),100,density=True)
plt.xlabel("WSS value(mPa)")
plt.ylabel("Probability")
plt.savefig(folder +fname.split(".tif")[0]+"WSS_Probability_distribution.png")
plt.close()


##Plot the coverage of wss for all frames
fig = plt.figure()
plt.bar(range(1,len(a)+1),coverage_all)
plt.savefig(folder +fname.split(".tif")[0]+"_coverage_Bar.png")
plt.close()
###Print the overall coverage
print("Full coverage is "+str(len(wssByEdgePointAll)/len(wall)))

##Save the wss to pkl file
import pickle
with open(folder+fname+"wss.pkl", 'wb') as f:
  pickle.dump([wssByEdgePoint_split,wssByEdgePointAll], f)
##Load wss from saved pkl file
#with open(folder + fname + "wss.pkl", 'rb') as f:
#    [wssByEdgePoint_split, wssByEdgePointAll]= pickle.load( f)
