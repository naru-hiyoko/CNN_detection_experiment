# encoding:utf-8

import numpy as np
import cv2
import sys

class BB:
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        
    def get_pt1(self):
        return (self.x, self.y)
    def get_pt2(self):
        return (self.x+self.width, self.y + self.height)
    def getWidth(self):
        return self.width
    def getHeight(self):
        return self.height
    def getCenter(self):
        return (self.x+self.width/2, self.y+self.height/2)
    def getRoi(self, image):
        max_width = image.shape[1]
        max_height = image.shape[0]
        if max_height <= self.y + self.height or max_width <= self.x + self.width or self.x < 0 or self.y < 0 or self.width < 20 or self.height < 20:
            return None
        else:
            return image[self.y:self.y+self.height,
                         self.x:self.x+self.width,:]

class Detect:
    def __init__(self, image):
        self.image = image
    def detectRegion(self):
        if '3.0' in cv2.__version__:
            msers = cv2.MSER_create().detectRegions(self.image, None)
        elif '2.4' in cv2.__version__:
            msers = cv2.MSER().detect(self.image, None)
        """ MSER + α　の候補を格納"""
        BBs = [] 
        for bb in msers:
            pt1 = tuple(np.min(bb, 0))
            pt2 = tuple(np.max(bb, 0))
            bb_width = np.abs(pt2[0] - pt1[0])
            bb_height = np.abs(pt2[1] - pt1[1])
            #BBs.append(BB(pt1[0], pt1[1], bb_width, bb_height))
            for pad in (10, 20, 30, 35):
                BBs.append(BB(pt1[0]-pad, pt1[1]-pad, bb_width+pad*2, bb_height+pad*2))
                BBs.append(BB(pt1[0]-pad, pt2[1]-pad, bb_width+pad*2, bb_width+pad*2))
                BBs.append(BB(pt1[0]-pad, pt1[1]-pad - bb_width, bb_width+pad*2, bb_width+pad*2))
                BBs.append(BB(pt1[0]-pad, pt1[1]-pad, bb_height+pad*2, bb_height+pad*2))
                BBs.append(BB(pt1[0] - bb_height- pad, pt1[1]-pad, bb_height+pad*2, bb_height+pad*2))                

        return BBs



if __name__ == '__main__':
    image = cv2.imread(sys.argv[1])

    detect = Detect(image)
    BBs = detect.detectRegion()
    for bb in BBs:
        crop = bb.getRoi(image)
        if not crop is None:
            cv2.imshow('sample', crop)
            q = cv2.waitKey(100)
            if q == ord('q'):
                sys.exit(0)
        #cv2.rectangle(image, bb.get_pt1(), bb.get_pt2(), (255, 0, 0))
    #cv2.imshow('sample', image)
    #cv2.waitKey(4000)        
