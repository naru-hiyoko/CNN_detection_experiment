# encoding: utf-8
import os
import sys
from xml.etree import ElementTree
from xml.etree.ElementTree import Element, SubElement
from xml.dom import minidom
import cv2
import wx
import uuid
from skimage.transform import rotate, rescale
from skimage.io import imsave
import numpy as np

DEBUG = True
""" バウンディングボックス　が順番に格納"""
result = []
"""最大アイテム数 """
max_item = 5
"""アノテーションラベルセット"""
obj_names = []


def disp():
    global obj_names
    tmp_image =  source_image.copy()

    i = -1
    for (x1, y1, x2, y2) in result:
        i = i + 1
        if i == max_item:
            break
        tmp_image[y1-20:y1, x1:x1+150, :] = 255
        cv2.rectangle(tmp_image, (x1,y1),(x2, y2),color=(255, 0, 0),thickness=1)
        cv2.putText(tmp_image,
                    str(i+1)+':'+obj_names[i].GetValue(),
                    (x1, y1-5),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, color = (255, 0, 0))
    cv2.imshow('viewer', tmp_image)
    
def on_openImage_button_clicked(event):
    global source_image
    global filename
    global image
    global result
    global id
    
    try:
        dir_class = wx.FileDialog(None,'select image file', filename,"","",wx.OPEN)
    except:
        dir_class = wx.FileDialog(None,'select image file', os.getcwd(),"","",wx.OPEN)


    if dir_class.ShowModal() == wx.ID_OK:
        print dir_class.GetPath()
        filename = dir_class.GetPath()
        if '.jpg' in filename or '.JPG' in filename or '.png' in filename:
            result = []
            image_name_textBox.SetValue(filename)
            image = cv2.imread(filename)
            image = rotate(image, -90) * 255.0
            image = image.astype(np.uint8)
            if image.shape[0]*image.shape[1] > 1200*1200:
                image = cv2.resize(image, (int(image.shape[1]*0.7), int(image.shape[0]*0.7)))
            cv2.imshow('viewer', image)
            source_image = image.copy()
        
        
    dir_class.Destroy()
    id = str(uuid.uuid4())
    
def on_mouse_clicked(event,x,y,flags,params):
    global cx,cy
    global clicked
    global source_image

    if event == cv2.EVENT_LBUTTONDOWN:
        clicked = True
        cx = x
        cy = y

        try:
            disp()
        except:
            pass
        
        return None

    if event == cv2.EVENT_LBUTTONUP:
        clicked = False
        wx = x - cx
        wy =  y - cy

        if wx < 5 or wy < 5:
            return
        
        try:
            result.append((cx, cy, x, y))
            disp()
        except:
            pass

        return

    try:
        if event == cv2.EVENT_MOUSEMOVE and clicked:
            tmp_image =  source_image.copy()
            cv2.rectangle(tmp_image,(cx,cy),(x,y),color=(0,255,0),thickness=1)
            cv2.imshow('viewer',tmp_image)
    except:
        clicked = False


def on_write_button(event):
    disp()
    """xml"""
    root = Element('annotation')
    """保存先のディレクトリ"""
    prefix = '../data/'
    #dir_prefix = 'price'
    try:
        os.makedirs(os.path.join(prefix, 'Annotations'))
    except:
        pass
    #SubElement(root, 'folder').text = dir_prefix
    #SubElement(root, 'filename').text = '_'.join([dir_prefix, id])

    _sizeNode = SubElement(root, 'size')
    SubElement(_sizeNode, 'width').text = str(image.shape[1])
    SubElement(_sizeNode, 'height').text = str(image.shape[0])
    SubElement(_sizeNode, 'depth').text = str(image.shape[2])

    for i in range(len(result)):
        if obj_names[i].GetValue() != '':
            obj_node = SubElement(root, 'object')
            SubElement(obj_node, 'name').text = obj_names[i].GetValue()
            SubElement(obj_node, 'pose').text = 'Unspecified'
            SubElement(obj_node, 'truncated').text = '0'
            SubElement(obj_node, 'difficult').text = '0'
            bndbox_node = SubElement(obj_node, 'bndbox')
            SubElement(bndbox_node, 'xmin').text = str(result[i][0])
            SubElement(bndbox_node, 'ymin').text = str(result[i][1])
            SubElement(bndbox_node, 'xmax').text = str(result[i][2])
            SubElement(bndbox_node, 'ymax').text = str(result[i][3])

    data =  minidom.parseString(ElementTree.tostring(root, 'utf-8')).toprettyxml(indent = '   ')
    print data
    with open(os.path.join(prefix, 'Annotations',  str(id)+'.xml'), 'w') as f: 
        f.write(data)
        cv2.imwrite(os.path.join(prefix, 'Images',  str(id) + '.JPEG'), source_image)

def on_undo(e):
    try:
        result.pop()
    except:
        pass
    disp()

def on_exit(e):
    sys.exit(1)

def on_refresh(e):
    disp()


if __name__ == '__main__':
    
    """ wx インターフェース"""
    app = wx.App()
    frame = wx.Frame(None, -1, 'controller')
    global panel
    panel = wx.Panel(frame, -1)
    panel.SetBackgroundColour('white')
    write_button = wx.Button(panel, -1, 'write', size = (100, -1))
    write_button.Bind(wx.EVT_BUTTON, on_write_button)
    undo_button = wx.Button(panel, -1, 'undo', size = (100, -1))
    undo_button.Bind(wx.EVT_BUTTON, on_undo)
    
    exit_button = wx.Button(panel, -1, 'exit', size = (100, -1))
    exit_button.Bind(wx.EVT_BUTTON, on_exit)
    
    refresh_button = wx.Button(panel, -1, 'refresh', size = (100, -1))
    refresh_button.Bind(wx.EVT_BUTTON, on_refresh)
    
    open_Image_button = wx.Button(panel,wx.ID_ANY,'open Image',size = (100,-1))
    open_Image_button.Bind(wx.EVT_BUTTON, on_openImage_button_clicked)
    image_name_textBox = wx.TextCtrl(panel,wx.ID_ANY,'select image',size = (490,-1))
    
    layout = wx.BoxSizer(wx.VERTICAL)
    layoutA = wx.BoxSizer(wx.HORIZONTAL)
    layoutA.Add(write_button, 1, wx.ALL, 1)
    layoutA.Add(undo_button, 1, wx.ALL, 1)
    layoutA.Add(exit_button, 1, wx.ALL, 1)
    layoutA.Add(refresh_button)
    

    layoutB = wx.BoxSizer(wx.HORIZONTAL)
    layoutB.Add(open_Image_button)
    layoutB.Add(image_name_textBox)

    layoutC = wx.BoxSizer(wx.VERTICAL)
    layoutC.Add(wx.StaticText(panel, -1, 'アノテーションラベルセット↓'))
    for i in range(max_item):
        layoutD = wx.BoxSizer(wx.HORIZONTAL)
        obj_names.append(wx.TextCtrl(panel, wx.ID_ANY, 'tag',size = (100,-1)))
        layoutD.Add(wx.StaticText(panel, -1, str(i+1)+' : '))
        layoutD.Add(obj_names[i])
        layoutC.Add(layoutD)
    
    layout.Add(layoutB)
    layout.Add(layoutA)
    layout.Add(layoutC)
    
    
    panel.SetSizer(layout)

    frame.Show(True)
    
    
    cv2.namedWindow('viewer')
    cv2.setMouseCallback('viewer', on_mouse_clicked)
    while(True):
        ret = cv2.waitKey(10)
        if ret == ord('q'):
            break
    
