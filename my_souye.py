import sys
import cv2
from PyQt5.Qt import QMessageBox   #QMessage用
from PyQt5 import QtCore,QtGui,QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import QFileDialog,QMainWindow
from repair import Ui_MainWindow
#from my_classify import My_classify  #引入另一个画面的类，可以不用
class My_souye(QMainWindow,Ui_MainWindow):
    def __init__(self):
        super(My_souye,self).__init__()
        self.setupUi(self)

        self.camera=cv2.VideoCapture(0)
        self.is_camera_opened=False# 摄像头有没有打开
        #定时器，30ms捕获一帧
        self._timer=QtCore.QTimer(self)
        self._timer.timeout.connect(self._queryFrame)
        self._timer.setInterval(30)
    #打开摄像头
    def open_camera_click(self):
        '''
        打开和关闭摄像头
        '''
        self.is_camera_opened = ~self.is_camera_opened
        if self.is_camera_opened:
            self.pushButton_5.setText("关闭摄像头")
            self._timer.start()
        else:
            self.pushButton_5.setText("打开摄像头")
            self._timer.stop()

    #从视频中截取图片的function
    def camera_click(self):
        if not self.is_camera_opened:
            return
        #捕获视频
        temp=self.src
        img_rows, img_cols, channels = temp.shape
        bytePerLine = channels * img_cols
        # Qt显示图片时，需要先转换成QImgage类型
        QImg=QImage(temp.data,img_cols,img_rows,bytePerLine,QImage.Format_RGB888)
        self.label_3.setPixmap(QPixmap.fromImage(QImg).scaled(
            self.label_3.size(),Qt.KeepAspectRatio,Qt.SmoothTransformation
        ))


    #打开文件,并显示原图
    def openfile_click(self):
        filename,_=QFileDialog.getOpenFileName(self,'打开图片','D:/aphoto',"JPEG Files(*.jpg);;PNG Files(*.png)")
        if filename:
            self.src=cv2.imread(str(filename))   #原图，以后很多要用,所有要用self
            #Opencv图像以BGR存储，显示的时候要从BGR转到RGB
            temp=cv2.cvtColor(self.src,cv2.COLOR_BGR2RGB)
            rows,cols,channels=temp.shape
            bytePerLine=channels*cols
            QImg=QImage(temp.data,cols,rows,bytePerLine,QImage.Format_RGB888)
            self.label_2.setPixmap(QPixmap.fromImage(QImg).scaled(
                self.label_2.size(),Qt.KeepAspectRatio,Qt.SmoothTransformation
            ))
        else:
            QMessageBox.information(self,"打开失败","没有选择图片")

    #从打开的图变为灰度图
    def gray(self):
        #hasattr判断这个对象是否有这个变量或属性
        if not hasattr(self,"src"):
            QMessageBox.information(self, "灰度失败", "请先打开图片")
            return
        self.temp=cv2.cvtColor(self.src,cv2.COLOR_BGR2GRAY)
        rows,columns=self.temp.shape
        bytePerLine=columns
        # 灰度图是单通道，所以需要用Format_Indexed8
        QImg=QImage(self.temp.data,columns,rows,bytePerLine,QImage.Format_Indexed8)
        self.label_3.setPixmap(QPixmap.fromImage(QImg).scaled(
            self.label_3.size(),Qt.KeepAspectRatio,Qt.SmoothTransformation
        ))

        #保存灰度图
    def save_huidu_click(self):
        savefilename,_=QFileDialog.getSaveFileName(self,"保存图片",'Image','*.png *.jpg *.bmp')
        if savefilename and (hasattr(self, "temp")):
            cv2.imwrite(savefilename, self.temp)
            QMessageBox.information(self, "保存灰度图", "保存成功")
        else:
            QMessageBox.information(self, "保存灰度图失败", "请选择有效路径或生成灰度图先")
            return


        #阈值化
    def threshed_click(self):
        if not hasattr(self,"src"):
            QMessageBox.information(self, "阈值化失败", "请先打开图片")
            return
        temp = cv2.cvtColor(self.src, cv2.COLOR_BGR2GRAY)
        _,thresh_img=cv2.threshold(temp,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        rows,columns=thresh_img.shape
        bytesPerLine=columns
        # 阈值分割图也是单通道，也需要用Format_Indexed8
        QImg=QImage(thresh_img.data,columns,rows,bytesPerLine,QImage.Format_Indexed8)
        self.label_3.setPixmap(QPixmap.fromImage(QImg).scaled(
            self.label_3.size(),Qt.KeepAspectRatio,Qt.SmoothTransformation
        ))

    #@QtCore.pyqtSlot()
    def _queryFrame(self):
        '''
        循环捕获图片
        '''
        import cv2 as cv
        import numpy as np
        #import Camera

        #cap = cv.VideoCapture(0)
        whT = 320 #
        confThreshold = 0.5
        nmsThreshold = 0.2

        #### LOAD MODEL
        ## Coco Names
        classesFile = "Fire.names"
        classNames = []#class of list
        with open(classesFile, 'rt') as f:
            classNames = f.read().rstrip('\n').split('\n')#split class of  name
        print(classNames)
        ## Model Files
        modelConfiguration = "yolov3-tiny-obj.cfg"#after training and  load configurations file
        modelWeights = "yolov3-tiny-obj_final.weights"#model of weight
        net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)#read Darknet of model Configuration and  modelWeights
        net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

        def findObjects(outputs, img):
            hT, wT, cT = img.shape # input picture of size
            bbox = [] #store
            classIds = [] # class of index
            confs = [] #confident value
            for output in outputs:
                for det in output:
                    scores = det[5:] #Delete the first five of the list
                    classId = np.argmax(scores)
                    confidence = scores[classId]
                    if confidence > confThreshold:
                        w, h = int(det[2] * wT), int(det[3] * hT)
                        x, y = int((det[0] * wT) - w / 2), int((det[1] * hT) - h / 2)#Object detection location
                        bbox.append([x, y, w, h])#add to bundbox
                        classIds.append(classId)
                        confs.append(float(confidence))

            #print(len(bbox))
            indices = cv.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)
            #Draw a frame in the picture
            for i in indices:
                i = i[0]
                box = bbox[i]
                x, y, w, h = box[0], box[1], box[2], box[3]
                # print(x,y,w,h)
                cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)
                cv.putText(img, f'{classNames[classIds[i]].upper()} {int(confs[i] * 100)}%',
                           (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

        while True:
            #success, img = cap.read()
            ret, self.src = self.camera.read()

            blob = cv.dnn.blobFromImage(self.src, 1 / 255, (whT, whT), [0, 0, 0], 1, crop=False)#Processing image size
            net.setInput(blob)#Put the read photos into the model
            layersNames = net.getLayerNames() #get model of layer name
            outputNames = [(layersNames[i[0] - 1]) for i in net.getUnconnectedOutLayers()] #get  detect object of index
            outputs = net.forward(outputNames)
            #print(outputNames[0][0])
            findObjects(outputs, self.src)
            #print(findObjects(outputs, self.src))

            #cv.imshow('Image', self.src)
            cv.waitKey(1)


        # ret,self.src=self.camera.read()
            img_rows,img_cols,channels=self.src.shape
            bytePerLine=channels*img_cols
            self.src=cv2.cvtColor(self.src,cv2.COLOR_BGR2RGB)
            QImg=QImage(self.src.data,img_cols,img_rows,bytePerLine,QImage.Format_RGB888)
            self.label_2.setPixmap(QPixmap.fromImage(QImg).scaled(
             self.label_2.size(),Qt.KeepAspectRatio,Qt.SmoothTransformation
         ))




if __name__=="__main__":
    app=QtWidgets.QApplication(sys.argv)
    ex=My_souye()
    ex.show()
    #另一个页面对象
    # child=My_classify()
    # btn=ex.pushButton_6
    # btn.clicked.connect(child.show)
    sys.exit(app.exec_())