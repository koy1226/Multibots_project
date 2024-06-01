import datetime
import os
import sys
import threading
import time
import numpy as np
# import serial.tools.list_ports
import qdarkstyle             # pip install qdarkstyle, if you want to use modern style.. please find moder...
import subprocess          

from PyQt5.QtWidgets import *
#from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox, QInputDialog
from PyQt5.QtCore import *
#from PyQt5.QtCore import pyqtSignal, Qt, QObject
from PyQt5.QtGui import *
from PyQt5 import uic
from UI_Multibot import Ui_MainWindow
from typing import Final
# from SerialManager import SerialManager
from datetime import datetime, timedelta

# import pandas as pd


# Signal to run 미아찾기 
class RunfindkidSig(QObject):
    runfindkidSignal = pyqtSignal()
    def run(self):
        self.runfindkidSignal.emit()
        
class RefreshButtonGui(QObject):
    refreshButton = pyqtSignal()
    def run(self):
        self.refreshButton.emit()

####################################################################
#################### Main Window Class #############################
####################################################################
class MyWindow(QMainWindow) :
    def __init__(self, parent = None):
        ########## Initialize Functions ##########
        super().__init__(parent)        
        
        self._gui = Ui_MainWindow()
        self._gui.setupUi(self)
        self.checked_texts =[]
        # GUI Init
        self.setWindowTitle("Multibot")

        self._findkidBusy = False
        self._findkidThreadStop = False
        self.runfind = RunfindkidSig()
        self.runfind.runfindkidSignal.connect(self.runfindkid)   
        self._findkidThread = threading.Thread(target=self.runfindkidTrigger)            
        self._findkidThread.start()
        self.step = 0
        self.substep = 0
        self.missing_kid = ''
        self.blank=' '

        self.initUI()

    def initUI(self):
        for i in range(1, 31):
            getattr(self._gui, f"checkBox_{i}").stateChanged.connect(self.updateText)

    def updateText(self):
        self.checked_texts = [getattr(self._gui, f"checkBox_{i}").text() for i in range(1, 31) if getattr(self._gui, f"checkBox_{i}").isChecked()]
        # QPlainTextEdit에 텍스트 설정
        self._gui.basket.setPlainText("\n".join(self.checked_texts))
        
    def savebasket(self):
        with open("shoppinglist.txt", 'w') as f:
            for text in self.checked_texts:
                f.write(str(text) + "\n")
        self.runOtherFile()  #db로 장바구니 보내는 함수 호출

    def clearbasket(self):
        for i in range(1, 31):
            getattr(self._gui, f"checkBox_{i}").setChecked(False)        
        self._gui.basket.clear()
        with open("shoppinglist.txt", 'w') as f:
            f.write("")



    def runOtherFile(self):
        # 다른 파이썬 파일 실행
        subprocess.run(["python", "send2cart.py"])   

    def runfindkidTrigger(self):
        while True:           
            if (False == self._findkidBusy):
                self.runfind.run()
            
            time.sleep(0.5)     # 500ms period
            
            if (self._findkidThreadStop == True):
                return
            
    def runfindkid(self):
        if self.step == 0:
            if self.substep == 0:
                with open("missing_kid.txt", "r") as f:
                    self.missing_kid = f.read()
                    if self.missing_kid == 'detect':
                        self.substep += 1
                    else:
                        self.substep = 0
            elif self.substep == 1:
                if self.missing_kid == 'detect':
                    self.substep = 0
                    self.step = 250
                    reply = QMessageBox.warning(self, "!! FIND !!", "!! I FOUND THE MISSING KID !!")
                    if reply:
                        self.Restore_search_missing_kid()
        else:
            pass

    def Restore_search_missing_kid(self):
        with open("missing_kid.txt", "r") as f:
            self.missing_kid = f.read()
            if self.missing_kid == 'detect':
                self.step = 0
                self.substep = 0
        with open("missing_kid.txt", 'w') as f:
            f.write("")

    def closeEvent(self, event):                                        # Override for 'X' button operation, 여기에서 serial port를 닫아줘야 함. 안그러면 꺼지지 않음
        self._isConnected = False  
        try:
            self._findkidThreadStop = True
            self._findkidThread.close()
            self._findkidThread.join()
        except:
            print("find kid thread close exception")
        else:
            print("find kid thread closed")              

        event.accept()

    
####################################################################
### Main Function should be located in bottom of the code ##########
####################################################################
if __name__ == "__main__":
    os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
    app = QApplication(sys.argv)    
    mw = MyWindow()    
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    mw.show()
    app.exec_()
