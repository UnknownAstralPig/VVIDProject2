import cv2
import sys
import subprocess
import re as regex

from PyQt5.QtWidgets import QApplication , QWidget , QMainWindow , QPushButton , QLabel , QLineEdit , QToolButton , QMenu , QAction , QFrame
from PyQt5.QtCore import QSize , QPoint , QTimer
from PyQt5.QtGui import QPixmap , QImage
from PyQt5 import QtCore

class MainWindow( QMainWindow ):
    def __init__( self ):
        super( ).__init__( )

        self.h = 500
        self.w = 700

        self.setWindowTitle( "Camera App" )
        self.setFixedSize( QSize( self.w , self.h ) )

        self.fpsLabel = QLabel( 'FPS' , self )
        self.fpsLabel.setGeometry( 10 , 0 , 100 , 20 )

        fpsMenu = QMenu( )
        defAct = fpsMenu.addAction( '60' )
        fpsMenu.addAction( '30' )
        fpsMenu.addAction( '1' )

        self.fpsButtonMenu = QToolButton( self )
        self.fpsButtonMenu.setMenu( fpsMenu )
        self.fpsButtonMenu.setGeometry( 10 , 20 , 100 , 20 )
        self.fpsButtonMenu.setPopupMode( QToolButton.MenuButtonPopup )
        self.fpsButtonMenu.setDefaultAction( defAct )
        self.fpsButtonMenu.triggered.connect( self.fpsButtonMenu.setDefaultAction )

        self.typeLabel = QLabel( 'Type' , self )
        self.typeLabel.setGeometry( 120 , 0 , 100 , 20 )

        typeMenu = QMenu( )
        defAct = typeMenu.addAction( 'Video' )
        typeMenu.addAction( 'RTSP' )

        self.typeButtonMenu = QToolButton( self )
        self.typeButtonMenu.setMenu( typeMenu )
        self.typeButtonMenu.setGeometry( 120 , 20 , 100 , 20 )
        self.typeButtonMenu.setPopupMode( QToolButton.MenuButtonPopup )
        self.typeButtonMenu.setDefaultAction( defAct )
        self.typeButtonMenu.triggered.connect( self.TypeSelector )

        self.videoLabel = QLabel( 'Video' , self )
        self.videoLabel.setGeometry( 230 , 0 , 100 , 20 )

        self.videoButtonMenu = QToolButton( self )
        self.videoButtonMenu.setGeometry( 230 , 20 , 100 , 20 )
        self.videoButtonMenu.setPopupMode( QToolButton.MenuButtonPopup )

        self.rtspLabel = QLabel( 'RTSP IP' , self )
        self.rtspLabel.setGeometry( 230 , 0 , 100 , 20 )

        self.rtspTextIPInput = QLineEdit( self )
        self.rtspTextIPInput.setGeometry( 230 , 20 , 100 , 20 )
        self.rtspTextIPInput.setMaxLength( 15 )
        self.rtspTextIPInput.setInputMask( '000.000.000.000;_' )

        self.startButton = QPushButton( 'Start' , self )
        self.startButton.setGeometry( 340 , 20 , 100 , 20 )
        self.startButton.clicked.connect( self.StartButtonPressed )
        
        self.videoImage = QLabel( )
        self.videoImage.setGeometry( 10 , 50 , self.w - 20 , self.h - 80 )
        self.videoImage.setFrameStyle( QFrame.Panel )

        self.rtspTextIPInput.hide( )
        self.rtspLabel.hide( )
        self.videoImage.setParent( self )
        self.show( )

        self.camera = ''
        self.timer = QTimer( )
        self.rate = 0

    def ConnectToRTSP( self , b ):
        self.label.setText( "New state" )

    def TypeSelector( self , action ):
        self.typeButtonMenu.setDefaultAction( action )
        if( action.text( ) == 'Video' ):
            self.InitVideoInputs( )
            self.videoButtonMenu.show( )
            self.videoLabel.show( )
            self.rtspTextIPInput.hide( )
            self.rtspLabel.hide( )
        else:
            self.videoButtonMenu.hide( )
            self.videoLabel.hide( )
            self.rtspTextIPInput.show( )
            self.rtspLabel.show( )

    def InitVideoInputs( self ):
        videos = str( subprocess.run( [ '/usr/bin/find' , '/dev/' , '-name' , 'video*' ] , stdout = subprocess.PIPE , stderr = subprocess.DEVNULL ).stdout )[2:-3]
        print( videos )
        if( len( videos ) == 0 ):
            self.videoButtonMenu.setMenu( None )
            self.videoButtonMenu.setDefaultAction( QAction( '' ) )
            return
        else:
            videos = videos.split( r'\n' )
            menu = QMenu( )

            for video in videos:
                menu.addAction( video )

            self.videoButtonMenu.setMenu( menu )
            self.videoButtonMenu.setDefaultAction( menu.actions( )[ 0 ] )
            self.videoButtonMenu.triggered.connect( self.videoButtonMenu.setDefaultAction )

    def StartButtonPressed( self ):
        self.rate = int( 1000 / int(self.fpsButtonMenu.defaultAction( ).text( ) ) )
        if( self.timer.isActive( ) ):
            self.timer.stop( )
            self.startButton.setText( 'Start' )
            self.videoImage.clear( )
            self.camera.release( )
            return
        if( self.typeButtonMenu.defaultAction( ).text( ) == 'Video' and len( self.videoButtonMenu.defaultAction( ).text( ) ) != 0 ):
            self.camera = cv2.VideoCapture( self.videoButtonMenu.defaultAction( ).text( ) )

            if( not self.camera.isOpened( ) ):
                return

            self.timer.timeout.connect( self.printNextFrame )
            self.timer.start( self.rate )
        elif( self.typeButtonMenu.defaultAction( ).text( ) == 'RTSP' ):
            ip = 'rtsp://admin:admin@' + self.rtspTextIPInput.text( ) + '/1'
            self.camera = cv2.VideoCapture( ip )

            if( not self.camera.isOpened( ) ):
                self.statusBar( ).showMessage( 'RTSP cannot be opened' )
                return

            self.timer.timeout.connect( self.printNextFrame )
            self.timer.start( self.rate )
        self.startButton.setText( 'Stop' )

    def printNextFrame( self ):
        rval, frame = self.camera.read()
        frame = cv2.cvtColor( frame , cv2.COLOR_BGR2RGB )
        image = QImage( frame , frame.shape[1] , frame.shape[0] , QImage.Format_RGB888 )
        pixmap = QPixmap.fromImage( image )
        pixmap = pixmap.scaled( self.videoImage.size( ) , QtCore.Qt.KeepAspectRatio )
        self.videoImage.setPixmap( pixmap )

            

if __name__ == '__main__':
    app = QApplication( sys.argv )
    execution = MainWindow()
    app.exec()
    
