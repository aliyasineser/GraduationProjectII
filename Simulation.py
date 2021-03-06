import numpy as np
import sys
import argparse
from PyQt5.QtCore import pyqtSignal, QThread, Qt, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QWidget, QDialog, QLabel, QVBoxLayout, QHBoxLayout, QMainWindow, QPushButton
from aruco_tracker import relativePosition, draw, calibrate, saveCoefficients, loadCoefficients
import cv2
import cv2.aruco as aruco
import pickle

cap = cv2.VideoCapture(0)

calibrationMarkerID = None
needleMarkerID = None
ultraSoundMarkerID = None

mtx, dist = None, None  # Camera parameters
pressedKey = None
save_Name = "savedCalibration.pkl"  # File to store calibration info

needleComposeRvec, needleComposeTvec = None, None  # Composed for needle
ultraSoundComposeRvec, ultraSoundComposeTvec = None, None  # Composed for ultrasound

class Thread(QThread):
    # Slots for the gui
    changePixmap = pyqtSignal(QImage)
    changeXDiff = pyqtSignal(str)
    changeYDiff = pyqtSignal(str)
    changeZDiff = pyqtSignal(str)
    calibrationType = pyqtSignal(str)

    # Thread function
    def run(self):
        self.track(mtx, dist)

    # Does all the tracking and updates the gui with slots
    def track(self, matrix_coefficients, distortion_coefficients):
        global pressedKey, needleComposeRvec, needleComposeTvec, ultraSoundComposeRvec, ultraSoundComposeTvec
        """ Real time ArUco marker tracking.  """
        TcomposedRvecNeedle, TcomposedTvecNeedle = None, None
        TcomposedRvecUltrasound, TcomposedTvecUltrasound = None, None

        # Behaviour is a key between calibration types.
        # No simulation is equal to 0
        # Needle Calibration is equal to 1
        # Ultrasound Calibration is equal to 2
        # Press
        behaviour = 0
        try:
            while True:
                # No marker detected
                isCalibrationMarkerDetected = False
                isNeedleDetected = False
                isUltraSoundDetected = False

                # Get the next frame to process
                ret, frame = cap.read()
                # operations on the frame come here
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Change grayscale
                aruco_dict = aruco.Dictionary_get(aruco.DICT_5X5_250)  # Use 5x5 dictionary to find markers
                parameters = aruco.DetectorParameters_create()  # Marker detection parameters

                # lists of ids and the corners beloning to each id
                corners, ids, rejected_img_points = aruco.detectMarkers(gray, aruco_dict,
                                                                        parameters=parameters,
                                                                        cameraMatrix=matrix_coefficients,
                                                                        distCoeff=distortion_coefficients)

                # Get the behaviour and update the gui
                if behaviour == 0:
                    self.calibrationType.emit('Simulation')
                elif behaviour == 1:
                    self.calibrationType.emit('Needle calibration')
                else:
                    self.calibrationType.emit('Ultrasound calibration')
                    pass

                if np.all(ids is not None):  # If there are markers found by detector
                    # sort the markers
                    zipped = zip(ids, corners)
                    ids, corners = zip(*(sorted(zipped)))
                    # 4 axis for ultrasound, 2axis for needle
                    axisForFourPoints = np.float32(
                        [[-0.02, -0.02, 0], [-0.02, 0.02, 0], [0.02, -0.02, 0], [0.02, 0.02, 0]]).reshape(-1,3)  # axis for a plan
                    axisForTwoPoints = np.float32([[0.01, 0, 0], [-0.01, 0, 0]]).reshape(-1, 3)  # axis for a line

                    for i in range(0, len(ids)):  # Iterate in markers
                        # Estimate pose of each marker and return the values rvec and tvec---different from camera coefficients
                        rvec, tvec, markerPoints = aruco.estimatePoseSingleMarkers(corners[i], 0.02, matrix_coefficients,
                                                                                   distortion_coefficients)


                        if ids[i] == calibrationMarkerID:
                            calibrationRvec = rvec
                            calibrationTvec = tvec
                            isCalibrationMarkerDetected = True
                            calibrationMarkerCorners = corners[i]
                        elif ids[i] == needleMarkerID:
                            needleRvec = rvec
                            needleTvec = tvec
                            isNeedleDetected = True
                            needleCorners = corners[i]
                        elif ids[i] == ultraSoundMarkerID:
                            ultraSoundRvec = rvec
                            ultraSoundTvec = tvec
                            isUltraSoundDetected = True
                            ultrasoundCorners = corners[i]

                        (rvec - tvec).any()  # get rid of that nasty numpy value array error
                        # aruco.drawAxis(frame, matrix_coefficients, distortion_coefficients, rvec, tvec, 0.01)
                        frame = aruco.drawDetectedMarkers(frame, corners)  # Draw A square around the markers

                    if isNeedleDetected and needleComposeRvec is not None and needleComposeTvec is not None:
                        info = cv2.composeRT(needleComposeRvec, needleComposeTvec, needleRvec.T, needleTvec.T)
                        TcomposedRvecNeedle, TcomposedTvecNeedle = info[0], info[1]
                        imgpts, jac = cv2.projectPoints(axisForTwoPoints, TcomposedRvecNeedle, TcomposedTvecNeedle, matrix_coefficients,
                                                        distortion_coefficients)
                        frame = draw(frame, imgpts, (200, 200, 220))

                    if isUltraSoundDetected and ultraSoundComposeRvec is not None and ultraSoundComposeTvec is not None:
                        info = cv2.composeRT(ultraSoundComposeRvec, ultraSoundComposeTvec, ultraSoundRvec.T, ultraSoundTvec.T)
                        TcomposedRvecUltrasound, TcomposedTvecUltrasound = info[0], info[1]
                        imgpts, jac = cv2.projectPoints(axisForFourPoints, TcomposedRvecUltrasound, TcomposedTvecUltrasound, matrix_coefficients,
                                                        distortion_coefficients)
                        frame = draw(frame, imgpts, (60, 200, 50))

                    # If the both markers can be seen by the camera, print the xyz differences between them
                    # So it will guide the user
                    if isNeedleDetected and needleComposeRvec is not None and needleComposeTvec is not None and \
                        isUltraSoundDetected and ultraSoundComposeRvec is not None and ultraSoundComposeTvec is not None:
                        xdiff = round((TcomposedTvecNeedle[0] - TcomposedTvecUltrasound[0])[0], 3)
                        ydiff = round((TcomposedTvecNeedle[1] - TcomposedTvecUltrasound[1])[0], 3)
                        zdiff = round((TcomposedTvecNeedle[2] - TcomposedTvecUltrasound[2])[0], 3)
                        self.changeXDiff.emit('X difference:' + str(xdiff))
                        self.changeYDiff.emit('Y difference:' + str(ydiff))
                        self.changeZDiff.emit('Z difference:' + str(zdiff))


                # Display the resulting frame
                rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                convertToQtFormat = QImage(rgbImage.data, rgbImage.shape[1], rgbImage.shape[0], QImage.Format_RGB888)
                p = convertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)
                self.changePixmap.emit(p)

                # Key handling. For marker calibration, saving marker etc.
                if pressedKey is None:  # No key pressed
                    pass
                elif pressedKey == Qt.Key_C:  # Button for calibration.
                    if ids is not None and len(ids) > 1:  # If there are two markers, reverse the second and get the difference
                        if isNeedleDetected and isCalibrationMarkerDetected and behaviour == 1:
                            needleComposeRvec, needleComposeTvec = relativePosition(calibrationRvec, calibrationTvec,
                                                                                    needleRvec, needleTvec)
                            savedNeedleRvec, savedNeedleTvec = needleComposeRvec, needleComposeTvec
                        elif isUltraSoundDetected and isCalibrationMarkerDetected and behaviour == 2:
                            ultraSoundComposeRvec, ultraSoundComposeTvec = relativePosition(calibrationRvec,
                                                                                            calibrationTvec,
                                                                                            ultraSoundRvec,
                                                                                            ultraSoundTvec)
                            savedUltraSoundRvec, savedUltraSoundTvec = ultraSoundComposeRvec, ultraSoundComposeTvec
                elif pressedKey == Qt.Key_U:  # Button for moving z axis 1 mm
                    if behaviour == 1 and needleComposeTvec is not None:
                        needleComposeTvec = needleComposeTvec + [[0], [0], [0.001]]
                    elif behaviour == 2 and ultraSoundComposeTvec is not None:
                        ultraSoundComposeTvec = ultraSoundComposeTvec + [[0], [0], [0.001]]
                elif pressedKey == Qt.Key_D:  # Button for moving z axis -1 mm
                    if behaviour == 1 and needleComposeTvec is not None:
                        needleComposeTvec = needleComposeTvec + [[0], [0], [-0.001]]
                    elif behaviour == 2 and ultraSoundComposeTvec is not None:
                        ultraSoundComposeTvec = ultraSoundComposeTvec + [[0], [0], [-0.001]]
                elif pressedKey == Qt.Key_R:  # Button for moving x axis 1 mm
                    if behaviour == 1 and needleComposeTvec is not None:
                        needleComposeTvec = needleComposeTvec + [[0.001], [0], [0]]
                    elif behaviour == 2 and ultraSoundComposeTvec is not None:
                        ultraSoundComposeTvec = ultraSoundComposeTvec + [[0.001], [0], [0]]
                elif pressedKey == Qt.Key_L:  # Button for moving x axis -1 mm
                    if behaviour == 1 and needleComposeTvec is not None:
                        needleComposeTvec = needleComposeTvec + [[-0.001], [0], [0]]
                    elif behaviour == 2 and ultraSoundComposeTvec is not None:
                        ultraSoundComposeTvec = ultraSoundComposeTvec + [[-0.001], [0], [0]]
                elif pressedKey == Qt.Key_B:  # Button for moving y axis -1 mm
                    if behaviour == 1 and needleComposeTvec is not None:
                        needleComposeTvec = needleComposeTvec + [[0], [-0.001], [0]]
                    elif behaviour == 2 and ultraSoundComposeTvec is not None:
                        ultraSoundComposeTvec = ultraSoundComposeTvec + [[0], [-0.001], [0]]
                elif pressedKey == Qt.Key_F:  # Button for moving y axis 1 mm
                    if behaviour == 1 and needleComposeTvec is not None:
                        needleComposeTvec = needleComposeTvec + [[0], [0.001], [0]]
                    elif behaviour == 2 and ultraSoundComposeTvec is not None:
                        ultraSoundComposeTvec = ultraSoundComposeTvec + [[0], [0.001], [0]]
                elif pressedKey == Qt.Key_P:  # print necessary information here, for debug
                    pass  # Insert necessary print here
                elif pressedKey == Qt.Key_S:  # Save the calibration vectors to a file.
                    if(needleComposeRvec is not None and needleComposeTvec is not None and ultraSoundComposeRvec is not None and ultraSoundComposeTvec is not None):
                        print(needleComposeRvec, needleComposeTvec, ultraSoundComposeRvec, ultraSoundComposeTvec)
                        fileObject = open(save_Name, 'wb')
                        data = [needleComposeRvec, needleComposeTvec, ultraSoundComposeRvec, ultraSoundComposeTvec]
                        pickle.dump(data, fileObject)
                        fileObject.close()
                elif pressedKey == Qt.Key_X:  # change simulation type, "Simulation, needle calibration, ultrasound calibration"
                    behaviour = (behaviour + 1) % 3
                    # print(behaviour)
                pressedKey = None

        except Exception:
            pass
        finally:
        # When everything done, release the capture
            cap.release()


        return frame


class App(QWidget):
    def __init__(self):
        super().__init__()
        self.title = "Augmented Biopsy"
        # Start points and the resolution
        self.left = 40
        self.top = 40
        self.width = 1024
        self.height = 768

        self.initUI()

    # slot for setting the image.
    @pyqtSlot(QImage)
    def setImage(self, image):
        self.streamLabel.setPixmap(QPixmap.fromImage(image))

    # slot for the label that shows which type is on. Simulation, needle calibration, ultrasound calibration.
    @pyqtSlot(str)
    def setCalibrationTypeLabel(self, labelText):
        self.calibrationType.setText(labelText)

    # slot for x diff label.
    @pyqtSlot(str)
    def setXDiffLabel(self, labelText):
        self.xDifflabel.setText(labelText)

    # slot for y diff label.
    @pyqtSlot(str)
    def setYDiffLabel(self, labelText):
        self.yDifflabel.setText(labelText)

    # slot for z diff label.
    @pyqtSlot(str)
    def setZDiffLabel(self, labelText):
        self.zDifflabel.setText(labelText)

    # Key press event to get the keys.
    def keyPressEvent(self, event):
        global pressedKey
        pressedKey = event.key()



    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        # create a label for image
        self.streamLabel = QLabel(self)
        self.streamLabel.resize(640, 480)
        self.streamLabel.setAlignment(Qt.AlignCenter)


        # create a label for showing the type of the program, calibration or simulation
        self.calibrationType = QLabel(self)
        self.calibrationType.setAlignment(Qt.AlignCenter)
        # create a label for x diff
        self.xDifflabel = QLabel(self)
        self.xDifflabel.setAlignment(Qt.AlignCenter)

        # create a label for y diff
        self.yDifflabel = QLabel(self)
        self.yDifflabel.setAlignment(Qt.AlignCenter)

        # create a label for z diff
        self.zDifflabel = QLabel(self)
        self.zDifflabel.setAlignment(Qt.AlignCenter)

        # self.run_button = QPushButton('Start')

        displayVbox = QVBoxLayout()

        displayVbox.addWidget(self.streamLabel)
        displayVbox.addWidget(self.calibrationType)
        displayVbox.addWidget(self.xDifflabel)
        displayVbox.addWidget(self.yDifflabel)
        displayVbox.addWidget(self.zDifflabel)

        self.setLayout(displayVbox)

        th = Thread(self)
        th.changePixmap.connect(self.setImage)
        th.changeXDiff.connect(self.setXDiffLabel)
        th.changeYDiff.connect(self.setYDiffLabel)
        th.changeZDiff.connect(self.setZDiffLabel)
        th.calibrationType.connect(self.setCalibrationTypeLabel)
        th.start()
        return



''' Main Function '''
if __name__ == '__main__':
    global window
    parser = argparse.ArgumentParser(description='Aruco Marker Tracking')
    parser.add_argument('--coefficients', metavar='bool', required=True,
                        help='File name for matrix coefficients and distortion coefficients')
    parser.add_argument('--calibrationMarker', metavar='int', required=True,
                        help='Marker ID for the calibration marker')
    parser.add_argument('--needleMarker', metavar='int', required=True,
                        help='Marker ID for the needle\'s marker')
    parser.add_argument('--ultrasoundMarker', metavar='int', required=True,
                        help='Marker ID for the needle\'s marker')
    parser.add_argument('--savedCalibration', metavar='int', required=False,
                        help='Calibration saves')

    # Parse the arguments and take action for that.
    args = parser.parse_args()
    calibrationMarkerID = int(args.calibrationMarker)
    needleMarkerID = int(args.needleMarker)
    ultraSoundMarkerID = int(args.ultrasoundMarker)

    # If marker calibration is already done and saved, you can load it.
    if args.savedCalibration == '1':
        fileObject = open(save_Name, 'rb')
        saved = pickle.load(fileObject)
        needleComposeRvec, needleComposeTvec, ultraSoundComposeRvec, ultraSoundComposeTvec = saved
        fileObject.close()
    # If camera calibration is done and saved, you can load it.
    if args.coefficients == '1':
        mtx, dist = loadCoefficients("calib_images/calibrationCoefficients.yaml")
        ret = True
    else:  # No calibration, use "calib_images" as directory.
        ret, mtx, dist, rvecs, tvecs = calibrate("calib_images")
        saveCoefficients(mtx, dist, "calib_images/calibrationCoefficients.yaml")
    print("Calibration is completed. Starting tracking sequence.")
    # If everything is okay, we can init the gui and start the stream.
    if ret:
        app = QApplication(sys.argv)
        main_window = QMainWindow()
        widget = App()
        widget.show()
        app.exec_()
        # Release the camera
        cap.release()
        cv2.destroyAllWindows()
