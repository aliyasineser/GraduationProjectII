import numpy as np
import cv2
import cv2.aruco as aruco
import glob
import argparse

# Marker id infos. Global to access everywhere. It is unnecessary to change it to local.
calibrationMarkerID = None
needleMarkerID = None
ultraSoundMarkerID = None

cap = cv2.VideoCapture(1)
image_width = 0
image_height = 0

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

def calibrate(dirpath):
    """ Apply camera calibration operation for images in the given directory path. """
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(8,6,0)
    objp = np.zeros((6*9, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    images = glob.glob(dirpath+'/*.png')

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

        # If found, add object points, image points (after refining them)
        if ret:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (9, 6), corners2, ret)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    return [ret, mtx, dist, rvecs, tvecs]


def saveCoefficients(mtx, dist, path):
    """ Save the camera matrix and the distortion coefficients to given path/file. """
    cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
    cv_file.write("camera_matrix", mtx)
    cv_file.write("dist_coeff", dist)
    # note you *release* you don't close() a FileStorage object
    cv_file.release()


def loadCoefficients(path):
    """ Loads camera matrix and distortion coefficients. """
    # FILE_STORAGE_READ
    cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)

    # note we also have to specify the type to retrieve other wise we only get a
    # FileNode object back instead of a matrix
    camera_matrix = cv_file.getNode("camera_matrix").mat()
    dist_matrix = cv_file.getNode("dist_coeff").mat()

    # Debug: print the values
    # print("camera_matrix : ", camera_matrix.tolist())
    # print("dist_matrix : ", dist_matrix.tolist())

    cv_file.release()
    return [camera_matrix, dist_matrix]


def inversePerspective(rvec, tvec):
    """ Applies perspective transform for given rvec and tvec. """
    R, _ = cv2.Rodrigues(rvec)
    R = np.matrix(R).T
    invTvec = np.dot(R, np.matrix(-tvec))
    invRvec, _ = cv2.Rodrigues(R)
    return invRvec, invTvec


def make_1080p():
    global image_width
    global image_height
    image_width = 1920
    image_height = 1080
    change_res(image_width, image_height)


def make_720p():
    global image_width
    global image_height
    image_width = 1280
    image_height = 720
    change_res(image_width, image_height)


def make_480p():
    global image_width
    global image_height
    image_width = 640
    image_height = 480
    change_res(image_width, image_height)


def change_res(width, height):
    cap.set(3, width)
    cap.set(4, height)


def relativePosition(rvec1, tvec1, rvec2, tvec2):
    """ Get relative position for rvec2 & tvec2. Compose the returned rvec & tvec to use composeRT with rvec2 & tvec2 """
    rvec1, tvec1 = rvec1.reshape((3, 1)), tvec1.reshape((3, 1))
    rvec2, tvec2 = rvec2.reshape((3, 1)), tvec2.reshape((3, 1))

    # Inverse the second marker, the right one in the image
    invRvec, invTvec = inversePerspective(rvec2, tvec2)

    info = cv2.composeRT(rvec1, tvec1, invRvec, invTvec)
    composedRvec, composedTvec = info[0], info[1]

    composedRvec = composedRvec.reshape((3, 1))
    composedTvec = composedTvec.reshape((3, 1))
    return composedRvec, composedTvec


def draw(img, imgpts, color):
    """ draw a line between given two points. """
    imgpts = np.int32(imgpts).reshape(-1, 2)
    for pointf in range(len(imgpts)):
        for points in range(len(imgpts)):
            img = cv2.line(img, tuple(imgpts[pointf]), tuple(imgpts[points]), color, 3)
    return img


def track(matrix_coefficients, distortion_coefficients):
    global image_width
    global image_height
    """ Real time ArUco marker tracking.  """
    needleComposeRvec, needleComposeTvec = None, None  # Composed for needle
    ultraSoundComposeRvec, ultraSoundComposeTvec = None, None  # Composed for ultrasound
    savedNeedleRvec, savedNeedleTvec = None, None  # Pure Composed
    savedUltraSoundRvec, savedUltraSoundTvec = None, None  # Pure Composed
    TcomposedRvecNeedle, TcomposedTvecNeedle = None, None
    TcomposedRvecUltrasound, TcomposedTvecUltrasound = None, None

    # Behaviour is a key between calibration types.
    # No simulation is equal to 0
    # Needle Calibration is equal to 1
    # Ultrasound Calibration is equal to 2
    # Press
    behaviour = 0
    make_480p()

    while True:
        isCalibrationMarkerDetected = False
        isNeedleDetected = False
        isUltraSoundDetected = False
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

        if behaviour == 0:
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, 'No Calibration', (10, 40), font, 0.7, (180, 250, 199), 2, cv2.LINE_AA)
        elif behaviour == 1:
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, 'Needle calibration', (10, 40), font, 0.7, (180, 250, 199), 2, cv2.LINE_AA)
        else:
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, 'Ultrasound calibration', (10, 40), font, 0.7, (180, 250, 199), 2, cv2.LINE_AA)
            pass



        if np.all(ids is not None):  # If there are markers found by detector
            zipped = zip(ids, corners)
            ids, corners = zip(*(sorted(zipped)))
            axisForFourPoints = np.float32([[-0.025, -0.025, 0], [-0.025, 0.025, 0], [0.025, -0.025, 0], [0.025, 0.025, 0]]).reshape(-1, 3)  # axis for a plane
            axisForTwoPoints = np.float32([[0.01, 0.01, 0], [-0.01, 0.01, 0]]).reshape(-1, 3)  # axis for a line
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
                # aruco.drawAxis(frame, matrix_coefficients, distortion_coefficients, rvec, tvec, 0.01)  # Draw Axis
                aruco.drawDetectedMarkers(frame, corners)  # Draw A square around the markers

            if isNeedleDetected and needleComposeRvec is not None and needleComposeTvec is not None:
                info = cv2.composeRT(needleComposeRvec, needleComposeTvec, needleRvec.T, needleTvec.T)
                TcomposedRvecNeedle, TcomposedTvecNeedle = info[0], info[1]
                imgpts, jac = cv2.projectPoints(axisForTwoPoints, TcomposedRvecNeedle, TcomposedTvecNeedle, matrix_coefficients,
                                                distortion_coefficients)
                frame = draw(frame, imgpts, (200, 200, 220))

            if isUltraSoundDetected and ultraSoundComposeRvec is not None and ultraSoundComposeTvec is not None:
                info = cv2.composeRT(ultraSoundComposeRvec, ultraSoundComposeTvec, ultraSoundRvec.T, ultraSoundTvec.T)
                TcomposedRvecUltrasound, TcomposedTvecUltrasound = info[0], info[1]
                imgpts, jac = cv2.projectPoints(axisForTwoPoints, TcomposedRvecUltrasound, TcomposedTvecUltrasound, matrix_coefficients,
                                                distortion_coefficients)
                frame = draw(frame, imgpts, (60, 200, 50))



            if isNeedleDetected and needleComposeRvec is not None and needleComposeTvec is not None and \
                isUltraSoundDetected and ultraSoundComposeRvec is not None and ultraSoundComposeTvec is not None:
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame, 'X difference:'+str(TcomposedTvecNeedle[0] - TcomposedTvecUltrasound[0]), (10, 70), font, 0.7, (0, 0, 200), 2, cv2.LINE_AA)
                cv2.putText(frame, 'y difference:'+str(TcomposedTvecNeedle[1] - TcomposedTvecUltrasound[1]), (10, 100), font, 0.7, (0, 200, 0), 2, cv2.LINE_AA)
                cv2.putText(frame, 'z difference:'+str(TcomposedTvecNeedle[2] - TcomposedTvecUltrasound[2]), (10, 130), font, 0.7, (200, 0, 0), 2, cv2.LINE_AA)

        # Display the resulting frame
        cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('frame', image_width, image_height)
        cv2.imshow('frame', frame)
        # Wait 3 milisecoonds for an interaction. Check the key and do the corresponding job.
        key = cv2.waitKey(3) & 0xFF
        if key == ord('q'):  # Quit
            break
        elif key == ord('c'):  # Calibration
            if len(ids) > 1:  # If there are two markers, reverse the second and get the difference
                if isNeedleDetected and isCalibrationMarkerDetected and behaviour == 1:
                    needleComposeRvec, needleComposeTvec = relativePosition(calibrationRvec, calibrationTvec, needleRvec, needleTvec)
                    savedNeedleRvec, savedNeedleTvec = needleComposeRvec, needleComposeTvec
                elif isUltraSoundDetected and isCalibrationMarkerDetected and behaviour == 2:
                    ultraSoundComposeRvec, ultraSoundComposeTvec = relativePosition(calibrationRvec, calibrationTvec, ultraSoundRvec, ultraSoundTvec)
                    savedUltraSoundRvec, savedUltraSoundTvec = ultraSoundComposeRvec, ultraSoundComposeTvec
        elif key == ord('u'):  # Up
            if behaviour == 1 and needleComposeTvec is not None:
                needleComposeTvec = needleComposeTvec + [[0], [0], [0.001]]
            elif behaviour == 2 and ultraSoundComposeTvec is not None:
                ultraSoundComposeTvec = ultraSoundComposeTvec + [[0], [0], [0.001]]
        elif key == ord('d'):  # Down
            if behaviour == 1 and needleComposeTvec is not None:
                needleComposeTvec = needleComposeTvec + [[0], [0], [-0.001]]
            elif behaviour == 2 and ultraSoundComposeTvec is not None:
                ultraSoundComposeTvec = ultraSoundComposeTvec + [[0], [0], [-0.001]]
        elif key == ord('r'):  # Right
            if behaviour == 1 and needleComposeTvec is not None:
                needleComposeTvec = needleComposeTvec + [[0.001], [0], [0]]
            elif behaviour == 2 and ultraSoundComposeTvec is not None:
                ultraSoundComposeTvec = ultraSoundComposeTvec + [[0.001], [0], [0]]
        elif key == ord('l'):  # Left
            if behaviour == 1 and needleComposeTvec is not None:
                needleComposeTvec = needleComposeTvec + [[-0.001], [0], [0]]
            elif behaviour == 2 and ultraSoundComposeTvec is not None:
                ultraSoundComposeTvec = ultraSoundComposeTvec + [[-0.001], [0], [0]]
        elif key == ord('b'):  # Back
            if behaviour == 1 and needleComposeTvec is not None:
                needleComposeTvec = needleComposeTvec + [[0], [-0.001], [0]]
            elif behaviour == 2 and ultraSoundComposeTvec is not None:
                ultraSoundComposeTvec = ultraSoundComposeTvec + [[0], [-0.001], [0]]
        elif key == ord('f'):  # Front
            if behaviour == 1 and needleComposeTvec is not None:
                needleComposeTvec = needleComposeTvec + [[0], [0.001], [0]]
            elif behaviour == 2 and ultraSoundComposeTvec is not None:
                ultraSoundComposeTvec = ultraSoundComposeTvec + [[0], [0.001], [0]]
        elif key == ord('p'):  # print necessary information here
            pass  # Insert necessary print here
        elif key == ord('x'):  # change simulation type
            behaviour = (behaviour+1) % 3
            print(behaviour)

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Aruco Marker Tracking')
    parser.add_argument('--coefficients', metavar='bool', required=True,
                        help='File name for matrix coefficients and distortion coefficients')
    parser.add_argument('--calibrationMarker', metavar='int', required=True,
                        help='Marker ID for the calibration marker')
    parser.add_argument('--needleMarker', metavar='int', required=True,
                        help='Marker ID for the needle\'s marker')
    parser.add_argument('--ultrasoundMarker', metavar='int', required=True,
                        help='Marker ID for the needle\'s marker')

    # Parse the arguments and take action for that.
    args = parser.parse_args()
    calibrationMarkerID = int(args.calibrationMarker)
    needleMarkerID = int(args.needleMarker)
    ultraSoundMarkerID = int(args.ultrasoundMarker)

    if args.coefficients == '1':
        mtx, dist = loadCoefficients("calib_images/calibrationCoefficients.yaml")
        ret = True
    else:
        ret, mtx, dist, rvecs, tvecs = calibrate("calib_images")
        saveCoefficients(mtx, dist, "calib_images/calibrationCoefficients.yaml")
    print("Calibration is completed. Starting tracking sequence.")
    if ret:
        track(mtx, dist)
