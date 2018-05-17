import numpy as np
import cv2
import cv2.aruco as aruco
import glob
import argparse

calibrationMarkerID = None
needleMarkerID = None

cap = cv2.VideoCapture(1)

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


def relativePosition(rvec1, tvec1, rvec2, tvec2):
    rvec1, tvec1 = rvec1.reshape((3, 1)), tvec1.reshape((3, 1))
    rvec2, tvec2 = rvec2.reshape((3, 1)), tvec2.reshape((3, 1))

    # Inverse the second marker, the right one in the image
    invRvec, invTvec = inversePerspective(rvec2, tvec2)

    info = cv2.composeRT(rvec1, tvec1, invRvec, invTvec)
    composedRvec, composedTvec = info[0], info[1]

    composedRvec = composedRvec.reshape((3, 1))
    composedTvec = composedTvec.reshape((3, 1))
    return composedRvec, composedTvec


def draw(img, imgpts):
    imgpts = np.int32(imgpts).reshape(-1, 2)
    # draw pillars in blue color
    img = cv2.line(img, tuple(imgpts[0]), tuple(imgpts[1]), (200, 200, 220), 3)
    # draw top layer in red color
    return img


def track(matrix_coefficients, distortion_coefficients):
    markerTvecList = []
    markerRvecList = []
    needleComposeRvec, needleComposeTvec = None, None  # Composed
    TcomposedRvec, TcomposedTvec = None, None  # Composed + second Marker
    savedNeedleRvec, savedNeedleTvec = None, None  # Pure Composed
    while True:
        isCalibrationMarkerDetected = False
        isNeedleDetected = False
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

        if np.all(ids is not None):  # If there are markers found by detector
            del markerTvecList[:]
            del markerRvecList[:]
            zipped = zip(ids, corners)
            ids, corners = zip(*(sorted(zipped)))
            # axis = np.float32([[-0.01, -0.01, 0], [-0.01, 0.01, 0], [0.01, -0.01, 0], [0.01, 0.01, 0]]).reshape(-1, 3)  # axis for a plane
            axisForTwoPoints = np.float32([[0.01, 0.01, 0], [-0.01, 0.01, 0]]).reshape(-1, 3)  # axis for a line
            for i in range(0, len(ids)):  # Iterate in markers
                # Estimate pose of each marker and return the values rvec and tvec---different from camera coefficients
                rvec, tvec, markerPoints = aruco.estimatePoseSingleMarkers(corners[i], 0.02, matrix_coefficients,
                                                                           distortion_coefficients)
                print(ids)
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

                (rvec - tvec).any()  # get rid of that nasty numpy value array error
                markerRvecList.append(rvec)
                markerTvecList.append(tvec)

                # aruco.drawAxis(frame, matrix_coefficients, distortion_coefficients, rvec, tvec, 0.01)  # Draw Axis
                aruco.drawDetectedMarkers(frame, corners)  # Draw A square around the markers

            if isNeedleDetected and needleComposeRvec is not None and needleComposeTvec is not None:
                info = cv2.composeRT(needleComposeRvec, needleComposeTvec, needleRvec.T, needleTvec.T)
                TcomposedRvec, TcomposedTvec = info[0], info[1]
                imgpts, jac = cv2.projectPoints(axisForTwoPoints, TcomposedRvec, TcomposedTvec, matrix_coefficients,
                                                distortion_coefficients)
                frame = draw(frame, imgpts)
                # aruco.drawAxis(frame, matrix_coefficients, distortion_coefficients, TcomposedRvec, TcomposedTvec, 0.01)  # Draw Axis

        # Display the resulting frame
        cv2.imshow('frame', frame)
        # Wait 3 milisecoonds for an interaction. Check the key and do the corresponding job.
        key = cv2.waitKey(3) & 0xFF
        if key == ord('q'):  # Quit
            break
        elif key == ord('c'):  # Calibration
            if len(ids) > 1:  # If there are two markers, reverse the second and get the difference
                needleComposeRvec, needleComposeTvec = relativePosition(calibrationRvec, calibrationTvec, needleRvec, needleTvec)
                savedNeedleRvec, savedNeedleTvec = needleComposeRvec, needleComposeTvec
        elif key == ord('u'):
            needleComposeTvec = needleComposeTvec + [[0], [0], [0.001]]
        elif key == ord('d'):
            needleComposeTvec = needleComposeTvec + [[0], [0], [-0.001]]
        elif key == ord('r'):
            needleComposeTvec = needleComposeTvec + [[0.001], [0], [0]]
        elif key == ord('l'):
            needleComposeTvec = needleComposeTvec + [[-0.001], [0], [0]]
        elif key == ord('b'):
            needleComposeTvec = needleComposeTvec + [[0], [-0.001], [0]]
        elif key == ord('f'):
            needleComposeTvec = needleComposeTvec + [[0], [0.001], [0]]
        elif key == ord('p'):
            print("composed vector to print")
            print(needleComposeTvec)
            print("calculated vector to print")
            print(savedNeedleTvec)

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
    args = parser.parse_args()
    calibrationMarkerID = int(args.calibrationMarker)
    needleMarkerID = int(args.needleMarker)
    if args.coefficients == '1':
        mtx, dist = loadCoefficients("calib_images/calibrationCoefficients.yaml")
        ret = True
    else:
        ret, mtx, dist, rvecs, tvecs = calibrate("calib_images")
        saveCoefficients(mtx, dist, "calib_images/calibrationCoefficients.yaml")
    print("Calibration is completed. Starting tracking sequence.")
    if ret:
        track(mtx, dist)
