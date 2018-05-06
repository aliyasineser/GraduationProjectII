import numpy as np
import cv2
import cv2.aruco as aruco
import glob
import argparse

cap = cv2.VideoCapture(1)
markerTvecList = []
markerRvecList = []
pointCircle = None

class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

def calibrate():
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(8,6,0)
    objp = np.zeros((6*9,3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    images = glob.glob('calib_images/*.png')

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


def saveCoefficients(mtx, dist):
    cv_file = cv2.FileStorage("calib_images/calibrationCoefficients.yaml", cv2.FILE_STORAGE_WRITE)
    cv_file.write("camera_matrix", mtx)
    cv_file.write("dist_coeff", dist)
    # note you *release* you don't close() a FileStorage object
    cv_file.release()


def loadCoefficients():
    # FILE_STORAGE_READ
    cv_file = cv2.FileStorage("calib_images/calibrationCoefficients.yaml", cv2.FILE_STORAGE_READ)

    # note we also have to specify the type to retrieve other wise we only get a
    # FileNode object back instead of a matrix
    camera_matrix = cv_file.getNode("camera_matrix").mat()
    dist_matrix = cv_file.getNode("dist_coeff").mat()

    # Debug: print the values
    # print("camera_matrix : ", camera_matrix.tolist())
    # print("dist_matrix : ", dist_matrix.tolist())

    cv_file.release()
    return [camera_matrix, dist_matrix]

def inversePerspectiveWithTransformMatrix(tvec, rvec):
    R, _ = cv2.Rodrigues(rvec)  # 3x3 representation of rvec
    R = np.matrix(R).T  # transpose of 3x3 rotation matrix
    transformMatrix = np.zeros((4, 4), dtype=float)  # Transformation matrix
    # Transformation matrix fill operation, matrix should be [R|t,0|1]
    transformMatrix[0:3, 0:3] = R  #
    transformMatrix[0:3, 3] = tvec
    transformMatrix[3, 3] = 1
    # Inverse the transform matrix to get camera centered Transform
    _transformMatrix = np.linalg.inv(transformMatrix)
    # Extract new rotation and translation vectors from transform matrix
    _R = _transformMatrix[0:3, 0:3]
    _tvec = _transformMatrix[0:3, 3]
    _rvec, _ = cv2.Rodrigues(_R)
    # return
    return _tvec, _rvec

def composeTransform(rvec1, tvec1, rvec2, tvec2):
    # Sum of the translation and multiplication of the rotation will give the composed vectors
    tvec3 = tvec1 + tvec2
    firstRvecSquareMatrix, _ = cv2.Rodrigues(rvec1)
    secondRvecSquareMatrix, _ = cv2.Rodrigues(rvec2)
    rvec3 = firstRvecSquareMatrix * secondRvecSquareMatrix
    rvec3, _ = cv2.Rodrigues(rvec3)
    return rvec3, tvec3

def inversePerspective(tvec, rvec):
    R, _ = cv2.Rodrigues(rvec)
    R = np.matrix(R).T
    invTvec = -R * np.matrix(tvec)
    invRvec = cv2.Rodrigues(R)
    return invTvec, invRvec


def track(matrix_coefficients, distortion_coefficients):
    pointCircle = (0, 0)
    first2print = (0, 0)
    sec2print = (0, 0)
    while True:
        ret, frame = cap.read()
        # operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Change grayscale
        aruco_dict = aruco.Dictionary_get(aruco.DICT_5X5_250)  # Use 5x5 dictionary to find markers
        parameters = aruco.DetectorParameters_create()  # Marker detection parameters

        # lists of ids and the corners beloning to each id
        corners, ids, rejected_img_points = aruco.detectMarkers(gray, aruco_dict, parameters=parameters, cameraMatrix=matrix_coefficients, distCoeff=distortion_coefficients)

        if np.all(ids is not None):  # If there are markers found by detector
            del markerTvecList[:]
            del markerRvecList[:]
            for i in range(0, len(ids)):  # Iterate in markers
                # Estimate pose of each marker and return the values rvec and tvec---different from camera coefficients
                rvec, tvec, markerPoints = aruco.estimatePoseSingleMarkers(corners[i], 0.02, matrix_coefficients, distortion_coefficients)
                # print(markerPoints)
                (rvec - tvec).any()  # get rid of that nasty numpy value array error
                # experimental(frame, corners[i], matrix_coefficients, distortion_coefficients)
                # print("translation: " , tvec)
                # print("rotation: ", rvec)
                markerTvecList.append(tvec)
                markerRvecList.append(rvec)
                objectPositions = np.array([(0, 0, 0)], dtype=np.float)
                imgpts, jac = cv2.projectPoints(objectPositions, rvec, tvec, matrix_coefficients, distortion_coefficients)
                aruco.drawAxis(frame, matrix_coefficients, distortion_coefficients, rvec[0], tvec[0], 0.01)  # Draw Axis , DEBUG : +[[[0, 0, 0.5]]]
                cv2.circle(frame, pointCircle, 6, (255, 0, 255), 3)
                cv2.line(frame, first2print, sec2print, (35, 0, 152), 3)
                # cv2.circle(frame, (int(imgpts[0][0][0]), int(imgpts[0][0][1])), 6, (255, 0, 255), 3)
                aruco.drawDetectedMarkers(frame, corners)  # Draw A square around the markers

        # Display the resulting frame
        cv2.imshow('frame', frame)
        # Wait 3 milisecoonds for an interaction. Check the key and do the corresponding job.
        key = cv2.waitKey(3) & 0xFF
        if key == ord('q'):  # Quit
            break
        elif key == ord('c'):  # Calibration
            if len(ids) > 1:  # If there are two markers, reverse the second and get the difference
                # Inverse the second marker, the right one in the image
                invTvec, invRvec = inversePerspectiveWithTransformMatrix(markerTvecList[1], markerRvecList[1])
                # Alternative for composeRT
                # composedRvec, composedTvec = composeTransform(markerRvecList[0], markerTvecList[0], markerRvecList[1], markerTvecList[1])
                markerRvecList[0], markerTvecList[0], markerRvecList[1], markerTvecList[1] = markerRvecList[0].reshape((3, 1)), markerTvecList[0].reshape((3, 1)), markerRvecList[1].reshape((3, 1)), markerTvecList[1].reshape((3, 1))
                info = cv2.composeRT(markerRvecList[0], markerTvecList[0], markerRvecList[1], markerTvecList[1])
                composedRvec, composedTvec = info[0], info[1]

                objectPositions = np.array([(0, 0, 0)], dtype=np.float)  # 3D point for projection
                # Get projected point and draw a circle
                imgpts, jac = cv2.projectPoints(objectPositions, composedRvec, composedTvec, matrix_coefficients, distortion_coefficients)

                # let's write a line to see it is really in the right direction
                first, jac = cv2.projectPoints(objectPositions, markerRvecList[0], markerTvecList[0], matrix_coefficients, distortion_coefficients)
                second, jac = cv2.projectPoints(objectPositions, markerRvecList[1], markerTvecList[1], matrix_coefficients, distortion_coefficients)
                cv2.line(frame, (int(first[0][0][0]), int(first[0][0][1])), (int(second[0][0][0]), int(second[0][0][1])), (255, 255, 255), 3)
                print(int(first[0][0][0]), int(first[0][0][1]), int(second[0][0][0]), int(second[0][0][1]))
                # pointCircle a global variable to store circle position to draw in every frame.
                # TODO: make a function and copy the calibration code, do the calibration in every frame.
                pointCircle = (int(imgpts[0][0][0]), int(imgpts[0][0][1]))
                first2print = (int(first[0][0][0]), int(first[0][0][1]))
                sec2print = (int(second[0][0][0]), int(second[0][0][1]))
                # print((int(imgpts[0][0][0]), int(imgpts[0][0][1])))
                cv2.circle(frame, pointCircle, 6, (255, 0, 255), 3)
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Aruco Marker Tracking')
    parser.add_argument('--coefficients', metavar='bool', required=True,
                        help='File name for matrix coefficients and distortion coefficients')
    args = parser.parse_args()
    if args.coefficients == '1':
        mtx, dist = loadCoefficients()
        ret = True
    else:
        ret, mtx, dist, rvecs, tvecs = calibrate()
        saveCoefficients(mtx, dist)
    print("Calibration is completed. Starting tracking sequence.")
    if ret:
        track(mtx, dist)
