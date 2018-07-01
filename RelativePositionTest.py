import numpy as np
import cv2
import cv2.aruco as aruco
import glob
import argparse

cap = cv2.VideoCapture(0)

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

def calibrate():
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

# Experimental inversion
def inversePerspectiveWithTransformMatrix(tvec, rvec):
    R, _ = cv2.Rodrigues(rvec)  # 3x3 representation of rvec
    R = np.matrix(R).T  # transpose of 3x3 rotation matrix
    transformMatrix = np.zeros((4, 4), dtype=float)  # Transformation matrix
    # Transformation matrix fill operation, matrix should be [R|t,0|1]
    transformMatrix[0:3, 0:3] = R
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


def inversePerspective(rvec, tvec):
    R, _ = cv2.Rodrigues(rvec)
    R = np.matrix(R).T
    invTvec = np.dot(-R, np.matrix(tvec))
    invRvec, _ = cv2.Rodrigues(R)
    return invRvec, invTvec


def relativePosition(rvec1, tvec1, rvec2, tvec2):
    rvec1, tvec1 = rvec1.reshape((3, 1)), tvec1.reshape(
        (3, 1))
    rvec2, tvec2 = rvec2.reshape((3, 1)), tvec2.reshape((3, 1))

    # Inverse the second marker, the right one in the image
    invRvec, invTvec = inversePerspective(rvec2, tvec2)

    orgRvec, orgTvec = inversePerspective(invRvec, invTvec)
    # print("rvec: ", rvec2, "tvec: ", tvec2, "\n and \n", orgRvec, orgTvec)

    info = cv2.composeRT(rvec1, tvec1, invRvec, invTvec)
    composedRvec, composedTvec = info[0], info[1]

    composedRvec = composedRvec.reshape((3, 1))
    composedTvec = composedTvec.reshape((3, 1))
    return composedRvec, composedTvec


def draw(img, corners, imgpts):
    imgpts = np.int32(imgpts).reshape(-1,2)
    # draw ground floor in green
    # img = cv2.drawContours(img, [imgpts[:4]],-1,(0,255,0),-3)
    # draw pillars in blue color
    for i,j in zip(range(4),range(4)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]),(255),3)
    # draw top layer in red color
    return img


def track(matrix_coefficients, distortion_coefficients):
    pointCircle = (0, 0)
    markerTvecList = []
    markerRvecList = []
    composedRvec, composedTvec = None, None
    while True:
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
            axis = np.float32([[-0.01, -0.01, 0], [-0.01, 0.01, 0], [0.01, -0.01, 0], [0.01, 0.01, 0]]).reshape(-1, 3)
            for i in range(0, len(ids)):  # Iterate in markers
                # Estimate pose of each marker and return the values rvec and tvec---different from camera coefficients
                rvec, tvec, markerPoints = aruco.estimatePoseSingleMarkers(corners[i], 0.02, matrix_coefficients,
                                                                           distortion_coefficients)
                # print(markerPoints)
                (rvec - tvec).any()  # get rid of that nasty numpy value array error
                markerRvecList.append(rvec)
                markerTvecList.append(tvec)

                # aruco.drawAxis(frame, matrix_coefficients, distortion_coefficients, rvec, tvec, 0.01)  # Draw Axis
                # cv2.circle(frame, pointCircle, 6, (255, 0, 255), 3)
                aruco.drawDetectedMarkers(frame, corners)  # Draw A square around the markers

            if len(ids) > 1 and composedRvec is not None and composedTvec is not None:
                info = cv2.composeRT(composedRvec, composedTvec, markerRvecList[1].T, markerTvecList[1].T)
                TcomposedRvec, TcomposedTvec = info[0], info[1]

                objectPositions = np.array([(0, 0, 0)], dtype=np.float)  # 3D point for projection
                imgpts, jac = cv2.projectPoints(axis, TcomposedRvec, TcomposedTvec, matrix_coefficients,
                                                distortion_coefficients)
                # print(imgpts)
                # frame = draw(frame, corners[0], imgpts)
                aruco.drawAxis(frame, matrix_coefficients, distortion_coefficients, TcomposedRvec, TcomposedTvec,
                               0.01)  # Draw Axis
                # relativePoint = (int(imgpts[0][0][0]), int(imgpts[0][0][1]))
                # cv2.circle(frame, relativePoint, 2, (255, 255, 0))



        # Display the resulting frame
        cv2.imshow('frame', frame)
        # Wait 3 milisecoonds for an interaction. Check the key and do the corresponding job.
        key = cv2.waitKey(3) & 0xFF
        if key == ord('q'):  # Quit
            break
        elif key == ord('c'):  # Calibration
            if len(ids) > 1:  # If there are two markers, reverse the second and get the difference
                markerRvecList[0], markerTvecList[0] = markerRvecList[0].reshape((3, 1)), markerTvecList[0].reshape(
                    (3, 1))
                markerRvecList[1], markerTvecList[1] = markerRvecList[1].reshape((3, 1)), markerTvecList[1].reshape((3, 1))

                composedRvec, composedTvec = relativePosition(markerRvecList[0], markerTvecList[0], markerRvecList[1], markerTvecList[1])
                # debug: get the relative again so you can be sure about you are doing it right!
                # info = cv2.composeRT(composedRvec, composedTvec, markerRvecList[1], markerTvecList[1])
                # TcomposedRvec, TcomposedTvec = info[0], info[1]
                # print("first: ", markerRvecList[0], markerTvecList[0])  # first marker vectors
                # print("second: ", markerRvecList[1], markerTvecList[1])  # second marker vectors
                # print("composed: ", composedRvec, composedTvec)  # relative marker vectors
                # print("test: ", TcomposedRvec, TcomposedTvec)  # second relative marker vectors, should be equal to first or second
                # aruco.drawAxis(frame, matrix_coefficients, distortion_coefficients, TcomposedRvec, TcomposedTvec, 0.01)  # Draw Axis for second relative!



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
