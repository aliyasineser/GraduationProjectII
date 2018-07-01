# GraduationProjectII
Visualization for Ultrasound Supported Biopsy

The goal of the project is to prepare 3D visualization software for the thyroid biopsy
operation.

The first prototype was made with TrakStar hardware and Unity software. Although
the TrakStar transmits the position and the rotation correctly, it is prototyped on the
camera-based system because it is affected from the magnetic field and metals.

Using ArUco and the OpenCV library, 3D positions and rotations of ArUco markers
were detected. Calibration was prepared using two markers. Tracking software was
designed by attaching a marker to the ultrasonic transducer and the needle.

The TrakStar equipment could not be implemented in the clinical environment
because it was affected from the magnetic field. Success was achieved with
millimetric error in the test environment. The user-assisted calibration of the
marker-based system resulted in a milimetric error.

Prepared with optimum calibrations and environment, the software prepared with
ArUco markers has succeeded with the millimetric error rate. Because lines are
drawn in two dimensions, the perspective problem is resolved by taking the position
differences of the markers in the three-dimensional environment.

