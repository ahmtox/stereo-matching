import cv2

def stereo_calibrate(objpoints, imgpointsL, imgpointsR, mtxL, distL, mtxR, distR, img_shape):
    flags = cv2.CALIB_FIX_INTRINSIC  
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
    retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpointsL, imgpointsR, mtxL, distL, mtxR, distR, img_shape,
        criteria=criteria, flags=flags)

    print("Rotation Matrix (R):\n", R)
    print("Translation Vector (T):\n", T)
    print("Essential Matrix (E):\n", E)
    print("Fundamental Matrix (F):\n", F)

    return R, T, E, F