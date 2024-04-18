import cv2
import numpy as np
import os
from pupil_apriltags import Detector
import matplotlib.pyplot as plt
import requests
import pickle
import glob

class CameraCalibration:

    def __init__(self):
        self.calObjPoints = []
        self.calImgPoints = []
        self.calMatrix = None
        self.distCoeffs = None
        self.calRotations = None
        self.calTranslations = None
        self.extrinsic_matrices = None

    
    def detect_fiducials(self, images_filepath, plot=False):

        response = requests.get("https://github.com/Harvard-CS283/pset-data/raw/f1a90573ae88cd530a3df3cd0cea71aa2363b1b3/april/AprilBoards.pickle")
        data = pickle.loads(response.content)

        at_coarseboard = data['at_coarseboard']
        at_fineboard = data['at_fineboard']

        at_detector = Detector(families='tag36h11',
                            nthreads=1,
                            quad_decimate=1.0,
                            quad_sigma=0.0,
                            refine_edges=1,
                            decode_sharpening=0.25,
                            debug=0)
        
        N = 70 # only use images with at least N detected objects for calibration
        total_valid = 0

        # Edit this line to point to the collection of input calibration image
        CALIBFILES = images_filepath

        # Uncomment one of the following two lines to indicate which AprilBoard is being used (fine or coarse)
        BOARD = at_fineboard
        #BOARD = at_coarseboard

        ###### BEGIN CALIBRATION SCRIPT

        # exit if no images are found or if BOARD is unrecognized
        images = glob.glob(CALIBFILES)
        assert images, "no calibration images matching: " + CALIBFILES
        assert BOARD==at_fineboard or BOARD==at_coarseboard, "Unrecognized AprilBoard"

        # else continue
        print("{} images:".format(len(images)))

        # initialize 3D object points and 2D image points
        calObjPoints = []
        calImgPoints = []

        # define the number of columns for the plot, then calculate number of rows
        num_plot_cols = 5
        num_plot_rows = (len(images) + num_plot_cols - 1) // num_plot_cols

        # create the figure and axes; flatten the axes array for convenvience
        if plot:
            fig, axs = plt.subplots(num_plot_rows, num_plot_cols, figsize=(12, 4*num_plot_rows))
            axs = axs.flatten()

        # loop through the images
        for count,fname in enumerate(images):

            # read image and convert to grayscale if necessary
            orig = cv2.imread(fname)
            if len(orig.shape) == 3:
                img = cv2.cvtColor(orig, cv2.COLOR_RGB2GRAY)
            else:
                img = orig

            # show image
            if plot:
                axs[count].imshow(img / 255.0, cmap="gray")
                axs[count].set_axis_off()
                axs[count].set_title("Image {}".format(count))

            # detect apriltags and report number of detections
            imgpoints, objpoints, tagIDs = self.detect_aprilboard(img,BOARD,at_detector)
            #print("{} {}: {} imgpts, {} objpts".format(count, fname, len(imgpoints),len(objpoints)))
            # print(f"{count} {fname}: {len(imgpoints)} imgpts, {len(objpoints)} objpts")

            # append detections if some are found
            if len(imgpoints) >= N and len(objpoints) >= N:
                total_valid += 1
                # display detected tag centers
                if plot:
                    axs[count].scatter(imgpoints[:,0], imgpoints[:,1], marker='o', color='#ff7f0e')

                # append points detected in all images, (there is only one image now)
                calObjPoints.append(objpoints.astype('float32'))
                calImgPoints.append(imgpoints.astype('float32'))

        return calObjPoints, calImgPoints, total_valid


    def detect_aprilboard(self, img, board, apriltag_detector):
        # Usage:  imgpoints, objpoints, tag_ids = detect_aprilboard(img,board,AT_detector)
        #
        # Input:
        #   image -- grayscale image
        #   board -- at_coarseboard or at_fineboard (list of dictionaries)
        #   AT_detector -- AprilTag Detector parameters
        #
        # Returns:
        #   imgpoints -- Nx2 numpy array of (x,y) image coords
        #   objpoints -- Nx3 numpy array of (X,Y,Z=0) board coordinates (in inches)
        #   tag_ids -- Nx1 list of tag IDs

        imgpoints=[]
        objpoints=[]
        tagIDs=[]

        # detect april tags
        imgtags = apriltag_detector.detect(img,
                                        estimate_tag_pose=False,
                                        camera_params=None,
                                        tag_size=None)

        if len(imgtags):
            # collect image coordinates of tag centers
            # imgpoints = np.vstack([ sub.center for sub in tags ])

            # list of all tag_id's that are in board
            brdtagIDs = [ sub['tag_id'] for sub in board ]

            # list of all detected tag_id's that are in image
            imgtagIDs = [ sub.tag_id for sub in imgtags ]

            # list of all tag_id's that are in both
            tagIDs = list(set(brdtagIDs).intersection(imgtagIDs))

            if len(tagIDs):
                # all board list-elements that contain one of the common tag_ids
                objs=list(filter(lambda tagnum: tagnum['tag_id'] in tagIDs, board))

                # their centers
                objpoints = np.vstack([ sub['center'] for sub in objs ])

                # all image list-elements that contain one of the detected tag_ids
                imgs=list(filter(lambda tagnum: tagnum.tag_id in tagIDs, imgtags))

                # their centers
                imgpoints = np.vstack([ sub.center for sub in imgs ])

        return imgpoints, objpoints, tagIDs

    def in2hom(self, X):
        return np.concatenate([X, np.ones((X.shape[0], 1), dtype=np.float32)], axis=1)

    # Convert from Nxm homogeneous to Nx(m-1) inhomogeneous coordinates
    def hom2in(self, X):
        return X[:, :-1] / X[:, -1:]

    def intersect_ray_plane(self, plane, impts, calMatrix, distCoeffs):
        """
        3D intersection points from back-projected camera rays and a 3D plane

        Args:
        plane:      (a,b,c,d) plane parameters in camera coordinate system
        impts:      Nx2 array of pixel coordinates (xi,yi) to be back-projected
        calMatrix:  K-matrix as returned by cv2.calibrateCamera()
        distCoeffs: optical distortion coefficients as returned by cv2.calibrateCamera()

        Returns:
        intersections:  Nx3 array of 3D intersection-point coordinates (Xi,Yi,Zi)
        """

        # compute normalized coordinates (analogous to K^{-1}*x)
        impts_norm = cv2.undistortPoints(impts, calMatrix, distCoeffs)

        # remove extraneous leading size-1 axis that openCV returns (annoying)
        impts_norm = np.squeeze(impts_norm)

        # back-projections are homogeneous versions of these
        backproj = self.in2hom(impts_norm)

        # ray-plane intersection (use equation derived from Class Session slides)
        intersections_lambda = -plane[3]/np.dot(backproj,plane[:3])
        intersections = np.expand_dims(intersections_lambda,axis=1)*backproj

        return intersections

    def calibrate_camera(self, filepath, img_height, img_width, plot=False):

        calObjPoints, calImgPoints, total_valid = self.detect_fiducials(filepath, plot)

        reprojerr, calMatrix, distCoeffs, calRotations, calTranslations = cv2.calibrateCamera(
            calObjPoints,
            calImgPoints,
            (img_height, img_width),    # uses image H,W to initialize the principal point to (H/2,W/2)
            None,         # no initial guess for the remaining entries of calMatrix
            None,         # initial guesses for distortion coefficients are all 0
            flags = None) # default contstraints (see documentation)

        # Print output, including reprojection error, which is the root mean square (RMS)
        #   re-projection error in pixels. If this value is much greater than 1, it is
        #   likely to be a bad calibration.  Examine the images and detections, and the
        #   options given to cv2.calibrateCamera() to figure out what went wrong.
        np.set_printoptions(precision=5, suppress=True)
        print('RMSE of reprojected points:', reprojerr)
        print('Distortion coefficients:', distCoeffs)

        np.set_printoptions(precision=2, suppress=True)
        print('Intrinsic camera matrix:\n', calMatrix)
        print('Total images used for calibration: ', total_valid)

        extrinsic_matrices = []

        for rotation, translation in zip(calRotations, calTranslations):
            rotation_matrix, _ = cv2.Rodrigues(rotation)
            extrinsic_matrix = np.hstack((rotation_matrix, translation))
            extrinsic_matrix = np.vstack((extrinsic_matrix, [0, 0, 0, 1]))
            extrinsic_matrices.append(extrinsic_matrix)

        for i, extrinsic_matrix in enumerate(extrinsic_matrices):
            print(f"Extrinsic matrix for image {i+1}:\n{extrinsic_matrix}")


        return reprojerr, calMatrix, distCoeffs, calRotations, calTranslations, extrinsic_matrices
