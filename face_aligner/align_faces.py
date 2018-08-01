# import the necessary packages
import os

import cv2
import dlib
import imutils
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb


# construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-p", "--shape-predictor", required=True,
#                 help="path to facial landmark predictor")
# ap.add_argument("-i", "--image", required=True,
#                 help="path to input image")
# args = vars(ap.parse_args())


def align_faces(in_dir, out_dir):
    # initialize dlib's face detector (HOG-based) and then create
    # the facial landmark predictor and the face aligner
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("face_aligner/shape_predictor_68_face_landmarks.dat")
    fa = FaceAligner(predictor, desiredFaceWidth=500, desiredFaceHeight=300)

    # load the input image, resize it, and convert it to grayscale
    # image = cv2.imread(args["image"])
    processed_count = 0
    file_count = len(os.listdir(in_dir))

    for filename in os.listdir(in_dir):
        image = cv2.imread(in_dir + filename)
        image = imutils.resize(image, width=800)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        for x in range(0, 4):

            # show the original input image and detect faces in the grayscale
            # image
            rects = detector(gray, 2)
            if len(rects) > 0:
                break

            image = imutils.rotate(image, 90)
            gray = imutils.rotate(gray, 90)

        processed_count = processed_count + 1
        print('Aligned frames: ' + str(processed_count) + '/' + str(file_count))
        # loop over the face detections
        for rect in rects:
            # extract the ROI of the *original* face, then align the face
            # using facial landmarks
            (x, y, w, h) = rect_to_bb(rect)
            faceOrig = imutils.resize(image[y:y + h, x:x + w], width=100)
            faceAligned = fa.align(image, gray, rect)

            # display the output images
            # cv2.imshow("Original", faceOrig)
            # cv2.imshow("Aligned", faceAligned)
            # cv2.waitKey(0)

            # save aligned image
            cv2.imwrite(out_dir + filename, faceAligned)
