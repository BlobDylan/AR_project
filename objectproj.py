import cv2
import numpy as np
import math
import os
from object_loader import *

# Minimum number of matches that have to be found
# to consider the recognition valid
DEFAULT_COLOR = (0, 0, 0)

# ======= constants
template_path = "Taki3.jpg"
video_path = "video_drawings3.mp4"
output_path = "output_ar.avi"
object_file_path = "yoshi.obj"
ratio_test_threshold = 0.7
min_good_matches = 11
template_width_cm = 9.4
template_height_cm = 12.1
history_size = 10
imgpts_history = []

camera_parameters = np.array(
    [
        [3.23807962e03, 0.00000000e00, 2.36625441e03],
        [0.00000000e00, 3.27509469e03, 9.54133944e02],
        [0.00000000e00, 0.00000000e00, 1.00000000e00],
    ]
)

distortion_coefficients = np.array(
    [3.89846165e-01, -3.27481974e00, -2.41473373e-02, -2.53291228e-03, 1.10716359e01]
)


def main():
    """
    This functions loads the target surface image,
    """
    homography = None
    # matrix of camera parameters (made up but works quite well for me)

    # create ORB keypoint detector
    orb = cv2.SIFT.create()
    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
    # load the reference surface that will be searched in the video stream
    template = cv2.imread(template_path)
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    # Compute model keypoints and its descriptors
    kp_model, des_model = orb.detectAndCompute(template, None)
    # Load 3D model from OBJ file
    obj = OBJ(object_file_path, swapyz=True, scale=50.0)
    # init video capture
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        # read the current frame
        ret, frame = cap.read()
        if not ret:
            break
        # find and draw the keypoints of the frame
        kp_frame, des_frame = orb.detectAndCompute(frame, None)
        # match frame descriptors with model descriptors
        matches = bf.match(des_model, des_frame)
        # sort them in the order of their distance
        # the lower the distance, the better the match

        matches = sorted(matches, key=lambda x: x.distance)

        # compute Homography if enough matches are found
        if len(matches) > min_good_matches:
            # differenciate between source points and destination points
            src_pts = np.float32([kp_model[m.queryIdx].pt for m in matches]).reshape(
                -1, 1, 2
            )
            dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in matches]).reshape(
                -1, 1, 2
            )
            # compute Homography
            homography, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            # if a valid homography matrix was found render cube on model plane
            if homography is not None:
                try:
                    # obtain 3D projection matrix from homography matrix and camera parameters
                    projection = projection_matrix(camera_parameters, homography)
                    # project cube or model
                    frame = render(frame, obj, projection, template, False)
                    # frame = render(frame, model, projection)
                except:
                    pass
            # show result and save
            out.write(frame)

            frame = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))

            cv2.imshow("frame", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    return 0


def render(img, obj, projection, template, color=False):
    """
    Render a loaded obj model into the current video frame
    """
    vertices = obj.vertices
    scale_matrix = np.eye(3) * 3
    h, w = template.shape

    for face in obj.faces:
        face_vertices = face[0]
        points = np.array([vertices[vertex - 1] for vertex in face_vertices])
        points = np.dot(points, scale_matrix)
        # render model in the middle of the reference surface. To do so,
        # model points must be displaced
        points = np.array([[p[0] + w / 2, p[1] + h / 2, p[2]] for p in points])
        dst = cv2.perspectiveTransform(points.reshape(-1, 1, 3), projection)
        imgpts = np.int32(dst)
        cv2.fillConvexPoly(img, imgpts, DEFAULT_COLOR)

    return img


def projection_matrix(camera_parameters, homography):
    """
    From the camera calibration matrix and the estimated homography
    compute the 3D projection matrix
    """
    # Compute rotation along the x and y axis as well as the translation
    homography = homography * (-1)
    rot_and_transl = np.dot(np.linalg.inv(camera_parameters), homography)
    col_1 = rot_and_transl[:, 0]
    col_2 = rot_and_transl[:, 1]
    col_3 = rot_and_transl[:, 2]
    # normalise vectors
    l = math.sqrt(np.linalg.norm(col_1, 2) * np.linalg.norm(col_2, 2))
    rot_1 = col_1 / l
    rot_2 = col_2 / l
    translation = col_3 / l
    # compute the orthonormal basis
    c = rot_1 + rot_2
    p = np.cross(rot_1, rot_2)
    d = np.cross(c, p)
    rot_1 = np.dot(
        c / np.linalg.norm(c, 2) + d / np.linalg.norm(d, 2), 1 / math.sqrt(2)
    )
    rot_2 = np.dot(
        c / np.linalg.norm(c, 2) - d / np.linalg.norm(d, 2), 1 / math.sqrt(2)
    )
    rot_3 = np.cross(rot_1, rot_2)
    # finally, compute the 3D projection matrix from the model to the current frame
    projection = np.stack((rot_1, rot_2, rot_3, translation)).T
    return np.dot(camera_parameters, projection)


if __name__ == "__main__":
    main()
