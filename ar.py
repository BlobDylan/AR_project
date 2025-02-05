# ======= imports
import cv2
import numpy as np

# ======= constants
template_path = "Taki3.jpg"
video_path = "video_drawings3.mp4"
output_path = "output_ar.avi"
ratio_test_threshold = 0.7
min_good_matches = 11
template_width_cm = 9.4
template_height_cm = 12.1
history_size = 10
imgpts_history = []

camera_matrix = np.array(
    [
        [3.23807962e03, 0.00000000e00, 2.36625441e03],
        [0.00000000e00, 3.27509469e03, 9.54133944e02],
        [0.00000000e00, 0.00000000e00, 1.00000000e00],
    ]
)
distortion_coefficients = np.array(
    [3.89846165e-01, -3.27481974e00, -2.41473373e-02, -2.53291228e-03, 1.10716359e01]
)

objectPoints = 3.0 * np.array(
    [
        [0, 0, 0],
        [0, 1, 0],
        [1, 1, 0],
        [1, 0, 0],
        [0, 0, -1],
        [0, 1, -1],
        [1, 1, -1],
        [1, 0, -1],
    ]
)


def draw(img, imgpts):
    imgpts = np.int32(imgpts).reshape(-1, 2)

    # draw ground floor in green
    img = cv2.drawContours(img, [imgpts[:4]], -1, (0, 255, 0), -1)

    # draw pillars in blue color
    for i, j in zip(range(4), range(4, 8)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), (255), 3)

    # draw top layer in red color
    img = cv2.drawContours(img, [imgpts[4:]], -1, (0, 0, 255), 3)

    return img


# === template image keypoint and descriptors
template_image = cv2.imread(template_path)
template_gray = cv2.cvtColor(template_image, cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT_create()
kp_template, des_template = sift.detectAndCompute(template_gray, None)

# ===== video input, output and metadata
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))


def is_far_from_history(imgpts, imgpts_history):
    if len(imgpts_history) == 0:
        return False

    imgpts_history_average = get_imgpts_history_average(imgpts_history)
    return np.linalg.norm(imgpts - imgpts_history_average) > 5.0


def get_imgpts_history_average(imgpts_history):
    return np.mean(imgpts_history, axis=0)


# ========== run on all frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # ====== find keypoints matches of frame and template
    # we saw this in the SIFT notebook
    kp_frame, des_frame = sift.detectAndCompute(frame_gray, None)
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des_template, des_frame, k=2)

    # ======== find homography
    # also in SIFT notebook
    good_matches = []
    for m, n in matches:
        if m.distance < ratio_test_threshold * n.distance:
            good_matches.append(m)

    # Find homography and warp overlay if enough good matches
    if len(good_matches) > min_good_matches:
        src_pts = np.float32(
            [kp_template[m.queryIdx].pt for m in good_matches]
        ).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good_matches]).reshape(
            -1, 1, 2
        )

        # Find homography
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # ++++++++ take subset of keypoints that obey homography (both frame and reference)
    # this is at most 3 lines- 2 of which are really the same
    # HINT: the function from above should give you this almost completely
    mask_inliers = mask.ravel() == 1
    src_pts_inliers = src_pts[mask_inliers]
    dst_pts_inliers = dst_pts[mask_inliers]

    # ++++++++ solve PnP to get cam pose (r_vec and t_vec)
    # `cv2.solvePnP` is a function that receives:
    # - xyz of the template in centimeter in camera world (x,3)
    # - uv coordinates (x,2) of frame that corresponds to the xyz triplets
    # - camera K
    # - camera dist_coeffs
    # and outputs the camera pose (r_vec and t_vec) such that the uv is aligned with the xyz.
    #
    # NOTICE: the first input to `cv2.solvePnP` is (x,3) vector of xyz in centimeter- but we have the template keypoints in uv
    # because they are all on the same plane we can assume z=0 and simply rescale each keypoint to the ACTUAL WORLD SIZE IN CM.
    # For this we just need the template width and height in cm.
    #
    # this part is 2 rows
    template_height_px, template_width_px = template_gray.shape[:2]
    scale_x = template_width_cm / template_width_px
    scale_y = template_height_cm / template_height_px

    # Convert template keypoints to real-world coordinates (in cm, z=0)
    src_pts_inliers_flat = src_pts_inliers.reshape(-1, 2)
    object_points_cm = np.zeros((src_pts_inliers_flat.shape[0], 3), dtype=np.float32)
    object_points_cm[:, 0] = src_pts_inliers_flat[:, 0] * scale_x  # x in cm
    object_points_cm[:, 1] = src_pts_inliers_flat[:, 1] * scale_y  # y in cm

    # Prepare image points (corresponding frame keypoints)
    image_points = dst_pts_inliers.reshape(-1, 2)

    # Solve PnP to get camera pose
    success, rvec, tvec = cv2.solvePnP(
        object_points_cm, image_points, camera_matrix, distortion_coefficients
    )

    # ++++++ draw object with r_vec and t_vec on top of rgb frame
    # We saw how to draw cubes in camera calibration. (copy paste)
    # after this works you can replace this with the draw function from the renderer class renderer.draw() (1 line)
    if success:
        # ++++++ draw object with r_vec and t_vec on top of rgb frame
        # Project 3D cube points to 2D image points
        adjusted_objectPoints = objectPoints.copy()
        adjusted_objectPoints[:, 0] += template_width_cm - 3.0
        adjusted_objectPoints[:, 1] += template_height_cm - 3.0

        # Project adjusted cube points
        imgpts, _ = cv2.projectPoints(
            adjusted_objectPoints, rvec, tvec, camera_matrix, distortion_coefficients
        )

        frame = draw(frame, imgpts)
        if is_far_from_history(imgpts, imgpts_history):
            imgpts = get_imgpts_history_average(imgpts_history)
        else:
            if len(imgpts_history) == history_size:
                imgpts_history.pop(0)
            imgpts_history.append(imgpts)

    # Write frame to output video
    out.write(frame)

    # Resize for visualization
    resized_frame = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))
    cv2.imshow("frame", resized_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ======== end all
cap.release()
out.release()  # Release the video writer
cv2.destroyAllWindows()
