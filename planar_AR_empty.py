# ======= imports
import cv2
import numpy as np

# ======= constants
template_path = "Taki.jpg"
video_path = "video_long.mp4"
output_path = "output2.avi"
ratio_test_threshold = 0.7
min_good_matches = 12

template_width_cm = 9.5  # Template real-world width (cm)
template_height_cm = 11.5  # Template real-world height (cm)

# Camera parameters
camera_matrix = np.array(
    [
        [3.23808077e03, 0.00000000e00, 2.36625417e03],
        [0.00000000e00, 3.27509568e03, 9.54132680e02],
        [0.00000000e00, 0.00000000e00, 1.00000000e00],
    ],
    dtype=np.float32,
)

dist_coeffs = np.array(
    [3.89845128e-01, -3.27480159e00, -2.41473881e-02, -2.53295027e-03, 1.10715584e01],
    dtype=np.float32,
)

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

# ========== run on all frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ====== find keypoints matches of frame and template
    kp_frame, des_frame = sift.detectAndCompute(frame_gray, None)
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des_template, des_frame, k=2)

    # Apply ratio test
    good_matches = [
        m for m, n in matches if m.distance < ratio_test_threshold * n.distance
    ]

    # ======== find homography
    if len(good_matches) > min_good_matches:
        src_pts = np.float32(
            [kp_template[m.queryIdx].pt for m in good_matches]
        ).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good_matches]).reshape(
            -1, 1, 2
        )

        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        if H is not None:
            # ++++++++ take subset of keypoints that obey homography
            src_pts_inliers = src_pts[mask.ravel() == 1]
            dst_pts_inliers = dst_pts[mask.ravel() == 1]

            # ++++++++ solve PnP to get cam pose (r_vec and t_vec)
            # Convert template keypoints to real-world 3D points (z=0 plane, scaled to cm)
            object_points = np.zeros((len(src_pts_inliers), 3), dtype=np.float32)
            object_points[:, :2] = src_pts_inliers[:, 0] * [
                template_width_cm / template_gray.shape[1],
                template_height_cm / template_gray.shape[0],
            ]

            # Solve PnP
            ret, r_vec, t_vec = cv2.solvePnP(
                object_points, dst_pts_inliers, camera_matrix, dist_coeffs
            )

            if ret:
                # ++++++ draw object with r_vec and t_vec on top of rgb frame
                cube_points = np.array(
                    [
                        [0, 0, 0],  # Bottom square
                        [template_width_cm, 0, 0],
                        [template_width_cm, template_height_cm, 0],
                        [0, template_height_cm, 0],
                        [0, 0, -template_width_cm],  # Top square
                        [template_width_cm, 0, -template_width_cm],
                        [template_width_cm, template_height_cm, -template_width_cm],
                        [0, template_height_cm, -template_width_cm],
                    ],
                    dtype=np.float32,
                )

                # Project cube points to 2D
                projected_points, _ = cv2.projectPoints(
                    cube_points, r_vec, t_vec, camera_matrix, dist_coeffs
                )
                projected_points = projected_points.astype(int).reshape(-1, 2)

                # Draw cube edges
                for i, j in zip(range(4), [1, 2, 3, 0]):  # Bottom square
                    cv2.line(
                        frame,
                        tuple(projected_points[i]),
                        tuple(projected_points[j]),
                        (255, 0, 0),
                        2,
                    )
                for i, j in zip(range(4, 8), [5, 6, 7, 4]):  # Top square
                    cv2.line(
                        frame,
                        tuple(projected_points[i]),
                        tuple(projected_points[j]),
                        (0, 255, 0),
                        2,
                    )
                for i in range(4):  # Vertical edges
                    cv2.line(
                        frame,
                        tuple(projected_points[i]),
                        tuple(projected_points[i + 4]),
                        (0, 0, 255),
                        2,
                    )

    # =========== plot and save frame
    out.write(frame)
    resized_frame = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))
    cv2.imshow("frame", resized_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ======== end all
cap.release()
out.release()
cv2.destroyAllWindows()
