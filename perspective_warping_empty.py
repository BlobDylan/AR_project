# ======= imports
import cv2
import numpy as np

# ======= constants
template_path = "Taki.jpg"
video_path = "video_long.mp4"
output_path = "output.avi"
ratio_test_threshold = 0.7
min_good_matches = 12

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

# Load overlay image and resize to template dimensions once
overlay_image = cv2.imread(template_path)
h, w = template_gray.shape
overlay_resized = cv2.resize(overlay_image, (w, h))

# ========== run on all frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Find keypoints matches of frame and template
    kp_frame, des_frame = sift.detectAndCompute(frame_gray, None)
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des_template, des_frame, k=2)

    # Apply ratio test
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

        if H is not None:
            # Draw detected template region
            template_corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(
                -1, 1, 2
            )
            frame_corners = cv2.perspectiveTransform(template_corners, H)
            frame = cv2.polylines(
                frame, [np.int32(frame_corners)], True, (0, 255, 0), 3
            )

            # Warp overlay onto frame
            warped_overlay = cv2.warpPerspective(
                overlay_resized, H, (frame.shape[1], frame.shape[0])
            )
            mask_overlay = cv2.warpPerspective(
                np.ones_like(overlay_resized, dtype=np.uint8),
                H,
                (frame.shape[1], frame.shape[0]),
            )
            frame = (
                cv2.bitwise_and(frame, cv2.bitwise_not(mask_overlay)) + warped_overlay
            )

    # Write original frame to output video
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
