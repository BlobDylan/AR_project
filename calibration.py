# camera_matrix = np.array(
#     [
#         [3.23807962e03, 0.00000000e00, 2.36625441e03],
#         [0.00000000e00, 3.27509469e03, 9.54133944e02],
#         [0.00000000e00, 0.00000000e00, 1.00000000e00],
#     ]
# )
# distortion_coefficients = np.array(
#     [3.89846165e-01, -3.27481974e00, -2.41473373e-02, -2.53291228e-03, 1.10716359e01]
# )
# %%
import cv2
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

# %%
square_size = 2.4  # cm
img_mask = "calibration_images/*.PNG"
pattern_size = (9, 6)  # corners
figsize = (20, 20)

img_names = glob(img_mask)
num_imgs = len(img_names)

pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
pattern_points *= square_size

obj_points = []
img_points = []
h, w = cv2.imread(img_names[0]).shape[:2]

# %%
plt.figure(figsize=figsize)

for i, fn in enumerate(img_names):
    print("processing %s... " % fn)
    imgBGR = cv2.imread(fn)

    if imgBGR is None:
        print("Failed to load", fn)
        continue

    imgRGB = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2RGB)
    img = cv2.cvtColor(imgRGB, cv2.COLOR_RGB2GRAY)

    assert w == img.shape[1] and h == img.shape[0], (
        f"size: {img.shape[1]} x {img.shape[0]}"
    )
    found, corners = cv2.findChessboardCorners(img, pattern_size)
    # if you want to better improve the accuracy... cv2.findChessboardCorners already uses cv2.cornerSubPix
    # if found:
    #     term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1)
    #     cv2.cornerSubPix(img, corners, (5, 5), (-1, -1), term)

    if not found:
        print("chessboard not found")
        continue

    if i < 12:
        img_w_corners = cv2.drawChessboardCorners(imgRGB, pattern_size, corners, found)
        plt.subplot(4, 3, i + 1)
        plt.imshow(img_w_corners)

    print(f"{fn}... OK")
    img_points.append(corners.reshape(-1, 2))
    obj_points.append(pattern_points)


plt.show()

# %%
rms, camera_matrix, dist_coefs, _rvecs, _tvecs = cv2.calibrateCamera(
    obj_points, img_points, (w, h), None, None
)

print("\nRMS:", rms)
print("camera matrix:\n", camera_matrix)
print("distortion coefficients: ", dist_coefs.ravel())

# %%
plt.figure(figsize=figsize)
for i, fn in enumerate(img_names):
    imgBGR = cv2.imread(fn)
    imgRGB = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2RGB)

    dst = cv2.undistort(imgRGB, camera_matrix, dist_coefs)

    if i < 12:
        plt.subplot(4, 3, i + 1)
        plt.imshow(dst)

plt.show()
print("Done")

# %%
objectPoints = (
    3
    * square_size
    * np.array(
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


plt.figure(figsize=figsize)
for i, fn in enumerate(img_names):
    imgBGR = cv2.imread(fn)
    imgRGB = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2RGB)

    imgpts = cv2.projectPoints(
        objectPoints, _rvecs[i], _tvecs[i], camera_matrix, dist_coefs
    )[0]
    drawn_image = draw(imgRGB, imgpts)

    if i < 12:
        plt.subplot(4, 3, i + 1)
        plt.imshow(drawn_image)

plt.show()

# %%
