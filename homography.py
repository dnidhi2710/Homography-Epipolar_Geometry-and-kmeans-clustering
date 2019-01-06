UBIT = "dhayanid";
import numpy as np;
np.random.seed(sum([ord(c) for c in UBIT]))
import cv2 as cv
import matplotlib as plt

# Read input images
mount2_color = cv.imread("data/mountain2.jpg")
mount2 = cv.imread("data/mountain2.jpg", cv.IMREAD_GRAYSCALE)
mount1_color = cv.imread("data/mountain1.jpg")
mount1 = cv.imread("data/mountain1.jpg", cv.IMREAD_GRAYSCALE)

# extracting sift features
sift = cv.xfeatures2d.SIFT_create()
key1, desc1 = sift.detectAndCompute(mount1, None)
key2, desc2 = sift.detectAndCompute(mount2, None)

# drawing the extracted key points in input image
task1_sift1 = cv.drawKeypoints(mount1_color, key1, mount1)
task1_sift2 = cv.drawKeypoints(mount2_color, key2, mount2)

# writing the sift image
cv.imwrite("task1_sift1.jpg", task1_sift1)
cv.imwrite("task1_sift2.jpg", task1_sift2)

# feature matching
# flann based matcher is used to get the matches betweent the keypoints of two images with k=2
index = dict(algorithm=0, trees=5)
search = dict()
flann = cv.FlannBasedMatcher(index, search)
feature_match = flann.knnMatch(desc1, desc2, k=2)

key_point_match = []

# this loop gets all the  good matches which satisfies the given condition
for x, y in feature_match:
    if x.distance < 0.75 * y.distance:
        key_point_match.append(x)

# this line randomly selects 10 good matches and return it to the variable
#random_key_point = np.random.choice(key_point_match, 10)

task1_matches_knn = cv.drawMatches(
    mount1_color, key1, mount2_color, key2, key_point_match, mount2)

# random points are drawn and printed in a output file
cv.imwrite("task1_matches_knn.jpg", task1_matches_knn)

# Homography matrix computation

if len(key_point_match) > 10:
    src_pts = np.float32(
        [key1[m.queryIdx].pt for m in key_point_match]).reshape(-1, 1, 2)
    target_pts = np.float32(
        [key2[m.trainIdx].pt for m in key_point_match]).reshape(-1, 1, 2)

    Matrix, mask = cv.findHomography(src_pts, target_pts, cv.RANSAC, 5.0)

    print("Homography matrix")
    print(Matrix)
    # convert the mask to a list
    inliers = mask.ravel().tolist()

# get random 10 values from the inliers/matches
random_inliers = np.random.choice(inliers, 10)

random_key_point2 = np.random.choice(key_point_match, 10)

height, width = mount2.shape

draw_parameters = dict(matchColor=(
    255, 0, 0), singlePointColor=None, matchesMask=random_inliers.tolist(), flags=2)

# draw the matched parameters with respect to both th eimages
task1_matches = cv.drawMatches(
    mount1_color, key1, mount2_color, key2, random_key_point2, None, **draw_parameters)

cv.imwrite("task1_matches.jpg", task1_matches)

M = np.float32([[1,0,517],[0,1,374],[0,0,1]])

dest = cv.warpPerspective(mount1_color,np.matmul(M,Matrix),(mount2_color.shape[1] + mount1_color.shape[1],mount1_color.shape[0] + mount2_color.shape[0]))
dest[374:748, 517:1034] = mount2_color
cv.imwrite("task1_pano.jpg", dest)
