UBIT = "dhayanid";
import numpy as np;
np.random.seed(sum([ord(c) for c in UBIT]))
import cv2 as cv
from matplotlib import pyplot as plt

def get_Max(matrix):
    largest_num = matrix[0][0]
    for row_idx, row in enumerate(matrix):
        for col_idx, num in enumerate(row):
            if num > largest_num:
                largest_num = num

    return largest_num


def Normalise_Matrix(Matrix):
    MAX_VALUE = get_Max(Matrix)
    row = len(Matrix)
    col = len(Matrix[0])
    for i in range(row):
        for j in range(col):
            Matrix[i][j] = (Matrix[i][j]/MAX_VALUE)*255

    return Matrix

#this funciton returns the 10 elements out of the given array
def gettenelem(random_array):
    count = 0
    random_ten = []
    for i in range(len(random_array)):
        count+=1
        if count<10:
            random_ten.append(random_array[i])

    return np.array(random_ten)

#this function draws the set of lines in the given color image
#color_count is used to show the same color for the same point pair in left and right image
def draw_epiline(img1,lines,pts1):
    row,col,d = img1.shape
    color_count = 0
    for row,pt1 in zip(lines,pts1):
            color_count += 17
            B=color_count
            G=color_count + 100
            R=color_count +31
            color_list= [B,G,R]
            color = tuple(color_list)
            x0,y0 = map(int, [0, -row[2]/row[1] ])
            x1,y1 = map(int, [col, -(row[2]+row[0]*col)/row[1] ])
            img1 = cv.line(img1, (x0,y0), (x1,y1), color,1)
            img1 = cv.circle(img1,tuple(pt1.flatten()),5,color,-1)
    return img1

# Read input images
left_img_color = cv.imread("data/tsucuba_left.png")
left_img = cv.imread("data/tsucuba_left.png", cv.IMREAD_GRAYSCALE)
right_img_color = cv.imread("data/tsucuba_right.png")
right_img = cv.imread("data/tsucuba_right.png", cv.IMREAD_GRAYSCALE)
# extracting sift features
sift = cv.xfeatures2d.SIFT_create()
key1, desc1 = sift.detectAndCompute(left_img, None)
key2, desc2 = sift.detectAndCompute(right_img, None)

# drawing the extracted key points in input image
task2_sift1 = cv.drawKeypoints(left_img_color, key1, left_img)
task2_sift2 = cv.drawKeypoints(right_img_color, key2, right_img)

# writing the sift image
cv.imwrite("task2_sift1.jpg", task2_sift1)
cv.imwrite("task2_sift2.jpg", task2_sift2)

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

task2_matches_knn = cv.drawMatches(
    left_img_color, key1, right_img_color, key2, key_point_match, right_img)

# random points are drawn and printed in a output file
cv.imwrite("task2_matches_knn.jpg", task2_matches_knn)

src_pts = np.float32([key1[m.queryIdx].pt for m in key_point_match]).reshape(-1,1,2)
target_pts = np.float32([key2[m.trainIdx].pt for m in key_point_match]).reshape(-1,1,2)

#fundamental matrix computation
#pts1 = np.int32(src_pts)
#pts2 = np.int32(target_pts)
pts1 = src_pts
pts2 = target_pts
F_Matrix, mask = cv.findFundamentalMat(pts1, pts2, cv.RANSAC,5.0)

print("Fundamental Matrix")
print(F_Matrix)

# We select only inlier points
pts1 = pts1[mask.ravel()==1]
pts2 = pts2[mask.ravel()==1]

#get random 10 points in the list and compute the line for each points in pts2
# draw the computed epilines over the left image
random_ten_pts2 = gettenelem(pts2)
lines1 = cv.computeCorrespondEpilines(np.array(random_ten_pts2), 2,F_Matrix).reshape(-1,3)
epi_left = draw_epiline(left_img_color,lines1,pts1)
#get random 10 points in the list and compute the line for each points pts1
# draw the computed epilines over the right image
random_ten_pts1 = gettenelem(pts1)
lines2 = cv.computeCorrespondEpilines(np.array(random_ten_pts1), 1,F_Matrix).reshape(-1,3)
epi_right = draw_epiline(right_img_color,lines2,pts2)

cv.imwrite("task2_epi_left.jpg",epi_left)
cv.imwrite("task2_epi_right.jpg",epi_right)

stereo = cv.StereoBM_create(numDisparities=80, blockSize=25)
disparity = stereo.compute(left_img,right_img)
disparity = Normalise_Matrix(disparity)
cv.imwrite("task2_disparity.jpg",disparity)
