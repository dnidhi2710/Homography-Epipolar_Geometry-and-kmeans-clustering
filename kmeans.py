UBIT = "dhayanid"
import numpy as np
np.random.seed(sum([ord(c) for c in UBIT]))
import math
from matplotlib import pyplot as plt
import cv2 as cv


def computeEuclideanDist(a, b):
    x1 = a[0]
    y1 = a[1]
    x2 = b[0]
    y2 = b[1]
    distance = math.sqrt(((x1-x2)**2)+((y1-y2)**2))
    return round(distance, 3)


# this function returns the classification vector
def computedistanceAndClassify(x,red,green,blue):
    class_vector = []
    for i in range(len(x)):
        lista = []
        red_dist = computeEuclideanDist(x[i], red)
        green_dist = computeEuclideanDist(x[i], green)
        blue_dist = computeEuclideanDist(x[i], blue)
        lista.append(red_dist)
        lista.append(green_dist)
        lista.append(blue_dist)
        min_value = min(lista)

        if min_value == red_dist:
            class_vector.append('r')
        elif min_value == blue_dist:
            class_vector.append('b')
        elif min_value == green_dist:
            class_vector.append('g')

    return class_vector

# calculate the average x and y value ofall the points in the cluster
def computeNewCentroid(x,code):
    sum_x = 0
    sum_y = 0
    count = 0
    for a in x:
        if a[2] == code:
            count += 1
            sum_x += a[0]
            sum_y += a[1]
    avg_x = sum_x/count
    avg_y = sum_y/count

    return [avg_x,avg_y]

# center locations
red = [6.2, 3.2]
green = [6.6, 3.7]
blue = [6.5, 3.0]

x = [[5.9, 3.2],
     [4.6, 2.9],
     [6.2, 2.8],
     [4.7, 3.2],
     [5.5, 4.2],
     [5.0, 3.0],
     [4.9, 3.1],
     [6.7, 3.1],
     [5.1, 3.8],
     [6.0, 3.0]]

cvectr = computedistanceAndClassify(x,red,green,blue)
print("classification vector")
print(cvectr)

for i in range(len(cvectr)):
    x[i].append(cvectr[i])

plt.figure()
for point in x:
    plt.scatter(point[0], point[1], c=point[2], marker='^')

plt.scatter(6.2, 3.2, c='r', marker='o')
plt.scatter(6.6, 3.7, c='g', marker='o')
plt.scatter(6.5, 3.0, c='b', marker='o')
plt.savefig("task3_iter1_a.jpg")


red_new_centroid = computeNewCentroid(x,'r')
blue_new_centroid = computeNewCentroid(x,'b')
green_new_centroid = computeNewCentroid(x,'g')

plt.figure()  
for point in x:
    plt.scatter(point[0], point[1], c=point[2], marker='^')

plt.scatter(red_new_centroid[0],red_new_centroid[1], c='r', marker='o')
plt.scatter(blue_new_centroid[0],blue_new_centroid[1], c='b', marker='o')
plt.scatter(green_new_centroid[0],green_new_centroid[1], c='g', marker='o')
plt.savefig("task3_iter1_b.jpg")

plt.figure()
cvectr1 = computedistanceAndClassify(x,red_new_centroid,green_new_centroid,blue_new_centroid)
print("classification vector after iteration 1")
print(cvectr1)

x1= []
for i in range(len(cvectr1)):
    arr = []
    arr.append(x[i][0])
    arr.append(x[i][1])
    arr.append(cvectr1[i])
    x1.append(arr)

plt.figure()
for point in x1:
    plt.scatter(point[0], point[1], c=point[2], marker='^')

plt.scatter(red_new_centroid[0],red_new_centroid[1], c='r', marker='o')
plt.scatter(blue_new_centroid[0],blue_new_centroid[1], c='b', marker='o')
plt.scatter(green_new_centroid[0],green_new_centroid[1], c='g', marker='o')
plt.savefig("task3_iter2_a.jpg")

red_new_centroid_1 = computeNewCentroid(x1,'r')
blue_new_centroid_1 = computeNewCentroid(x1,'b')
green_new_centroid_1 = computeNewCentroid(x1,'g')

plt.figure()  
for point in x1:
    plt.scatter(point[0], point[1], c=point[2], marker='^')

plt.scatter(red_new_centroid_1[0],red_new_centroid_1[1], c='r', marker='o')
plt.scatter(blue_new_centroid_1[0],blue_new_centroid_1[1], c='b', marker='o')
plt.scatter(green_new_centroid_1[0],green_new_centroid_1[1], c='g', marker='o')
plt.savefig("task3_iter2_b.jpg")

img = cv.imread("data/baboon.jpg")
print(img.shape)
# compute euclidean distance of 3d image
def computeEuclideanDist3d(a, b):
    x1 = a[0]
    y1 = a[1]
    z1 = a[2]
    x2 = b[0]
    y2 = b[1]
    z2 = b[2]
    distance = math.sqrt(((x1-x2)**2)+((y1-y2)**2)+((z1-z2)**2))
    return round(distance, 1)

#find the minimum value of the list and returns the min value and the centroid
def find_centroid(a):
    temp_list = []
    for i in range(len(a)):
        temp_list.append(a[i][0])
    min_value = min(temp_list)
    
    for i in range(len(a)):
        if min_value == a[i][0]:
            centroid = a[i][1]
    return centroid
# this function returns the classification vector
def computedistanceAndClassifyGeneric(x,centroids_Array):
    class_vector = []
    lista = []
    for center in centroids_Array:
        lista.append([computeEuclideanDist3d(x,center),center[3]])
            
    centroid = find_centroid(lista)
    if len(x)<4:
        x.append(centroid)
    else:
        x[3] = centroid
            
    return x
# calculate the average x and y value ofall the points in the cluster
def computeNewCentroidGeneric(x,code):
    sum_x = 0
    sum_y = 0
    sum_z = 0
    count = 0
    for row in range(len(x)):
        if x[row][3] == code:
            count += 1
            sum_x += x[row][0]
            sum_y += x[row][1]
            sum_z += x[row][2]
    avg_x = round(sum_x/count, 1)
    avg_y = round(sum_y/count, 1)
    avg_z = round(sum_z/count, 1)

    return [avg_x,avg_y,avg_z,code]
def getcentroid_value(img_coord,final_centroid):
    for i in final_centroid:
        if img_coord[3] == i[3]:
            arr=[]
            arr.append(math.floor(i[0]))
            arr.append(math.floor(i[1]))
            arr.append(math.floor(i[2]))
            return arr
                    
def adjust_centroid(centroid_array,img_wvector):
    
    new_centroid =[]
    for x in centroid_array:
        new_centroid.append(computeNewCentroidGeneric(img_wvector,x[3]))  
    return new_centroid

def classify(centroid_array,img_wvector):
    for row in range(len(img_wvector)):
        img_wvector[row]= computedistanceAndClassifyGeneric(img_wvector[row],centroid_array)
        
    new_centroid = adjust_centroid(centroid_array,img_wvector)
    print(new_centroid)
    if np.array_equal(new_centroid,centroid_array) != True:
        centroid_array = new_centroid
        classify(centroid_array,img_wvector)
    else:
        print("centroid computed")
        
    return centroid_array,img_wvector


def trainKmeans(img,k):
    centroid_array = []
    img_w = img
    img_wvector = []

  
    for row in range(len(img_w)):
        for col in range(len(img_w[0])):
            arr = []
            arr.append(img_w[row][col][0])
            arr.append(img_w[row][col][1])
            arr.append(img_w[row][col][2])
            #arr.append(random.randint(0,k-1))
            img_wvector.append(arr)

    rand_index = np.random.choice(len(img_wvector), k)
    
    for i in range(k):
        arr = []
        j = rand_index[i]
        for value in img_wvector[j]:
            arr.append(value)
        arr.append(i)
        centroid_array.append(arr)
  
            
    final_centroid,final_wvector = classify(centroid_array,img_wvector)

    count =0 
    for row in range(len(img_w)):
        for col in range(len(img_w[0])):
            img_w[row][col] = getcentroid_value(final_wvector[count],final_centroid)
            count +=1

    return img_w

task3_baboon_3=trainKmeans(img,3)
cv.imwrite("task3_baboon_3.jpg",task3_baboon_3)

task3_baboon_5=trainKmeans(img,5)
cv.imwrite("task3_baboon_5.jpg",task3_baboon_5)

task3_baboon_10=trainKmeans(img,10)
cv.imwrite("task3_baboon_10.jpg",task3_baboon_10)

task3_baboon_20=trainKmeans(img,20)
cv.imwrite("task3_baboon_20.jpg",task3_baboon_20)


