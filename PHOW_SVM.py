import cv2
import numpy as np
import glob
import math
import os
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score


# configuration variables
BASE_PATH = "D:\\path\\to\\\\project\\root\\"
TRAININNG_DATASET_PATH = BASE_PATH+"training\\"
TESTING_DATASET_PATH = BASE_PATH+"testing\\"
SEED = 42 # random seed value
TEST_RATIO = 0.07 # test ratio
SVC_C = 0.0005379 # C constant for SVM classifier
k = 200 # for K-Means
DSIFT_STEP_SIZE = 12

# ---------------------------------------------------------------------------------

# function to load dataset using the training dataset root path and class names
def load_train_dataset(path, classes):
    """Loads a dataset with one level nested class directories. Returns data (images) and labels (associated classes)."""
    data = []
    labels = []
    for id, class_name in classes.items():
        img_path_class = glob.glob(path + class_name + '\\*.jpg')
        labels.extend([id]*len(img_path_class))
        for filename in img_path_class:
            data.append(cv2.imread(filename, 0))
    return data, labels

def load_test_data(path):
    """Loads a dataset with images in the given path. Returns the filenames and images data."""
    data = []
    img_path_class = glob.glob(path + '\\*.jpg')
    for filename in img_path_class:
        data.append(cv2.imread(filename, 0))
    return [os.path.basename(x) for x in img_path_class], data

def compute_denseSIFT(img, dsift_step_size):
    """Compute dense SIFT descriptors for given images and step size"""
    sift = cv2.xfeatures2d.SIFT_create()
    keypoints = [cv2.KeyPoint(x, y, dsift_step_size)
            for y in range(0, img.shape[0], dsift_step_size)
                for x in range(0, img.shape[1], dsift_step_size)]

    descriptors = sift.compute(img, keypoints)[1]
    
    return descriptors

def compute_denseSIFT_all(images):
    """Compute dense SIFT for all images"""
    ftrs = []
    for i in range(0, len(images)):
        ds = compute_denseSIFT(images[i], DSIFT_STEP_SIZE)
        ftrs.append(ds)
    return ftrs

def getImageFeaturesPHOW(L, img, kmeans, k):
    """Get Pyramid Histogram of dense SIFT for a single image using the clustered mean of SIFT"""
    height = img.shape[0]
    width = img.shape[1] 
    hg = []
    for l in range(L+1):
        ht_step = math.floor(height/(2**l))
        wt_step = math.floor(width/(2**l))
        x = 0
        y = 0
        for i in range(1,2**l + 1):
            x = 0
            for j in range(1, 2**l + 1):                
                desc = compute_denseSIFT(img[y:y+ht_step, x:x+wt_step], DSIFT_STEP_SIZE)
                km = kmeans.predict(desc)
                hgram = np.bincount(km, minlength=k).reshape(1,-1).ravel()
                weight = 2**(l-L)
                hg.append(weight*hgram)
                x = x + wt_step
            y = y + ht_step
    # get combined histogram
    histogram = np.array(hg).ravel()
    # normalize histogram
    histogram -= np.mean(histogram)
    histogram /= np.std(histogram)
    return histogram

def clusterFeatures(features, k):
    """Cluster features using KMeans"""
    kmeans = KMeans(n_clusters=k, random_state=0).fit(features)
    return kmeans

def getPHOW(L, images, kmeans, k):
    """Get pyramid histogram of dense SIFT for all images"""
    phow = []
    for i in range(len(images)):        
        hist = getImageFeaturesPHOW(L, images[i], kmeans, k)        
        phow.append(hist)
    return np.array(phow)

#----------------------------------------------------------------------------------

# Create the enumerated dictionary of classes
classes_dict = [dname[len(TRAININNG_DATASET_PATH):] for dname in glob.glob(TRAININNG_DATASET_PATH+'*')]
classes_dict = dict(zip(range(0, len(classes_dict)), classes_dict))
print(classes_dict)

# load training dataset
train_data, train_label = load_train_dataset(TRAININNG_DATASET_PATH, classes_dict)
print("Loaded training dataset = ", len(train_label))

# create a training and validation split
train_data, val_data, train_label, val_label = train_test_split(train_data, train_label, test_size=TEST_RATIO, stratify=train_label, random_state=SEED)

print("Train = ", len(train_data), len(train_label))
print("Test = ", len(val_data), len(val_label))

x_train = compute_denseSIFT_all(train_data)

# prepare for clustering - array of full training set features
full_train_dsifts = []
for i in range(len(x_train)):
    for j in range(x_train[i].shape[0]):
        full_train_dsifts.append(x_train[i][j,:])

full_train_dsifts = np.array(full_train_dsifts)

# do kmeans clustering
kmeans = clusterFeatures(full_train_dsifts, k)

# create spatially pooled histogram of dense SIFTs for train and validation data
train_phows = getPHOW(5, train_data, kmeans, k)
val_phows = getPHOW(5, val_data, kmeans, k)

# train classifier
clf = LinearSVC(random_state=SEED, C=SVC_C)
clf.fit(train_phows, train_label)
predict = clf.predict(val_phows)
print("Validation accuracy:", accuracy_score(val_label, predict)*100, "%")

# Generating run 3 test output
real_test_filenames, real_test_data = load_test_data(TESTING_DATASET_PATH)
print("Loaded run 3 test data: ", len(real_test_data))
real_test_phows = getPHOW(5, real_test_data, kmeans, k)

# predict classes for test set
real_predict = clf.predict(real_test_phows)

# create prediction file with test data
f = open(BASE_PATH+"run3.txt", "w")
for i in range(len(real_predict)):
    f.write(real_test_filenames[i]+" "+classes_dict[real_predict[i]]+"\n")
f.close()
