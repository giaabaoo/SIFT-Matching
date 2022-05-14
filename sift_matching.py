import cv2
import numpy as np
from scipy.spatial import distance_matrix
import pdb
import os
from pathlib import Path

# if you can not import library, type the following command
# pip install scipy opencv-python numpy

def matchsift(D1, D2, alpha=0.07):
    # Compute Pairwise Distance
    distance = distance_matrix(D1, D2) # Fast Pairwise Distance

    matches = []
    dists = []
    for i in range(len(D1)):
        # Find the first closest D2 to D1[i] and its index
        d1, idx1 = min(distance[i]), np.argmin(distance[i]) # Your Code Here
        # Find the first closest D2 to D1[i] and its index
        d2, idx2 = sum(D1[i]), i # Your Code Here

        if ((d1 / d2) <= alpha): # Heuristic Checking
            matches.append([i, idx1])
            dists.append(d1)

    return matches, dists

def find_matching(img1, img2, test_id):    
    image1 = cv2.imread(img1)
    image2 = cv2.imread(img2)
    
    try:
        image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))
    except:
        return

    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    kp1, D1 = sift.detectAndCompute(gray1,None)
    kp2, D2 = sift.detectAndCompute(gray2,None)

    print("Found %d keypoints in image 1" % len(D1))
    print("Found %d keypoints in image 2" % len(D2))

    matches, dists = matchsift(D1, D2)

    return matches, [kp1, kp2], image1, image2

if __name__ == "__main__":
    objet_image_path = "./Code/objets"
    test_image_path = "./Code/images_test"
    Path("./Q3").mkdir(parents=True, exist_ok=True)
            
    # Best matching
    for img2 in os.listdir(test_image_path):
        if ".jpg" in img2: 
            img2_path = os.path.join(test_image_path, img2)

            matching_list = []
            kp_list = []
            img_list = []
            for img1 in os.listdir(objet_image_path):
                img1_path = os.path.join(objet_image_path, img1)
                print("Matching image {} and image {}".format(img1, img2))
                one_matches, kps, image1, image2 = find_matching(img1_path, img2_path, img2.replace(".jpg", ""))

                matching_list.append(one_matches)
                kp_list.append(kps)
                img_list.append(image1)
            # pdb.set_trace()
            matching_length_list = [len(i) for i in matching_list]
            matches = matching_list[np.argmax(matching_length_list)]
            kp1, kp2 = kp_list[np.argmax(matching_length_list)]

            image1 = img_list[np.argmax(matching_length_list)]
            image2 = cv2.imread(img2_path)
            try:
                image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))
            except:
                continue
            image = np.concatenate([image1, image2], 1)
            for i, j in matches:
                x1, y1 = kp1[i].pt[:2]
                x2, y2 = kp2[j].pt[:2]
                x2 = x2 + image1.shape[1]

                cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

            cv2.imwrite("./Q3/q3_2_match{}.jpg".format(img2.replace(".jpg", "")), image)
        