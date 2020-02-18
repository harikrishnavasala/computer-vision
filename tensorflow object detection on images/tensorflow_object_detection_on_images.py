# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 11:39:20 2020

@author: DELL
"""

# importing required packages
from object_detection.utils import label_map_util
import tensorflow as tf
import numpy as np
import argparse
import imutils
import cv2
#from google.colab.patches import cv2_imshow
from imutils import paths
#save_imgs = 'E:\\TensorFlow\\models\\research\\object_detection\\test_images\\'
import pandas as pd
results = { 'filename':[],'class':[],'xmin':[],'ymin':[],'xmax':[],'ymax':[],'score':[]}
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,help="base path for frozen checkpoint detection graph")
ap.add_argument("-l", "--labels", required=True,help="labels file")
ap.add_argument("-i", "--images", required=True,help="path to input image")
ap.add_argument("-o", "--output",required=True,help="path to output images")
#ap.add_argument("-n", "--num-classes", type=int, required=True,help="# of class labels")
#ap.add_argument("-c", "--min-confidence", type=float, default=0.5,help="minimum probability used to filter weak detections")
args = vars(ap.parse_args())
# initialize a set of colors for our class labels
#COLORS = np.random.uniform(0, 255, size=(args["num_classes"], 3))
COLORS = np.random.uniform(0, 255, size=(1, 3))
print('loading model')

# initialize the model
model = tf.Graph()

# create a context manager that makes this model the default one for
# execution
with model.as_default():
    # initialize the graph definition
    graphDef = tf.GraphDef()

    # load the graph from disk
    #with tf.gfile.GFile(args["model"], "rb") as f:
    with tf.gfile.GFile(args["model"],"rb") as f: #'E:\\TensorFlow\\models\\research\\object_detection\\test_images\\frozen_inference_graph_lights.pb', "rb") as f:
        serializedGraph = f.read()
        graphDef.ParseFromString(serializedGraph)
        tf.import_graph_def(graphDef, name="")
# load the class labels from disk
labelMap = label_map_util.load_labelmap(args["labels"])#'E:\\TensorFlow\\models\\research\\object_detection\\test_images\\label_map_lights.pbtxt')
#categories = label_map_util.convert_label_map_to_categories(labelMap, max_num_classes=args["num_classes"],use_display_name=True)
categories = label_map_util.convert_label_map_to_categories(labelMap, max_num_classes=43,use_display_name=True)
categoryIdx = label_map_util.create_category_index(categories)

# create a session to perform inference
with model.as_default():
    with tf.Session(graph=model) as sess:
        # grab a reference to the input image tensor and the boxes
        # tensor
        imageTensor = model.get_tensor_by_name("image_tensor:0")
        boxesTensor = model.get_tensor_by_name("detection_boxes:0")
        # for each bounding box we would like to know the score
        # (i.e., probability) and class label
        scoresTensor = model.get_tensor_by_name("detection_scores:0")
        classesTensor = model.get_tensor_by_name("detection_classes:0")
        numDetections = model.get_tensor_by_name("num_detections:0")
        images = sorted(list(paths.list_images(args["images"])))
        # load the image from disk
        for i in range(len(images)):
            image = cv2.imread(images[i])
            
        #image = cv2.imread('/content/drive/My Drive/computer_vision/traffic sign detection/images/Hyderabad to Bangalore Center lane Centre Camera 0000517.jpg')
            (H, W) = image.shape[:2]
        # check to see if we should resize along the width
            if W > H and W > 1000:
                image = imutils.resize(image, width=1000)
        # otherwise, check to see if we should resize along the# height
            elif H > W and H > 1000:
                image = imutils.resize(image, height=1000)
        # prepare the image for detection
            (H, W) = image.shape[:2]
            output = image.copy()
            image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
            image = np.expand_dims(image, axis=0)
        # perform inference and compute the bounding boxes,
        # probabilities, and class labels
            (boxes, scores, labels, N) = sess.run([boxesTensor, scoresTensor, classesTensor, numDetections],feed_dict={imageTensor: image})
        # squeeze the lists into a single dimension
            boxes = np.squeeze(boxes)
            scores = np.squeeze(scores)
            labels = np.squeeze(labels)
            # loop over the bounding box predictions
            for (box, score, label) in zip(boxes, scores, labels):
            # if the predicted probability is less than the minimum
            # confidence, ignore it
                if score < 0.5:
                    continue
            # scale the bounding box from the range [0, 1] to [W, H]
                (startY, startX, endY, endX) = box
                startX = int(startX * W)
                startY = int(startY * H)
                endX = int(endX * W)
                endY = int(endY * H)
            # draw the prediction on the output image
                label = categoryIdx[label]
                label_a = label # this for my analysation #########################3
                idx = int(label["id"]) - 1  
                label = "{}: {:.2f}".format(label["name"], score)
                cv2.rectangle(output, (startX, startY), (endX, endY),COLORS[idx], 2)
                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.putText(output, label, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.3, COLORS[idx], 1)
                ################################################################3
                # below is for my requirement
                results['filename'].append(images[i])
                results['xmin'].append(startX)
                results['ymin'].append(startY)
                results['xmax'].append(endX)
                results['ymax'].append(endY)
                results['score'].append(score)
                results['class'].append(label_a['name'])
                
        # show the output image
            #cv2_imshow(output)
            cv2.imwrite(args["output"] + '\\gamenous' + str(i) + '.jpeg',output)
            
            print("completed:",i)
        results_frame = pd.DataFrame(results)
        results_frame.to_csv(args["output"] + '\\results_data.csv')
        #cv2.waitKey(0)
        #cv2.resize((500,500),output)
        
