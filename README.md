##Writeup Template
###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car.jpg
[image2]: ./output_images/car_hog.jpg                            # 4 images show hog features from both car class and non-car class
[image3]: ./output_images/non_car.jpg
[image4]: ./output_images/non_car_hog.jpg
[image5]: ./output_images/scaled_features.png                    # shows the comparsion of non-scaled features and scaled features for a car image which dominated by red color.
[image6]: ./output_images/windows_search_test1.jpg               # 6 images show the results of Hog Sub-sampling Window Search
[image7]: ./output_images/windows_search_test2.jpg
[image8]: ./output_images/windows_search_test3.jpg
[image9]: ./output_images/windows_search_test4.jpg
[image10]: ./output_images/windows_search_test5.jpg              # PS. this image has a false detection 
[image11]: ./output_images/windows_search_test6.jpg
[image12]: ./output_images/1_frame_stacked_heatmap.jpg           # one stacked frame resulting heatmap and labelled image
[image13]: ./output_images/1_frame_stacked_label_image.jpg
[image14]: ./output_images/2_frame_stacked_heatmap.jpg           # two stacked frames resulting heatmap and labelled image, PS. the orignal false detection filtered out
[image15]: ./output_images/2_frame_stacked_label_image.jpg
[image16]: ./output_images/3_frame_stacked_heatmap.jpg           # three stacked frames resulting heatmap and labelled image
[image17]: ./output_images/3_frame_stacked_label_image.jpg

[video1]: https://www.youtube.com/watch?v=bbmTBb1zpho            # the proccessed project video

[code1]:  The code is in  vehicle_detection.ipynb

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

I used YCrCb color space and set skimage.hog() parameters. I grabbed one images from each of the two classes and displayed them to get a feel for what the skimage.hog() output looks like.

Here is an example using the Y channel of YCrCb color space and HOG parameters of orientations=8, pixels_per_cell=(8, 8) and cells_per_block=(2, 2):

1) output_images/car.jpg
2) output_images/car_hog.jpg
3) output_images/non_car.jpg
4) output_images/non_car_hog.jpg

####2. Explain how you settled on your final choice of HOG parameters.

The code is in function hog_features()

I tried some combinations, and looked like Y channel gave the best figure. But when later moved to train a SVM, I realized all 3 channels were more useful than color space features

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I combined both HOG features, which located in function hog_features_3(), and color space features, which located in functions color_features(), hist_features().

In the initial try, I grapped hog features only from Y channel. But when trained the classifier, I realized too many color space features would gave lots of false detections. In order to increase the propotions of hog features, I then grapped them from all 3 channels. The result looked much better.

Then in function process_features() , I used StandardScaler() to normalize all the features, and prepared corresponding labels.

To verify the trainning result, I used train_test_split() to split features to trainning set and testing set.

Finally I used a liner SVC to do the trainning, the verifying result showed a 99.2% accuracy which was surprising.

1) output_images/scaled_features.png  
shows the comparsion of non-scaled features and scaled features for a car image which dominated by red color.

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I implemented Hog Sub-sampling Window Search in function find_cars() , and set scale=1.5, orient=8, pix_per_cell=8, cell_per_block=2 . It extracted hog features once and then sub-sampled to get all of its overlaying windows. Each window is defined by a scaling factor where a scale of 1 would result in a window that's 8 x 8 cells then the overlap of each window is in terms of the cell distance. This means that a cells_per_step = 2 would result in a search window overlap of 75%.

These parameters were decided by experiements, since they were good enough from the beginning, I didn't do many changes.

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector. To minimize the workload, I clipped image on Y-axis in range (300, 700)

Example images:

1) output_images/windows_search_test1.jpg
2) output_images/windows_search_test2.jpg
3) output_images/windows_search_test3.jpg
4) output_images/windows_search_test4.jpg
5) output_images/windows_search_test5.jpg
6) output_images/windows_search_test6.jpg
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result] https://www.youtube.com/watch?v=bbmTBb1zpho 


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

My design was based on a frames FIFO as the filter, heat values in the FIFO would be summed up before sent to threshold and label functions. The FIFO windows was 48 depth, and the heat threshold was 60 which worked well on the project video.

In the example, I chose test images 4, 5, 6 which looked like in consecutive to test my FIFO design.

1) one stacked frame result:
    a) output_images/1_frame_stacked_heatmap.jpg
    b) output_images/1_frame_stacked_label_image.jpg
    
2) two stacked frames result:
    a) output_images/2_frame_stacked_heatmap.jpg
    b) output_images/2_frame_stacked_label_image.jpg
    
3) three stacked frames result:
    a) output_images/3_frame_stacked_heatmap.jpg
    b) output_images/3_frame_stacked_label_image.jpg

Obervered the stacked ( filtered ) result images, the false detection in test_images_5 disappeared.


---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

In the initial try, I used only Y channel to grap HOG features. Then found dominant color features gave too many false detections. Then I changed to grap HOG features from all 3 channels to make it dominating instead.

To get rid of false detections in the video, I chose to implment a FIFO with 48 depth ( 2 seconds ), which I believed enough to filter out the noises. But the heat threshold turned out to be hard to adjust, I needed to find a balance point which could satify both removing false detections and retaining true detections. The parameter 60 I finnally chose might not work on other videos.

FIFO size 48 raised another problem, the system became not insensitive to sudden change, e.g. a very high speed vehicle poped up. I need to think about it. 

