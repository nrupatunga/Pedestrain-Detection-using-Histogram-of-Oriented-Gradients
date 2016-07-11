# Pedestrain-Detection-using-Histogram-of-Oriented-Gradients

HOG is a visual descriptor i.e., it describes the content of an image in a single feature vector. 
The idea behind HOG is that local object appearance and shape within an image can be described by 
the distribution of intensity gradients or edge directions.

## Dataset
|   |Training Set   |Test Set   |Total |
|---|---|---|---|---|
| Positive Samples  |3977   |442   |4419   |
| Negative Samples  |4842   |538   |5380   |

Each sample is of dimension 64x128

## SVM model

we train a soft-margin linear SVM classifier. We use [svmlight](http://svmlight.joachims.org/) to train the SVM model. 

##Accuracy on train and test set

|   |Accuracy (%)   |Precision/Recall (%) |
|---|---|---|---|
| Training Set  |99.07   |99.67/98.27 |
| Test Set  |99.18  |100/98.19|

## Dependency

OpenCV

## Sample Detections
![](https://github.com/nrupatunga/nrupatunga.github.io/blob/master/project/hog/pedestriandetection.PNG)

