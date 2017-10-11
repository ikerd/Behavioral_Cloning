
# **Behavioral Cloning**

## Writeup


---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/center.jpg "Center drive"
[image2]: ./images/recovery_1.jpg "Recovery 1"
[image3]: ./images/recovery_2.jpg "Recovery 2"
[image4]: ./images/recovery_3.jpg "Recovery 3"
[image5]: ./images/Flipped.jpg "Flipped"
[image6]: ./images/unflipped.jpg "Unflipped"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model. Generators have been used as suggested in classroom, in order to obtain the maximum accuracy the proposed NVIDIA® architecture has been used. The code has the following structure:

* Lines 30-80 --> Training and validation data load and augmentation fliping the images and steering measurements and using the three available cameras. Generators creation in order to achive a memory efficient training.
* Lines 80-128 --> NVIDIA® architecture implementation using information from the [paper](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)
* Lines 128-135 --> Neural network training


### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My sequential model consists of a convolution neural network following the NVIDIA® architecture (model.py lines 80-135), in the  table below a detailed architecture description can be found.

| Layer		            |     Description	        					|
|:---------------------:|:---------------------------------------------:|
| 1. Normalization      |Lambda layer for data normalization |
| 2. Cropping			|Cropping2D layer for useless image area cropping 		|
| 3. Resizeing	      	|Cropped image resizeing using Lambda layer and resize_func function|
| 4. Convolution		|5x5 Convolution2D with 2x2 stride, Valid padding, RELU activation and output depth of 24|
| 5. Convolution		|5x5 Convolution2D with 2x2 stride, Valid padding, RELU activation and output depth of 36|
| 6. Convolution		|5x5 Convolution2D with 2x2 stride, Valid padding, RELU activation and output depth of 48|
| 7. Convolution		|3x3 Convolution2D with Valid padding, RELU activation and output depth of 64|
| 7. Convolution		|3x3 Convolution2D with Valid padding, RELU activation and output depth of 64|
| 8. Flattenning        |Flatten layer|
| 9. Fully connected    |Dense layer with a depth of 100 |
| 10. Dropout           |Dropout layer with 0.5 keep probability |
| 11. Fully connected    |Dense layer with an output depth of 50 |
| 12. Fully connected    |Dense layer with an output depth of 10 |

The model includes RELU activations to introduce nonlinearity, and dropout layer to prevent the overffitting of the network 

#### 2. Attempts to reduce overfitting in the model

The model contains a dropout layer with a 50% keep probability, in order to reduce overfitting (model.py lines 117).

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 30-80). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 127).

The other paramethers as the steering correction for the side cameras fixed in 0.23 or the number of epochs were fine-tuned via try-error aproach running simulations for different steering angle corrections and reducing the epoch number to the point where the validation loss is smaller than 0.01.

#### 4. Appropriate training data

Training data was choosen to keep the vehicle driving on the road. I used a combination of center lane driving (2 laps), recovering from the left and right sides of the road, curve driving to prevent the network to be biased to drive straight due to the unbalanced amount of straight driving data, curve recovering from left and right sides of the curve and dirt curve side recovering.

#### 5. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image1]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to correct its trajectory if in the start of a curve the vehicle aproaches any of the right or left sides. These images show what a recovery looks like starting from the right side of the road and the centering process using negative steering angles :

![alt text][image2]
![alt text][image3]
![alt text][image4]

To augment the data set, I also flipped images and angles thinking that this would replace the counter clockwise laps, and generalize the training data examples, as for example in the track one, there is only one right hand curve. For example, here is an image that has then been flipped:

![alt text][image5]
![alt text][image6]

Side-cameras images have been used and flipped also appying a steering angle correction of 0.23, added on the left-side camera and substracted on the right-side camera images.

After the collection process, I had 8599 data points. I then preprocessed this data, adding the side-camera images and flipping all the data set obtaining a much bigger and generalized data set composed of 51594 data points.


I finally randomly shuffled the data set and put 20% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3 as evidenced by try-error loops where the validation accuracy stalled or started to reduce due possible data set overfitting. I used an adam optimizer so that manually training the learning rate wasn't necessary.


```python

```
