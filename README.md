# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/cnn.png "Model Visualization"
[image2]: ./images/center.jpg "center"
[image3]: ./images/right.jpg "Recovery Image"
[image4]: ./images/left.jpg "Recovery Image"
[image5]: ./images/architecture.png "Recovery Image"


---

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 and 3x3 filter sizes and depths between 24 and 64 (model.py lines 55-58). The model includes RELU layers to introduce nonlinearity (code line 55-60), and the data is normalized in the model using a Keras lambda layer (code line 54). 

The normalized data is the input to a stack of 3 convolution layers with 5x5 filter sizes and depths between 24 and 48 (model.py lines 55-57) that include a RELU activation layer each. Then, the model has two convolution layers with 3x3 filter sizes and depths of 64 that also include a RELU activation layer each. Finally, 4 fully-connected layers complete the model with a number of neurons of 100, 50, 10, and 1, respectively.

#### 2. Attempts to reduce overfitting in the model

The data set is randomly shuffled to avoid overfitting (code line 69). Early termination is the strategy adopted to also avoid overfitting. After experimentation, the number of epochs is set to 3 due to the fact that the mse is no longer considerably minimnized (code line 69). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 68).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road and driving to the opposite direction. 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was based on NVIDIA's architecture (https://devblogs.nvidia.com/deep-learning-self-driving-cars/).
![alt text][image1]

My first step was to use a convolution neural network model similar to the Traffic signs classifier. It was a LeNet architecture. I thought this model might be appropriate because it extracts relevant features from the images that can help to come up with an acceptable regression model. I used 10 epochs.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that I reduced epochs to 3. I got a non-everfitted model after this because the validation mse and the testing mse where similar.

The next step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. To improve the driving behavior in these cases, I changed the architeture for the NVIDIA one. By making this choice, I enhanced the performance of the vehicle of the tracked. However, the car still had problems by turning on pronounced curves. I decided to augment my training and validation sets by considereng the images from the right side and left side cameras with a factor correction of 0.2. 

At the end of the process, the vehicle was able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 52-66) consisted of a convolution neural network with the following layers and layer sizes:

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image5]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to come bach to the center of the lane.




To augment the data sat, I used the images from the left-side camera and right-side camera with a correction of 0.2 degrees.

#### Left-side camera
![alt text][image4]
#### Right-side camera
![alt text][image3]


After the collection process, I had 4512 number of data points. I then preprocessed this data by cropping the upper part of the image where the threes and sky of the road are in order to delete useless information. 

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3 as evidenced by the matching of the validation set and training set accuracy. I used an adam optimizer so that manually training the learning rate wasn't necessary.
