#**Traffic Sign Recognition** 

---

[//]: # (Image References)

[image1]: ./examples/dataset_visualization.JPG "Visualization"
[image2]: ./examples/original.JPG "Original"
[image3]: ./examples/hist_eq.JPG "Histogram Equalization"
[image4]: ./examples/grayscale.JPG "Grayscale"
[image5]: ./examples/image3.JPG "Preprocessing Result"
[image6]: ./web_signs/09_no_passing.JPG "Traffic Sign 1"
[image7]: ./web_signs/14_stop2.JPG "Traffic Sign 2"
[image8]: ./web_signs/18_attention.JPG "Traffic Sign 3"
[image9]: ./web_signs/14_stop.JPG "Traffic Sign 4"
[image10]: ./web_signs/18_attention2.JPG "Traffic Sign 5"
[image11]: ./web_signs/13_yield.JPG "Traffic Sign 6"
[image12]: ./web_signs/05_80.JPG "Traffic Sign 7"
[image13]: ./web_signs/17_noentry.JPG "Traffic Sign 8"

---
###Writeup / README

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used a set of python libraries including pandas, numpy to calculate summary statistics of the traffic signs dataset [cell 2]:

* The size of the training set is ? 34799
* The size of the validation set is ? 4410
* The size of the test set is ? 12630
* The shape of a traffic sign image is? 32x32x3
* The number of unique classes/labels in the data set is? 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data is distributed among all classes in the dataset. 

![alt text][image1]

The distribution presented in the figure above clearly indicates that some the class has less than 250 images and some of them have more than 2000. It is possible to say that this dataset a highly unbalanced dataset. 

Using this dataset without augmentation may lead the model to be more biased in relation to the classes with more number of images in training set.


###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)
Original images
![alt text][image2]

As a first step, I decided to apply histogram equalization. It was applied because there are a lot of dark images in the dataset.  

Here is an example of a traffic sign image before and after histogram equalization.

![alt text][image3]

As a second step, I decided to convert the images to grayscale because it was suggested in the Yann LeCun's article 

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image4]

As the last step, I normalized the image data because it recommended for better model performance. I tried to normalize around zero(-1,1) and around 0.5 (0.1, 0.9). The normalization around 0 gives me a better result. 

![alt text][image5]

####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        							| 
|:---------------------:|:---------------------------------------------:		| 
| Input         		| 32x32x1 RGB image   									| 
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 28x28x16 			|
| RELU					|														|
| Max pooling	      	| 2x2 stride,  outputs 14x14x16 						|
| Convolution 3x3	    | 1x1 stride, valid padding, outputs 10x10x32			|
| RELU					|														|
| Max pooling	      	| 2x2 stride,  outputs 5x5x32 							|
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 1x1x800 			|
| RELU					|														|
| Flatten				| input 1x1x800 , output 800							|
| Flatten				| 3 dimension -> 1 dimesion  input 3x3x32, output 800	|
| Concatenate			|			800+800=1600								|
| Fully connected		| 1600 to 43 (output=number of trafic signs in dataset	|
|						|														|
|						|														|
 


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To final training of the model, I used the following parameters:

rate = 0.00097
EPOCHS = 35
BATCH_SIZE = 156

All weights were generated using the normal distribution. 
AdamOptimizer was used for optimization because it is simple and computationally effective.


####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results of the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well-known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.



My final model results were:
* validation set accuracy of ? 0.956
* test set accuracy of ? 0.942
Please investigate cell numbers [17,18]


If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
First of all I trained default LeNet and got the following results:
epochs=20
Batch=64
rate=0.0001
Validation Accuracy = 0.938
Test Accuracy = 0.903

* What were some problems with the initial architecture?
According to Yann LeCun's article, a multi-scale architecture has better performance than classic LeNet architecture. Because of that, I trained Advanced LeCun-like multi-scale network with default parameters. 
epochs=50
Batch=128
rate=0.001
Validation Accuracy = 0.943
Test Accuracy = 0.922

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates overfitting; a low accuracy on both sets indicates underfitting.

During training I experimented only with enabling/disabling dropout. It was found that with dropout turned on Validation accuracy becomes lower than when dropout turned off. Because of that, I commented string with dropout in the code. 

* Which parameters were tuned? How were they adjusted and why?

I tried to change learning rate. The experiment with 
epochs=50
batch=128
learning_rate=0.0005
gives me 
Validation Accuracy = 0.930
Test Accuracy = 0.907

Increasing the batch size with the same condition doesn't reach better performance.

Because playing with general parameters didn't reach project requirements. I started to change Network parameters - a number of features. In LeCun's article, there are hundreds of features are described.
On every layer, I increased the size of features twice. 
This solution increased performance of the architecture:
Validation Accuracy = 0.952
Test accuracy = 0.936

At the last step I reduced the number of epochs, increased the size of the batch, set learning rate 0.00097, turned off dropout and got the final result:
Validation accuracy = 0.956
Test accuracy = 0.942

If a well-known architecture was chosen:
* What architecture was chosen?
LeCun-like multi-scale architecture
* Why did you believe it would be relevant to the traffic sign application?
Because this approach demonstrated the good result in the competition.
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
Classic LeCun-like multi-scale architecture gives a performance about 98%. But they experiment with a much wider set of parameters. Because of that, I think that for current configuration my approach gives well results.
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are eight German traffic signs that I found on the web:

![alt text][image6] ![alt text][image7] ![alt text][image8] ![alt text][image9] ![alt text][image10]
![alt text][image11] ![alt text][image12] ![alt text][image13] 

The second image might be difficult to classify because the Stop sign is located behind objects on the first plane and point of view is unusual. 
The fourth image might be difficult to classify because the image point of view is unusual.
The fifth image might be difficult to classify because the part of the image is hidden under a black shadow.
The sixth image might be difficult to classify because there is a painting on the sign.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| No passing      		| Stop   										| 
| Stop     				| Ahead Only 									|
| General caution		| General caution								|
| Stop     				| Beware of ice/snow 							|
| General caution		| General caution								|
| Yield					| Yield      									|
| 80 km/h	      		| End of speed limit (80km/h)	 				|
| No entry	      		| No entry						 				|


The model was able to correctly guess 4 of the 8 traffic signs, which gives an accuracy of 50%. 

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 23rd cell of the Ipython notebook.

For the first image, the model is absolutely sure that this is a Stop sign (probability of 1), and the image doesn't contain a Stop sign. The top five softmax probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1         			| Stop sign   									| 
| 0     				| No entry 										|
| 0						| Speed limit (70km/h)							|
| 0	      				| Keep right					 				|
| 0					    | Priority road      							|


For the second image, the model is relatively sure that this is an Ahead only (probability of 0.85), and the image doesn't contain an Ahead only sign. 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.85         			| Ahead only   									| 
| 0.07    				| Stop sign										|
| 0.03					| Priority road									|
| 0.02	   				| Turn left ahead				 				|
| 0.01				    | Yield      									|

For the third image, the model is absolutely sure that this is a General caution (probability of 1), and the image does contain a General caution sign. 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1         			| General caution								| 
| 0     				| Pedestrians									|
| 0						| Road work										|
| 0	      				| Right-of-way at the next intersection			|
| 0					    | Speed limit (50km/h) 							|

For the fourth image the model is relatively sure that this is the End of all speed and passing limits (probability of 0.4), but the image does contain a Stop sign.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.4         			| End of all speed and passing limits			| 
| 0.33    				| Speed limit (30km/h)							|
| 0.2					| Yield											|
| 0.03	   				| End of speed limit (80km/h)					|
| 0.03				    | No entry      								|


For the fifth image, the model is absolutely sure that this is a General caution (probability of 1), and the image does contain a General caution sign. 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1         			| General caution								| 
| 0     				| Roundabout mandatory							|
| 0						| Go straight or left							|
| 0	      				| Double curve									|
| 0					    | Traffic signals 								|

For the sixth image, the model is absolutely sure that this is a Yield (probability of 1), and the image does contain a Yield sign. 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1         			| Yield											| 
| 0     				| Priority road									|
| 0						| No entry										|
| 0	      				| Right-of-way at the next intersection			|
| 0					    | End of no passing 							|

For the seventh image, the model is relatively sure that this is the End of speed limit (80km/h) (probability of 0.85), but the image does contain a Speed limit (80km/h) sign. 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.85         			| End of speed limit (80km/h)   				| 
| 0.15    				| Speed limit (50km/h)							|
| 0.0					| Speed limit (80km/h)							|
| 0.0	   				| Speed limit (30km/h)				 			|
| 0.0				    | Speed limit (60km/h)      					|

For the seventh image the model is relatively sure that this is a No entry (probability of 1), and the image does contain a No entry sign. 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1         			| No entry										| 
| 0     				| Turn left ahead								|
| 0						| Yield											|
| 0	      				| Keep right									|
| 0					    | Bicycles crossing 							|

