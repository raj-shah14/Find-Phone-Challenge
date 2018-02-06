# Find-Phone-Challenge
The task was to find the location of cell phone in a given image

## Data Preprocessing
The dataset consists of 134 images and text file with labels for each image. Initial step was to create a tfrecord file for the data and splitting the data into train and test. Record file is generated using the tfrecord.py file. The record file generated is then used in the model config file as input along with *.pbtxt file which contains list of all the objects to be detected. You can find the config file in training folder. 

## Training
I have used Google Tensorflow Object Detection API to detect the phone.  I am using SSD mobilenet pre-trained model, and then using transfer learning to train the model to detect another object, in this case, phone. The benefit of transfer learning is training can be quicker and few sample images are required. SSD mobilenet is used for Real-time Detection of objects in Video due to its high FPS the speed of detection is faster as compared to other models. But there is a trade off in terms of accuracy and speed. For this case, it works well.
After Training for 1700 time-steps
![alt text](https://github.com/raj-shah14/Find-Phone-Challenge/blob/master/Train_Phone/loss.jpg)
![alt text](https://github.com/raj-shah14/Find-Phone-Challenge/blob/master/Train_Phone/graph_loss.jpg)

## Testing
After training the model, it is saved and modelname.ckpt, model.index and model.meta file are used to generate the frozen_inference.pb file which is used for testing. The inference file is generated using export_inference_graph.py file.

### Sample images

![alt text](https://github.com/raj-shah14/Find-Phone-Challenge/blob/master/Find_Phone/106.jpg)![alt t7xt](https://github.com/raj-shah14/Find-Phone-Challenge/blob/master/Find_Phone/107.jpg)![alt text](https://github.com/raj-shah14/Find-Phone-Challenge/blob/master/Find_Phone/108.jpg)

### Output

![alt text](https://github.com/raj-shah14/Find-Phone-Challenge/blob/master/Find_Phone/image_106.jpg)![alt_text](https://github.com/raj-shah14/Find-Phone-Challenge/blob/master/Find_Phone/image_107.jpg)![alt text](https://github.com/raj-shah14/Find-Phone-Challenge/blob/master/Find_Phone/image_108.jpg)

## Result
![alt text](https://github.com/raj-shah14/Find-Phone-Challenge/blob/master/Find_Phone/results.jpg)

