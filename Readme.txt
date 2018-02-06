Directory Structure

find_phone_task_1
	|
	|----->Extras
	|
	|----->Find_Phone
	|
	|----->models
	|
	|----->protoc
	|
	|----->Screenshots
	|
	|----->Sift_phone
	|
	|----->Train_phone
----------------------------------------------------------------
Python 3.x

Libraries Required

tensorflow 1.4.0
Numpy
Scipy
Matplotlib
glob
-----------------------------------------------------------------
Errors Encountered
No "deployment" module or No "nets" Module
Solution
Set the path for python in Environment Variable
PYTHONPATH = <Path to file>/find_phone_task_1/models/research;<Path to file>/find_phone_task_1/models/research/slim
-----------------------------------------------------------------
There are two main folders : Find_Phone and Train_phone
Initially, open the "Train_phone" directory,it contains several folders and packages. Run the train_phone_finder.py file from command line by giving proper arguement

eg: python train_phone_finder.py --input_path=find_phone/

It will start training from the last checkpoint file. Once you are done with the training, the model is saved in the directory and inference graph needs to be exported.
That is done using the export_inference_graph.py in models->research->object_detection folder.

Next for testing, go to "Find_phone" directory. It contains the model, data and training directory where all the information fom training step is copied.

eg: python find_phone --images_path=test_images

Just add all images you want to test into the Find_phone->test_images folder. The results are saved in Find_phone->Results folder and cordinates are displayed on the command line.