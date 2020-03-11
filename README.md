# Deep-Learning-Image-Classification-Keras-MATLAB-CNN


To train model on RPCG16 data for the experiment: (Inspired from blog of François Chollet who used keras for classification of two class problem using CNN)

Rock-Paper-Scissors images colleceted by my self with phone camera and robot camera.

Image processing using MATLAB and Image augmentation done using Keras library.
	
Image data has been kept in respective folder type.

Image data first split into three folder(Training,test, validation) and later three subfolder in each folder for Rock, Paper and Scissors images

folder are /home/Rock, /home/Paper, /home/Scissors for three types of images in Rock-Paper-Scissors game.


	1. Version: tensorflow 1.0.0 and Keras 2.0.0. First one need to install tensorflow or rent a machine having tensorflow installed. Then installed Keras 2.0.0 in the machine.
	2. please make sure the version of both the framework should be same as mentioned above.
	3. This Keras code can be run on machines with or without GPU.
	4. Need to change few things in code like location of data and h5 file.
	
	After having above things, one can run code just by writing:
	
	python CNN_Keras_DL.py
	
	It would take around 20 minutes to train until 50 epochs and you will have results.
	
