## Deep Learning Project ##

### Project Overview

#### What is this?
This is my implementation of the Udacity Robotics Nanodegree Deep Learning Project

#### Problem statement
A quadrotor drone is flying in a simulated environment. What needs to be done is for it to be able to distinguish its a designated person from among a multitude of people, and from significant distances. Once the person has been identified and located, the drone is meant to fly towards and follow this person throughout the environment. As of now, it is incapable of doing so.  However, with the help of Fully Convolutional Deep Neural Networks, we can train models that will not only be able to identify the contents of an image, it will also be able to figure out what part of the image the object is located at. We can then proceed to use this trained model with our quadrotor drone.  

#### Solution and files of note
- A working solution was achieved with the help of Keras in writing the Fully Convolutional Neural Network. Training the model was performed using AWS p2.xlarge instances.
- The writeup of this project, which includes the diagrams of the network architecture, parameters used, as well as exploration of the concepts and discussion of the results can be found here: 
    - [WRITEUP.md](./submission_requirements/WRITEUP.md).
- The trained model files are linked below.  The final trained score of  0.4796 was able to exceed the required final score of 0.40 by a fair amount.
    - [model_weights.h5](./submission_requirements/model_files/model_weights.h5)
    - [config_model_weights.h5](./submission_requirements/model_files/config_model_weights.h5)
- Exported html version of the jupyter notebook that was used as our "workbook" as well as the notebook itself
    - [model_training.html](./submission_requirements/model_training.html)
    - [model_training.ipynb](./submission_requirements/model_training.ipynb)