# Image_Classification_Talent_Squad_DS 


# Talent Squad - Data Science II
--------

## Table of Contents

- [Repository Content](#repository-content)
- [Quick Start](#quick-start)
- [Description of the Task](#description-of-the-task)
- [Decision-making process](#decision-making-process)
    - [Data Pipeline](#data-pipeline)

- [Final Thoughts](#final-thoughts)



## Repository Content 

    - Predictions in json and cvs
    - Jupyter Notebook with the script
    - app.py with script

#### [top](#table-of-contents)
--------
## Quick Start

This project was done in a Jupyter Notebook but the model training was done from google Colab since they offer free GPUs and TPUs[^1] in case you want to save some time :wink:.

* <a href="https://colab.research.google.com/github/Freegalado/NoSupervisat_Agrupament/blob/main/S11_T01_Unsupervised_Learning_Grouping.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

if you want to open it in locally don't forget to have installed:

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
![OpenCV](https://img.shields.io/badge/opencv-%23white.svg?style=for-the-badge&logo=opencv&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![Seaborn](https://img.shields.io/badge/-Seaborn-blue?style=for-the-badge&logo=seaborn) 

Inside the repository you will find:

    - The Jupyter notebook (.ipynb)
    - A folder with images for each sport.
    - CSV file with the result of the classification

#### [top](#table-of-contents)
--------



### Description of the task

The project consists of classifying the different images by the type of sport to which they correspond, with the following: baseball, cricket and football.

The labels we will have for each group of images will be in alphabetical order, as follows:  

    - Baseball will be represented with the digit 0.

    - Cricket will be represented with the digit 1.

    - Football will be represented with the digit 2.


**The evaluation objectives will be:**

    1. Increase of samples in the images. 

    2. Calculate the macro F1-score. 






  #### [top](#table-of-contents)
--------

 ### Decision-making process
  

We have a classification problem so it is appropriate to use an Artificial neural networks (ANNs) with this we can detect patterns and use them to solve problems where we have to detect patterns. ANNs are composed of a layer of nodes, which contains an input layer, one more hidden layers and an output layer, each node is connected to another letting information pass to a greater or lesser extent depending on the final prediction, the values that indicate the level at which they let the information pass are called weights. 

 

For the solution of our problem we will use a convolutional neural network (CNNs), the difference is that a new mathematical operation is introduced.
called convolution. This new operation is the one that will allow us to find patterns in the images, and it is these patterns that are later used to solve our classification problem.

  #### [top](#table-of-contents)
--------
#### Data Pipeline

  ![Data-pipeline](https://user-images.githubusercontent.com/91080406/191982034-bd65086b-8e39-4e3c-a59d-986e32251e3c.png)


To perform the task, different activities were carried out in 4 stages:

- In the first stage, the libraries needed to perform the different tasks are loaded, the images (files) are loaded and an exploratory analysis will be performed on these, then they will be transformed to data suitable for the task in this case a matrix, finally some kind of pre-processing will be performed.

- In the second stage, different CNN models are built, a pre-trained model is loaded to perform a model with learning transfer, the error function to be used in both types of models is defined.

- In the third stage, the different models are trained and evaluated to understand their behavior.

- In the fourth stage, the model that has delivered the best F1-macro score is selected to subsequently classify the images.


#### [top](#table-of-contents)
---------
### Final Thoughts

It is my first time to make an image classification model, which helped me to learn a lot about the subject, there are processes to improve. It is the first time that I use the Google Colab platform, I usually work on my laptop, which already needs a replacement, but this time I had to look for another alternative to carry out the project. The models are not complex models but the difference in training time between local and cloud-based service is noticeable.

This is my fisrt time to participate the [NUWE](https://nuwe.io/dev/challenges) platform and I think I will become a regular user of its mini-projects (challenges) in order to keep improving my skills.

#### [top](#table-of-contents)
 

[^1]: In order to offer computational resources at no cost, Colab needs to retain the flexibility to adjust usage limits and hardware availability at any time 
