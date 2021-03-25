# PLAsTiCC-Kaggle-Classification

This directory of this repository holds the code used for the PLAsTiCC LSST data classification competition in 2018 (never submitted for score). This will also be used as a submission for the final project for the Big Data course offered at Drexel University in the Physics department.

This project involved classifying astronomical sources. There is time series data as well as time-independent data available. The strategy employed for classification was to calculate features (averages, standard deviations, etc.) of the time series data. Then the combined use of the time series features and time independent data are used to train logistic regression models for each individual class. 80% of the competition training set was used for training and 20% was used for testing. For testing, each object in the test set was classified by each model trained (for each class) and the classification was assigned via the model output with the highest probability. This strategy is called one vs. all binarization.

Details about the competition can be found here: https://www.kaggle.com/c/PLAsTiCC-2018

Sadly, most of the data files are too large to be stored on GitHub but can be found via the above link!
