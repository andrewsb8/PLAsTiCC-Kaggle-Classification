"""
This script takes data from a Kaggle competition (link below) which provides features and time series
data with the goal of classification.  Only the training data is used from the competition here.  This data
set is split into a training and test set (80/20 split).  After time series featurization, a one vs all
binarization LASSO logistic regression technique is used to find defining features and construct a model
for each class provided in the competition training data set.  These models are then applied to the test set
The results are then placed in a matrix where the diagonal shows the percent of correctly identified objects.

Link to data: https://www.kaggle.com/c/PLAsTiCC-2018/data

Author: Brian Andrews
Date: 12/3/2018
"""

import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas
from cesium.featurize import TimeSeries
import cesium.featurize as featurize
from sklearn import linear_model as lm
from scipy.stats import norm
import scipy as sps
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


#**************************************************************************************************************
#Start by collecting data and meta_data into dataframes.
"""
#Windows data paths
train_meta_data = pandas.read_csv("D:\\all\\training_set_metadata.csv")
train_data = pandas.read_csv("D:\\all\\training_set.csv")
"""
#Linux data paths
train_meta_data = pandas.read_csv("/media/sf_D_DRIVE/all/training_set_metadata.csv")
train_data = pandas.read_csv("/media/sf_D_DRIVE/all/training_set.csv")

#There are NaNs everywhere in the distmod column.  Going to replace them with the mean of the column
#using the Dataframe.fillna method from pandas below.  The models will not train without this.
train_meta_data.fillna(train_meta_data.mean(), inplace = True)

#create a dictionary for the object IDs, target classes, and features of time series
tsdict = {
    "object_ID":[], #object_id
    "target":[], #classification of object where the index corresponds to each entry of object_ID
    "target_list":[], #list of all possible classifications
    "time_series_objects":[],
    "feature_titles": [], #all of the feature titles, features of time series and static variables
    "features":[], #nested lists of feature values, indexes of each list element are tied to feature_titles
    "y_values":[], #stores y values (0s or 1s) based on what target is being analyzed
    "model_probabilities":[], #stores the coefficients of each model in an array or list
    "object_classification_percentages":[] #matrix showing percent accuracy
}

#list of features to capture from a time series
feature_list = ["amplitude",
                   "percent_beyond_1_std",
                   "maximum",
                   "max_slope",
                   "median",
                   "median_absolute_deviation",
                   "percent_close_to_median",
                   "minimum",
                   "skew",
                   "std",
                   "weighted_average"]

tsdict['feature_titles'] = feature_list + list(train_meta_data.columns[1:11]) #combine time series feature and static feature titles

#**************************************************************************************************************
#Start main loop to split data up for each individual object_id.  Three arrays
#are created (t,m,e) in order to extract features from the respective time series
#data.  These features may include average values or other statistical values.
#Now want to featurize the time series data and organize all of the data into the dictionary above

#calculate the number of objects now to avoid repeat calculations
total_number_of_objects = len(train_meta_data)
print(total_number_of_objects)
#loop to featurize all of the data according to the feature list above
#N = 100000 #set smaller number than above to do analysis first
for i in range(total_number_of_objects): #start with a smaller number of objects
    current_object_id = train_meta_data['object_id'][i] #identify object_id of interest
    current_object_target = train_meta_data['target'][i] #identify object targe
    print(current_object_id,current_object_target)

    #loop through target list to see if this classification has already been recorded
    found = 0
    for z in range(len(tsdict['target_list'])):
        if current_object_target == tsdict['target_list'][z]:
            found = 1
    if found == 0:
        tsdict['target_list'].append(current_object_target)
    if found == 1:
        found = 0

    indices_with_object_id = ((train_data['object_id'] == current_object_id)) #identify time series indices associated with object_id

    t = list(train_data['mjd'][indices_with_object_id]) #assign values from those indices to each of these arrays
    m = list(train_data['flux'][indices_with_object_id])
    e = list(train_data['flux_err'][indices_with_object_id])
    list_of_indices = np.arange(0,len(t)) #array of indicies for plotting purposes

    tsdict['object_ID'].append(current_object_id) #add id and target data to dictionary
    tsdict['target'].append(current_object_target)

    #create time series object for the source in question and store for transformation of test data
    timeobj = TimeSeries(t=t, m=m, e=e, label=current_object_target, name=current_object_id)

    #featurize the time series object from above
    features_of_time_series = featurize.featurize_single_ts(timeobj, features_to_use = feature_list,
        raise_exceptions=False)


    #print(features_of_time_series.values)
    tsdict['features'].append(list(features_of_time_series.values)) #add the list of time series features to the dictionary
    tsdict['features'][i] += list(train_meta_data.iloc[i,1:11]) #add the static data features


#Going to delete the dataframes with all of the data now that it is all organized in a dictionary
del(train_data)
del(train_meta_data)

#*******************************************************************************************************************
#Now the data is organized in the tsdict dictionary object and will be used
#for the construction of a logistic regression model with l1 (LASSO) regularization
#for each class as a binary multivariate classification problem. Here is the start
#of the model construction.

#going to split the training data up so we can see how well this method performs on data
#where the classification is known.  The test set does not provide this scenario.
tsdict['y_values'] = np.zeros(len(tsdict['target']))
x_train,x_test,y_train_target,y_test_target = train_test_split(tsdict['features'],tsdict['target'],test_size=.2,random_state=42)

#Scale the data
scales = StandardScaler()
scales.fit(x_train)
x_train = scales.transform(x_train)
x_test = scales.transform(x_test)

for i in range(len(tsdict['target_list'])): #loop to create a model for each class in data set
    print(tsdict['target_list'][i])
    y_train = np.zeros(len(y_train_target)) #create arrays to change labels to binary classifications
    y_test = np.zeros(len(y_test_target))
    this_model = LogisticRegression(penalty = 'l1') #LASSO Regression Object
    valuesInClass = ((y_train_target == tsdict['target_list'][i]))
    valuesInClass_test = ((y_test_target == tsdict['target_list'][i]))
    y_train[valuesInClass] = 1 #positive lables to the class in question for train and test set
    y_test[valuesInClass_test] = 1

    model_y_train = np.reshape(y_train,(-1,1)) #reshape y values so sklearn will accept it
    model_y_test = np.reshape(y_test,(-1,1))
    #print(model_y)
    this_model.fit(x_train,model_y_train) #fit the training data
    #print(this_model.coef_,this_model.intercept_)

    probs = this_model.predict_proba(x_test) #predict the probabilities for each entry being in the class
    #print(probs)
    #print(y_test_target)

    #place probabilities of each object being in class i as an array in the dictionary
    tsdict['model_probabilities'].append(probs[:,1])

#print(tsdict['model_probabilities'])

#***************************************************************************************************************************************************
#now we are going to go through 'model_probabilities' and say that we classify the object based
#on which of the models gave the highest probability.  Then this classification will be added to the
#correct position in 'object_classification_percentages' which is a matrix where the rows are the target
#and the columns are what the models classified the object as.

tsdict['object_classification_percentages'] = [np.zeros(len(tsdict['model_probabilities'])) for i in range(len(tsdict['model_probabilities']))]

#populate the above table by selecting the class for which the associated model provided the largest
#probability of an objecct being in that class
for g in range(len(y_test)):
    maximum = 0
    for f in range(len(tsdict['model_probabilities'])):
        if tsdict['model_probabilities'][f][g] > maximum:
            index_of_max = f
            maximum = tsdict['model_probabilities'][f][g]
    class_index = ((y_test_target[g] == tsdict['target_list']))
    #print(maximum,index_of_max,class_index,y_test_target)
    tsdict['object_classification_percentages'][index_of_max][class_index] += 1

#print(tsdict['object_classification_percentages'])

#normalize each row
for b in range(len(tsdict['object_classification_percentages'])):
    sum1 = sum(tsdict['object_classification_percentages'][b])
    if sum1 != 0:
        tsdict['object_classification_percentages'][b] = tsdict['object_classification_percentages'][b]/sum1

#print(tsdict['object_classification_percentages'])

#write the outputs to a csv file (you'll have to go through and remove some brackets but it's fine)
with open("/media/sf_D_DRIVE/all/result_table.csv",'w') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    spamwriter.writerow(["",list(tsdict['target_list'])])
    for h in range(len(tsdict['target_list'])):
        spamwriter.writerow([tsdict['target_list'][h], list(tsdict['object_classification_percentages'][h])])















#ignore this comment
