"""
This script takes data from a Kaggle competition (link below) which provides features and time series
data with the goal of classification.  This script is the beginning of this process and shows mostly
visualization of the data for presentation and exploratory data analysis.

Link to data: https://www.kaggle.com/c/PLAsTiCC-2018/data

Author: Brian Andrews
Date: 12/3/2018
"""

import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas
from cesium.featurize import TimeSeries
import cesium.featurize as featurize
from sklearn import linear_model as lm
from scipy.stats import norm
import scipy as sps
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


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

#**************************************************************************************************************
#Start main loop to split data up for each individual object_id.  Three arrays
#are created (t,m,e) in order to extract features from the respective time series
#data.  These features may include average values or other statistical values.

#create a dictionary for the object IDs, target classes, and features of time series
tsdict = {
    "object_ID":[], #object_id
    "target":[], #classification of object where the index corresponds to each entry of object_ID
    "target_list":[], #list of all possible classifications
    "time_series_objects":[],
    "feature_titles": [], #all of the feature titles, features of time series and static variables
    "features":[], #nested lists of feature values, indexes of each list element are tied to feature_titles
    "y_values":[], #stores y values (0s or 1s) based on what target is being analyzed
    "model_coefficients":[] #stores the coefficients of each model in an array or list
}

tsdict_test = {
    "object_ID":[], #object_id
    "features_test":[],
    "test_result_percentages":[] #each array, corresponding to each object, expresses the percent chance that an object is of each class
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

#***************************************************************************************************************************************
#Now want to featurize the time series data and organize all of the data into the dictionary above

#calculate the number of objects now to avoid repeat calculations
total_number_of_objects = len(train_meta_data)

#loop to featurize all of the data according to the feature list above
N = 50 #set smaller number than above to do analysis first
for i in range(N): #start with a smaller number of objects
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

    """
    #plot figures of time series data for first three objects
    if i < 3:
        plt.figure(1)
        plt.plot(list_of_indices,t,'-',label="{}".format(current_object_id))
        plt.legend()
        plt.figure(2)
        plt.plot(list_of_indices,m,'-',label="{}".format(current_object_id))
        plt.legend()
        plt.figure(3)
        plt.plot(list_of_indices,e,'-',label="{}".format(current_object_id))
        plt.legend()

#plot the various figures
plt.figure(1)
plt.title("MJD")
plt.figure(2)
plt.title("Flux")
plt.figure(3)
plt.title("Flux Error")
plt.show()
"""

#Going to delete the dataframes with all of the data now that it is all organized in a dictionary
del(train_data)
del(train_meta_data)

#Because of RAM limitations, I am going to repeat the process after deleting the training dataframes
test_meta_data = pandas.read_csv("/media/sf_D_DRIVE/all/test_set_metadata.csv",nrows=1000)
test_data = pandas.read_csv("/media/sf_D_DRIVE/all/test_set.csv",nrows=100000)
test_meta_data.fillna(test_meta_data.mean(),inplace = True)

print(test_meta_data)
#starting the loop over again
for j in range(N):
    current_ob_id_test = test_meta_data['object_id'][j] #organize test data
    print(current_ob_id_test)
    indices_with_object_id_test = ((test_data['object_id'] == current_ob_id_test)) #same with test data
    t_test = list(test_data['mjd'][indices_with_object_id_test])
    m_test = list(test_data['flux'][indices_with_object_id_test])
    e_test = list(test_data['flux_err'][indices_with_object_id_test])
    tsdict_test['object_ID'].append(current_ob_id_test) #same for test minus target data because that is not available
    timeobj_test = TimeSeries(t=t_test,m=m_test,e=e_test,label=current_ob_id_test,name=current_ob_id_test)
    features_of_time_series_test = featurize.featurize_single_ts(timeobj_test,features_to_use = feature_list,
        raise_exceptions=False)
    tsdict_test['features_test'].append(list(features_of_time_series_test.values))
    tsdict_test['features_test'][j] += list(test_meta_data.iloc[j,1:11])

#delete large dataframes again
del(test_data)
del(test_meta_data)

print(tsdict)
print(tsdict_test)
"""
#*******************************************************************************************************************
#Now want to utilize some data visualization in order to get a feeling for what really separates these classes
#Going to make plots for each feature where the classes will be color coded in the legend.

#create dummy arrays for demo plot of how logistic regression would work
dummy_array_y = []
dummy_array_x = []
dummy_array_y_16 = []
dummy_array_x_16 = []

#add another sum term to avoid repeat calculations
number_of_features_used = len(tsdict['feature_titles'])

#loop to create plots for each of the features
for k in range(number_of_features_used): #loop through features
    maxy = -100000.0 #variables to set reasonable ranges for the probability distribution plots
    miny = 100000.0
    plt.figure(4+k)
    plt.suptitle(tsdict['feature_titles'][k],fontsize="x-large")

    for v in range(len(tsdict['target_list'])): #loop through all of the classes
        array_for_y_values = []
        array_for_x_values = []
        for w in range(len(tsdict['target'])): #find each data point for each type of class and insert to a list for plotting
            if tsdict['target'][w] == tsdict['target_list'][v]: #if target of current object is the classification in question, add these values to the array for plotting
                array_for_y_values.append(tsdict['features'][w][k])
                array_for_x_values.append(tsdict['target_list'][v])

        #this block is meant to reconfigure the data to show the logistic regression fitting problem
        if tsdict['feature_titles'][k] == 'skew': #data point manually decided upon
            if tsdict['target_list'][v] == 16: #condition for class in question
                dummy_array_y_16 += [1] * len(array_for_y_values) #this group is labeled 1
                dummy_array_x_16 += array_for_y_values #class 16 skew values
            if tsdict['target_list'][v] != 16:
                dummy_array_y += [0] * len(array_for_y_values) #this group is labeled 2
                dummy_array_x += array_for_y_values #every other class skew values

        #set up the distributions and the plot ranges
        dist = norm(sps.mean(array_for_y_values),sps.std(array_for_y_values))
        plt.subplot(1,2,1)
        plt.scatter(array_for_x_values,array_for_y_values,label=tsdict['target_list'][v])
        plt.subplot(1,2,2)
        min_try = min(array_for_y_values)
        max_try = max(array_for_y_values)

        if min_try < miny:
            miny = min(array_for_y_values)
        if max_try > maxy:
            maxy = max(array_for_y_values)


        plt.subplot(1,2,2)
        #set a range for the distributions
        x = np.arange(-1000,1000,.05)
        #plot the distributions
        plt.plot(dist.pdf(x), x)

    plt.subplot(1,2,1)
    plt.title("Values for Each Class")
    plt.legend(loc=2,bbox_to_anchor=(2.2,1))
    plt.xlabel("Class")
    plt.ylabel("Value")
    plt.subplot(1,2,2)
    plt.xlim(-.0005,.05)
    plt.ylim(miny,maxy)
    plt.title("Probability Distribution")
    plt.show()

#make a logistic regression plot for demo plot
xrange = np.arange(-15,22,.05)
logreg = LogisticRegression()
X = np.array(dummy_array_x + dummy_array_x_16)
X = np.reshape(X,(-1,1))
y = np.array(dummy_array_y + dummy_array_y_16)
y = np.reshape(y,(-1,1))

logreg.fit(X,y)

def plotsigmoid(coeff,intercept,x):
    z = (coeff*x) + intercept
    value = []
    for i in range(len(z)):
        if z[i] < 0:
            value.append(1 - 1/(1 + math.exp(z[i])))
        if z[i] >= 0:
            value.append(1/(1 + math.exp(-z[i])))
    return value

values = plotsigmoid(logreg.coef_[0],logreg.intercept_,xrange)

#plot the data from the dummy arrays and the fitted sigmoid for the demo plot
plt.figure(999)
plt.title("Conversion of Data into a\n Binary Classification Problem")
plt.plot(dummy_array_x_16,dummy_array_y_16,'o',color='red',label='Class 16')
plt.plot(dummy_array_x,dummy_array_y,'o',color='blue',label='All Other Classes')
plt.plot(xrange,values,'-',color='green')
plt.legend()
plt.show()

"""

















#ignore this comment
