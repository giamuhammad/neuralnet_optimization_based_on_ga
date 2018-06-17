
# coding: utf-8

# In[2]:

#Common Library
import numpy as np
import pandas as pd
from __future__ import division
from datetime import datetime, timedelta
from dateutil import parser
import time
#Sklearn
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
#Matplotlib
import matplotlib.pyplot as plt
# TPOT
from tpot import TPOTRegressor
# NeuPy
from neupy import algorithms, layers, environment


# In[3]:

def DataPrepare(df):
    data = df.drop(df.columns[[15,16]],axis=1)
    #target PT08.S1, PT08.S2, PT08.S3, PT08.S4, PT08.S5
    col_target = ['PT08.S1(CO)', 'PT08.S2(NMHC)', 'PT08.S3(NOx)', 'PT08.S4(NO2)', 'PT08.S5(O3)']
    target = data[col_target]

    data.drop(['PT08.S1(CO)', 'PT08.S2(NMHC)', 'PT08.S3(NOx)', 'PT08.S4(NO2)', 'PT08.S5(O3)'], axis=1, inplace=True)

    data['Date'] = df.Date.astype(str).str.cat(df.Time.astype(str), sep=' ')
    data = data.drop(['Time'], axis=1)
    data = data.rename(columns={'Date': 'DateTime'})
    data['DateTime'] = pd.to_datetime(data['DateTime'])
    return data, target

def totimestamp(dt, epoch=datetime(1970,1,1)):
    td = dt - epoch
    # return td.total_seconds()
    return (td.microseconds + (td.seconds + td.days * 86400) * 10**6) / 10**6 

def ConvertDateTime(source):
    X = source.values
    for i in xrange(0, len(source['DateTime'])):
        #del X[i,:]
        tmp = parser.parse(str(source['DateTime'][i]))
        #print str(int(totimestamp(tmp)))
        #timestamp.append(str(int(totimestamp(tmp))))
        X[i,0] = float(totimestamp(tmp))
        X[i,1] = X[i,1].replace(',', '.')
        X[i,1] = float(X[i,1])
        X[i,3] = X[i,3].replace(',', '.')
        X[i,3] = float(X[i,3])
        X[i,6] = X[i,6].replace(',', '.')
        X[i,6] = float(X[i,6])
        X[i,7] = X[i,7].replace(',', '.')
        X[i,7] = float(X[i,7])
        X[i,8] = X[i,8].replace(',', '.')
        X[i,8] = float(X[i,8])
    return X

def Clean(X, Y):
    mx = []
    my = []
    
    for i in xrange(0, len(X[0])):
        mx.append(np.mean(X[:,i]))
        
    for i in xrange(0, len(Y[0])):
        my.append(np.mean(Y[:,i]))
    
    for i in xrange(0, len(X)):
        for j in xrange(1, len(X[0])):
            if X[i,j] == -200.0:
                X[i,j] = mx[j]
                #print(str(i) + str(j))
    for i in xrange(0, len(Y)):
        for j in xrange(1, len(Y[0])):
            if Y[i,j] == -200.0:
                Y[i,j] = my[j]
                #print(str(i) + " " + str(j))
    return X, Y

def DataPreparePipeline(source, showData=False):
    data, target = DataPrepare(source);
    X = ConvertDateTime(data)
    X, Y = Clean(X, target.values)
    return X, Y


# In[4]:

def FeatureSelectionModel(x, y):
    max_depth = 30
    #regr_multirf = MultiOutputRegressor(RandomForestRegressor(max_depth=max_depth, random_state=0))
    #regr_multirf.fit(x, y)
    regr_rf = RandomForestRegressor(max_depth=max_depth, random_state=2)
    regr_rf.fit(x, y)
    model = SelectFromModel(regr_rf, prefit=True)
    return model

def FeatureSelectionTransform(fs_model, x):
    x_reduced = fs_model.transform(x)
    print(x_reduced.shape)
    return x_reduced

def FeatureSelectionInverse(fs_model, x_reduced):
    x = fs_model.inverse_transform(x_reduced)
    print(x.shape)
    return x


# In[5]:

def BuildDataScale(X, Y):
    xs = preprocessing.MinMaxScaler()
    ys = preprocessing.MinMaxScaler()
    xs.fit(X)
    ys.fit(Y)
    return xs, ys

def DataScaleTransform(scale_model, data):
    data_scaled = scale_model.transform(data)
    return data_scaled

def DataScaleInverse(scale_model, data_scaled):
    data = scale_model.inverse_transform(data_scaled)
    return data


# In[6]:

def ANNForecastBuild(layer, step):
    return algorithms.ConjugateGradient(
        connection=layer,
        search_method='golden',
        show_epoch=25,
        verbose=True,
        step=step,
        addons=[algorithms.LinearSearch],
    )

def TrainANN(model, x, y, e = 1000):
    environment.reproducible()
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, test_size=0.2)
    model.train(x_train, y_train, x_test, y_test, epochs=e)
    return x_train, x_test, y_train, y_test


# In[7]:

def Evaluation(y_true, y_pred):
    print("R2")
    print(r2_score(y_true, y_pred))
    print("RMSE")
    print(np.sqrt(mean_squared_error(y_true, y_pred)))
    #print(mean_squared_error(y_true, y_pred))
    print("RMSE per output")
    print(np.sqrt(mean_squared_error(y_true, y_pred, multioutput='raw_values')))
    x = range(0, len(y_pred))
    for i in range(0, len(y_pred[0])):   
        plt.figure(figsize=(20,10))
        plt.scatter(x, y_pred[:,i], c='r', marker='^')
        plt.scatter(x, y_true[:,i], c='b', marker='o')
        plt.show()


# In[8]:

df = pd.read_csv('AirQualityUCI/AirQualityUCI.csv', delimiter=';')
df = df[0:9357]
X, Y = DataPreparePipeline(df)


# In[11]:

#test pipeline 1 - Feature Selection -> ANN
#pca = MyPCA(X)
fs_model = FeatureSelectionModel(X, Y)
X_reduced = FeatureSelectionTransform(fs_model, X)
xs, ys = BuildDataScale(X_reduced, Y)
x_scale = DataScaleTransform(xs, X_reduced)
y_scale = DataScaleTransform(ys, Y)
layer = [
            layers.Input(len(X_reduced[0])),
            layers.Sigmoid(15),
            layers.Sigmoid(5),
        ]
net = ANNForecastBuild(layer, 0.1)
start_time = time.time()
x_train, x_test, y_train, y_test = TrainANN(net, x_scale, y_scale, e=100)
elapsed_time = time.time() - start_time
print(time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

#evaluation
y_pred = net.predict(x_test)
y_pred = DataScaleInverse(ys, y_pred)
y_true = DataScaleInverse(ys, y_test)
Evaluation(y_true, y_pred)


# In[9]:

#test pipeline 2 - ANN
xs, ys = BuildDataScale(X, Y)
x_scale = DataScaleTransform(xs, X)
y_scale = DataScaleTransform(ys, Y)
layer = [
            layers.Input(len(X[0])),
            layers.Sigmoid(15),
            layers.Sigmoid(5),
        ]
net = ANNForecastBuild(layer, 0.1)
start_time = time.time()
x_train, x_test, y_train, y_test = TrainANN(net, x_scale, y_scale, e=100)
elapsed_time = time.time() - start_time
print(time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

#evaluation
y_pred = net.predict(x_test)
y_pred = DataScaleInverse(ys, y_pred)
y_true = DataScaleInverse(ys, y_test)
Evaluation(y_true, y_pred)


# In[9]:

import math

from optimal import GenAlg
from optimal import Problem
from optimal import helpers

err_threshold = 140


def decode_param(binary):
    learning_rate = helpers.binary_to_float(binary[0:16], 0.4, 1.0)
    neuron = helpers.binary_to_int(binary[16:32], 10, 15)
    return learning_rate, neuron

def ann_fs_fitness(solution):
    learning_rate, neuron = solution
    
    #
    fs_model = FeatureSelectionModel(X, Y)
    X_reduced = FeatureSelectionTransform(fs_model, X)
    xs, ys = BuildDataScale(X_reduced, Y)
    x_scale = DataScaleTransform(xs, X_reduced)
    y_scale = DataScaleTransform(ys, Y)
    layer = [
                layers.Input(len(X_reduced[0])),
                layers.Sigmoid(neuron),
                layers.Sigmoid(5),
            ]
    net = ANNForecastBuild(layer, learning_rate)
    start_time = time.time()
    x_train, x_test, y_train, y_test = TrainANN(net, x_scale, y_scale, e=100)
    elapsed_time = time.time() - start_time
    print(time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

    #evaluation
    y_pred = net.predict(x_test)
    y_pred = DataScaleInverse(ys, y_pred)
    y_true = DataScaleInverse(ys, y_test)
    #Evaluation(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    m_rmse = np.sqrt(mean_squared_error(y_true, y_pred, multioutput='raw_values'))
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print('m_rmse : ' + str(m_rmse));
    print('r2 : ' + str(r2));
    print('rmse : ' + str(rmse));
    print('lr : ' + str(learning_rate) + ', hidden neuron : ' + str(neuron));
    output = rmse

    finished = output <= err_threshold
    #finished = output <= 0.01
    fitness = 1 / output
    print(finished)
    print(fitness)
    return fitness, finished
    
def ann_fitness(solution):
    learning_rate, neuron = solution
    
    #
    xs, ys = BuildDataScale(X, Y)
    x_scale = DataScaleTransform(xs, X)
    y_scale = DataScaleTransform(ys, Y)
    layer = [
                layers.Input(len(X[0])),
                layers.Sigmoid(neuron),
                layers.Sigmoid(5),
            ]
    net = ANNForecastBuild(layer, learning_rate)
    start_time = time.time()
    x_train, x_test, y_train, y_test = TrainANN(net, x_scale, y_scale, e=100)
    elapsed_time = time.time() - start_time
    print(time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

    #evaluation
    y_pred = net.predict(x_test)
    y_pred = DataScaleInverse(ys, y_pred)
    y_true = DataScaleInverse(ys, y_test)
    #Evaluation(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    m_rmse = np.sqrt(mean_squared_error(y_true, y_pred, multioutput='raw_values'))
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print('m_rmse : ' + str(m_rmse));
    print('r2 : ' + str(r2));
    print('rmse : ' + str(rmse));
    print('lr : ' + str(learning_rate) + ', hidden neuron : ' + str(neuron));
    output = rmse

    finished = output <= err_threshold
    #finished = output <= 0.01
    fitness = 1 / output
    print(finished)
    print(fitness)
    return fitness, finished


# In[10]:

#Pipeine 3 - FS->ANN->GA
ann_ml = Problem(ann_fs_fitness, decode_function=decode_param)
my_genalg = GenAlg(32, mutation_chance=0.1, crossover_chance=0.8)
start_time = time.time()
best_solution = my_genalg.optimize(ann_ml, max_iterations=15, n_processes=1)
elapsed_time = time.time() - start_time
print(time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

print best_solution


# In[15]:

print best_solution


# In[11]:

#Pipeine 4 - ANN->GA
ann_ml = Problem(ann_fitness, decode_function=decode_param)
my_genalg = GenAlg(32, mutation_chance=0.1, crossover_chance=0.8)
start_time = time.time()
best_solution = my_genalg.optimize(ann_ml, max_iterations=15, n_processes=1)
elapsed_time = time.time() - start_time
print(time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

print best_solution


# In[ ]:



