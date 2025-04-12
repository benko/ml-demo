#!/bin/false
# ^^^ this just means don't allow this to be executed as a stand-alone script

# the basic imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
# but also reporting on the model
from sklearn.metrics import classification_report, confusion_matrix

# load data, extract just the features, and just the labels
wine_data = pd.read_csv("./WineQT.csv", delimiter=",")
wine_features = wine_data.drop("quality", axis=1).drop("Id", axis=1)
wine_labels = np.ravel(wine_data['quality'])

# split the dataset into train and test subsets
# note, while it may be tempting to get creative with variable names, such as
# features_train, features_test, labels_train, labels_test...
# it's WAY TOO MUCH typing, and most examples use x for features (as in, input
# data) and y for labels (as in, result)
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(wine_features, wine_labels, test_size=0.5, random_state=50)

# normalise the data (meaning spread it ALL out on a scale between a..b)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler().fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# train the SVC model
print("**** TESTING C-Support Vector Classification ****")

from sklearn.svm import SVC

svc_model = SVC()
svc_model.fit(x_train, y_train)

# now test the fitness with the test subset
svc_y_predict = svc_model.predict(x_test)

# visualise it
svc_cm = np.array(confusion_matrix(y_test, svc_y_predict, labels=[0,1,2,3,4,5,6,7,8,9,10]))
svc_conf_matrix = pd.DataFrame(svc_cm)
print(svc_conf_matrix)

# visualise it in a nice picture
sns.heatmap(svc_conf_matrix, annot=True, fmt='g')
plt.show()

# # train the NuSVC model
# print("**** TESTING Nu-Support Vector Classification ****")

# from sklearn.svm import NuSVC

# nusvc_model = NuSVC(nu=0.2)
# nusvc_model.fit(x_train, y_train)

# # now test the fitness with the test subset
# nusvc_y_predict = svc_model.predict(x_test)

# # visualise it
# nu_cm = np.array(confusion_matrix(y_test, nusvc_y_predict, labels=[0,1,2,3,4,5,6,7,8,9,10]))
# nu_conf_matrix = pd.DataFrame(nu_cm)
# print(nu_conf_matrix)

# # visualise it in a nice picture
# sns.heatmap(nu_conf_matrix, annot=True, fmt='g')
# plt.show()
