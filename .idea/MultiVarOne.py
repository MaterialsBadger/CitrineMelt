# This method works for a single variable linear regression

import matplotlib.pyplot as plt
import numpy as np
import pandas
import math
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error

remove_number = 999999.9 # the number that represents NaN in csv file

# Import the CSV using pandas
allData = pandas.read_csv("NaN_Corrected_MT_v2.0.csv")

# Read the melting temp and another variable (in this case, the atomic weight) in from the CSV using the column header
# print(len(allData))
# testData = allData[allData.AtomicVolume_stoich_avg != remove_number]
# print(len(allData))
# print(len(testData))


# Read the melting temp and another variable (in this case, the atomic weight) in from the CSV using the column header
# meltingTemp = allData['MeltingT_stoich_avg']

selectedColumns = []
for i in range(128):
    bestScore = 1000000
    bestColumn = None
    for column in allData:
        if column in selectedColumns or column == 'value' or column == 'formula': continue
        testData = allData.copy()
        currentColumns = list(selectedColumns)
        currentColumns.append(column)

        for item in currentColumns:
            testData = testData[testData[item] != remove_number]

        trainingX = testData[currentColumns]
        trainingX = trainingX.as_matrix()
        trainingY = testData['value']
        trainingY = trainingY.as_matrix()

        #full fit
        regr = linear_model.LinearRegression()
        regr.fit(trainingX,trainingY)
        predictionY = regr.predict(trainingX)


        score = rmse = np.sqrt(mean_squared_error(trainingY, predictionY))

        if (score<bestScore):
            bestColumn = column
            bestScore = score
    selectedColumns.append(bestColumn)

    print(bestColumn)

testData = allData.copy()
for item in selectedColumns:
    testData = testData[testData[item] != remove_number]

trainingX = testData[selectedColumns]
trainingX = trainingX.as_matrix()
trainingY = testData['value']
trainingY = trainingY.as_matrix()
regr = linear_model.LinearRegression()
regr.fit(trainingX,trainingY)
predictedY = regr.predict(trainingX)

# Create a scatter plot of the data
#plt.scatter(trainingY, predictedY)
#axes = plt.gca()
#m, b = np.polyfit(trainingY, predictedY, 1)
#X_plot = np.linspace(axes.get_xlim()[0],axes.get_xlim()[1],100)
#plt.plot(X_plot, m*X_plot + b, '-')
#plt.rcParams.update({'font.size': 18})
#plt.title('3 Best Properties')
#plt.ylabel('Predicted Melting Temp (K)')
#plt.xlabel('Measured Melting Temp(K)')
#plt.ylim(0,3500)
#plt.xlim(0,3500)
#plt.grid()
#plt.show()

# Create a histogram of the features
plt.hist()





'''
for each trial:
    testData = allData[allData.something != remove_number and somethingelse != remove_number]
print(len(testData))
meltingTemp = allData['MeltingT_stoich_avg']
atomicWeight = allData['AtomicWeight_stoich_avg']
otherProperties = allData['AtomicVolume_stoich_avg']


dataProp =['AtomicVolume_stoich_avg']
for dataProp in range(len(meltingTemp)):
    if dataProp = ['% NaN']
        otherProperties = otherProperties


# Convert from pandas dataframe to numpy array
meltingTemp = meltingTemp.as_matrix()
atomicWeight = atomicWeight.as_matrix()
otherProperties = otherProperties.as_matrix()

# Remove instances of NAN from both arrays
indexesToRemove = []
# for each element in the array
for index in range(len(meltingTemp)):
    # If either element is NAN
    if np.isnan(meltingTemp[index]) or np.isnan(atomicWeight[index]) or np.isnan(otherProperties[index]):
        # Add the index to the indexes to remove array
        indexesToRemove.append(index)

#Remove the instances of NaN from both arrays
meltingTemp = np.delete(meltingTemp, indexesToRemove)
atomicWeight = np.delete(atomicWeight, indexesToRemove)
otherProperties = np.delete(otherProperties, indexesToRemove)

regr.fit(meltingTemp, otherProperties)
predictMT = regr.predict(meltingTemp)

# Create a scatter plot of the data
plt.scatter(meltingTemp,predictMT)
axes = plt.gca()
m, b = np.polyfit(meltingTemp, predictMT, 1)
X_plot = np.linspace(axes.get_xlim()[0],axes.get_xlim()[1],100)
plt.plot(X_plot, m*X_plot + b, '-')
plt.rcParams.update({'font.size': 24})
plt.title('Actual Melting Temperature vs. Predicted Melting Temperature with Linear Regression Line')
plt.ylabel('Predicted Melting Temp (K)')
plt.xlabel('Measured Melting Temp(K)')
plt.ylim(0,3500)
plt.xlim(0,3500)
plt.grid()
plt.show()

'''