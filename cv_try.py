
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cross_validation
from sklearn.metrics import mean_squared_error
import pandas
from sklearn import datasets, linear_model

def mean_error(prediction,actual):
    mean_error = 0
    for i ,j in zip(np.asarray(prediction).ravel(),np.asarray(actual).ravel()):
        mean_error += i-j
    return mean_error/len(prediction.ravel())

allData = pandas.read_csv("MT2.csv")
testData = allData.copy()
currentColumns = ['IsNonmetal_stoich_avg','NUnfilled_stoich_avg','IsNonmetal_max']
trainingX = testData[currentColumns]
trainingX = trainingX.as_matrix()
trainingY = testData['value']
trainingY = trainingY.as_matrix()

model = linear_model.LinearRegression()

num_runs = 100
num_folds = 5

Ydata = trainingY
Xdata = trainingX

Y_predicted_best = []
Y_predicted_worst = []

maxRMS = 1
minRMS = 100000000000000000000000

RMS_List = []
ME_List = []
for n in range(num_runs):
    kf = cross_validation.KFold(len(Xdata), n_folds=num_folds, shuffle=True)
    K_fold_rms_list = []
    K_fold_me_list = []
    Overall_Y_Pred = np.zeros(len(Xdata))
    # split into testing and training sets
    for train_index, test_index in kf:
        X_train, X_test = Xdata[train_index], Xdata[test_index]
        Y_train, Y_test = Ydata[train_index], Ydata[test_index]
        # train on training sets
        model.fit(X_train, Y_train)
        Y_test_Pred = model.predict(X_test)
        rms = np.sqrt(mean_squared_error(Y_test, Y_test_Pred))
        me = mean_error(Y_test_Pred,Y_test)
        K_fold_rms_list.append(rms)
        K_fold_me_list.append(me)
        Overall_Y_Pred[test_index] = Y_test_Pred

    RMS_List.append(np.mean(K_fold_rms_list))
    ME_List.append(np.mean(K_fold_me_list))
    if np.mean(K_fold_rms_list) > maxRMS:
        maxRMS = np.mean(K_fold_rms_list)
        Y_predicted_worst = Overall_Y_Pred

    if np.mean(K_fold_rms_list) < minRMS:
        minRMS = np.mean(K_fold_rms_list)
        Y_predicted_best = Overall_Y_Pred

avgRMS = np.mean(RMS_List)
medRMS = np.median(RMS_List)
sd = np.std(RMS_List)
meanME = np.mean(ME_List)

print("Using {}x {}-Fold CV: ".format(num_runs, num_folds))
print("The average RMSE was {:.3f}".format(avgRMS))
print("The median RMSE was {:.3f}".format(medRMS))
print("The max RMSE was {:.3f}".format(maxRMS))
print("The min RMSE was {:.3f}".format(minRMS))
print("The std deviation of the RMSE values was {:.3f}".format(sd))

print(len(Y_predicted_best))
print(len(Ydata))

f, ax = plt.subplots(1, 2, figsize=(11, 5))
ax[0].scatter(Ydata, Y_predicted_best, c='black', s=10)
ax[0].plot(ax[0].get_ylim(), ax[0].get_ylim(), ls="--", c=".3")
ax[0].set_title('Best Fit')
ax[0].text(.05, .88, 'Min RMSE: {:.2f} MPa'.format(minRMS), transform=ax[0].transAxes)
ax[0].text(.05, .81, 'Mean RMSE: {:.2f} MPa'.format(avgRMS), transform=ax[0].transAxes)
ax[0].text(.05, .74, 'Std. Dev.: {:.2f} MPa'.format(sd), transform=ax[0].transAxes)
ax[0].text(.05, .67, 'Mean Mean Error.: {:.2f} MPa'.format(meanME), transform=ax[0].transAxes)
ax[0].set_xlabel('Measured (Mpa)')
ax[0].set_ylabel('Predicted (Mpa)')

ax[1].scatter(Ydata, Y_predicted_worst, c='black', s=10)
ax[1].plot(ax[1].get_ylim(), ax[1].get_ylim(), ls="--", c=".3")
ax[1].set_title('Worst Fit')
ax[1].text(.05, .88, 'Max RMSE: {:.3f}'.format(maxRMS), transform=ax[1].transAxes)
ax[1].set_xlabel('Measured (Mpa)')
ax[1].set_ylabel('Predicted (Mpa)')

f.tight_layout()
f.savefig("cv_best_worst", dpi=200, bbox_inches='tight')
plt.clf()
plt.close()