import numpy as np
import pandas as pd
from pandas import DataFrame
from itertools import islice
import pwlf
from sklearn.neighbors import LocalOutlierFactor
from sklearn import preprocessing
from sklearn import ensemble
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score,mean_squared_error, silhouette_score, calinski_harabasz_score, davies_bouldin_score,accuracy_score,accuracy_score, confusion_matrix,precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import mutual_info_classif
import statsmodels.api as sm




class DataEngineering:
    def NormalizeData(self, dataset):
        """
        Normalize the data using Min-Max scaling.
        """
        x = dataset.values  # returns a numpy array
        scaler = preprocessing.MinMaxScaler()
        scaled_x = scaler.fit_transform(x)
        data = pd.DataFrame(scaled_x, columns=dataset.columns)
        return data

    def CleanData(self, dataset,threshold=0.8):
        """This function takes in as input a dataset, and returns a clean dataset.

        :param data: This is the dataset that will be cleaned.
        :param treshold: This is the treshold that decides whether columns are deleted or their missing values filled.
        :return: A dataset that does not have any missing values.
        """
        data = dataset.replace('?', np.nan)
        data = data.loc[:, data.isnull().mean() < threshold]  # filter data
        imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        for i in data.columns:
            imputer = imputer.fit(data[[i]])
            data[[i]] = imputer.transform(data[[i]])
        return data

    def create_dummy_variables(self,dataset, variables):
        for var in variables:
            cat_list = 'var' + '_' + var
            cat_list = pd.get_dummies(dataset[var], prefix=var)
            datatemp = dataset.join(cat_list)
            dataset = datatemp
        data_vars = dataset.columns.values.tolist()
        to_keep = [i for i in data_vars if i not in variables]
        data = dataset[to_keep]
        return (data)

    def remove_outliers(self,dataset, Xcol, ycol, testsize=0.33):
        X = dataset[Xcol].values
        y = dataset[ycol].values
        # split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testsize, random_state=1)
        # identify outliers in the training dataset
        lof = LocalOutlierFactor()
        yhat = lof.fit_predict(X_train)
        # select all rows that are not outliers
        mask = yhat != -1
        X_train, y_train = X_train[mask, :], y_train[mask]
        # Set data back
        data = pd.DataFrame(X_train, columns=Xcol)
        data[ycol] = y_train
        return (data)


class ModelFitting:
    def LinearSKDefaultModel(self,data, Xcol,ycol):
        # use sm for P-values
        X = data[Xcol].values
        y = data[ycol]
        X = sm.add_constant(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
        smmodel = sm.OLS(y_train, X_train).fit()

        y = y.values.reshape(-1, 1)
        # perform linear regression
        model = LinearRegression().fit(X, y)

        # get the coefficient
        coef = model.coef_
        columns = {'coeff': coef[0][1:], 'pvalue': smmodel.pvalues.round(4).values[1:]}
        linearData = DataFrame(data=columns, index=Xcol)
        # calculate the r-squared value
        y_pred = model.predict(X)
        r2 = r2_score(y, y_pred)
        return (linearData, r2)

    def LinearDefaultModel(self,data, Xcol,ycol):
        X = data[Xcol].values
        y = data[ycol]
        X = sm.add_constant(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
        model = sm.OLS(y_train, X_train).fit()

        return (model)

    def LogisticrDefaultModel(self,data, Xcol,ycol):
        X = data[Xcol].values
        y = data[ycol]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
        model = sm.Logit(y_train, X_train).fit()

        logisticRegr = LogisticRegression()
        logisticRegr.fit(X_train, y_train)
        deviance = 2 * logisticRegr.score(X, y) * len(y)  # 2*(-log-likelihood of fitted model)
        df = len(y) - logisticRegr.coef_.shape[1] - 1
        devDdf = deviance / df

        return (model, devDdf)

    def GradientBoostingDefaultModel(self,data, Xcol,ycol, gbr_params):
        X = data[Xcol].values
        y = data[ycol]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
        model = ensemble.GradientBoostingRegressor(**gbr_params)
        model.fit(X_train, y_train)

        mse = mean_squared_error(y_test, model.predict(X_test))
        rmse = mse ** (1 / 2.0)
        r2 = model.score(X_test, y_test)
        importance = model.feature_importances_
        train_errors = []
        test_errors = []

        for i, y_pred in enumerate(model.staged_predict(X_train)):
            if i % 10 == 0:
                mse_train = mean_squared_error(y_train, y_pred)
                train_errors.append(np.round(mse_train, 3))
                y_pred_test = model.staged_predict(X_test)
                mse_test = mean_squared_error(y_test, next(islice(y_pred_test, i + 1)))
                test_errors.append(np.round(mse_test, 3))
        columns = {'important': importance}
        DTData = DataFrame(data=columns, index=Xcol)
        imp = ""
        for ind in DTData.index:
            if DTData['important'][ind] == max(DTData['important']):
                imp = ind
        return (model, mse, rmse, r2, imp, train_errors, test_errors)

    def RandomForestDefaultModel(self,X, y, Xcol, n_estimators, max_depth):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
        # Limit depth of tree to 3 levels
        rf_small = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth)
        rf_small.fit(X_train, y_train)
        # Extract the small tree
        tree_small = rf_small.estimators_[5]
        # R2
        r2 = rf_small.score(X_train, y_train)
        # Use the forest's predict method on the test data
        predictions = rf_small.predict(X_test)
        # Calculate the absolute errors
        errors = abs(predictions - y_test)
        # Calculate mean absolute percentage error (MAPE)
        mape = 100 * (abs(predictions - y_test) / y_test)
        # Calculate and display accuracy
        accuracy = 100 - np.mean(mape)
        mae = metrics.mean_absolute_error(y_test, predictions)
        mse = metrics.mean_squared_error(y_test, predictions)
        rmse = mse ** (1 / 2.0)
        importance = rf_small.feature_importances_
        columns = {'important': importance}
        DTData = DataFrame(data=columns, index=Xcol)
        return (tree_small, rf_small, DTData, r2, mse, rmse)

    def DecisionTreeDefaultModel(self,X, y, Xcol, max_depth):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
        model = DecisionTreeRegressor(random_state=0, max_depth=max_depth)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        r2 = model.score(X_train, y_train)
        mse = mean_squared_error(y_test, predictions)
        rmse = mse ** (1 / 2.0)
        importance = model.feature_importances_
        columns = {'important': importance}
        DTData = DataFrame(data=columns, index=Xcol)

        return (model, r2, mse, rmse, DTData)

    def piecewise_linear_fit(self,data, Xcol, ycol, num_breaks):
        """
        Perform piecewise linear fit using the pwlf library.

        Parameters:
        - dataframe: pandas DataFrame
        - Xcol: Name of the column to be used as X data
        - ycol: Name of the column to be used as y data
        - num_breaks: Number of breaks (segments + 1)

        Returns:
        - model: Fitted pwlf model
        - slopes: Slopes of each segment
        - segment_points: Start and end x points of each segment
        - segment_values: y values at the start and end points of each segment
        - max_slope_segment: Details (slope and start-end points) of the segment with the maximum absolute slope
        """

        # Extract data
        x = data[Xcol].values
        y = data[ycol].values

        # Fit the model
        my_pwlf = pwlf.PiecewiseLinFit(x, y)
        breaks = my_pwlf.fit(num_breaks)

        # Calculate slopes
        slopes = my_pwlf.slopes

        # Get segment start and end points and their corresponding y values
        segment_points = [(breaks[i], breaks[i + 1]) for i in range(len(breaks) - 1)]
        segment_values = [(my_pwlf.predict([segment_points[i][0]])[0], my_pwlf.predict([segment_points[i][1]])[0]) for i in
                          range(len(breaks) - 1)]

        # Identify the segment with the maximum absolute slope
        max_slope_idx = abs(slopes).argmax()
        max_slope_segment = {
            'slope': slopes[max_slope_idx],
            'start_end_points': segment_points[max_slope_idx],
            'start_end_values': segment_values[max_slope_idx]
        }
        r2 = my_pwlf.r_squared()

        return my_pwlf, slopes, segment_points, segment_values, max_slope_segment

    def RidgeClassifierModel(self,dataset, Xcol, ycol,class1,class2,cvnum=5):
        X = dataset[Xcol]
        y = dataset[ycol]
        y = y.map({class1: 0, class2: 1})
        # Split the data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Instantiate the RidgeClassifier model
        rclf = RidgeClassifier()

        # Fit the model to the training data
        rclf.fit(X_train, y_train)

        # Make predictions on the test data
        y_pred = rclf.predict(X_test)

        # Compute accuracy, precision, recall, F1-score
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        # Compute area under the ROC curve
        y_prob = rclf.decision_function(X_test)
        roc_auc = roc_auc_score(y_test, y_prob)
        confusionmatrix = confusion_matrix(y_test, y_pred)
        # Perform PCA for dimensionality reduction
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_test)

        # Compute feature importances
        importances = rclf.coef_[0]
        cv_scores = cross_val_score(rclf, X, y, cv=cvnum)
        return (rclf,pca,y_test, y_prob,roc_auc,X_pca,accuracy,importances,confusionmatrix,cv_scores)


    def KNeighborsClassifierModel(self,dataset, Xcol, ycol,Knum=3,cvnum=5):
        # Extract the feature matrix X and target vector y
        X = dataset[Xcol]
        y = dataset[ycol]
        # Split the data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        # Instantiate the K Neighbors Classifier model
        k = Knum  # Number of neighbors to consider
        clf = KNeighborsClassifier(n_neighbors=k)
        # Fit the model to the training data
        clf.fit(X_train, y_train)
        # Make predictions on the test data
        y_pred = clf.predict(X_test)
        # Compute accuracy, precision, recall, F1-score
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        # Calculate mutual information between each feature and the target variable
        feature_importances = mutual_info_classif(X, y)
        # Calculate confusion matrix
        confusionmatrix = confusion_matrix(y_test, y_pred)
        # Calculate cross-validation scores
        cv_scores = cross_val_score(clf, X, y, cv=cvnum)
        return (accuracy,precision,feature_importances,recall,f1,confusionmatrix,cv_scores)

    def SVCClassifierModel(self,dataset, Xcol, ycol,kernel='linear', C=1.0,cvnum=5):
        # Extract the feature matrix X and target vector y
        X = dataset[Xcol]
        y = dataset[ycol]
        # Split the data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        # Instantiate the Support Vector Machine (SVM) model
        clf = SVC(kernel=kernel, C=C)  # Linear kernel with regularization parameter C=1.0
        # Fit the model to the training data
        clf.fit(X_train, y_train)
        # Make predictions on the test data
        y_pred = clf.predict(X_test)
        # Compute accuracy, precision, recall, F1-score
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        # Calculate confusion matrix
        confusionmatrix = confusion_matrix(y_test, y_pred)
        cv_scores = cross_val_score(clf, X, y, cv=cvnum)  # 5-fold cross-validation
        return (accuracy,precision,recall,f1,confusionmatrix,cv_scores)

