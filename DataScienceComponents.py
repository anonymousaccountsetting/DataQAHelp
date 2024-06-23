import numpy as np
import pandas as pd
from pandas import DataFrame
from itertools import islice
import pwlf
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn import ensemble
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score,mean_squared_error, mean_absolute_error,silhouette_score, calinski_harabasz_score, davies_bouldin_score,accuracy_score,accuracy_score, confusion_matrix,precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import mutual_info_classif
import statsmodels.api as sm
from pycaret import classification
from pycaret import regression
import cv2
import statistics
import heapq
from statsmodels.stats.outliers_influence import variance_inflation_factor




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

    def fill_missing_with_frequently_values(self,data,colname):
        data[colname].fillna(data[colname].value_counts().index[0], inplace=True)
        return (data)

    def fill_missing_with_mean(self,data,colname):
        data[colname] = data[colname].fillna(data[colname].mean())
        return (data)

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


    def calculate_vif(self,data, Xcol):
        X = data[Xcol]
        vif_data = pd.DataFrame()
        vif_data["feature"] = X.columns
        vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        return vif_data

    def mean_absolute_percentage_error(self,y_true, y_pred):

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        non_zero_mask = y_true != 0
        y_true = y_true[non_zero_mask]
        y_pred = y_pred[non_zero_mask]

        mape = np.mean(np.abs((y_true - y_pred) / y_true))
        return mape

    def MseRmseMae(self,y_true, y_pred):
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        return (mse,rmse,mae)

    def set_values_to_0_and_1(self,data,colname,valuesShouldBeZero):
        data[colname] = data[colname].apply(
            lambda x: 0 if x in valuesShouldBeZero else 1)
        return (data)

    def set_date_columns_to_datetime(self,data,colname):
        data[colname] = data[colname].apply(pd.to_datetime)
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
        mape=DataEngineering().mean_absolute_percentage_error(y_test, y_pred)
        r2 = r2_score(y, y_pred)
        mse,rmse,mae=DataEngineering().MseRmseMae(y_test, y_pred)
        vif=DataEngineering().calculate_vif(data,Xcol)
        return (linearData, r2,mape,mse,rmse,mae,vif)

    def LinearDefaultModel(self,data, Xcol,ycol):
        X = data[Xcol].values
        y = data[ycol]
        X = sm.add_constant(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
        model = sm.OLS(y_train, X_train).fit()
        mape=DataEngineering().mean_absolute_percentage_error(y_test, model.predict(X_test))
        mse, rmse, mae = DataEngineering().MseRmseMae(y_test, model.predict(X_test))
        vif=DataEngineering().calculate_vif(data,Xcol)

        return (model,mape,mse,rmse,mae,vif)


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

        y_pred = logisticRegr.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        y_pred_prob = logisticRegr.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_pred_prob)

        return (model, devDdf,accuracy,auc)

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

        return my_pwlf, slopes, segment_points, segment_values, max_slope_segment,breaks

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

    def PValueCalculation(self,data,Xcol,ycol):
        X = data[Xcol].values
        y = data[ycol]
        X = sm.add_constant(X)
        model = sm.OLS(y, X).fit()
        p_values = model.pvalues

        p_values_df = pd.DataFrame({
            'Feature': ['const'] + Xcol,
            'P-Value': p_values
        })

        p_values_df = p_values_df[p_values_df['Feature'] != 'const']
        # print(p_values_df)
        return p_values_df

class DataDescription:
    def general_description(self, data, Xcol, ycol):
        X = data[Xcol].values
        y = data[ycol].values
        last_X = X[-1]
        last_X2=X[-2]
        last_y = y[-1]
        difference = y[-2] - y[-1]
        percentage_change = (difference / y[-2]) * 100
        max_value = np.max(y)
        max_y_X = X[np.argmax(y)]
        return (last_X,last_X2,last_y,difference,percentage_change,max_value,max_y_X)

    def loop_mean_compare(self,dataset, Xcol, ycol):
        diff = [0] * np.size(Xcol)
        i = 0
        for ind in Xcol:
            diff[i] = (statistics.mean(dataset[ind]) - statistics.mean(dataset[ycol]))
            i = i + 1
        return (diff)

    def find_row_n_max(self,dataset, Xcol, r=0, max_num=5):
        row_data = dataset[Xcol].values[0:r + 1][r]
        max_data = (heapq.nlargest(max_num, row_data))
        max_factor = []
        for ind in Xcol:
            if dataset[ind].values[0:r + 1][r] in max_data:
                max_factor.append(ind)
        return (max_factor)

    def detect_same_elements(self,list1, list2):
        same_element = 0
        for i in list1:
            if i in list2:
                same_element = same_element + 1
        return (same_element)

    def select_one_element(self,dataset, Xcol, ycol):
        datasize = np.size(dataset[Xcol])
        y = dataset[ycol].values[0:datasize][datasize - 1]
        X = dataset[Xcol].values[0:datasize][datasize - 1]
        return (X, y)

    def find_all_zero_after_arow(self,dataset, Xcol, ycol):
        period = dataset[ycol]
        zero_lastdata = ""
        for ind in Xcol:
            remain_num = dataset[ind].values
            for i in range(np.size(remain_num)):
                if remain_num[i] == 0:
                    zero_lastdata = period[i]
        return (zero_lastdata)

    def find_column_mean(self,dataset):
        meancol = statistics.mean(dataset)
        return (meancol)

    def two_point_percent_differ(self,data,point1,point2):
        data['Change_Percentage'] = (data[point2] - data[point1]) / data[point1] * 100
        return (data)

class FindBestModel:
    def __init__(self):
        pass

    def SHAP_interpretion(self,imp_shap, imp_var):
        m = 0
        n = 0
        imp_pos_sum = 0
        imp_pos_value_sum = 0
        imp_neg_sum = 0
        imp_neg_value_sum = 0
        for i in imp_shap['NUM']:
            if float(imp_shap['SHAP Value'].loc[imp_shap['NUM'] == i]) >= 0:
                imp_pos_sum = imp_pos_sum + float(imp_shap['SHAP Value'].loc[imp_shap['NUM'] == i])
                imp_pos_value_sum = imp_pos_value_sum + float(imp_shap[imp_var].loc[imp_shap['NUM'] == i])
                m = m + 1
            else:
                imp_neg_sum = imp_neg_sum + float(imp_shap['SHAP Value'].loc[imp_shap['NUM'] == i])
                imp_neg_value_sum = imp_neg_value_sum + float(imp_shap[imp_var].loc[imp_shap['NUM'] == i])
                n = n + 1
        imp_pos_ave = imp_pos_sum / m
        imp_pos_value_ave = imp_pos_value_sum / m
        imp_neg_ave = imp_neg_sum / n
        imp_neg_value_ave = imp_neg_value_sum / n
        return (imp_pos_ave, imp_pos_value_ave, imp_neg_ave, imp_neg_value_ave)
    def pycaret_create_model(self,types, modelname):
        if types == 0:
            model = classification.create_model(modelname)
            tuned_model = classification.tune_model(model)
            classification.plot_model(tuned_model, plot='error', save=True)
            classification.plot_model(tuned_model, plot='feature', save=True)
            if modelname in ['dt', 'rf', 'et', 'lightgbm','xgboost']:
                classification.interpret_model(tuned_model, save=True)
            if modelname in ['lr', 'lda', 'ridge', 'svm']:
                absolute_values = np.abs(tuned_model.coef_)
                column_means = np.mean(absolute_values, axis=0)
                # print(absolute_values)
                # print(column_means)
                importance = pd.DataFrame({'Feature': classification.get_config('X_train').columns,
                                       'Value': column_means}).sort_values(by='Value',
                                                                                                    ascending=False)
            elif modelname in ['rf', 'et', 'gbc', 'xgboost', 'lightgbm', 'catboost', 'ada', 'dt']:
                importance = pd.DataFrame({'Feature': classification.get_config('X_train').columns,
                                       'Value': abs(tuned_model.feature_importances_)}).sort_values(by='Value',ascending=False)
            else:
                importance=''

            # importance = pd.DataFrame({'Feature': classification.get_config('X_train').columns,
            #                            'Value': abs(tuned_model.feature_importances_)}).sort_values(by='Value',
            #                                                                                         ascending=False)
            # importance = pd.DataFrame({'Feature': classification.get_config('X_train').columns,
            #                            'Value': abs(tuned_model.coef_[0])}).sort_values(by='Value', ascending=False)

            print(modelname)
            for ind in importance.index:
                if importance['Value'][ind] == max(importance['Value']):
                    imp_var = importance['Feature'][ind]
            classification.predict_model(tuned_model)
            results = classification.pull(tuned_model)

            # imp_figure = cv2.imread('Feature Importance.png')
            # Error_figure = cv2.imread('Prediction Error.png')
            imp_figure = ''
            Error_figure = ''
            # shap_values = shap.TreeExplainer(tuned_model).shap_values(regression.get_config('X_train'))
            # imp_shap = pd.DataFrame({imp_var: regression.get_config('X_train')[imp_var],
            #                          'SHAP Value': shap_values[:, 0], 'NUM': range(len(shap_values[:, 0]))})
            # SHAP_figure = cv2.imread('SHAP summary.png')
            # imp_pos_ave, imp_pos_value_ave, imp_neg_ave, imp_neg_value_ave = FindBestModel.SHAP_interpretion(self,imp_shap, imp_var)
            return (importance['Feature'], imp_var, results['Accuracy'][0], results['AUC'][0], imp_figure, Error_figure,importance)
        elif types == 1:
            model = regression.create_model(modelname)
            tuned_model = regression.tune_model(model)
            regression.plot_model(tuned_model, plot='error', save=True)
            regression.plot_model(tuned_model, plot='feature', save=True)
            print(modelname)
            if modelname in ['dt', 'rf', 'et', 'lightgbm']:
                regression.interpret_model(tuned_model, save=True)
            if modelname in ['gbr', 'rf', 'catboost', 'lightgbm', 'et', 'ada', 'xgboost', 'dt']:
                importance = pd.DataFrame({'Feature': regression.get_config('X_train').columns,
                                       'Value': abs(tuned_model.feature_importances_)}).sort_values(by='Value',
                                                                                                    ascending=False)
            elif modelname in ['llar', 'ridge', 'br', 'lar', 'lasso', 'lr', 'huber', 'omp', 'par', 'en']:
                importance = pd.DataFrame({'Feature': regression.get_config('X_train').columns,
                                       'Value': abs(tuned_model.coef_)}).sort_values(by='Value',ascending=False)
            else:
                importance=''
            for ind in importance.index:
                if importance['Value'][ind] == max(importance['Value']):
                    imp_var = importance['Feature'][ind]
            regression.predict_model(tuned_model)
            results = regression.pull(tuned_model)
            # imp_figure = cv2.imread('Feature Importance.png')
            # Error_figure = cv2.imread('Prediction Error.png')
            imp_figure = ''
            Error_figure = ''

            # shap_values = shap.TreeExplainer(tuned_model).shap_values(regression.get_config('X_train'))
            # print(shap_values)
            # if modelname in ['gbr','rf','catboost','lightgbm','et','xgboost','dt']:
            #     explainer = shap.Explainer(tuned_model)
            # else:
            #     print(regression.get_config('X_train'))
            #     masker = shap.maskers.Independent(regression.get_config('X_train'))
            #     explainer = shap.KernelExplainer(model.predict, masker)
            # shap_values=explainer(regression.get_config('X_test'))
            # # imp_shap = pd.DataFrame({imp_var: regression.get_config('X_train')[imp_var],
            # #                          'SHAP Value': shap_values[:, 0], 'NUM': range(len(shap_values[:, 0]))})
            # print(shap_values)
            # imp_shap = pd.DataFrame({
            #     imp_var: regression.get_config('X_train')[:, 0],
            #     'SHAP Value': shap_values[:, 0].values,
            #     'NUM': range(len(shap_values[:, 0]))
            # })
            #
            # imp_pos_ave, imp_pos_value_ave, imp_neg_ave, imp_neg_value_ave = FindBestModel.SHAP_interpretion(self,imp_shap, imp_var)
            # SHAP_figure = cv2.imread('SHAP summary.png')
            return (importance['Feature'], imp_var, results['R2'][0], results['MAPE'][0], imp_figure, Error_figure)

    def find_best_regression(self,X,y,selected_dependent_var,selected_criterion,selected_independent_vars,exclude):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        dataset = pd.concat([X_train, y_train], axis=1)
        reg = regression.setup(data=dataset, target=selected_dependent_var)
        best_model = regression.compare_models(exclude=exclude, n_select=1, sort=selected_criterion)
        comapre_results = regression.pull()
        p_values = sm.OLS(y, sm.add_constant(X)).fit().pvalues
        #coefficients = np.append(best_model.intercept_, best_model.coef_)
        # r_squared = r2_score(y_test, best_model.predict(X_test))
        # data_dict = {'Coefficients': coefficients[1:]}
        # data_dict['P-values'] =p_values[1:]
        # coef_pval_df = pd.DataFrame(data_dict, index=selected_independent_vars)
        # coef_pval_df.index.name = "Xcol"
        # coef_pval_df = coef_pval_df.reset_index()
        modeldetail=str(best_model)
        pycaretname = FindBestModel.readable_name_converted_input_name(self,best_model)
        modelname=FindBestModel.more_readable_model_name(self,modeldetail)
        return (modelname,modeldetail,selected_criterion,comapre_results,pycaretname)

    def find_best_classifier(self, X, y, selected_dependent_var, selected_criterion, selected_independent_vars,exclude):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        dataset = pd.concat([X_train, y_train], axis=1)
        clf = classification.setup(data=dataset, target=selected_dependent_var)
        best_model = classification.compare_models(exclude=exclude, n_select=1, sort=selected_criterion)
        comapre_results = classification.pull()
        # print(comapre_results)

        # coefficients=best_model.coef_

        target_names = best_model.classes_
        # if len(target_names) == 2 and best_model.coef_.shape[0] == 1:
        #     # Special handling for binary classification case
        #     coeff_df = pd.DataFrame(best_model.coef_, columns=selected_independent_vars, index=[target_names[0]])
        # else:
        #     # The usual case for multiclass classification
        #     coeff_df = pd.DataFrame(best_model.coef_, columns=selected_independent_vars, index=target_names)
        # # Make predictions on the test set
        # # print(coeff_df)
        # y_pred = best_model.predict(X_test)
        # # Calculate and output the accuracy
        # accuracy = accuracy_score(y_test, y_pred)
        modeldetail = str(best_model)
        pycaretname = FindBestModel.readable_name_converted_input_name(self, best_model)
        modelname=FindBestModel.more_readable_model_name(self,modeldetail)
        return (modelname,modeldetail,selected_criterion,comapre_results,pycaretname)


    def more_readable_model_name(self,modeldetail):
        modeldetail=modeldetail
        if "Ridge" in modeldetail and "BayesianRidge" not in modeldetail:
            translatedmodel = "Ridge Model"
        elif "LinearDiscriminant" in modeldetail:
            translatedmodel = "Linear Discriminant Analysis"
        elif "GradientBoosting" in modeldetail:
            translatedmodel = "Gradient Boosting Model"
        elif "AdaBoost" in modeldetail:
            translatedmodel = "Ada Boost"
        elif "LGBMClassifier" in modeldetail:
            translatedmodel = "Light Gradient Boosting Machine Classifier"
        elif "DummyClassifier" in modeldetail:
            translatedmodel = "Dummy Classifier"
        elif "KNeighborsClassifier" in modeldetail:
            translatedmodel = "K Neighbors Classifier"
        elif "SGDClassifier" in modeldetail:
            translatedmodel = "SGD Classifier"
        elif "LGBMRegressor" in modeldetail:
            translatedmodel = "Light Gradient Boosting Machine"
        elif "RandomForest" in modeldetail:
            translatedmodel = "Random Forest Model"
        elif "XGBRegressor" in modeldetail:
            translatedmodel = "Extreme Gradient Boosting"
        elif "XGBClassifier" in modeldetail:
            translatedmodel = "Extreme Gradient Boosting Classifier"
        elif "Logistic" in modeldetail:
            translatedmodel = "Logistic Model"
        elif "QuadraticDiscriminant" in modeldetail:
            translatedmodel = "Quadratic Discriminant Analysis"
        elif "GaussianNB" in modeldetail:
            translatedmodel = "Naive Bayes"
        elif "ExtraTrees" in modeldetail:
            translatedmodel = "Extra Trees model"
        elif "DecisionTree" in modeldetail:
            translatedmodel = "Decision Tree Model"
        elif "Lasso" in modeldetail and "LassoLars" not in modeldetail:
            translatedmodel = "Lasso Regression"
        elif "LassoLars" in modeldetail:
            translatedmodel = "Lasso Least Angle Regression"
        elif "BayesianRidge" in modeldetail:
            translatedmodel = "Bayesian Ridge"
        elif "LinearRegression" in modeldetail:
            translatedmodel = "Linear Regression"
        elif "HuberRegressor" in modeldetail:
            translatedmodel = "Huber Regressor"
        elif "PassiveAggressiveRegressor" in modeldetail:
            translatedmodel = "Passive Aggressive Regressor"
        elif "OrthogonalMatchingPursuit" in modeldetail:
            translatedmodel = "Orthogonal Matching Pursuit"
        elif "AdaBoostRegressor" in modeldetail:
            translatedmodel = "AdaBoost Regressor"
        elif "KNeighborsRegressor" in modeldetail:
            translatedmodel = "K Neighbors Regressor"
        elif "ElasticNet" in modeldetail:
            translatedmodel = "Elastic Net"
        elif "DummyRegressor" in modeldetail:
            translatedmodel = "Dummy Regressor"
        elif "Lars" in modeldetail:
            translatedmodel = "Least Angle Regression"
        modelname=translatedmodel
        return modelname

    def readable_name_converted_input_name(self,modeldetail):
        modeldetail = str(modeldetail)
        if "Ridge" in modeldetail and "BayesianRidge" not in modeldetail:
            pycaretname = "ridge"
        elif "LinearDiscriminant" in modeldetail:
            pycaretname = "lda"
        elif "GradientBoosting" in modeldetail:
            pycaretname = "gbr"
        elif "AdaBoost" in modeldetail:
            pycaretname = "ada"
        elif "LGBMClassifier" in modeldetail:
            pycaretname = "lightgbm"
        elif "DummyClassifier" in modeldetail:
            pycaretname = "dummy"
        elif "KNeighborsClassifier" in modeldetail:
            pycaretname = "knn"
        elif "SGDClassifier" in modeldetail:
            pycaretname = "svm"
        elif "LGBMRegressor" in modeldetail:
            pycaretname = "lightgbm"
        elif "RandomForest" in modeldetail:
            pycaretname = "rf"
        elif "XGBRegressor" in modeldetail:
            pycaretname = "xgboost"
        elif "XGBClassifier" in modeldetail:
            pycaretname = "xgboost"
        elif "Logistic" in modeldetail:
            pycaretname = "lr"
        elif "QuadraticDiscriminant" in modeldetail:
            pycaretname = "qda"
        elif "GaussianNB" in modeldetail:
            pycaretname = "nb"
        elif "ExtraTrees" in modeldetail:
            pycaretname = "et"
        elif "DecisionTree" in modeldetail:
            pycaretname = "dt"
        elif "Lasso" in modeldetail and "LassoLars" not in modeldetail:
            pycaretname = "lasso"
        elif "LassoLars" in modeldetail:
            pycaretname = "llar"
        elif "BayesianRidge" in modeldetail:
            pycaretname = "br"
        elif "LinearRegression" in modeldetail:
            pycaretname = "lr"
        elif "HuberRegressor" in modeldetail:
            pycaretname = "huber"
        elif "PassiveAggressiveRegressor" in modeldetail:
            pycaretname = "par"
        elif "OrthogonalMatchingPursuit" in modeldetail:
            pycaretname = "omp"
        elif "AdaBoostRegressor" in modeldetail:
            pycaretname = "ada"
        elif "KNeighborsRegressor" in modeldetail:
            pycaretname = "knn"
        elif "ElasticNet" in modeldetail:
            pycaretname = "en"
        elif "DummyRegressor" in modeldetail:
            pycaretname = "dummy"
        elif "Lars" in modeldetail:
            pycaretname = "lar"
        return (pycaretname)