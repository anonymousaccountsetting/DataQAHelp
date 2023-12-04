import DataScienceComponents as DC
import NLGComponents as NC


class general_datastory_pipeline:
    def LinearFit(self,data, Xcol, ycol, Xnewname="", ynewname="", questionset=[1, 1, 1, 1], expect="",skmodel=0,portnum=8050):
        # "expect" is a list of size 3:
        # The first value: 0 means that the user wants to explore how to make the dependent variable as small as possible, and 1 means how to make the dependent variable as large as possible.
        # The second value: 0 means that the user expects a weak relationship between the dependent variable and the independent variable, and 1 means a strong relationship.
        # The third value: 0 means that the user expects that each independent variable has no significant impact on the dependent variable, and 1 means that there is a significant impact.
        # Their default value is "" to ignore user expectations.
        if Xnewname != "" or ynewname != "":
            data, Xcol, ycol = NC.Microplanning().variablenamechange(data, Xcol, ycol, Xnewname, ynewname)
        if skmodel==1:
            linearData,r2=DC.ModelFitting().LinearSKDefaultModel(data, Xcol, ycol)
        elif skmodel==0:
            model = DC.ModelFitting().LinearDefaultModel(data, Xcol, ycol)
            columns = {'coeff': model.params.values[1:], 'pvalue': model.pvalues.round(4).values[1:]}
            linearData = pd.DataFrame(data=columns, index=Xcol)
            r2 = model.rsquared
        NC.DocumentplanningandDashboard().LinearModelStats_view( linearData,r2,data, Xcol, ycol, questionset, expect,portnum)

    def LogisticFit(self,data, Xcol, ycol, Xnewname="", ynewname="", questionset=[1, 1, 1, 1],portnum=8050):
        if Xnewname != "" or ynewname != "":
            data, Xcol, ycol = NC.Microplanning().variablenamechange(data, Xcol, ycol, Xnewname, ynewname)
        model, devDdf = DC.ModelFitting().LogisticrDefaultModel(data, Xcol, ycol)
        NC.DocumentplanningandDashboard().LogisticModelStats_view(model,data, Xcol, ycol, devDdf, questionset,portnum)

    def GradientBoostingFit(self,data, Xcol, ycol, Xnewname="", ynewname="", questionset=[1, 1, 1],
                                   gbr_params={'n_estimators': 500, 'max_depth': 3, 'min_samples_split': 5,
                                               'learning_rate': 0.01},portnum=8050):
        if Xnewname != "" or ynewname != "":
            data, Xcol, ycol = NC.Microplanning().variablenamechange(data, Xcol, ycol, Xnewname, ynewname)

        model, mse, rmse, r2,imp,train_errors,test_errors = DC.ModelFitting().GradientBoostingDefaultModel(data, Xcol,ycol, gbr_params)
        NC.DocumentplanningandDashboard().GradientBoostingModelStats_view(data, Xcol, ycol, model, mse, r2,imp, questionset, gbr_params,train_errors,test_errors,portnum)

    def piecewiselinearFit(self,data, Xcol, ycol, num_breaks,Xnewname="", ynewname="",portnum=8050):
        if Xnewname != "" or ynewname != "":
            data, Xcol, ycol = NC.Microplanning().variablenamechange(data, Xcol, ycol, Xnewname, ynewname)
        my_pwlf, slopes, segment_points, segment_values, max_slope_segment=DC.ModelFitting().piecewise_linear_fit(data, Xcol, ycol, num_breaks)
        NC.DocumentplanningandDashboard().piecewise_linear_view(data, Xcol, ycol,my_pwlf, slopes, segment_points, segment_values, max_slope_segment,portnum)

    def RidgeClassifierFit(self,data,Xcol,ycol,class1,class2,Xnewname="", ynewname=""):
        if Xnewname != "" or ynewname != "":
            data, Xcol, ycol =  NC.Microplanning().variablenamechange(data, Xcol, ycol, Xnewname, ynewname)
        rclf,pca,y_test, y_prob,roc_auc,X_pca,accuracy,importances,confusionmatrix,cv_scores=DC.ModelFitting().RidgeClassifierModel(data, Xcol, ycol,class1,class2)
        NC.DocumentplanningandDashboard().RidgeClassifier_view(data,Xcol,ycol,rclf,pca,y_test, y_prob,roc_auc,X_pca,accuracy,importances,class1,class2,confusionmatrix,cv_scores)
    def KNeighborsClassifierFit(self,data,Xcol,ycol,Xnewname="", ynewname="",Knum=3,cvnum=5):
        if Xnewname != "" or ynewname != "":
            data, Xcol, ycol =  NC.Microplanning().variablenamechange(data, Xcol, ycol, Xnewname, ynewname)
        accuracy, precision, feature_importances, recall, f1, confusionmatrix, cv_scores=DC.ModelFitting().KNeighborsClassifierModel(data, Xcol, ycol,Knum,cvnum)
        NC.DocumentplanningandDashboard().KNeighborsClassifier_view(data,Xcol,ycol,accuracy,precision,feature_importances,recall,f1,confusionmatrix,cv_scores)
    def SVMClassifierFit(self,data,Xcol,ycol,Xnewname="", ynewname="",kernel='linear', C=1.0,cvnum=5):
        if Xnewname != "" or ynewname != "":
            data, Xcol, ycol =  NC.Microplanning().variablenamechange(data, Xcol, ycol, Xnewname, ynewname)
        accuracy,precision,recall,f1,confusionmatrix,cv_scores = DC.ModelFitting().SVCClassifierModel(data, Xcol, ycol,kernel=kernel, C=C,cvnum=cvnum)
        NC.DocumentplanningandDashboard().SVCClassifier_view(data,Xcol,ycol,accuracy,precision,recall,f1,confusionmatrix,cv_scores)
    def RandomForestFit(self,data, Xcol, ycol, Xnewname="", ynewname="", questionset=[1, 1, 1], n_estimators=10,
                               max_depth=3,portnum=8050):
        if Xnewname != "" or ynewname != "":
            data, Xcol, ycol = NC.Microplanning().variablenamechange(data, Xcol, ycol, Xnewname, ynewname)
        X = data[Xcol].values
        y = data[ycol]
        tree_small, rf_small, DTData, r2, mse, rmse = DC.ModelFitting().RandomForestDefaultModel(X, y, Xcol, n_estimators, max_depth)
        NC.DocumentplanningandDashboard().RandomForestModelStats_view(data, Xcol, ycol, tree_small, rf_small, DTData, r2, mse, questionset,portnum)

    def DecisionTreeFit(self,data, Xcol, ycol, Xnewname="", ynewname="", questionset=[1, 1, 1], max_depth=3,portnum=8050):
        if Xnewname != "" or ynewname != "":
            data, Xcol, ycol = NC.Microplanning().variablenamechange(data, Xcol, ycol, Xnewname, ynewname)
        X = data[Xcol].values
        y = data[ycol]
        DTmodel, r2, mse, rmse, DTData = DC.ModelFitting().DecisionTreeDefaultModel(X, y, Xcol, max_depth)
        NC.DocumentplanningandDashboard().DecisionTreeModelStats_view(data, Xcol, ycol, DTData, DTmodel, r2, mse, questionset,portnum)

