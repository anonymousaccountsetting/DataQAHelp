import pandas as pd
import DataScienceComponents as DC
import NLGComponents as NC
from docx.shared import RGBColor
from docx import Document


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
            linearData,r2,mape,mse,rmse,mae,vif=DC.ModelFitting().LinearSKDefaultModel(data, Xcol, ycol)
        elif skmodel==0:
            model,mape,mse,rmse,mae,vif = DC.ModelFitting().LinearDefaultModel(data, Xcol, ycol)
            columns = {'coeff': model.params.values[1:], 'pvalue': model.pvalues.round(4).values[1:]}
            linearData = pd.DataFrame(data=columns, index=Xcol)
            r2 = model.rsquared
        NC.DocumentplanningandDashboard().LinearModelStats_view( linearData,r2,mape,mse,rmse,mae,vif,data, Xcol, ycol, questionset, expect,portnum)


    def LogisticFit(self, data, Xcol, ycol, Xnewname="", ynewname="", pos_class_mean='', questionset=[1, 1, 1, 1],
                    portnum=8050):
        if Xnewname != "" or ynewname != "":
            data, Xcol, ycol = NC.Microplanning().variablenamechange(data, Xcol, ycol, Xnewname, ynewname)
        model, devDdf, accuracy, auc = DC.ModelFitting().LogisticrDefaultModel(data, Xcol, ycol)
        NC.DocumentplanningandDashboard().LogisticModelStats_view(model, data, Xcol, ycol, devDdf, accuracy, auc,
                                                                  pos_class_mean, questionset, portnum)

    def GradientBoostingFit(self, data, Xcol, ycol, Xnewname="", ynewname="", questionset=[1, 1, 1],
                            gbr_params={'n_estimators': 500, 'max_depth': 3, 'min_samples_split': 5,
                                        'learning_rate': 0.01}, portnum=8050):
        if Xnewname != "" or ynewname != "":
            data, Xcol, ycol = NC.Microplanning().variablenamechange(data, Xcol, ycol, Xnewname, ynewname)

        model, mse, mae, r2, imp, lessimp, train_r2, test_r2, train_mae, test_mae, cv_r2_scores, cv_r2_mean, mape,imp_pos_ave, imp_pos_value_ave, imp_neg_ave, imp_neg_value_ave,imp_var,explainer,shap_values,X_test = DC.ModelFitting().GradientBoostingDefaultModel(
            data,
            Xcol,
            ycol,
            gbr_params)
        NC.DocumentplanningandDashboard().GradientBoostingModelStats_view(data, Xcol, ycol, model, mse, mae, r2, imp,
                                                                          lessimp,
                                                                          questionset, gbr_params, train_r2, test_r2,
                                                                          train_mae, test_mae, cv_r2_scores, cv_r2_mean,
                                                                          mape, imp_pos_ave, imp_pos_value_ave, imp_neg_ave, imp_neg_value_ave,imp_var,explainer,shap_values,X_test,portnum)

    def RandomForestFit(self, data, Xcol, ycol, Xnewname="", ynewname="", questionset=[1, 1, 1], n_estimators=10,
                        max_depth=3, portnum=8050):
        if Xnewname != "" or ynewname != "":
            data, Xcol, ycol = NC.Microplanning().variablenamechange(data, Xcol, ycol, Xnewname, ynewname)
        X = data[Xcol].values
        y = data[ycol]
        rf_small, DTData, r2, mse, mae, mape, train_r2, test_r2, train_mae, test_mae, cv_r2_scores, cv_r2_mean,imp_pos_ave, imp_pos_value_ave, imp_neg_ave, imp_neg_value_ave,imp_var,explainer,shap_values,X_test = DC.ModelFitting().RandomForestRegressionDefaultModel(
            X, y, Xcol, n_estimators, max_depth)
        NC.DocumentplanningandDashboard().RandomForestRegressionModelStats_view(data, Xcol, ycol, rf_small, DTData, r2,
                                                                                mse, mae, mape, train_r2, test_r2,
                                                                                train_mae, test_mae, cv_r2_scores,
                                                                                cv_r2_mean, imp_pos_ave, imp_pos_value_ave, imp_neg_ave, imp_neg_value_ave,imp_var, explainer,shap_values,X_test,portnum)

    def DecisionTreeFit(self, data, Xcol, ycol, Xnewname="", ynewname="", questionset=[1, 1, 1], max_depth=3,
                        portnum=8050):
        if Xnewname != "" or ynewname != "":
            data, Xcol, ycol = NC.Microplanning().variablenamechange(data, Xcol, ycol, Xnewname, ynewname)
        X = data[Xcol].values
        y = data[ycol]
        model, r2, mse, mae, mape, train_r2, test_r2, train_mae, test_mae, cv_r2_scores, cv_r2_mean, DTData,imp_pos_ave, imp_pos_value_ave, imp_neg_ave, imp_neg_value_ave,imp_var,explainer,shap_values,X_test = DC.ModelFitting().DecisionTreeRegressionDefaultModel(
            X, y, Xcol, max_depth)
        NC.DocumentplanningandDashboard().DecisionTreeRegressionModelStats_view(data, Xcol, ycol, DTData, model, r2,
                                                                                mse, mae, mape, train_r2, test_r2,
                                                                                train_mae, test_mae, cv_r2_scores,
                                                                                cv_r2_mean, imp_pos_ave, imp_pos_value_ave, imp_neg_ave, imp_neg_value_ave,imp_var,explainer,shap_values,X_test, portnum)

    def piecewiselinearFit(self, data, Xcol, ycol, num_breaks, Xnewname="", ynewname="", portnum=8050):
        if Xnewname != "" or ynewname != "":
            data, Xcol, ycol = NC.Microplanning().variablenamechange(data, Xcol, ycol, Xnewname, ynewname)
        my_pwlf, slopes, segment_points, segment_values, max_slope_segment, breaks, segment_r2_values, mse, mae, bic, aic = DC.ModelFitting().piecewise_linear_fit(
            data, Xcol, ycol, num_breaks)
        NC.DocumentplanningandDashboard().piecewise_linear_view(data, Xcol, ycol, my_pwlf, slopes, segment_points,
                                                                segment_values, max_slope_segment, breaks,
                                                                segment_r2_values, mse, mae, bic, aic, portnum)

    def RidgeClassifierFit(self, data, Xcol, ycol, class1, class2, Xnewname="", ynewname=""):
        if Xnewname != "" or ynewname != "":
            data, Xcol, ycol = NC.Microplanning().variablenamechange(data, Xcol, ycol, Xnewname, ynewname)
        rclf, pca, y_test, y_prob, roc_auc, X_pca, accuracy, precision, recall, f1, importances, confusionmatrix, cv_scores = DC.ModelFitting().RidgeClassifierModel(
            data, Xcol, ycol, class1, class2)
        NC.DocumentplanningandDashboard().RidgeClassifier_view(data, Xcol, ycol, rclf, pca, y_test, y_prob, roc_auc,
                                                               X_pca, accuracy, precision, recall, f1, importances,
                                                               class1, class2,
                                                               confusionmatrix, cv_scores)

    def KNeighborsClassifierFit(self, data, Xcol, ycol, Xnewname="", ynewname="", Knum=3, cvnum=5):
        if Xnewname != "" or ynewname != "":
            data, Xcol, ycol = NC.Microplanning().variablenamechange(data, Xcol, ycol, Xnewname, ynewname)
        accuracy, precision, feature_importances, recall, f1, confusionmatrix, cv_scores = DC.ModelFitting().KNeighborsClassifierModel(
            data, Xcol, ycol, Knum, cvnum)
        NC.DocumentplanningandDashboard().KNeighborsClassifier_view(data, Xcol, ycol, accuracy, precision,
                                                                    feature_importances, recall, f1, confusionmatrix,
                                                                    cv_scores)

    def SVMClassifierFit(self, data, Xcol, ycol, Xnewname="", ynewname="", kernel='linear', C=1.0, cvnum=5):
        if Xnewname != "" or ynewname != "":
            data, Xcol, ycol = NC.Microplanning().variablenamechange(data, Xcol, ycol, Xnewname, ynewname)
        accuracy, precision, recall, f1, confusionmatrix, cv_scores, classes, total_influence, most_influential_feature, coefficients = DC.ModelFitting().SVCClassifierModel(
            data, Xcol,
            ycol,
            kernel=kernel,
            C=C,
            cvnum=cvnum)
        NC.DocumentplanningandDashboard().SVCClassifier_view(data, Xcol, ycol, accuracy, precision, recall, f1,
                                                             confusionmatrix, cv_scores, classes, total_influence,
                                                             most_influential_feature, coefficients)

    def DecisionTreeClassifierFit(self, data, Xcol, ycol, Xnewname="", ynewname="", cvnum=5):
        if Xnewname != "" or ynewname != "":
            data, Xcol, ycol = NC.Microplanning().variablenamechange(data, Xcol, ycol, Xnewname, ynewname)
        accuracy, precision, feature_importances, recall, f1, confusionmatrix, cv_scores, clf = DC.ModelFitting().DecisionTreeClassifierModel(
            data, Xcol, ycol, cvnum)
        NC.DocumentplanningandDashboard().DecisionTreeClassifier_view(data, Xcol, ycol, accuracy, precision,
                                                                      feature_importances, recall, f1, confusionmatrix,
                                                                      cv_scores, clf)

    def RandomForestClassifierFit(self, data, Xcol, ycol, Xnewname="", ynewname="", n_estimators=100, cvnum=5):
        if Xnewname != "" or ynewname != "":
            data, Xcol, ycol = NC.Microplanning().variablenamechange(data, Xcol, ycol, Xnewname, ynewname)
        accuracy, precision, feature_importances, recall, f1, confusionmatrix, cv_scores = DC.ModelFitting().RandomForestClassifierModel(
            data, Xcol, ycol, n_estimators, cvnum)
        NC.DocumentplanningandDashboard().RandomForestClassifier_view(data, Xcol, ycol, accuracy, precision,
                                                                      feature_importances, recall, f1, confusionmatrix,
                                                                      cv_scores)

    def RidgeRegressionFit(self, data, Xcol, ycol, Xnewname="", ynewname="", ridge_params=None):
        if Xnewname != "" or ynewname != "":
            data, Xcol, ycol = NC.Microplanning().variablenamechange(data, Xcol, ycol, Xnewname, ynewname)
        DTData, mse, mae, r2, imp, lessimp, train_r2, test_r2, train_mae, test_mae, cv_r2_scores, cv_r2_mean, mape,vif = DC.ModelFitting().RidgeDefaultModel(
            data, Xcol, ycol, ridge_params)
        NC.DocumentplanningandDashboard().RidgeRegressionModel_view(data, Xcol, ycol, DTData, mse, mae, r2, imp,
                                                                    lessimp, train_r2, test_r2, train_mae, test_mae,
                                                                    cv_r2_scores, cv_r2_mean, mape,vif)

    def LassoRegressionFit(self, data, Xcol, ycol, Xnewname="", ynewname="", lasso_params=None):
        if Xnewname != "" or ynewname != "":
            data, Xcol, ycol = NC.Microplanning().variablenamechange(data, Xcol, ycol, Xnewname, ynewname)
        DTData, mse, mae, r2, imp, lessimp, train_r2, test_r2, train_mae, test_mae, cv_r2_scores, cv_r2_mean, mape,vif = DC.ModelFitting().LassoDefaultModel(
            data, Xcol, ycol, lasso_params)
        NC.DocumentplanningandDashboard().LassoRegressionModel_view(data, Xcol, ycol, DTData, mse, mae, r2, imp,
                                                                    lessimp, train_r2, test_r2, train_mae, test_mae,
                                                                    cv_r2_scores, cv_r2_mean, mape,vif)

    def ElasticNetRegressionFit(self, data, Xcol, ycol, Xnewname="", ynewname="", enet_params=None):
        if Xnewname != "" or ynewname != "":
            data, Xcol, ycol = NC.Microplanning().variablenamechange(data, Xcol, ycol, Xnewname, ynewname)
        DTData, mse, mae, r2, imp, lessimp, train_r2, test_r2, train_mae, test_mae, cv_r2_scores, cv_r2_mean, mape,vif = DC.ModelFitting().ElasticNetDefaultModel(
            data, Xcol, ycol, enet_params)
        NC.DocumentplanningandDashboard().ElasticNetModel_view(data, Xcol, ycol, DTData, mse, mae, r2, imp, lessimp,
                                                               train_r2, test_r2, train_mae, test_mae, cv_r2_scores,
                                                               cv_r2_mean, mape,vif)

    def LeastAngleRegressionFit(self, data, Xcol, ycol, Xnewname="", ynewname="", lars_params=None):
        if Xnewname != "" or ynewname != "":
            data, Xcol, ycol = NC.Microplanning().variablenamechange(data, Xcol, ycol, Xnewname, ynewname)
        DTData, mse, mae, r2, imp, lessimp, train_r2, test_r2, train_mae, test_mae, cv_r2_scores, cv_r2_mean, mape,vif = DC.ModelFitting().LeastAngleRegressionDefaultModel(
            data, Xcol, ycol, lars_params)
        NC.DocumentplanningandDashboard().LeastAngleRegressionModel_view(data, Xcol, ycol, DTData, mse, mae, r2, imp,
                                                                         lessimp, train_r2, test_r2, train_mae,
                                                                         test_mae, cv_r2_scores, cv_r2_mean, mape,vif)

    def AdaBoostRegressionFit(self, data, Xcol, ycol, Xnewname="", ynewname="", adaboost_params=None):
        if Xnewname != "" or ynewname != "":
            data, Xcol, ycol = NC.Microplanning().variablenamechange(data, Xcol, ycol, Xnewname, ynewname)
        DTData, mse, mae, r2, imp, lessimp, train_r2, test_r2, train_mae, test_mae, cv_r2_scores, cv_r2_mean, mape = DC.ModelFitting().AdaBoostDefaultModel(
            data, Xcol, ycol, adaboost_params)
        NC.DocumentplanningandDashboard().AdaBoostRegressionModel_view(data, Xcol, ycol, DTData, mse, mae, r2, imp,
                                                                       lessimp, train_r2, test_r2, train_mae, test_mae,
                                                                       cv_r2_scores, cv_r2_mean, mape)

    def KNeighborsRegressionFit(self, data, Xcol, ycol, Xnewname="", ynewname="", adaboost_params=None):
        if Xnewname != "" or ynewname != "":
            data, Xcol, ycol = NC.Microplanning().variablenamechange(data, Xcol, ycol, Xnewname, ynewname)
        model, mse, mae, r2, train_r2, test_r2, train_mae, test_mae, cv_r2_scores, cv_r2_mean, mape= DC.ModelFitting().KNeighborsDefaultModel(
            data, Xcol, ycol, adaboost_params)
        NC.DocumentplanningandDashboard().KNeighborsRegressionModel_view(data, Xcol, ycol, model, mse, mae, r2,
                                                                         train_r2, test_r2, train_mae, test_mae,
                                                                         cv_r2_scores, cv_r2_mean, mape)


class casestudy_datastory_pipeline:
    def CPpiecewiselinearFit(self, data, Xcol, ycol, num_breaks, Xnewname="", ynewname=""):
        if Xnewname != "" or ynewname != "":
            data, Xcol, ycol = NC.Microplanning().variablenamechange(data, Xcol, ycol, Xnewname, ynewname)
        my_pwlf, slopes, segment_points, segment_values, max_slope_segment,breaks,segment_r2_values,mse,mae,bic,aic = DC.ModelFitting().piecewise_linear_fit(
            data, Xcol, ycol, num_breaks)
        last_X,last_X2,last_y,difference,percentage_change,max_value,max_y_X=DC.DataDescription().general_description(data, Xcol, ycol)
        summary=NC.DocumentplanningNoDashboard().CP_general_description(ycol,last_X,last_X2,last_y,difference,percentage_change,max_value,max_y_X)
        summary=summary+' '+NC.DocumentplanningNoDashboard().CP_piecewise_linear(data, Xcol, ycol, slopes, breaks)
        print(summary)

    def DRDpiecewiselinearFit(self, data, Xcol, ycol, num_breaks, Xnewname="", ynewname=""):
        if Xnewname != "" or ynewname != "":
            data, Xcol, ycol = NC.Microplanning().variablenamechange(data, Xcol, ycol, Xnewname, ynewname)
        my_pwlf, slopes, segment_points, segment_values, max_slope_segment,breaks,segment_r2_values,mse,mae,bic,aic = DC.ModelFitting().piecewise_linear_fit(
            data, Xcol, ycol, num_breaks)
        summary=NC.DocumentplanningNoDashboard().DRD_piecewise_linear(data, Xcol, ycol,my_pwlf, slopes, segment_points, segment_values, max_slope_segment,breaks)
        print(summary)

class find_best_mode_pipeline:

    def pycaret_model_fit(self,dataset,types,pycaretname,comparestory,comapre_results,target,modelname):
        if types==0:
            independent_var, imp, Accuracy, AUC, imp_figure, Error_figure,importance = DC.FindBestModel().pycaret_create_model(types, pycaretname)
            fitstory = NC.AutoFindBestModel().pycaret_classificationmodel_summary_view(imp, Accuracy, AUC,target,modelname)
            # app_name, listTabs = NC.start_app()
            # NC.dash_with_table(app_name, listTabs, comparestory, dataset, "Model Compare Overview")
            # _base64 = []
            # _base64 = NC.read_figure(_base64, "Prediction Error")
            # _base64 = NC.read_figure(_base64, "Feature Importance")
            # _base64 = NC.read_figure(_base64, "SHAP summary")
            # NC.dash_with_table(app_name, listTabs, fitstory, comapre_results, "Model credibility")
            # # VW.dash_with_figure(app_name, listTabs, impstory, 'Variables Summary', _base64[1])
            # NC.dash_with_two_figure(app_name, listTabs, impstory, 'Important Variables Summary', _base64[1],_base64[2])
            # NC.run_app(app_name, listTabs)
            return (fitstory,importance)
        elif types==1:
            independent_var, imp,r2, mape, imp_figure, Error_figure = DC.FindBestModel().pycaret_create_model(types, pycaretname)
            fitstory= NC.AutoFindBestModel().pycaret_model_summary_view(imp, r2, mape,target,modelname)
            # app_name, listTabs = NC.start_app()
            # NC.dash_with_table(app_name, listTabs, comparestory, dataset, "Model Compare Overview")
            # _base64 = []
            # _base64 = NC.read_figure(_base64, "Prediction Error")
            # _base64 = NC.read_figure(_base64, "Feature Importance")
            # _base64 = NC.read_figure(_base64, "SHAP summary")
            # NC.dash_with_table(app_name, listTabs, fitstory, comapre_results, "Model credibility")
            # # VW.dash_with_figure(app_name, listTabs, impstory, 'Variables Summary', _base64[1])
            # NC.dash_with_two_figure(app_name, listTabs, impstory, 'Important Variables Summary', _base64[1],_base64[2])
            # NC.run_app(app_name, listTabs)
            return (fitstory)
    def FindBestRegressionPipeline(self,data,Xcol,ycol,selected_criterion="r2",Xnewname="", ynewname="",exclude=['rf','gbr','catboost','lightgbm','et','ada','xgboost','llar','lar','huber','dt','omp','par','en','knn','dummy']):
        if Xnewname != "" or ynewname != "":
            data, Xcol, ycol =  NC.Microplanning().variablenamechange(data, Xcol, ycol, Xnewname, ynewname)
        X = data[Xcol]
        y = data[ycol]

        modelname, modeldetail, selected_criterion,comapre_results,pycaretname=DC.FindBestModel().find_best_regression(X,y,ycol,selected_criterion,Xcol,exclude)
        modelcomparestory=NC.AutoFindBestModel().model_compare(modelname, modeldetail, selected_criterion,1)
        fitstory=find_best_mode_pipeline.pycaret_model_fit(self,data,1, pycaretname, modelcomparestory, comapre_results,ycol,modelname)
        p_values_df=DC.ModelFitting().PValueCalculation(data,Xcol,ycol)
        NC.DocumentplanningandDashboard().FindBestRegression_view(data,Xcol,ycol,comapre_results,fitstory,modelcomparestory,p_values_df)

    def FindBestClassifierPipeline(self,data,Xcol,ycol,selected_criterion="Accuracy",Xnewname="", ynewname="",exclude = ['rf','dt','qda','knn','lightgbm','et','catboost','xgboost','gbc','ada','nb','dummy']):
        if Xnewname != "" or ynewname != "":
            data, Xcol, ycol =  NC.Microplanning().variablenamechange(data, Xcol, ycol, Xnewname, ynewname)
        X = data[Xcol]
        y = data[ycol]

        modelname, modeldetail, selected_criterion,comapre_results,pycaretname=DC.FindBestModel().find_best_classifier(X,y,ycol,selected_criterion,Xcol,exclude)
        modelcomparestory=NC.AutoFindBestModel().model_compare(modelname, modeldetail, selected_criterion,1)
        fitstory,importance = find_best_mode_pipeline.pycaret_model_fit(self, data, 0, pycaretname, modelcomparestory,
                                                             comapre_results, ycol, modelname)
        print(fitstory)
        NC.DocumentplanningandDashboard().FindBestClassifier_view(data, Xcol, ycol, comapre_results, fitstory,
                                                                  modelcomparestory,importance,pycaretname,modelname)


class special_pipelines_for_ACCCP:
    def register_question1(self, app_name, listTabs, register_dataset, per1000inCity_col, per1000nation_col,
                           table_col=['Period', 'Registrations In Aberdeen City',
                                      'Registrations per 1000 population in Aberdeen City',
                                      'Compared with last year for Aberdeen City'],
                           label='What are the emerging trends or themes emerging from local and comparators data?'):
        diff = DC.DataDescription().loop_mean_compare(register_dataset, per1000inCity_col, per1000nation_col)
        app, listTabs, text = NC.special_view_for_ACCCP().register_question1_view(register_dataset, per1000inCity_col,
                                                                                  diff, table_col, label, app_name,
                                                                                  listTabs)
        return (app, listTabs, text)

    def riskfactor_question1(self, app_name, listTabs, risk_factor_dataset, risk_factor_col, cityname="Aberdeen City",
                             max_num=5,
                             label='What are the emerging trends or themes emerging from local single agency data?'):
        row = 0
        max_factor = DC.DataDescription().find_row_n_max(risk_factor_dataset, risk_factor_col, row, max_num)
        row = 1
        max_factor_lastyear = DC.DataDescription().find_row_n_max(risk_factor_dataset, risk_factor_col, row, max_num)
        same_factor = DC.DataDescription().detect_same_elements(max_factor, max_factor_lastyear)
        app, listTabs, text = NC.special_view_for_ACCCP().riskfactor_question1_view(risk_factor_dataset, max_factor,
                                                                                    same_factor, label, cityname,
                                                                                    app_name, listTabs)
        return (app, listTabs, text)

    def re_register_question4(self, app_name, listTabs, register_dataset, reregister_col, period_col='Period',
                              national_average_reregistration='13 - 16%',
                              table_col=['Period', 'Re-Registrations In Aberdeen City',
                                         'Re-registrations as a % of registrations in Aberdeen City',
                                         'Largest family for Aberdeen City',
                                         'Longest gap between registrations of Aberdeen City',
                                         'Shortest gap between registrations of Aberdeen City'],
                              label='To what extent is Aberdeen City consistent with the national and comparator averages for re-registration?  Can the CPC be assured that deregistered children receive at least 3 months’ post registration multi-agency support?'):
        reregister_lastyear, period = DC.DataDescription().select_one_element(register_dataset, reregister_col,
                                                                              period_col)
        app, listTabs, text = NC.special_view_for_ACCCP().re_register_question4_view(register_dataset,
                                                                                     national_average_reregistration,
                                                                                     reregister_lastyear, period,
                                                                                     table_col, label, app_name,
                                                                                     listTabs)
        return (app, listTabs, text)

    def remain_time_question5(self, app_name, listTabs, remain_data, check_col, period_col='Period',
                              label='What is the number of children remaining on the CPR for more than 1 year and can the CPC be assured that it is necessary for any child to remain on the CPR for more than 1 year?'):
        zero_lastdata = DC.DataDescription().find_all_zero_after_arow(remain_data, check_col, period_col)
        app, listTabs, text = NC.special_view_for_ACCCP().remain_time_question5_view(remain_data, zero_lastdata, label,
                                                                                     app_name, listTabs)
        return (app, listTabs, text)

    def enquiries_question6(self, app_name, listTabs, enquiries_data, AC_enquiries, AS_enquiries, MT_enquiries,
                            period_col='Period',
                            label='To what extent do agencies make use of the CPR?  If they are not utilising it, what are the reasons for that?'):
        period = enquiries_data[period_col]
        ACdata = enquiries_data[AC_enquiries].values
        ASdata = enquiries_data[AS_enquiries].values
        MTdata = enquiries_data[MT_enquiries].values
        ACmean = DC.DataDescription().find_column_mean(ACdata)
        ASmean = DC.DataDescription().find_column_mean(ASdata)
        MTmean = DC.DataDescription().find_column_mean(MTdata)
        app, listTabs, text = NC.special_view_for_ACCCP().enquiries_question6_view(ACmean, ASmean, MTmean, ACdata,
                                                                                   ASdata, MTdata, period, label,
                                                                                   app_name, listTabs)
        return (app, listTabs, text)

    def national_question(self, nationaldata, Authority1, Authority2, Authority3, point1, point2, IP1, IP2,
                          IPchangeType):
        data = DC.DataDescription().two_point_percent_differ(nationaldata, point1, point2)
        initial_year_value = data[data['Authority'] == Authority1][point1].values[0]
        final_year_value = data[data['Authority'] == Authority1][point2].values[0]
        change_percentage = data[data['Authority'] == Authority1]['Change_Percentage'].values[0]
        shirechange_percentage = data[data['Authority'] == Authority2]['Change_Percentage'].values[0]
        Moraychange_percentage = data[data['Authority'] == Authority3]['Change_Percentage'].values[0]
        data = DC.DataDescription().two_point_percent_differ(nationaldata, IP1, IP2)
        IPchange_percentage = int(data[data['Authority'] == IPchangeType]['Change_Percentage'].values[0])
        story = NC.special_view_for_ACCCP().national_data_view(int(change_percentage), initial_year_value,
                                                               final_year_value,
                                                               int(shirechange_percentage), int(Moraychange_percentage),
                                                               IPchange_percentage)
        return (story)

    def timescales_question(self, data, colname1, colname2, colname3, colname4, colname5, colname6, colname7,
                            colname8, colname9, colname10, colname11, colname12, colname13,
                            colname14, colname15, colname16, colname17, colname18, colname19,
                            colname20, colname21, colname22, colname23, colname24):
        story = NC.special_view_for_ACCCP().timescales_description(data, colname1, colname2, colname3, colname4,
                                                                   colname5, colname6, colname7,
                                                                   colname8, colname9, colname10, colname11, colname12,
                                                                   colname13,
                                                                   colname14, colname15, colname16, colname17,
                                                                   colname18, colname19,
                                                                   colname20, colname21, colname22, colname23,
                                                                   colname24)
        return (story)

    def SCRA_question(self, data, Authority, point1, point2):
        total_1 = int(data[data['Authority'] == Authority][point1].values[0])
        total_2 = int(data[data['Authority'] == Authority][point2].values[0])
        story = NC.special_view_for_ACCCP().SCRA_description(total_1, total_2, point1)
        return (story)

    def ACCCP_questions(self, register_dataset, risk_factor_dataset, remain_data, enquiries_data, IPdata, SCRAdata,
                        timedata, per1000inCity_col,
                        per1000nation_col, risk_factor_col, reregister_col, check_col, period_col, AC_enquiries,
                        AS_enquiries, MT_enquiries, Authority1, Authority2, Authority3, point1, point2, IP1, IP2,
                        IPchangeType, Authority, time1, time2, colname1, colname2, colname3, colname4, colname5,
                        colname6, colname7,
                        colname8, colname9, colname10, colname11, colname12, colname13,
                        colname14, colname15, colname16, colname17, colname18, colname19,
                        colname20, colname21, colname22, colname23, colname24, template_path,
                        output_path='./output.docx'):
        app_name, listTabs = NC.start_app()
        app, listTabs, story3 = special_pipelines_for_ACCCP.register_question1(self, app_name, listTabs,
                                                                               register_dataset, per1000inCity_col,
                                                                               per1000nation_col)
        app, listTabs, story4 = special_pipelines_for_ACCCP.riskfactor_question1(self, app_name, listTabs,
                                                                                 risk_factor_dataset, risk_factor_col)
        app, listTabs, story6 = special_pipelines_for_ACCCP.re_register_question4(self, app_name, listTabs,
                                                                                  register_dataset, reregister_col)
        app, listTabs, story7 = special_pipelines_for_ACCCP.remain_time_question5(self, app_name, listTabs, remain_data,
                                                                                  check_col, period_col)
        app, listTabs, story8 = special_pipelines_for_ACCCP.enquiries_question6(self, app_name, listTabs,
                                                                                enquiries_data, AC_enquiries,
                                                                                AS_enquiries, MT_enquiries, period_col)
        story1 = special_pipelines_for_ACCCP.national_question(self, IPdata, Authority1, Authority2, Authority3,
                                                               point1,
                                                               point2, IP1, IP2, IPchangeType)
        story2 = special_pipelines_for_ACCCP.SCRA_question(self, SCRAdata, Authority, time1, time2)

        story5 = special_pipelines_for_ACCCP.timescales_question(self, timedata, colname1, colname2, colname3,
                                                                 colname4, colname5, colname6, colname7,
                                                                 colname8, colname9, colname10, colname11,
                                                                 colname12, colname13,
                                                                 colname14, colname15, colname16, colname17,
                                                                 colname18, colname19,
                                                                 colname20, colname21, colname22, colname23,
                                                                 colname24)
        print("Based on the data set, the application automatically generates the following content. The corresponding content and tables have been saved in a new document.")
        stories = [story1, story2, story3, story4, story5, story6, story7, story8]

        for story in stories:
            print(story)

        for col in risk_factor_dataset.columns:
            if col != period_col:
                risk_factor_dataset[col] = risk_factor_dataset[col].round(0).astype(int)
        table1=NC.reshape_data(register_dataset,period_col)
        table2=NC.reshape_data(risk_factor_dataset,period_col)
        table3=NC.reshape_data(remain_data,period_col)

        tables = {
            "{{table1}}": table1,
            "{{table2}}": table2,
            "{{table3}}": table3,
        }

        doc=Document(template_path)
        for placeholder, df in tables.items():
            NC.replace_placeholder_with_table(doc, placeholder, df)
        doc.save(output_path)

        replacements = {
            "{{story1}}": story1,
            "{{story2}}": story2,
            "{{story3}}": story3,
            "{{story4}}": story4,
            "{{story5}}": story5,
            "{{story6}}": story6,
            "{{story7}}": story7,
            "{{story8}}": story8
        }
        NC.fill_template(output_path, output_path, replacements, RGBColor(255, 0, 0))
        print(f'New document saved to ({output_path}). Please review the report and make any necessary changes.')
        app, listTabs = NC.special_view_for_ACCCP().unfinished_report(app_name, listTabs, story1, story2, story3,
                                                                      story4, story5, template_path)
        NC.run_app(app_name, listTabs)
