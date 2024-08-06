[//]: # (<p align="center"><a href="#"><img width=60% alt="" src="https://github.com/lux-org/lux-resources/blob/master/readme_img/logo.png?raw=true"></a></p>)
<h1 align="center">DataQAHelper</h1>
<h3 align="center">A Framework for Data-to-Text Application Development</h3>
<img src="https://github.com/tangjikededela/DataQAHelp/blob/main/readmepic/cover.jpg"
  alt="DataQAHelper Cover"
  style="width:1920px" />

The DataQAHelper is a Python-based framework that integrates commonly used data science algorithms with model-based question banks and answer templates. This framework aims to facilitate a quick start in developing data-to-text applications. With just a few lines of code, users can quickly build applications to support data visualization, model evaluation, and data interpretation (See Example 1 and 2 below). The unique feature of this framework is its ability to transform complex data analysis results into understandable FAQ-like textual reports, enabling users to quickly grasp the insights from the dataset.

Another significant advantage of the framework is its support for reporting and narrative tasks in data science. When presented with a dataset and a set of business questions from a client, data scientists can select the appropriate data science model and determine the suitable data analysis algorithm. In this scenario, data scientists as the developers can match specific business questions with those in the question bank and call the corresponding pipelines to swiftly develop data-to-text applications. Additionally, the framework encourages developers to continuously refine the application and improve the quality of the generated reports based on their requirements. This allows for the repeated use of the application in future projects (See Example 3 below).

## Simple Examples

Here are some examples of simple applications based on DataQAHelper in action.

1. Check out [this Colab notebook](https://github.com/tangjikededela/DataQAHelp/blob/main/tutorial/Tutorial_ModelComparisonInterpretation.ipynb) for examples of how applications based on DataQAHelper recommend the most suitable machine learning model for a dataset.

### Figure 1: DataQAHelper Application Demo 1
<img src="https://github.com/tangjikededela/DataQAHelp/blob/main/readmepic/demo2.gif?raw=true"
  alt="DataQAHelper Application Demo 1"
  style="width:1920px" />

2. And check out [this Colab notebook](https://github.com/tangjikededela/DataQAHelp/blob/main/tutorial/Tutorial_ModelFittingInterpretation.ipynb) for examples of how applications based on DataQAHelper automatically interpret analysis results of different datasets.

### Figure 2: DataQAHelper Application Demo 2
<img src="https://github.com/tangjikededela/DataQAHelp/blob/main/readmepic/demo.gif?raw=true"
  alt="DataQAHelper Application Demo 2"
  style="width:1920px" />

3. Finally, check out [this Colab notebook](https://github.com/tangjikededela/DataQAHelp/blob/main/tutorial/Tutorial_aSpecialApplicationCaseStudy.ipynb) for an example showcasing a real case study involving the cyclical reporting project of Aberdeen's child protection services. This example demonstrates how the specialized application based on DataQAHelper, after completing the iterative refinement process, can automatically generate a substantial amount of content and corresponding tables for the report, significantly reducing the workload for the report writers.

### Figure 3: DataQAHelper Application Demo 3
<img src="https://github.com/tangjikededela/DataQAHelp/blob/main/readmepic/demo3.gif?raw=true"
  alt="DataQAHelper Application Demo 3"
  style="width:1920px" />

The functionality of the framework is far more than the examples above. The content below provides a more comprehensive view of the main machine learning models currently available and the questions that can be answered.

## Available Models and Questions

The figure below illustrates the primary models available in the current framework along with some of the questions they can answer. While the framework supports a wider range of models and questions, the following table highlights only some of the main ones. The details of the figure are as follows: 

### Table 1: Available Models

| **ID** | **Model**                         |
|--------|-----------------------------------|
| M1     | Linear regression                 |
| M2     | Random forest regression          |
| M3     | Decision tree regression          |
| M4     | Gradient boosting                 |
| M5     | Ridge regression                  |
| M6     | Lasso regression                  |
| M7     | Elastic Net Regression            |
| M8     | Least angle regression            |
| M9     | Ada boost regression              |
| M10    | K-neighbors regression            |
| M11    | Piecewise linear regression       |
| M12    | Logistic regression               |
| M13    | Ridge classifier                  |
| M14    | K-neighbours classifier           |
| M15    | Support vector machines           |
| M16    | Decision tree classifier          |
| M17    | Random forest classifier          |
| M18    | Model comparison                  |

### Table 2: Available Questions

| **ID** | **Question**                                                                                       |
|--------|----------------------------------------------------------------------------------------------------|
| Q1     | Is there a strong relationship between X and y? (R2)                                               |
| Q2     | How well does the model's prediction compare to actual values in percentage terms? (MAPE)          |
| Q3     | What is the average squared deviation between the predicted and actual values, and how does it indicate prediction accuracy? (MSE\&RMSE) |
| Q4     | What is the average absolute deviation between the predicted and actual values, and how accurately does it measure prediction performance? (MAE) |
| Q5     | What is the contribution of each feature to the model's predictions, and how do they impact the prediction for individual samples? (SHAP) |
| Q6     | How does the model balance goodness-of-fit with complexity, and which model should be preferred based on the trade-off between model fit and simplicity? (AIC\&BIC) |
| Q7     | Where are the breakpoints located?                                                                 |
| Q8     | How does an increase in X affect y?                                                                |
| Q9     | Which X has a significant effect on y?                                                             |
| Q10    | How independent are the predictor variables, and is multicollinearity present in the regression model? (VIF) |
| Q11    | Which X is most important to y?                                                                    |
| Q12    | Is the model at risk of overfitting or underfitting?                                               |
| Q13    | How to interpret the dendrogram in the model?                                                      |
| Q14    | How accurate is the classifier?                                                                    |
| Q15    | What proportion of positive identifications was actually correct? (Precision)                      |
| Q16    | What proportion of actual positives was correctly identified by the classifier? (Recall)           |
| Q17    | How does the model balance precision and recall, providing a single score for overall classification performance? (F1) |
| Q18    | How well does the model fit the data, considering the complexity of the model relative to the number of observations? (deviance divided by the degree of freedom) |
| Q19    | How much does a one-unit increase in X increase the probability of y?                              |
| Q20    | How to interpret the confusion matrix of the model?                                                |
| Q21    | How to interpret the decision boundary in the model?                                               |
| Q22    | How does the ROC curve of the model explain the classifier's performance? (ROC)                    |
| Q23    | What effect will each X have on each classification outcome?                                       |
| Q24    | Which regression is best for this dataset? And why?                                                |
| Q25    | Which classifier is best for this dataset? And why?                                                |

### Figure 4: Available questions and data science components: ‘Q’ indicates a question, ‘M’ indicates a model, and the check mark indicates that the question and its answer template are available for this model.
<img src="https://github.com/tangjikededela/DataQAHelp/blob/main/readmepic/available.PNG"
  alt="Available Models and Questions"
  style="width:1920px" />

## System Requirements 

* Python version  - '3.10'

## Requirement
### The following packages are required to run the prototype:
```
numpy==1.23.5
seaborn==0.12.2
pandas==1.5.3
matplotlib==3.7.1
matplotlib-inline==0.1.6
scikit-learn==1.2.2
scikit-plot==0.3.7
statsmodels==0.13.5
Jinja2==3.1.2
jupyter-dash==0.4.2
dash==2.9.1
dash-bootstrap-components==1.4.1
dash-core-components==2.0.0
dash-html-components==2.0.0
dash-table==5.0.0
plotly==5.13.1
language-tool-python==2.7.1
iteration-utilities==0.11.0
pwlf==2.2.1
pycaret==3.0.0
joblib==1.2.0
opencv-python==4.7.0.72
python-docx==0.8.11
```
