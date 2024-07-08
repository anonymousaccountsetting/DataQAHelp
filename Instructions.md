# Overview:
This document aims to outline several main functions and integrated pipelines that can be directly invoked to quickly build initial applications. These can take datasets as input, perform the corresponding model fitting, and ultimately present the data analysis results to the user in the form of intuitive textual reports.

Note: This tutorial assumes that you have already cloned the package, if you have not done so already, please check out this [page](https://github.com/tangjikededela/DataQAHelp).

# Introduction to ready-to-use pipeline:
The framework has three core components: the data science components, the NLG components, and the integrated pipelines. Quickly constructing an initial application often only requires a simple call to the integrated pipelines.

To enable the integrated pipelines, create a new .py file in the root folder and add the import statement. (If the .py file is created in another folder, ensure the import path is correct.)

Similarly, the data science components and the NLG components can also provide functions separately, such as data cleaning or checking for missing values. These components can be imported in the same manner.

```
import IntegratedPipelines as IP
import DataScienceComponents as DC
import NLGComponents as NC
```
### Textual Report of Rapid Data Exploration
NLGComponents is mainly used to use the dashboard to present pictures of data analysis results and related FAQ-like reports. However, it also has some functions to support users to quickly check a data set.

If users want to quickly understand the data, they can use pandas to read the dataset first then run the following code, which will construct a histogram and describe in text how many missing and unique data each row has.

```
from pandas import read_csv
df=read_csv(dataset path,header=0)
NC.DocumentplanningNoDashboard().dataset_information(df)
```
Users can also generate a heatmap with the following code to further understand which variables have positive or negative relationships.


```
NC.DocumentplanningNoDashboard().simple_dataset_description(df)
```
Similarly, the following code not only generates a heatmap but also provides a more detailed description of whether the pairwise relationships between all variables are strong.


```
NC.DocumentplanningNoDashboard().dataset_description(df)
```

### Automatic Model Recommendation
If the user is not sure which machine learning model to use, there are ready-to-use integrated pipelines in the framework to provide recommendations. First, read the dataset.


```
from pandas import read_csv
df=read_csv(DatasetPath,header=0)
Xcol=['Independent variable 1', 'Independent variable 2'...]
ycol='Dependent variable'
```

Then set up the pipeline that can automatically recommend a model based on the dataset.


```
pipeline=IP.find_best_mode_pipeline()
```

Next, enter the corresponding dataset, independent variables, and dependent variable to run the pipeline. The results will be displayed on the dashboard.

```
# Recommend the most suitable regression model
pipeline.FindBestRegressionPipeline(data,Xcol,ycol)
# Recommend the most suitable classification model
pipeline.FindBestClassifierPipeline(data,Xcol,ycol)
```


### FAQ-like Report of Machine Learning Models

The integrated pipelines combine multiple machine learning models and corresponding question-answering templates. They are designed based on the same principle, so the usage is roughly the same. The following shows several use cases.


First, read the dataset and set the dependent and independent variable names.


```
from pandas import read_csv
df=read_csv(DatasetPath,header=0)
Xcol=['Independent variable 1', 'Independent variable 2'...]
ycol='Dependent variable'
```

Then, set up the pipeline that interprets the results of the machine learning model analysis.


```
pipeline=IP.general_datastory_pipeline()
```

Next, simply select the corresponding machine learning model to complete the fitting, and a series of data analysis images and corresponding FAQ-like reports will be displayed on the dashboard.


For example, using a linear regression model, you can use the following code:

```
pipeline.LinearFit(df,Xcol,ycol)
```

These pipelines also support using more readable column names to replace the original column names.

```
NewXcolName=['More readable X 1', 'More readable X 2'...]
NewycolName='More readable y'
pipeline.LinearFit(df,Xcol,ycol,NewXcolName,NewycolName)
```
Users can also replace 'LinearFit' with any of the following models to get the corresponding data analysis results, visualization images, and FAQ-like reports on the dashboard:

- LogisticFit
- GradientBoostingFit
- RandomForestFit
- DecisionTreeFit
- piecewiselinearFit
- RidgeClassifierFit
- KNeighborsClassifierFit
- SVMClassifierFit

If users have enough knowledge about the components of the framework, they can also easily add new models for data science, NLG components, or integration pipelines.

