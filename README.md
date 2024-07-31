[//]: # (<p align="center"><a href="#"><img width=60% alt="" src="https://github.com/lux-org/lux-resources/blob/master/readme_img/logo.png?raw=true"></a></p>)
<h1 align="center">DataQAHelper</h1>
<h3 align="center">A Framework for Data-to-Text Application Development</h3>


The DataQAHelper is a Python-based framework that integrates commonly used data science algorithms with model-based question banks and answer templates. This framework aims to facilitate a quick start in developing data-to-text applications. With just a few lines of code, users can quickly build applications to support data visualization, model evaluation, and data interpretation. The unique feature of this framework is its ability to transform complex data analysis results into understandable FAQ-like textual reports, enabling users to quickly grasp the insights from the dataset.

Another significant advantage of the framework is its support for reporting and narrative tasks in data science. When presented with a dataset and business questions from a client, data scientists can select the appropriate data science model and determine the suitable data analysis algorithm. In this scenario, data scientists as the developers can match specific business questions with those in the question bank and call the corresponding pipelines to swiftly develop data-to-text applications. Additionally, the framework encourages developers to continuously refine the application and improve the quality of the generated reports based on their requirements. This allows for the repeated use of the application in future projects.

## Simple Examples

Here are some examples of simple applications based on DataQAHelper in action.

Check out [this Colab notebook](https://github.com/tangjikededela/DataQAHelp/blob/main/tutorial/Tutorial_ModelComparisonInterpretation.ipynb) for examples of how applications based on DataQAHelper recommend the most suitable machine learning model for a dataset.
<img src="https://github.com/tangjikededela/DataQAHelp/blob/main/readmepic/demo2.gif?raw=true"
  alt="DataQAHelper Application Demo 1"
  style="width:1920px" />

And check out [this Colab notebook](https://github.com/tangjikededela/DataQAHelp/blob/main/tutorial/Tutorial_ModelFittingInterpretation.ipynb) for examples of how applications based on DataQAHelper automatically interpret analysis results of different datasets.
<img src="https://github.com/tangjikededela/DataQAHelp/blob/main/readmepic/demo.gif?raw=true"
  alt="DataQAHelper Application Demo 2"
  style="width:1920px" />

Finally, check out [this Colab notebook](https://github.com/tangjikededela/DataQAHelp/blob/main/tutorial/Tutorial_aSpecialApplicationCaseStudy.ipynb) for an example showcasing a real case study involving the cyclical reporting project of Aberdeen's child protection services. This example demonstrates how the specialized application based on DataQAHelper, after completing the iterative refinement process, can automatically generate a substantial amount of content and corresponding tables for the report, significantly reducing the workload for the report writers.
<img src="https://github.com/tangjikededela/DataQAHelp/blob/main/readmepic/demo3.gif?raw=true"
  alt="DataQAHelper Application Demo 3"
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
