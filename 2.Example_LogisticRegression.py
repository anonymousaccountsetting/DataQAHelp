from pandas import read_csv
import IntegratedPipelines as IP

# # # # Example 1: A simple example. Just choose a model,
# # input data, independent and dependent variables,
# # the output will be a series of stories about fitting the data with this model.

# # Step 1: Read the example dataset about diabetes, the dependent column (target variable) should use 0 or 1 to represent not having diabetes or having diabetes.
col_names = ['pregnant', 'glucose level', 'blood pressure', 'skin', 'insulin level', 'BMI', 'pedigree', 'age',
             'diabetes']
diabetes_dataset = read_csv("./data/diabetes.csv", header=None, names=col_names)
# # Step 2: Choose the model (which is logistic regression here) and the independent and dependent variables, the stories will be generated.
pipeline = IP.general_datastory_pipeline()
pipeline.LogisticFit(diabetes_dataset,
                     ['pregnant', 'glucose level', 'blood pressure', 'skin', 'insulin level', 'pedigree', 'BMI', 'age'],
                     'diabetes')
