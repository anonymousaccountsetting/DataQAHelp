from pandas import read_csv
import IntegratedPipelines as IP
import DataScienceComponents as DC

# # Before everything starts: Set up a pipeline.
pipeline=IP.general_datastory_pipeline()

# # Example 1: A simple example.
# # Just choose a model, input data, independent and dependent variables,
# # the output will be a series of stories about fitting the data with this model.

# # # Step 1: Read the example dataset about red wine quality
# Car_dataset = read_csv('./data/car data.csv')
#
# Xcol = ['Present_Price', 'Kms_Driven', 'Year']
# ycol = 'Selling_Price'
#
# # # Step 2: Choose the model (which is linear regression here) and the independent and dependent variables,
# # # the stories will be generated.
# pipeline.LinearFit(Car_dataset,Xcol,ycol)

# # # Example 2: A more complex example.
# # Choose a model, do the dataset cleaning before fitting it to a model.
# # Input data, independent and dependent variables.
# # The following are optional:
# # Set more readable names for variables.
# # Select the question you want the system to answer.
# # Choose your overall expectations for the fit.

# Step 1: Read the example dataset about crime rate and drop some columns
features = read_csv('./data/attributes.csv', delim_whitespace=True)
dataset = read_csv('./data/communities.data', names=features['attributes']).drop(
    columns=['state', 'county', 'community', 'communityname', 'fold', 'racepctblack', 'racePctWhite', 'racePctAsian',
             'racePctHisp'], axis=1)
# Step 2: Data cleaning
dataset = DC.DataEngineering().CleanData(dataset, 0.8)  # Clear data with a threshold of 80%
# Setting the more readable variable names
readable_names = dict((kv.split(': ') for kv in (l.strip(' \n') for l in open('./data/readableNames.txt'))))
# Step 3: Choose the model, the independent and dependent variables,
# replace the independent and dependent variables, set questions, and the expectation.
pipeline.LinearFit(dataset, ['pctWPubAsst', 'PctHousLess3BR', 'PctPersOwnOccup'], 'ViolentCrimesPerPop',
                   [readable_names.get(key) for key in ['pctWPubAsst', 'PctHousLess3BR', 'PctPersOwnOccup']],
                   readable_names.get('ViolentCrimesPerPop'), questionset=[1, 1, 1, 1], expect=[1,1,1])
