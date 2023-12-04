import numpy as np
import seaborn as sns
from pandas import DataFrame
import matplotlib.pyplot as plt
from sklearn import tree
from jinja2 import Environment, FileSystemLoader
import math
from jupyter_dash import JupyterDash
from sklearn.metrics import roc_curve
from dash import Dash, html, dcc, dash_table, callback
import plotly.express as px
import base64
import language_tool_python

# Loading the folder that contains the txt templates

file_loader = FileSystemLoader('./templates')

# Creating a Jinja Environment

env = Environment(loader=file_loader)

# Loading the Jinja templates from the folder
# For the regression
get_correlation = env.get_template('getcorrelation.txt')
model_comparison = env.get_template('modelcomparison.txt')
correlation_state = env.get_template('correlation.txt')
prediction_results = env.get_template('prediction.txt')
linearSummary = env.get_template('linearSummary.txt')
linearSummary2 = env.get_template('linearSummary2.txt')
linearSummary3 = env.get_template('linearSummary3.txt')
linearQuestion = env.get_template('linearQuestionset.txt')
DecisionTree1 = env.get_template('decisiontree1.txt')
DecisionTree2 = env.get_template('decisiontree2.txt')
DecisionTree3 = env.get_template('decisiontree3.txt')
DecisionTreeQuestion = env.get_template('decisiontreequestion.txt')
gamStory = env.get_template('gamStory.txt')
GAMslinear_stats = env.get_template('GAMsLinearL1')
GAMslinear_R2 = env.get_template('GAMsLinearL2')
GAMslinear_P = env.get_template('GAMsLinearL3')
GAMslinear_sum = env.get_template('GAMsLinearL4')
GB1 = env.get_template('GB1')
GB2 = env.get_template('GB2')
GB3 = env.get_template('GB3')
piecewiseQuestion=env.get_template('piecewiseQuestion.txt')
piecewiseSummary = env.get_template('piecewiseSummary.txt')
piecewiseSummary2 = env.get_template('piecewiseSummary2.txt')
piecewiseSummary3 = env.get_template('piecewiseSummary3.txt')

# For the classifier
logisticSummary = env.get_template('logisticSummary.txt')
logisticSummary2 = env.get_template('logisticSummary2')
logisticSummary3 = env.get_template('logisticSummary3.txt')
logisticQuestion = env.get_template('logisticQuestionset.txt')
classifieraccuracy = env.get_template('classifierAccuracy.txt')
classifierauc = env.get_template('classifierAUC.txt')
classifiercv = env.get_template('classifierCvscore.txt')
classifierf1 = env.get_template('classifierF1score.txt')
classifierimp = env.get_template('classifierImportant.txt')
ridgequestionset = env.get_template('ridgeQuestionset.txt')
ridgedecision = env.get_template('ridgeDecision.txt')
classifierquestionset = env.get_template('classifierQuestionset.txt')
# For the cluster
clusterQuestion = env.get_template('clusterQuestionset.txt')
clusterBestnum = env.get_template('clusterWSSS.txt')
clusterDaCa = env.get_template('clusterDaviesCalinski.txt')
clusterSil = env.get_template('clusterSilhouette.txt')
clusterGroup = env.get_template('clusterGroup.txt')
# For some basic function
basicdescription = env.get_template('basicdescription.txt')
simpletrend = env.get_template('simpletrend.txt')
modeloverfit = env.get_template('modeloverfit.txt')

# variables which each load a different segmented regression template
segmented_R2P = env.get_template('testPiecewisePwlfR2P')
segmented_R2 = env.get_template('testPiecewisePwlfR2')
segmented_P = env.get_template('testPiecewisePwlfP')
segmented_B = env.get_template('testPiecewisePwlfB')
segmented_GD1 = env.get_template('drugreport1')
segmented_GC1 = env.get_template('childreport1')

# For Aberdeen City CP
register_story = env.get_template('register.txt')
risk_factor_story = env.get_template('risk_factor.txt')
reregister_story = env.get_template('reregister.txt')
remain_story = env.get_template('remain_story.txt')
enquiries_story = env.get_template('enquiries_story.txt')

# For different dependent variables compared DRD
dc1 = env.get_template('dependentmagnificationcompare')
dc2 = env.get_template('samedependentmagnificationcompare')
dc3 = env.get_template('dependentquantitycompare')
dc4 = env.get_template('trendpercentagedescription')
dct = env.get_template('trenddescription')
tppc = env.get_template('twopointpeak_child')

# for different independent variables compared
idc1 = env.get_template('independentquantitycompare')
idtpc = env.get_template('independenttwopointcomparison')

# for batch processing
bp1 = env.get_template('batchprocessing1')
bp2 = env.get_template('batchprocessing2')

# for ChatGPT
databackground = env.get_template('databackground')
questionrequest = env.get_template('question_request.txt')

# for pycaret
automodelcompare1 = env.get_template('AMC1.txt')
automodelcompare2 = env.get_template('AMC2.txt')
pycaretimp = env.get_template('pycaret_imp.txt')
pycaretmodelfit = env.get_template('pycaret_modelfit.txt')
pycaretclassificationimp = env.get_template('pycaret_classificationimp.txt')
pycaretclassificationmodelfit = env.get_template('pycaret_classificationmodelfit.txt')
# for SKpipeline
pipeline_interpretation = env.get_template('pipeline_interpretation.txt')


def start_app():
    app_name = JupyterDash(__name__)
    listTabs = []
    return (app_name, listTabs)


def dash_tab_add(listTabs, label, child):
    listTabs.append(dcc.Tab(label=label, children=child))


def run_app(app_name, listTabs, portnum=8050):
    app_name.layout = html.Div([dcc.Tabs(listTabs)])
    app_name.run_server(mode='inline', debug=True, port=portnum)

def TreeExplain(model, Xcol):
    n_nodes = model.tree_.node_count
    children_left = model.tree_.children_left
    children_right = model.tree_.children_right
    feature = model.tree_.feature
    threshold = model.tree_.threshold
    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, 0)]  # start with the root node id (0) and its depth (0)
    explain = ""
    while len(stack) > 0:
        # `pop` ensures each node is only visited once
        node_id, depth = stack.pop()
        node_depth[node_id] = depth
        # If the left and right child of a node is not the same we have a split
        # node
        is_split_node = children_left[node_id] != children_right[node_id]
        # If a split node, append left and right children and depth to `stack`
        # so we can loop through them
        if is_split_node:
            stack.append((children_left[node_id], depth + 1))
            stack.append((children_right[node_id], depth + 1))
        else:
            is_leaves[node_id] = True

    explain = explain + (
        "The binary tree structure has {n} nodes and has "
        "the following tree structure:\n ".format(n=n_nodes)
    )
    for i in range(n_nodes):
        if is_leaves[i]:
            explain = explain + (
                "{space}node={node} is a leaf node.\n".format(
                    space=node_depth[i] * "\t", node=i
                )
            )
        else:
            explain = explain + (
                "{space}node={node} is a split node: "
                "go to node {left} if {feature} <= {threshold} "
                "else to node {right}.\n".format(
                    space=node_depth[i] * "\t",
                    node=i,
                    left=children_left[i],
                    feature=Xcol[feature[i]],
                    threshold=threshold[i],
                    right=children_right[i],
                )
            )
    return (explain)


class Microplanning:
    def variablenamechange(self, dataset, Xcol, ycol, Xnewname, ynewname):
        if Xnewname != "":
            if np.size(Xnewname) != np.size(Xcol):
                raise Exception(
                    "The column name of the replacement X is inconsistent with the size of the column name of the original data X.")
            for i in range(np.size(Xnewname)):
                if (Xnewname[i] != ''):
                    dataset.rename(columns={Xcol[i]: Xnewname[i]}, inplace=True)
                else:
                    Xnewname[i] = Xcol[i]
        elif type(Xcol) == str and Xnewname == "":
            Xnewname = Xcol
        if (ynewname != ''):
            dataset.rename(columns={ycol: ynewname}, inplace=True)
        else:
            ynewname = ycol
        return (dataset, Xnewname, ynewname)

    def GrammarCorrection(self, text):
        tool = language_tool_python.LanguageTool('en-US')
        # get the matches
        matches = tool.check(text)
        my_mistakes = []
        my_corrections = []
        start_positions = []
        end_positions = []

        for rules in matches:
            if len(rules.replacements) > 0:
                start_positions.append(rules.offset)
                end_positions.append(rules.errorLength + rules.offset)
                my_mistakes.append(text[rules.offset:rules.errorLength + rules.offset])
                my_corrections.append(rules.replacements[0])

        my_new_text = list(text)

        for m in range(len(start_positions)):
            for i in range(len(text)):
                my_new_text[start_positions[m]] = my_corrections[m]
                if (i > start_positions[m] and i < end_positions[m]):
                    my_new_text[i] = ""
        my_new_text = "".join(my_new_text)
        return (my_new_text)



class DocumentplanningandDashboard:
    def LinearModelStats_view(self, linearData, r2, data, Xcol, ycol, questionset, expect, portnum):

        if expect == "":
            expect = ["", "", ""]

        # Store results for xcol
        for ind in linearData.index:
            ax = sns.regplot(x=ind, y=ycol, data=data)
            plt.savefig('pictures/{}.png'.format(ind))
            plt.clf()
        # Create Format index with file names
        _base64 = []
        for ind in linearData.index:
            _base64.append(base64.b64encode(open('pictures/{}.png'.format(ind), 'rb').read()).decode('ascii'))

        linear_app, listTabs = start_app()

        # Add to dashbord Linear Model Statistics
        fig = px.bar(linearData)
        question = linearQuestion.render(xcol=Xcol, ycol=ycol, qs=questionset, section=1, indeNum=np.size(Xcol),
                                         trend=expect[0])
        intro = linearSummary2.render(r2=r2, indeNum=np.size(Xcol), modelName="Linear Model", Xcol=Xcol,
                                      ycol=ycol, qs=questionset, t=expect[0], expect=expect[1])
        # intro = MicroLexicalization(intro)
        # set chatGPT
        aim = Xcol
        aim.insert(0, ycol)

        children = [html.P(question), html.Br(), html.P(intro),
                    dash_table.DataTable(data[aim].to_dict('records'),
                                         [{"name": i, "id": i} for i in data[aim].columns],
                                         style_table={'height': '400px', 'overflowY': 'auto'})]
        dash_tab_add(listTabs, 'LinearModelStats', children)
        aim.remove(ycol)

        pf, nf, nss, ss, imp, i = "", "", "", "", "", 0
        # Add to dashbord Xcol plots and data story

        for ind in linearData.index:
            question = linearQuestion.render(xcol=ind, ycol=ycol, qs=questionset, section=2, indeNum=1, trend=expect[0])
            conflict = linearSummary.render(xcol=ind, ycol=ycol, coeff=linearData['coeff'][ind],
                                            p=linearData['pvalue'][ind], qs=questionset, expect=expect[2])

            # newstory = MicroLexicalization(story)
            if abs(linearData['coeff'][ind]) == max(abs(linearData['coeff'])):
                imp = ind
            if linearData['coeff'][ind] > 0:
                pf = pf + "the " + ind + ", "
            elif linearData['coeff'][ind] < 0:
                nf = nf + "the " + ind + ", "
            if linearData['pvalue'][ind] > 0.05:
                nss = nss + "the " + ind + ", "
            else:
                ss = ss + "the " + ind + ", "

            if questionset[1] == 1 or questionset[2] == 1:
                children = [
                    html.Img(src='data:image/png;base64,{}'.format(_base64[i])), html.P(question), html.Br(),
                    html.P(conflict)
                ]
                dash_tab_add(listTabs, ind, children)

            i = i + 1
        question = linearQuestion.render(xcol="", ycol=ycol, qs=questionset, section=3, indeNum=1, trend=expect[0])
        summary = linearSummary3.render(imp=imp, ycol=ycol, nss=nss, ss=ss, pf=pf, nf=nf, t=expect[0], r2=r2,
                                        qs=questionset)

        children = [dcc.Graph(figure=fig), html.P(question), html.Br(), html.P(summary)]
        dash_tab_add(listTabs, 'Summary', children)

        run_app(linear_app, listTabs, portnum)

    def LogisticModelStats_view(self, model,data, Xcol, ycol, devDdf, questionset, portnum):


        # Create data frames for coefficients, p-values and importance scores.
        columns1 = {'coeff': model.params.values, 'pvalue': model.pvalues.round(4).values}
        logisticData1 = DataFrame(data=columns1, index=Xcol)
        columns2 = {'importance score': abs(model.params.values)}
        logisticData2 = DataFrame(data=columns2, index=Xcol)

        # Store results for xcol
        for ind in logisticData1.index:
            ax = sns.regplot(x=ind, y=ycol, data=data, logistic=True)
            plt.savefig('pictures/{}.png'.format(ind))
            plt.clf()

        # Create Format index with file names
        _base64 = []

        for ind in logisticData1.index:
            _base64.append(base64.b64encode(open('pictures/{}.png'.format(ind), 'rb').read()).decode('ascii'))
        logistic_app, listTabs = start_app()
        i = 0

        # Add to dashbord Model Statistics
        question = logisticQuestion.render(indeNum=np.size(Xcol), xcol=Xcol, ycol=ycol, qs=questionset, section=1)
        intro = logisticSummary3.render(r2=devDdf, indeNum=np.size(Xcol), modelName="Logistic Model", Xcol=Xcol,
                                        ycol=ycol, qs=questionset, t=9)
        aim = Xcol
        aim.insert(0, ycol)

        children = [html.P(question), html.Br(), html.P(intro), dash_table.DataTable(data[aim].to_dict('records'),
                                                                                     [{"name": i, "id": i} for i in
                                                                                      data[aim].columns],
                                                                                     style_table={'height': '400px',
                                                                                                  'overflowY': 'auto'})]

        dash_tab_add(listTabs, 'LogisticModelStats', children)
        aim.remove(ycol)

        pos_eff, neg_eff, nss, ss, imp = "", "", "", "", ""
        # Add to dashbord Xcol plots and data story

        for ind in logisticData1.index:
            question = logisticQuestion.render(indeNum=1, xcol=ind, ycol=ycol, qs=questionset, section=2)
            # independent_variable_story
            independent_variable_story = logisticSummary.render(xcol=ind, ycol=ycol,
                                                                odd=abs(
                                                                    100 * (math.exp(logisticData1['coeff'][ind]) - 1)),
                                                                coeff=logisticData1['coeff'][ind],
                                                                p=logisticData1['pvalue'][ind],
                                                                qs=questionset)
            # independent_variable_story = model.MicroLexicalization(independent_variable_story)
            if logisticData1['coeff'][ind] == max(logisticData1['coeff']):
                imp = ind
            if logisticData1['coeff'][ind] > 0:
                pos_eff = pos_eff + ind + ", "
            else:
                neg_eff = neg_eff + ind + ", "
            if logisticData1['pvalue'][ind] > 0.05:
                nss = nss + ind + ", "
            else:
                ss = ss + ind + ", "
            if questionset[1] == 1 or questionset[2] == 1:
                children = [html.Img(src='data:image/png;base64,{}'.format(_base64[i])), html.P(question),
                            html.Br(),
                            html.P(independent_variable_story)]
                dash_tab_add(listTabs, ind, children)
            i = i + 1
        fig = px.bar(logisticData2)
        plt.savefig('pictures/{}.png'.format(imp))
        plt.clf()
        question = logisticQuestion.render(indeNum=1, xcol=ind, ycol=ycol, qs=questionset, section=3)
        summary = logisticSummary2.render(pos=pos_eff, neg=neg_eff, ycol=ycol, nss=nss, ss=ss, imp=imp,
                                          r2=devDdf, qs=questionset)
        # summary = model.MicroLexicalization(summary)

        children = [dcc.Graph(figure=fig), html.P(question), html.Br(), html.P(summary)]
        dash_tab_add(listTabs, 'Summary', children)

        run_app(logistic_app, listTabs, portnum)

    def GradientBoostingModelStats_view(self,data, Xcol, ycol, GBmodel, mse,r2, imp, questionset, gbr_params,
                                        train_errors, test_errors,portnum):

        # Store importance figure
        plt.bar(Xcol, GBmodel.feature_importances_)

        plt.title("Importance Score")
        plt.savefig('pictures/{}.png'.format("GB1"))
        plt.clf()

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 10))  # Set the figure size as desired
        tree.plot_tree(GBmodel.estimators_[0][0], ax=ax, feature_names=Xcol, rounded=True, precision=1, node_ids=True)
        plt.savefig('pictures/tree_figure.png')
        encoded_image = base64.b64encode(open("pictures/tree_figure.png", 'rb').read()).decode('ascii')

        _base64 = []
        _base64.append(base64.b64encode(open('pictures/{}.png'.format("GB1"), 'rb').read()).decode('ascii'))
        # Training & Test Deviance Figure
        test_score = np.zeros((gbr_params['n_estimators'],), dtype=np.float64)
        fig = plt.figure(figsize=(8, 8))
        plt.title('Deviance')
        plt.plot(np.arange(gbr_params['n_estimators']) + 1, GBmodel.train_score_, 'b-',
                 label='Training Set Deviance')
        plt.plot(np.arange(gbr_params['n_estimators']) + 1, test_score, 'r-',
                 label='Test Set Deviance')
        plt.legend(loc='upper right')
        plt.xlabel('Boosting Iterations')
        plt.ylabel('Deviance')
        fig.tight_layout()
        plt.savefig('pictures/{}.png'.format("GB2"))
        plt.clf()
        _base64.append(base64.b64encode(open('pictures/{}.png'.format("GB2"), 'rb').read()).decode('ascii'))

        plt.plot(train_errors, label='Training MSE')
        plt.plot(test_errors, label='Testing MSE')
        plt.legend()
        plt.savefig('pictures/{}.png'.format("GB3"))
        plt.clf()
        _base64.append(base64.b64encode(open('pictures/{}.png'.format("GB3"), 'rb').read()).decode('ascii'))

        GB_app, listTabs = start_app()
        # Add to dashbord Model Statistics
        question1 = DecisionTreeQuestion.render(q=1, m="gb")
        intro = DecisionTree2.render(r2=r2, qs=questionset, indeNum=np.size(Xcol), modelName="Gradient Boosting",
                                     Xcol=Xcol,
                                     ycol=ycol, )
        # micro planning
        # intro = model.MicroLexicalization(intro)
        aim = Xcol
        aim.insert(0, ycol)


        children = [html.P(question1), html.Br(), html.P(intro),
                        dash_table.DataTable(data[aim].to_dict('records'),
                                             [{"name": i, "id": i} for i in data[aim].columns],
                                             style_table={'height': '400px', 'overflowY': 'auto'})]
        dash_tab_add(listTabs, 'Gradient Boosting Stats', children)
        aim.remove(ycol)

        explain = TreeExplain(GBmodel.estimators_[0][0], Xcol)
        # listTabs.append(dcc.Tab(label="Training & Test Deviance", children=[
        #     html.Img(src='data:image/png;base64,{}'.format(_base64[1])), html.P(explain)
        # ]))
        question2 = DecisionTreeQuestion.render(q=2)
        children = [html.Img(src='data:image/png;base64,{}'.format(encoded_image)), html.P(question2), html.Br(),
                        html.Pre(explain)]
        dash_tab_add(listTabs, 'Tree Explanation', children)

        summary = DecisionTree3.render(imp=imp, ycol=ycol, r2=round(r2, 3), qs=questionset, mse=round(mse, 3))
        question3 = DecisionTreeQuestion.render(q=3)

        children = [html.Img(src='data:image/png;base64,{}'.format(_base64[0])), html.P(question3), html.Br(),
                        html.P(summary), ]
        dash_tab_add(listTabs, 'Summary', children)

        overfit = modeloverfit.render()
        question4 = DecisionTreeQuestion.render(q=4)


        children = [html.Img(src='data:image/png;base64,{}'.format(_base64[2])), html.P(question4), html.Br(),
                        html.P(overfit), ]
        dash_tab_add(listTabs, 'Model Fitting', children)

        run_app(GB_app, listTabs, portnum)


    def piecewise_linear_view(self,data, Xcol, ycol,model, slopes, segment_points, segment_values, max_slope_info,portnum):

        r2 = model.r_squared()

        piecewise_app, listTabs = start_app()

        # Add to dashbord Linear Model Statistics
        question = piecewiseQuestion.render(xcol=Xcol, ycol=ycol, section=1)
        intro = piecewiseSummary.render(r2=r2, modelName="Piecewise Linear Model")
        # intro = MicroLexicalization(intro)

        # aim = Xcol
        # aim.insert(0, ycol)

        children = [html.P(question), html.Br(), html.P(intro)]
        dash_tab_add(listTabs, 'PiecewiseStats', children)
        # aim.remove(ycol)


        question = piecewiseQuestion.render(xcol=Xcol, ycol=ycol, section=2)
        summary=""
        for i in range(len(slopes)):

            summary = summary+piecewiseSummary2.render(startPoint=segment_points[i][0],endPoint=segment_points[i][1],slope=slopes[i])+"\n"

        children = [html.P(question), html.Br(), html.P(summary)]
        dash_tab_add(listTabs, 'Slopes', children)


        print(segment_values)
        print(max_slope_info)
        question = piecewiseQuestion.render(xcol="", ycol=ycol, section=3)
        summary = piecewiseSummary3.render(xcol=Xcol, ycol=ycol,startPoint=max_slope_info['start_end_points'][0],endPoint=max_slope_info['start_end_points'][1],slope=max_slope_info['slope'],yValue1=max_slope_info['start_end_values'][0],yValue2=max_slope_info['start_end_values'][1])

        children = [ html.P(question), html.Br(), html.P(summary)]
        dash_tab_add(listTabs, 'Summary', children)

        run_app(piecewise_app, listTabs, portnum)

    def RandomForestModelStats_view(self,data, Xcol, ycol, tree_small, rf_small, DTData, r2, mse, questionset, portnum=8050):

        # Extract one of the decision trees from the Random Forest model
        tree_idx = 0  # Index of the tree to visualize
        estimator = rf_small.estimators_[tree_idx]
        # Create a tree figure
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 10))
        tree.plot_tree(estimator, ax=ax, feature_names=Xcol, rounded=True, precision=1, node_ids=True)

        # Save the tree figure as a PNG image
        plt.savefig('pictures/tree_figure.png')
        # Encode the image as base64
        encoded_image = base64.b64encode(open("pictures/tree_figure.png", 'rb').read()).decode('ascii')

        # Explain of the tree
        explain = TreeExplain(rf_small.estimators_[0], Xcol)
        # Importance score Figure
        imp = ""
        fig = px.bar(DTData)
        for ind in DTData.index:
            if DTData['important'][ind] == max(DTData['important']):
                imp = ind
        # print(DTData['important'])
        RF_app, listTabs = start_app()
        # Add to dashbord Model Statistics
        intro = DecisionTree2.render(r2=r2, qs=questionset, indeNum=np.size(Xcol), modelName="Random Forest", Xcol=Xcol,
                                     ycol=ycol, )
        question1 = DecisionTreeQuestion.render(q=1, m="rf")
        # intro = MicroLexicalization(intro)
        aim = Xcol
        aim.insert(0, ycol)
        children = [html.P(question1), html.Br(), html.P(intro), dash_table.DataTable(data[aim].to_dict('records'),
                                                                                      [{"name": i, "id": i} for i in
                                                                                       data[aim].columns],
                                                                                      style_table={'height': '400px',
                                                                                                   'overflowY': 'auto'})]
        dash_tab_add(listTabs, 'RandomForestModelStats', children)

        aim.remove(ycol)
        question2 = DecisionTreeQuestion.render(q=2)
        tree_explain_story = explain
        children = [html.Img(src='data:image/png;base64,{}'.format(encoded_image)),
                    html.P(question2), html.Br(), html.Pre(tree_explain_story)]
        dash_tab_add(listTabs, 'Tree Explanation', children)

        summary = DecisionTree3.render(imp=imp, ycol=ycol, r2=round(r2, 3), qs=questionset, mse=mse)
        question3 = DecisionTreeQuestion.render(q=3)
        children = [dcc.Graph(figure=fig), html.P(question3), html.Br(), html.P(summary)]
        dash_tab_add(listTabs, 'Summary', children)

        run_app(RF_app, listTabs, portnum)

    def RidgeClassifier_view(self,data, Xcol, ycol, rclf, pca, y_test, y_prob, roc_auc, X_pca, accuracy, importances, class1,
                             class2,confusionmatrix,cv_scores):
        _base64 = []
        ridge_app, listTabs = start_app()
        # Plot ROC curve
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label='ROC Curve (area = {:.2f})'.format(roc_auc))
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc='lower right')
        plt.savefig('pictures/{}.png'.format("ROC"))
        _base64.append(base64.b64encode(open('pictures/{}.png'.format("ROC"), 'rb').read()).decode('ascii'))
        plt.clf()

        # Plot decision boundaries
        sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y_test, palette='husl')
        ax = plt.gca()
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 500),
                             np.linspace(ylim[0], ylim[1], 500))
        Z = rclf.predict(pca.inverse_transform(np.c_[xx.ravel(), yy.ravel()])).reshape(xx.shape)
        ax.contourf(xx, yy, Z, alpha=0.8, cmap='coolwarm')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        # plt.title('Decision Boundaries (Malignant vs. Benign)')
        plt.title('Decision Boundaries')
        plt.savefig('pictures/{}.png'.format("DecisionBoundaries"))
        _base64.append(
            base64.b64encode(open('pictures/{}.png'.format("DecisionBoundaries"), 'rb').read()).decode('ascii'))

        # Create a dataframe to store feature importances along with their corresponding feature names
        X = data[Xcol]
        importances_df = DataFrame({'Feature': X.columns, 'Importance': importances})
        # Sort the dataframe by importance in descending order
        importances_df = importances_df.sort_values(by='Importance', ascending=False)
        # Plot feature importances using a bar plot
        plt.figure(figsize=(10, 6))
        plt.bar(importances_df['Feature'], importances_df['Importance'])
        plt.xlabel('Feature')
        plt.ylabel('Importance')
        plt.title('Feature Importances')
        plt.xticks(rotation=90)
        plt.savefig('pictures/{}.png'.format("FeatureImportances"))

        # Plot confusion matrix
        sns.heatmap(confusionmatrix, annot=True, fmt="d", cmap="Blues", cbar=False)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.savefig('pictures/{}.png'.format("confusionmatrix"))
        _base64.append(base64.b64encode(open('pictures/{}.png'.format("confusionmatrix"), 'rb').read()).decode('ascii'))
        plt.clf()

        _base64.append(
            base64.b64encode(open('pictures/{}.png'.format("FeatureImportances"), 'rb').read()).decode('ascii'))
        for i, importance in enumerate(importances):
            if abs(importance) == max(abs(importances)):
                # print("Feature {}: {}".format(X.columns[i], importance))
                imp = X.columns[i]

        # Extract coefficients from the trained model
        coefs = rclf.coef_[0]
        intercept = rclf.intercept_[0]

        # Construct the linear equation for the decision boundaries
        equation = 'Decision Boundary Equation: '
        for i in range(len(Xcol)):
            equation += '({:.4f} * {}) + '.format(coefs[i], Xcol[i])
        equation += '{:.4f}'.format(intercept)

        intro = classifieraccuracy.render(accuracy=round(accuracy, 3), classifiername="ridge classifier")
        question = ridgequestionset.render(section=1)
        aim = Xcol
        aim.insert(0, ycol)
        children = [html.P(question), html.Br(), html.P(intro), dash_table.DataTable(data[aim].to_dict('records'),
                                                                                     [{"name": i, "id": i} for i in
                                                                                      data[aim].columns],
                                                                                     style_table={'height': '400px',
                                                                                                  'overflowY': 'auto'})]

        dash_tab_add(listTabs, 'RidgeClassifierStats', children)
        aim.remove(ycol)

        question = ridgequestionset.render(section=2)
        DecisionBoundaryStory = ridgedecision.render(equation=equation, class1=class1, class2=class2)
        children = [html.Img(src='data:image/png;base64,{}'.format(_base64[1])), html.P(question), html.Br(),
                    html.P(DecisionBoundaryStory)]
        dash_tab_add(listTabs, "Decision Boundary Equation", children)

        question = ridgequestionset.render(section=3)
        aucStory = classifierauc.render(AUC=roc_auc)
        children = [html.Img(src='data:image/png;base64,{}'.format(_base64[0])), html.P(question), html.Br(),
                    html.P(aucStory)]
        dash_tab_add(listTabs, "Area under the Receiver Operating Characteristic curve", children)

        question = classifierquestionset.render(section=3)
        crossvalidStory = classifiercv.render(cv=round(cv_scores.mean(), 3), cm=confusionmatrix)
        children = [html.Img(src='data:image/png;base64,{}'.format(_base64[2])), html.P(question), html.Br(),
                    html.P(crossvalidStory)]
        dash_tab_add(listTabs, "Confusion Matrix and Cross-validation", children)

        question = ridgequestionset.render(section=4)
        ImpStory = classifierimp.render(imp=imp)
        children = [html.Img(src='data:image/png;base64,{}'.format(_base64[3])), html.P(question), html.Br(),
                    html.P(ImpStory)]
        dash_tab_add(listTabs, "Feature Importances", children)

        run_app(ridge_app, listTabs)

    def KNeighborsClassifier_view(self,data, Xcol, ycol, accuracy, precision, feature_importances, recall, f1,
                                  confusionmatrix, cv_scores):
        _base64 = []
        KNei_app, listTabs = start_app()
        # Print feature importances with column names
        for i in range(len(feature_importances)):
            if abs(feature_importances[i]) == max(abs(feature_importances)):
                print("Feature {}: {} - {:.2f}".format(i + 1, Xcol[i], feature_importances[i]))
                imp = Xcol[i]

        # Create a dictionary to store the evaluation metrics
        metrics = {"Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1-score": f1}
        # Plot the evaluation metrics
        plt.bar(metrics.keys(), metrics.values())
        plt.xlabel("Metrics")
        plt.ylabel("Score")
        plt.title("Model Evaluation Metrics")
        plt.savefig('pictures/{}.png'.format("Metrics"))
        _base64.append(base64.b64encode(open('pictures/{}.png'.format("Metrics"), 'rb').read()).decode('ascii'))
        plt.clf()

        # Plot confusion matrix
        sns.heatmap(confusionmatrix, annot=True, fmt="d", cmap="Blues", cbar=False)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.savefig('pictures/{}.png'.format("confusionmatrix"))
        _base64.append(base64.b64encode(open('pictures/{}.png'.format("confusionmatrix"), 'rb').read()).decode('ascii'))
        plt.clf()

        # Plot feature importances
        plt.barh(Xcol, feature_importances)
        plt.xlabel("Mutual Information")
        plt.title("Feature Importances")
        plt.savefig('pictures/{}.png'.format("imp"))
        _base64.append(base64.b64encode(open('pictures/{}.png'.format("imp"), 'rb').read()).decode('ascii'))
        plt.clf()

        question = classifierquestionset.render(section=1)
        intro = classifieraccuracy.render(accuracy=round(accuracy, 3), classifiername="K neighbors classifier")
        aim = Xcol
        aim.insert(0, ycol)
        children = [html.P(question), html.Br(), html.P(intro), dash_table.DataTable(data[aim].to_dict('records'),
                                                                                     [{"name": i, "id": i} for i in
                                                                                      data[aim].columns],
                                                                                     style_table={'height': '400px',
                                                                                                  'overflowY': 'auto'})]

        dash_tab_add(listTabs, 'KNeighborsClassifierStats', children)
        aim.remove(ycol)

        question = classifierquestionset.render(section=2)
        modelStory = classifierf1.render(f1=round(f1, 3))
        children = [html.Img(src='data:image/png;base64,{}'.format(_base64[0])), html.P(question), html.Br(),
                    html.P(modelStory)]
        dash_tab_add(listTabs, "Model Evaluation Metrics", children)

        question = classifierquestionset.render(section=3)
        crossvalidStory = classifiercv.render(cv=round(cv_scores.mean(), 3), cm=confusionmatrix)
        children = [html.Img(src='data:image/png;base64,{}'.format(_base64[1])), html.P(question), html.Br(),
                    html.P(crossvalidStory)]
        dash_tab_add(listTabs, "Confusion Matrix and Cross-validation", children)

        question = classifierquestionset.render(section=4)
        ImpStory = classifierimp.render(imp=imp)
        children = [html.Img(src='data:image/png;base64,{}'.format(_base64[2])), html.P(question), html.Br(),
                    html.P(ImpStory)]
        dash_tab_add(listTabs, "Feature Importances", children)

        run_app(KNei_app, listTabs)

    def SVCClassifier_view(self,data, Xcol, ycol, accuracy, precision, recall, f1, confusionmatrix, cv_scores):
        _base64 = []
        svm_app, listTabs = start_app()

        # Create a dictionary to store the evaluation metrics
        metrics = {"Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1-score": f1}
        # Plot the evaluation metrics
        plt.bar(metrics.keys(), metrics.values())
        plt.xlabel("Metrics")
        plt.ylabel("Score")
        plt.title("Model Evaluation Metrics")
        plt.savefig('pictures/{}.png'.format("Metrics"))
        _base64.append(base64.b64encode(open('pictures/{}.png'.format("Metrics"), 'rb').read()).decode('ascii'))
        plt.clf()

        # Plot confusion matrix
        sns.heatmap(confusionmatrix, annot=True, fmt="d", cmap="Blues", cbar=False)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.savefig('pictures/{}.png'.format("confusionmatrix"))
        _base64.append(base64.b64encode(open('pictures/{}.png'.format("confusionmatrix"), 'rb').read()).decode('ascii'))
        plt.clf()

        question = classifierquestionset.render(section=1)
        intro = classifieraccuracy.render(accuracy=round(accuracy, 3), classifiername="Support Vector Machine model")
        aim = Xcol
        aim.insert(0, ycol)
        children = [html.P(question), html.Br(), html.P(intro), dash_table.DataTable(data[aim].to_dict('records'),
                                                                                     [{"name": i, "id": i} for i in
                                                                                      data[aim].columns],
                                                                                     style_table={'height': '400px',
                                                                                                  'overflowY': 'auto'})]

        dash_tab_add(listTabs, 'SupportVectorMachineModelStats', children)
        aim.remove(ycol)

        question = classifierquestionset.render(section=2)
        modelStory = classifierf1.render(f1=round(f1, 3))
        children = [html.Img(src='data:image/png;base64,{}'.format(_base64[0])), html.P(question), html.Br(),
                    html.P(modelStory)]
        dash_tab_add(listTabs, "Model Evaluation Metrics", children)

        question = classifierquestionset.render(section=3)
        crossvalidStory = classifiercv.render(cv=round(cv_scores.mean(), 3), cm=confusionmatrix)
        children = [html.Img(src='data:image/png;base64,{}'.format(_base64[1])), html.P(question), html.Br(),
                    html.P(crossvalidStory)]
        dash_tab_add(listTabs, "Confusion Matrix and Cross-validation", children)

        run_app(svm_app, listTabs)

    def DecisionTreeModelStats_view(self,data, Xcol, ycol, DTData, DTmodel, r2, mse, questionset, portnum=8050):
        # Importance score Figure
        imp = ""
        fig = px.bar(DTData)
        for ind in DTData.index:
            if DTData['important'][ind] == max(DTData['important']):
                imp = ind

        DT_app, listTabs = start_app()

        # Add to dashbord Model Statistics
        question1 = DecisionTreeQuestion.render(q=1, m="dt")
        intro = DecisionTree2.render(r2=r2, qs=questionset, indeNum=np.size(Xcol), modelName="Decision Tree", Xcol=Xcol,
                                     ycol=ycol, )
        # intro = MicroLexicalization(intro)
        aim = Xcol
        aim.insert(0, ycol)
        children = [html.P(question1), html.Br(), html.P(intro),
                    dash_table.DataTable(data[aim].to_dict('records'),
                                         [{"name": i, "id": i} for i in data[aim].columns],
                                         style_table={'height': '400px', 'overflowY': 'auto'})]
        dash_tab_add(listTabs, 'DecisionTreeModelStats', children)
        aim.remove(ycol)

        # Figure of the tree
        fig2, axes = plt.subplots()
        tree.plot_tree(DTmodel,
                       feature_names=Xcol,
                       class_names=ycol,
                       filled=True,
                       node_ids=True);
        fig2.savefig('pictures/{}.png'.format("DT"))
        encoded_image = base64.b64encode(open("pictures/DT.png", 'rb').read()).decode('ascii')

        # # Explain of the tree
        explain = TreeExplain(DTmodel, Xcol)
        # Text need to fix here
        tree_explain_story = explain
        question2 = DecisionTreeQuestion.render(q=2)
        children = [html.Img(src='data:image/png;base64,{}'.format(encoded_image)),
                    html.P(question2), html.Br(), html.Pre(tree_explain_story)]
        dash_tab_add(listTabs, 'Tree Explanation', children)

        summary = DecisionTree3.render(imp=imp, ycol=ycol, r2=round(r2, 3), qs=questionset, mse=mse)
        question3 = DecisionTreeQuestion.render(q=3)
        children = [dcc.Graph(figure=fig), html.P(question3), html.Br(), html.P(summary)]
        dash_tab_add(listTabs, 'Summary', children)

        run_app(DT_app, listTabs, portnum)

