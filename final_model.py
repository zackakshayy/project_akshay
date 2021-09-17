import pandas as pd
import numpy as np
from scipy import stats
from matplotlib import pyplot
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
#!{sys.executable} -m pip install xgboost
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier


df=pd.read_csv('Train_final.csv')
print(df.shape)

#######Null check
print(df.isnull().sum().sum())
df.replace(to_replace='?', value=np.nan, inplace=True)

summary=df.describe()
print(summary)


#######class count
class_counts = df.groupby('class').size()
print(class_counts)


#######skewness check
skew = df.skew()
print(skew)

#######correlation check
correlations = df.corr()
correlations.drop(correlations.tail(1).index,inplace=True) # drop last n rows
correlations.drop('class', axis=1, inplace=True)

# plot correlation matrix
fig = pyplot.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,9,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
#ax.set_xticklabels(names)
#ax.set_yticklabels(names)
pyplot.show()

###correlation with class
##The point biserial correlation is used to measure the 
##relationship between a binary variable, x, and a continuous variable, y
cor=[]
for i in range(0,df.shape[1]-1):
    r= stats.pointbiserialr(df['class'], df[df.columns[i]])[0]
    cor.append(r)

pyplot.bar(np.arange(28),cor)

#######PCA
X=df.iloc[ : , :-1].values
Y=df.iloc[ : ,-1].values 

# Standardizing the features
scaler = StandardScaler().fit(X)
rescaledX = scaler.transform(X)

################################################################################
##Model comparision with k-fold validation
#!{sys.executable} -m pip install imblearn
from imblearn.ensemble import RUSBoostClassifier
from imblearn.ensemble import EasyEnsembleClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.ensemble import BalancedBaggingClassifier
models = []
models.append(('LR', LogisticRegression(class_weight = 'balanced')))
models.append(('DT', DecisionTreeClassifier(class_weight = 'balanced')))
models.append(('BDT',BalancedBaggingClassifier(base_estimator=DecisionTreeClassifier(),sampling_strategy='auto',replacement=False,random_state=0)))
models.append(('NB', GaussianNB()))
models.append(('RF', RandomForestClassifier(class_weight = 'balanced')))
models.append(('BRF',BalancedRandomForestClassifier(random_state=0))) #each tree of the forest will be provided a balanced bootstrap sample
models.append(('GB', GradientBoostingClassifier()))
models.append(('SGD', SGDClassifier(class_weight = 'balanced')))
models.append(('ADA', AdaBoostClassifier()))
models.append(('EEC',EasyEnsembleClassifier(random_state=0))) #bag AdaBoost learners which are trained on balanced bootstrap samples 
models.append(('XGB', XGBClassifier()))
models.append(('RUS',RUSBoostClassifier(random_state=0))) #randomly under-sample the dataset before to perform a boosting iteration 
models.append(('MLP',MLPClassifier()))
    
# evaluate each model in turn
results = []
names = []
scoring = 'roc_auc'
for name, model in models:
    kfold = StratifiedKFold(n_splits=5, random_state=2)
    cv_results = cross_val_score(model, rescaledX, Y, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    print(name, cv_results.mean(), cv_results.std())

# boxplot algorithm comparison
fig = pyplot.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()

#####################################################################
##########Parameter Tuning
#xgb_model = XGBClassifier(random_state=10)
#from sklearn.model_selection import train_test_split
#from sklearn.model_selection import GridSearchCV
#from sklearn.model_selection import StratifiedKFold
#
#parameters = {'nthread':[4], #when use hyperthread, xgboost may become slower
#              'objective':['binary:logistic'],
#              'learning_rate': [0.05, 0.1], #so called `eta` value
#              'max_depth': [4,5,6],
#              'min_child_weight': [1],
#              'silent': [1],
#              'gamma':[0],
#              'subsample': [0.6,0.8,0.9],
#              'colsample_bytree': [0.7],
#              'n_estimators': [1000], #number of trees, change it to 1000 for better results
#              'missing':[-999],
#              'scale_pos_weight':[1],
#              'seed': [10]}
#
#
#clf = GridSearchCV(xgb_model, parameters, n_jobs=-1, 
#                   cv=StratifiedKFold(n_splits=5, shuffle=True), 
#                   scoring='roc_auc',
#                   verbose=2, refit=True)
#
#clf.fit(rescaledX, Y)
#
#means = clf.cv_results_['mean_test_score']
#stds = clf.cv_results_['std_test_score']
#
#
#for mean, std, params in zip(means, stds, clf.cv_results_['params']):
#    print("%0.3f (+/-%0.03f) for %r"
#          % (mean, std * 2, params))

#test_probs = clf.predict_proba(X_test)[:,1]

#0.889 (+/-0.021) 
#'colsample_bytree': 0.7
#'gamma': 0
#'learning_rate': 0.1
#'max_depth': 5
#'min_child_weight': 1
#'missing': -999
#'n_estimators': 1000
#'nthread': 4
#'objective': 'binary:logistic'
#'scale_pos_weight': 1
#'seed': 10
#'silent': 1
#'subsample': 0.6
    
##########################################################################
###feature importance
# fit model no training data
xgb_model = XGBClassifier(learning_rate =0.1, n_estimators=1000, max_depth=5,
 min_child_weight=1, gamma=0, subsample=0.6, colsample_bytree=0.7,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=10, missing=-999, silent=1)

xgb_model.fit(rescaledX, Y)
feat_imp = pd.DataFrame({'importance':xgb_model.feature_importances_})    
feat_imp['feature'] = df.columns[:-1]
feat_imp.sort_values(by='importance', ascending=False, inplace=True)
feat_imp = feat_imp.iloc[:10]
    
feat_imp.sort_values(by='importance', inplace=True)
feat_imp = feat_imp.set_index('feature', drop=True)
feat_imp.plot.barh(title="Feature Importances")
pyplot.xlabel('Feature Importance Score')
pyplot.show()

#########################################################################################
##################EVALUATION METRICS
from sklearn.model_selection import cross_validate

score=['roc_auc','f1_weighted']

kfold = StratifiedKFold(n_splits=5, random_state=10)
cv_results = cross_validate(xgb_model, rescaledX, Y, cv=kfold, scoring=score)
print(cv_results)


##################################################################learning curve
from sklearn.model_selection import learning_curve
 # Create CV training and test scores for various training set sizes
train_sizes, train_scores, test_scores = learning_curve(XGBClassifier(), 
                                               rescaledX, Y, cv=kfold, scoring='roc_auc', n_jobs=-1, 
                                               # 50 different sizes of the training set
                                               train_sizes=np.linspace(0.5, 1.0, 50))
    
# Create means and standard deviations of training set scores
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)

# Create means and standard deviations of test set scores
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)
    
# Draw lines
pyplot.subplots(figsize=(12,12))
pyplot.plot(train_sizes, train_mean, '--', color="#111111",  label="Training score")
pyplot.plot(train_sizes, test_mean, color="#111111", label="Cross-validation score")

# Draw bands
pyplot.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color="#DDDDDD")
pyplot.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color="#DDDDDD")
    
# Create plot
pyplot.title("Learning Curve")
pyplot.xlabel("Training Set Size") 
pyplot.ylabel("Accuracy Score") 
pyplot.legend(loc="best")
pyplot.tight_layout() 
pyplot.show()

import graphviz
from xgboost import plot_tree
plot_tree(xgb_model, num_trees=1)
fig = pyplot.gcf()
fig.set_size_inches(150, 100)
fig.savefig('tree.png')


###############Save load and testing the model
from sklearn.pipeline import Pipeline
import pickle
pipeline = Pipeline([('scaler', StandardScaler()), ('xgb_model', XGBClassifier(learning_rate =0.1, n_estimators=1000, max_depth=5,
 min_child_weight=1, gamma=0, subsample=0.6, colsample_bytree=0.7,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=10, missing=-999, silent=1))])
pipeline.fit(X, Y)
pickle.dump(pipeline, open('comp', 'wb'))
fin = pickle.load(open('comp', 'rb'))
#ynew = fin.predict(np.array(X_test).reshape(1,28))
#ynew_prob = fin.predict_proba(np.array(X_test).reshape(1,28))[:,1]
#print(ynew,ynew_prob)


df2=pd.read_csv('test.csv')
print(df2.shape)
cl=[]
prob=[]
for i in range(df2.shape[0]):
    a=df2.iloc[ i , :-1].values
    ynew = fin.predict(np.array(a).reshape(1,28))
    ynew_prob = fin.predict_proba(np.array(a).reshape(1,28))[:,1]
    print(ynew,ynew_prob)
    cl.append(ynew)
    prob.append(ynew_prob)
    
accuracy=sum(cl)/df2.shape[0]*100
print(sum(cl), df2.shape[0], accuracy)

    