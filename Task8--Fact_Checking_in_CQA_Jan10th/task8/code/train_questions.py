import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
import numpy as np
import os

from keras import layers, models
from keras.utils import np_utils

import xgboost as xgb
from xgboost import XGBClassifier

curr_dir = os.path.dirname(os.path.abspath(__file__))

def get_data_from_pickle_files():
    print "Reading train data:"
    with open('questions_dict_train.pickle', 'rb') as handle:
        data_dict = pickle.load(handle)
    print len(data_dict)

    with open('X_data_combined_concat_train.pickle', 'rb') as handle:
        x_data = pickle.load(handle)

    # get relevant labels
    y_data = []
    for qid in x_data['ids']:
        y_data.append(data_dict[qid]['label'])
    # print y_data
    print len(y_data)

    return x_data['data'], y_data

def train_extra_trees_classifier():

    x_data, y_data = get_data_from_pickle_files()

    print "Training started:"

    # clf = ExtraTreesClassifier()
    # params = {'n_estimators': [1, 3, 9, 15, 20], 'max_features': range(1, 50, 2)}
    # grid = GridSearchCV(estimator=clf, param_grid=params, cv=10)
    # grid.fit(x_data['data'], y_data)
    #
    # best_estimator  = grid.best_estimator_
    # best_parameters = grid.best_params_
    # best_score      = grid.best_score_
    # print "GridSearch Results:"
    # print best_parameters
    # print best_score
    #
    # #GridSearch Results:
    # #{'max_features': 37, 'n_estimators': 20}
    # #0.5590339892665475

    clf = ExtraTreesClassifier(n_estimators=20, max_features=37)
    scores = cross_val_score(clf, x_data, y_data, cv=10)
    print scores.mean()

    clf.fit(x_data, y_data)

    print "Training completed...!"
    return clf

def train_decision_tree_classifier():

    x_data, y_data = get_data_from_pickle_files()

    print "Training started:"

    clf = DecisionTreeClassifier(max_depth=None, min_samples_split=2, random_state=0)
    scores = cross_val_score(clf, x_data, y_data, cv=10)
    print scores.mean()

    clf.fit(x_data, y_data)

    print "Training completed...!"
    return clf

def train_random_forest_classifier():

    x_data, y_data = get_data_from_pickle_files()

    print "Training started:"

    # clf = RandomForestClassifier()
    # params = {'n_estimators': [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]}
    # grid = GridSearchCV(estimator=clf, param_grid=params, cv=10)
    # grid.fit(x_data, y_data)
    #
    # best_estimator  = grid.best_estimator_
    # best_parameters = grid.best_params_
    # best_score      = grid.best_score_
    # print "GridSearch Results:"
    # print best_parameters
    # print best_score
    # print best_estimator

    # GridSearch Results:
    # {'n_estimators': 25}
    # 0.5608228980322003
    # RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
    #                        max_depth=None, max_features='auto', max_leaf_nodes=None,
    #                        min_impurity_decrease=0.0, min_impurity_split=None,
    #                        min_samples_leaf=1, min_samples_split=2,
    #                        min_weight_fraction_leaf=0.0, n_estimators=25, n_jobs=None,
    #                        oob_score=False, random_state=None, verbose=0,
    #                        warm_start=False)

    clf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                                max_depth=None, max_features='auto', max_leaf_nodes=None,
                                min_impurity_decrease=0.0, min_impurity_split=None,
                                min_samples_leaf=1, min_samples_split=2,
                                min_weight_fraction_leaf=0.0, n_estimators=25, n_jobs=None,
                                oob_score=False, random_state=None, verbose=0,
                                warm_start=False)
    scores = cross_val_score(clf, x_data, y_data, cv=10)
    print scores.mean()

    clf.fit(x_data, y_data)

    print "Training completed...!"
    return clf

def train_xgboost_classifier():

    x_data, y_data = get_data_from_pickle_files()

    print "Training xgboost started:"

    ipt_dmatrix = xgb.DMatrix(x_data, label=y_data)

    # param = {'max_depth': 2, 'eta': 1, 'silent': 1, 'eval_metric': 'auc'}
    # trained_model = xgb.train(params=param, dtrain=ipt_dmatrix)
    # trained_model.save_model("depth2_eta1_obj_log.model")
    # return trained_model

    clf = XGBClassifier(learning_rate=0.1, n_estimators=100, max_depth=5, min_child_weight=3, gamma=0.2, subsample=0.6,
                        objective='multi:softprob', colsample_bytree=1.0, nthread=4, scale_pos_weight=1, seed=27, num_class = 3)

    # Setting parameters to classifier using cross validation --- Didn't get better results
    # xgb_params = clf.get_xgb_params()
    # num_boost_rounds = clf.get_params()['n_estimators']
    # cvresult = xgb.cv(xgb_params, ipt_dmatrix, num_boost_round = num_boost_rounds, nfold = 10, early_stopping_rounds = 50)
    # clf.set_params(n_estimators = cvresult.shape[0])

    clf.fit(x_data, y_data, eval_metric="auc")

    est = clf.get_params()['n_estimators']
    lr  = clf.get_params()['learning_rate']

    print "Estimators:", est, "Learning Rate: ", lr

    # Saving model
    print "Saving model"
    file_name = "est_" + str(est) + "_eta_" + str(lr) + "_model.model"
    saved_model_file = os.path.join(curr_dir, "saved_models", "xgboost", file_name)
    clf.save_model(saved_model_file)
    print "Model saved as", saved_model_file, "....!"

    return clf

def train_keran_nn():

    x_data, y_data = get_data_from_pickle_files()
    ipt_shape = len(x_data[0])
    print ipt_shape
    print x_data.shape
    print y_data.shape

    y_data = np_utils.to_categorical(y_data)
    print y_data.shape

    # Building model
    model = models.Sequential()
    # Input - Layer
    model.add(layers.Dense(50, activation="relu", input_shape=(ipt_shape,)))
    # Hidden - Layers
    model.add(layers.Dropout(0.3, noise_shape=None, seed=None))
    model.add(layers.Dense(50, activation="relu"))
    model.add(layers.Dropout(0.2, noise_shape=None, seed=None))
    model.add(layers.Dense(50, activation="relu"))
    # Output- Layer
    model.add(layers.Dense(3, activation="softmax"))
    model.summary()

    # compiling the model
    model.compile(optimizer="adam", loss="categorical_crossentropy")
    print "Fitting"
    print x_data.shape
    print y_data.shape
    results = model.fit(x=x_data, y=y_data, epochs=20)
    print "FITTTTED"

    print results

    return model


def write_to_output_xml(clf, model):
    print "Reading dev data:"
    with open('questions_dict_dev.pickle', 'rb') as handle:
        data_dict = pickle.load(handle)
    print len(data_dict)
    print data_dict.keys()

    with open('X_data_combined_concat_dev.pickle', 'rb') as handle:
        x_data = pickle.load(handle)

    print "Prediction started:"
    y_pred = clf.predict(x_data['data'])

    # print "Prediction results:"
    # print y_pred
    # print type(y_pred)
    print x_data['data'].shape
    print y_pred.shape

    if model == "keras_nn": # NN uses categorical data for labels
        y_pred = np.argmax(y_pred, axis=1)
    print y_pred.shape

    file_name = 'predict_questions_' + model + '.txt'
    file_name = os.path.join(curr_dir, "predict_output", "questions", file_name)

    with open(file_name, 'w') as handle:
        for ind, qid in enumerate(x_data['ids']):
            handle.write(qid+"\t"+str(y_pred[ind])+"\n")

    print "Writing " + file_name + " completed...!"


def write_to_output_xml_from_model(model_file_name, model):
    # Just give the file name (.model) for model_file_name. Path of it is hardcoded at 'saved_models/model/'.
    print "Reading dev data:"
    with open('questions_dict_dev.pickle', 'rb') as handle:
        data_dict = pickle.load(handle)
    print len(data_dict)
    print data_dict.keys()

    with open('X_data_combined_concat_dev.pickle', 'rb') as handle:
        x_data = pickle.load(handle)

    # loading model

    saved_model_file = os.path.join(curr_dir, "saved_models", model, model_file_name)
    clf = xgb.Booster({'nthread': 4})  # init model
    clf.load_model(saved_model_file)  # load data

    print saved_model_file

    print "Prediction started:"
    y_pred = clf.predict(xgb.DMatrix(x_data['data']))

    # print "Prediction results:"
    # print y_pred
    # print type(y_pred)
    print x_data['data'].shape
    print y_pred.shape

    if model in ["keras_nn", "xgboost"]: # NN uses categorical data for labels
        y_pred = np.argmax(y_pred, axis=1)
    print y_pred.shape

    file_name = 'predict_questions_' + model + '_from_model.txt'
    file_name = os.path.join(curr_dir, "predict_output", "questions", file_name)

    with open(file_name, 'w') as handle:
        for ind, qid in enumerate(x_data['ids']):
            handle.write(qid+"\t"+str(y_pred[ind])+"\n")

    print "Writing " + file_name + " completed...!"


write_to_output_xml(train_xgboost_classifier(), model="xgboost")
# write_to_output_xml_from_model("est_100_eta_0.1_model.model", "xgboost")

# write_to_output_xml(train_decision_tree_classifier(), model="decision_tree")

# write_to_output_xml(train_extra_trees_classifier(), model="extra_trees_old")

# write_to_output_xml(train_random_forest_classifier(), model="random_forest")

# write_to_output_xml(train_keran_nn(), model="keras_nn")

'''
0.517889442934286 - RF
0.4804127522782231 - DT
0.5338687787723663 - ET
0.5546013321136639 - AB
'''

"""
>>> from sklearn.model_selection import cross_val_score
>>> from sklearn.datasets import make_blobs
>>> from sklearn.ensemble import RandomForestClassifier
>>> from sklearn.ensemble import ExtraTreesClassifier
>>> from sklearn.tree import DecisionTreeClassifier

>>> X, y = make_blobs(n_samples=10000, n_features=10, centers=100,
...     random_state=0)

>>> clf = DecisionTreeClassifier(max_depth=None, min_samples_split=2,
...     random_state=0)
>>> scores = cross_val_score(clf, X, y, cv=5)
>>> scores.mean()                               
0.98...

>>> clf = RandomForestClassifier(n_estimators=10, max_depth=None,
...     min_samples_split=2, random_state=0)
>>> scores = cross_val_score(clf, X, y, cv=5)
>>> scores.mean()                               
0.999...

>>> clf = ExtraTreesClassifier(n_estimators=10, max_depth=None,
...     min_samples_split=2, random_state=0)
>>> scores = cross_val_score(clf, X, y, cv=5)
>>> scores.mean() > 0.999
True

>>> clf = AdaBoostClassifier(n_estimators=100)
>>> scores = cross_val_score(clf, X, y, cv=5)
>>> print scores.mean()

"""