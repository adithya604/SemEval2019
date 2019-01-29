import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier

from keras import layers, models
from keras.utils import np_utils
from keras.callbacks import EarlyStopping, ModelCheckpoint

import  numpy as np
import os

import xgboost as xgb
from xgboost import XGBClassifier

np.random.seed(0)

curr_dir = os.path.dirname(os.path.abspath(__file__))

def get_data_from_pickle_files():

    print "Reading train data:"
    with open('answers_dict_train.pickle', 'rb') as handle:
        data_dict = pickle.load(handle)
    print len(data_dict)

    with open('answers_X_data_combined_concat_train.pickle', 'rb') as handle:
        x_data = pickle.load(handle)

    # get relevant labels
    y_data = []
    for qid in x_data['ids']:
        y_data.append(data_dict[qid]['label'])
    # print y_data
    print len(y_data)

    return x_data['data'], np.asarray(y_data)


def train_adaboost_classifer():
    x_data, y_data = get_data_from_pickle_files()

    print "Training Adaboost started:"

    # clf = AdaBoostClassifier()
    # params = {'n_estimators': range(5, 55, 5)}
    #
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
    #
    # # GridSearch
    # # Results:
    # # {'n_estimators': 45}
    # # 0.4121212121212121
    # # AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None,
    # #                    learning_rate=1.0, n_estimators=45, random_state=None)
    # # 0.357426097711812

    clf =     AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None,
                       learning_rate=1.0, n_estimators=45, random_state=None)

    scores = cross_val_score(clf, x_data, y_data, cv=10)
    print scores.mean()

def train_extra_trees_classifier():

    x_data, y_data = get_data_from_pickle_files()

    print "Training Extra Trees started:"

    # clf = ExtraTreesClassifier()
    # params = {'n_estimators': [1, 3, 9, 15, 20], 'max_features': range(1, 30, 2)}
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
    #
    # # GridSearch
    # # Results:
    # # {'max_features': 27, 'n_estimators': 9}
    # # 0.3939393939393939
    # # ExtraTreesClassifier(bootstrap=False, class_weight=None, criterion='gini',
    # #                      max_depth=None, max_features=27, max_leaf_nodes=None,
    # #                      min_impurity_decrease=0.0, min_impurity_split=None,
    # #                      min_samples_leaf=1, min_samples_split=2,
    # #                      min_weight_fraction_leaf=0.0, n_estimators=9, n_jobs=None,
    # #                      oob_score=False, random_state=None, verbose=0, warm_start=False)
    # # 0.37154021608643456

    clf =     ExtraTreesClassifier(bootstrap=False, class_weight=None, criterion='gini',
                         max_depth=None, max_features=27, max_leaf_nodes=None,
                         min_impurity_decrease=0.0, min_impurity_split=None,
                         min_samples_leaf=1, min_samples_split=2,
                         min_weight_fraction_leaf=0.0, n_estimators=9, n_jobs=None,
                         oob_score=False, random_state=None, verbose=0, warm_start=False)

    scores = cross_val_score(clf, x_data, y_data, cv=10)
    print scores.mean()

    clf.fit(x_data, y_data)

    print "Training completed...!"
    return clf

def train_decision_tree_classifier():

    x_data, y_data = get_data_from_pickle_files()

    print "Training Decision tree started:"

    # clf = DecisionTreeClassifier()
    # params = {'max_depth': [1, 2, 4, 8, 12], 'min_samples_split': range(2, 20, 2)}
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
    #
    # # GridSearch
    # # Results:
    # # {'min_samples_split': 18, 'max_depth': 12}
    # # 0.4202020202020202
    # # DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=12,
    # #                        max_features=None, max_leaf_nodes=None,
    # #                        min_impurity_decrease=0.0, min_impurity_split=None,
    # #                        min_samples_leaf=1, min_samples_split=18,
    # #                        min_weight_fraction_leaf=0.0, presort=False, random_state=None,
    # #                        splitter='best')
    # # 0.4103065226090436

    clf =     DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=12,
                           max_features=None, max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=18,
                           min_weight_fraction_leaf=0.0, presort=False, random_state=None,
                           splitter='best')

    scores = cross_val_score(clf, x_data, y_data, cv=10)
    print scores.mean()

    clf.fit(x_data, y_data)

    print "Training completed...!"
    return clf

def train_random_forest_classifier():

    x_data, y_data = get_data_from_pickle_files()

    print "Training Random Forest started:"

    # clf = RandomForestClassifier()
    # params = {'n_estimators': range(5, 55, 5)}
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
    #
    # # GridSearch
    # # Results:
    # # {'n_estimators': 45}
    # # 0.397979797979798
    # # RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
    # #                        max_depth=None, max_features='auto', max_leaf_nodes=None,
    # #                        min_impurity_decrease=0.0, min_impurity_split=None,
    # #                        min_samples_leaf=1, min_samples_split=2,
    # #                        min_weight_fraction_leaf=0.0, n_estimators=45, n_jobs=None,
    # #                        oob_score=False, random_state=None, verbose=0,
    # #                        warm_start=False)
    # # 0.36041586634653855

    clf =     RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                           max_depth=None, max_features='auto', max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, n_estimators=45, n_jobs=None,
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
    file_name = "answers_est_" + str(est) + "_eta_" + str(lr) + "_model.model"
    saved_model_file = os.path.join(curr_dir, "saved_models", "xgboost", file_name)
    clf.save_model(saved_model_file)
    print "Model saved as", saved_model_file, "....!"

    return clf

def train_keran_nn():
    x_data, y_data = get_data_from_pickle_files()
    ipt_shape = len(x_data[0])
    print type(x_data)
    print type(y_data)
    print ipt_shape
    print x_data.shape
    print y_data.shape

    y_data = np_utils.to_categorical(y_data)
    print y_data.shape

    print "Training Keras NN Started:"

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

    callbacks = [EarlyStopping(monitor='val_loss', patience=7),
                 ModelCheckpoint(filepath='saved_models/keras_nn_best_model.h5', monitor='val_loss', save_best_only=True)]
    # This saved model hasn't performed well. Hence, not using this model.
    
    print "Fitting"
    print x_data.shape
    print y_data.shape
    # results = model.fit(x=x_data, y=y_data, epochs=50, callbacks=callbacks, validation_split=0.10)
    results = model.fit(x=x_data, y=y_data, epochs=50)
    print "FITTTTED"

    print results

    return model


def write_to_output_xml(clf, model):
    opt_folder = "predict_output/answers/"
    print "Reading dev data:"
    with open('answers_dict_dev.pickle', 'rb') as handle:
        data_dict = pickle.load(handle)
    print len(data_dict)
    print data_dict.keys()

    with open('answers_X_data_combined_concat_dev.pickle', 'rb') as handle:
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

    file_name = opt_folder + 'predict_answers_' + model + '.txt'

    with open(file_name, 'w') as handle:
        for ind, qid in enumerate(x_data['ids']):
            handle.write(qid+"\t"+str(y_pred[ind])+"\n")

    print "Writing " + file_name + " completed...!"



try:
    write_to_output_xml(train_adaboost_classifer(), model="adaboost")
except Exception:
    print "Exception at adaboost"
    pass

try:
    write_to_output_xml(train_extra_trees_classifier(), model="extra_trees")
except Exception:
    print "Exception at extra trees"
    pass

try:
    write_to_output_xml(train_decision_tree_classifier(), model="decision_tree")
except Exception:
    print "Exception at decision tree"
    pass

try:
    write_to_output_xml(train_random_forest_classifier(), model="random_forest")
except Exception:
    print "Exception at random forest"
    pass

try:
    write_to_output_xml(train_xgboost_classifier(), model="xgboost")
except Exception:
    print "Exception at xgboost"
    pass

# try:
#     write_to_output_xml(train_keran_nn(), model="keras_nn")
# except Exception:
#     print "Exception at Keras NN"
#     pass

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
"""