'''
Description:
Author: xueyuxin_jane
Date: 2020-09-01 21:05:56
LastEditTime: 2022-06-21 01:01:15
LastEditors: xueyuxin_jane
'''
from copy import deepcopy
import pandas as pd
import numpy as np
import openpyxl
from patsy import dmatrices
import statsmodels.api as sm
import xgboost as xgb
# from sklearn.xgboost import XGBClassifier
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
from sklearn import tree
from sklearn.metrics import mean_squared_error, accuracy_score,precision_score,f1_score, recall_score, roc_auc_score, precision_recall_curve
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LassoCV, LogisticRegression, RidgeClassifierCV, Lasso
from sklearn.feature_selection import SelectFromModel, VarianceThreshold, RFECV
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score, ShuffleSplit, train_test_split, StratifiedShuffleSplit

def save_data(df, excel_path, sheet):
    '''
    description: save new df_data to new sheet and resume previous sheet
    param :
    df:new data needs to be saved
    excel_path: excel_path(can be empty or not)
    sheet: sheet_name of df to be save in
    return {None}
    '''
    # 写入数据 encoding="utf-8-sig" 看情况而用哦
    writer = pd.ExcelWriter(excel_path, engine='openpyxl')
    book = openpyxl.load_workbook(writer.path)
    writer.book = book
    # df.to_excel(excel_writer=writer, sheet_name=sheet, index=None)
    df.to_excel(excel_writer=writer, sheet_name=sheet)
    writer.save()
    writer.close()

def VIF_feature_reduction(df, y_name,step_size=4, thres=5.0):
    '''
    description: You need to set an inside step to calculate
    the VIF value (don't make the step too big or the VIF value
    will go to infinity). An upper threshold needs to be set.
    The characteristics of VIF beyond the secondary threshold are
    considered multicollinearity. The feature with the highest VIF
    value in the group is removed iteratively, and the iteration
    is repeated until all VIF values are less than this threshold。
    param : df: original_features
    step_size: feature number for VIF calculating
    thres: uppper VIF
    return {DataFrame}
    '''
    drop = True
    iter_time = 0
    y_data = df[y_name]
    df.drop(y_name,axis=1,inplace=True)
    while drop:
        print(" iter time is ", iter_time)
        df_copy = deepcopy(df)
        feature_num = df_copy.shape[1]
        start_list = [x for x in range(
            0, feature_num) if x % step_size == 0 or x == 0]
        drop_list = []
        for num in start_list:
            X = sm.add_constant(df_copy.iloc[:, num:num+step_size])
            # vif = pd.DataFrame()
            vif_feature = list(X.columns)
            vif_value = [variance_inflation_factor(
                X.values, i) for i in range(X.shape[1])]
            del(vif_value[0], vif_feature[0])
            max_value = max(vif_value)
            if max_value > thres:
                max_feature = vif_feature[vif_value.index(max_value)]
                print(max_feature)
                df.drop(max_feature, axis=1, inplace=True)
                drop_list.append(1)
            else:
                drop_list.append(0)
        print(drop_list)
        if all(x == 0 for x in drop_list):
            drop = False
        else:
            drop = True
            print(df.shape)
        iter_time = iter_time + 1
    df[y_name]=y_data
    return df

    # if names.all() == None:
    #     names = ["X%s" % x for x in range(len(coefs))]
    # lst = zip(coefs, names)
    # if sort:
    #     lst = sorted(lst, key=lambda x: -np.abs(x[0]))
    # return " + ".join("%s * %s" % (round(coef, 3), name) for coef, name in lst)

def removing_features_with_low_variance(df, names, y_name,threshold):
    x_columns = [x for x in df.columns if x != y_name]
    x_names = [name for name in names.columns if name != y_name]
    X = df[x_columns]
    Y = df[y_name]
    selector = VarianceThreshold(threshold)
    selector.fit(X)
    # print("After transform is %s"%selector.transform(X))
    # print("The surport is %s"%selector.get_support(True))
    # print("After reverse transform is %s"%selector.inverse_transform(selector.transform(X)))
    new_x = selector.fit_transform(X)
    selected_indexes= selector.get_support(indices = True)  # Returns array of indexes of nonremoved features
    selected_features= [x_names[index] for index in selected_indexes]
    new_features = {}
    for feature in selected_features:
        feature_index = selected_features.index(feature)
        new_features[feature]=new_x[:,feature_index]
    new_features[y_name] = Y
    # print(new_features)
    new_features_df = pd.DataFrame(new_features)
    return new_features_df
    
def recursive_feature_elimination(df, names, y_name):
    x_columns = [x for x in df.columns if x != y_name]
    x_names = [name for name in names.columns if name != y_name]
    X = df[x_columns]
    Y = df[y_name]
    std = StandardScaler()
    x_train_s= std.fit_transform(X)
    # rfc = RandomForestClassifier(n_estimators=2000)
    xgbc = xgb.XGBClassifier(n_estimators=2000, objective='multi:softmax',num_class=3)
    rfecv = RFECV(estimator=xgbc, scoring='neg_mean_squared_error')
    rfecv.fit(x_train_s, Y)
    print ("Features sorted by their rank:")
    print (sorted(zip(map(lambda x: round(x, 4), rfecv.ranking_), x_names)))
    top_50list = sorted(zip(map(lambda x: round(x, 4), rfecv.ranking_), x_names))[:10]
    feature_list = [x[1] for x in top_50list]
    sele_features = df[feature_list]
    sele_features[y_name] = Y
    print(feature_list)
    return sele_features
    # return selected_features_df

def random_forest_selection(df, names, y_name):
    # Load boston housing dataset as an example
    x_columns = [x for x in df.columns if x != y_name]
    x_names = [name for name in names.columns if name != y_name]
    X = df[x_columns]
    Y = df[y_name]
    x_names = set(x_names)
    # rf = RandomForestRegressor(n_estimators=20, max_depth=4)
    for i in range(20):                           #这里我们进行十次循环取交集
        rfc = RandomForestClassifier(n_estimators=1000)
        rfc.fit(X, Y)
        print("training finished")
        tmp = set()
        importances = rfc.feature_importances_
        indices = np.argsort(importances)[::-1]   # 降序排列
        for f in range(X.shape[1]):
            if f < 11:                            #选出前50个重要的特征
                tmp.add(X.columns[indices[f]])
                print(X.columns[indices[f]],importances[indices[f]])
            # print("%2d) %-*s %f" % (f + 1, 30, X.columns[indices[f]], importances[indices[f]]))
        x_names &= tmp
        print("while calculating","round",i,len(x_names))       
    print(x_names, len(x_names), "features are selected")
    new_features = df[x_names]
    new_features[y_name]=Y
    return new_features

def Grid_Search_CV_RFR(X_train, y_train, estimator):
    n_estimators = [int(x) for x in np.linspace(start = 800, stop = 1200, num = 5)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt',"log2"]
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(8, 30, num = 5)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 3, 5]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True,False]
    
    random_grid = {'n_estimators': n_estimators, #5,10,15
                'max_features':max_features,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf,
                'bootstrap': bootstrap}
    # Create the random grid
    # random_grid = {'n_estimators': [1000,800,1200], #5,10,15
    #             'max_features': [2,3,4],
    #             'max_depth': [5,15,25],
    #             'min_samples_split': [2,3],
    #             'min_samples_leaf': [3,4,5,6],
    #             'bootstrap': [False,True]}
    # Use the random grid to search for best hyperparameters
    # First create the base model to tune
    # Random search of parameters, using 3 fold cross validation, 
    # search across 100 different combinations, and use all available cores
    grid_search = RandomizedSearchCV(estimator=estimator, param_distributions=random_grid,
                                scoring='neg_mean_absolute_error', 
                                cv=3, n_iter=300, verbose=2, n_jobs=-1)
    # Fit the random search model
    grid_search.fit(X_train, y_train)
    print(grid_search.best_params_)
    return grid_search.best_estimator_

def random_forest_prediction(df,names,y_name):
    # x_columns = [x for x in df.columns if x != y_name]
    # x_names = [name for name in names.columns if name != y_name]
    # X = df[x_columns]
    # Y = df[y_name]
    X = df.drop(y_name, axis=1)
    Y = df[y_name]
    X = np.array(X)
    std = StandardScaler()
    X_s= std.fit_transform(X)
    pre_scores = []
    ac_scores = []
    re_scores = []
    f1_scores = []
    auc_scores = []
    roc_scores = []
    rf = RandomForestClassifier(n_estimators=5000,min_samples_split=3,
    min_samples_leaf=1,max_features="sqrt",max_depth=15,bootstrap=True)  
    # Log.fit(X_train,y_train)  #训练模型
    # rf = RandomForestClassifier(n_estimators=1000)
    # random_state = np.random.randint(10,370864)
    ss=StratifiedShuffleSplit(n_splits=5,test_size=0.2,random_state=2)
    for train_index, valid_index in ss.split(X_s, Y):
        print(valid_index)
        x_train, x_valid = X[train_index], X[valid_index]
        y_train, y_valid = Y[train_index], Y[valid_index]
        rf.fit(x_train, y_train)
        y_pred = rf.predict(x_valid)
        print(y_valid)
        print(y_pred)
        pre_scores.append(precision_score(y_valid, y_pred, average='micro'))
        ac_scores.append(accuracy_score(y_valid, y_pred))
        re_scores.append(recall_score(y_valid, y_pred))
        f1_scores.append(f1_score(y_valid, y_pred))
        auc_scores.append(roc_auc_score(y_valid, y_pred))
        roc_scores.append(precision_recall_curve(y_valid, y_pred))
    
    # x_train, x_test, y_train, y_test = train_test_split(X_s, Y, stratify=Y, shuffle=True, random_state=1,train_size=0.7)
    # std = StandardScaler()
    # x_train_s,x_test_s = std.fit_transform(x_train),std.fit_transform(x_test)
    # best_model = Grid_Search_CV_RFR(x_train_s,y_train,RandomForestClassifier())
    # # # # best_model = logistic_regression(x_train_s,y_train)
    # predictions = best_model.predict(x_test_s)
    # print("label",y_test)
    # print("predic",predictions)
    # ac_score = accuracy_score(y_test,predictions)
    # re_score = recall_score(y_test,predictions)
    # print(roc_scores)
    ac_score = np.mean(ac_scores)
    re_score = np.mean(re_scores)
    pre_score = np.mean(pre_scores)
    auc_score = np.mean(auc_scores)
    # roc_score = np.mean(roc_scores)
    f_score = np.mean(f1_scores)
    print("accuracy",ac_score)
    print("precision",pre_score)
    print("recall",re_score)
    print("f1",f_score)
    print("auc",auc_score)
    # print("roc",roc_score)

def xgboost_prediction(df,names,y_name):
    X = df.drop(y_name, axis=1)
    Y = df[y_name]
    X = np.array(X)
    std = StandardScaler()
    # X_s= std.fit_transform(X)
    pre_scores = []
    ac_scores = []
    re_scores = []
    f1_scores = []
    auc_scores = []
    roc_scores = []
    cv_num = 1
    ss=StratifiedShuffleSplit(n_splits=5,test_size=0.2,random_state=2)  
    for train_index, valid_index in ss.split(X, Y):
        print(valid_index)
        x_train, x_valid = X[train_index], X[valid_index]
        y_train, y_valid = Y[train_index], Y[valid_index]
        dtrain = xgb.DMatrix(x_train, label=y_train)
        dtest = xgb.DMatrix(x_valid, label=y_valid)
        param = {'max_depth': 4, 'eta': 0.01, 'silent': 1, 'objective': 'binary:logistic'}
        param['nthread'] = 4
        param['eval_metric'] = 'auc'
        evallist = [(dtest, 'eval'), (dtrain, 'train')]
        num_round = 100
        bst = xgb.train(param, dtrain, num_round, evallist, early_stopping_rounds=20)
        # bst = xgb.train(param, dtrain, num_round, evallist, early_stopping_rounds=10)
        bst.save_model('model/'+str(cv_num)+'.model')
        # dump model
        bst.dump_model('dump.raw.txt')
        # dump model with feature map
        # bst.dump_model('dump.raw.txt', 'featmap.txt')
        bst = xgb.Booster({'nthread': 4})  # init model
        bst.load_model('model/'+str(cv_num)+'.model')  # load data
        cv_num = cv_num+1
        ypred = bst.predict(dtest)
        y_pred = [round(value) for value in ypred]
        print(accuracy_score(y_valid, y_pred))
        print(roc_auc_score(y_valid, y_pred))
        pre_scores.append(precision_score(y_valid, y_pred, average='micro'))
        ac_scores.append(accuracy_score(y_valid, y_pred))
        re_scores.append(recall_score(y_valid, y_pred))
        f1_scores.append(f1_score(y_valid, y_pred))
        auc_scores.append(roc_auc_score(y_valid, y_pred))
        roc_scores.append(precision_recall_curve(y_valid, y_pred))

    ac_score = np.mean(ac_scores)
    re_score = np.mean(re_scores)
    pre_score = np.mean(pre_scores)
    auc_score = np.mean(auc_scores)
    # roc_score = np.mean(roc_scores)
    f_score = np.mean(f1_scores)
    print("accuracy",ac_score)
    print("precision",pre_score)
    print("recall",re_score)
    print("f1",f_score)
    print("auc",auc_score) 

def multi_class_xgboost_prediction(df,names,y_name):
    X = df.drop(y_name, axis=1)
    Y = df[y_name]
    X = np.array(X)
    std = StandardScaler()
    X_s= std.fit_transform(X)
    pre_scores = []
    ac_scores = []
    re_scores = []
    f1_scores = []
    auc_scores = []
    roc_scores = []
    cv_num = 1
    ss=StratifiedShuffleSplit(n_splits=5,test_size=0.2,random_state=20)  
    for train_index, valid_index in ss.split(X_s, Y):
        print(valid_index)
        x_train, x_valid = X[train_index], X[valid_index]
        y_train, y_valid = Y[train_index], Y[valid_index]
        dtrain = xgb.DMatrix(x_train, label=y_train)
        dtest = xgb.DMatrix(x_valid, label=y_valid)
        # label = dtest.get_label
        # print(label)
        param = {'max_depth': 10, 'eta': 0.5,'objective':'multi:softmax','num_class':3, 'gamma': 0.1}
        param['nthread'] = 4
        # param['eval_metric'] = 'mlogloss'
        evallist = [(dtest, 'eval'), (dtrain, 'train')]
        num_round = 100
        bst = xgb.train(param, dtrain, num_round, evallist, early_stopping_rounds=20)
        # bst = xgb.train(param, dtrain, num_round, evallist, early_stopping_rounds=10)
        bst.save_model('model/'+str(cv_num)+'.model')
        # dump model
        bst.dump_model('dump.raw.txt')
        # dump model with feature map
        # bst.dump_model('dump.raw.txt', 'featmap.txt')
        bst = xgb.Booster({'nthread': 4})  # init model
        bst.load_model('model/'+str(cv_num)+'.model')  # load data
        cv_num = cv_num+1
        ypred = bst.predict(dtest)
        y_pred = [round(value) for value in ypred]
        print(y_valid)
        print(ypred)
        pre_scores.append(precision_score(y_valid, y_pred, average='weighted'))
        ac_scores.append(accuracy_score(y_valid, y_pred))
        re_scores.append(recall_score(y_valid, y_pred,average='weighted'))
        f1_scores.append(f1_score(y_valid, y_pred,average='weighted'))
        # auc_scores.append(roc_auc_score(y_valid, y_pred))
        # roc_scores.append(precision_recall_curve(y_valid, y_pred))

    ac_score = np.mean(ac_scores)
    re_score = np.mean(re_scores)
    pre_score = np.mean(pre_scores)
    # auc_score = np.mean(auc_scores)
    # roc_score = np.mean(roc_scores)
    f_score = np.mean(f1_scores)
    print("accuracy",ac_score)
    print("precision",pre_score)
    print("recall",re_score)
    print("f1",f_score)
    # print("roc",roc_score)
    # print("auc",auc_score)

def xgboost_train(df,names,y_name):
    X = df.drop(y_name, axis=1)
    Y = df[y_name]
    X = np.array(X)
    std = StandardScaler()
    X_s= std.fit_transform(X)
    pre_scores, pre_final= [],[]
    ac_scores, ac_final= [],[]
    re_scores, re_final= [],[]
    f1_scores, f1_final= [],[]
    auc_scores, auc_final= [],[]
    roc_scores, roc_final = [],[]
    cv_num = 1
    for i in range(1,101):
        ss=StratifiedShuffleSplit(n_splits=5,test_size=0.2,random_state=i)  
        for train_index, valid_index in ss.split(X_s, Y):
            print(valid_index)
            x_train, x_valid = X[train_index], X[valid_index]
            y_train, y_valid = Y[train_index], Y[valid_index]
            dtrain = xgb.DMatrix(x_train, label=y_train)
            dtest = xgb.DMatrix(x_valid, label=y_valid)
            param = {'max_depth': 4, 'eta': 0.01, 'silent': 1, 'objective': 'binary:logistic'}
            param['nthread'] = 4
            param['eval_metric'] = 'auc'
            evallist = [(dtest, 'eval'), (dtrain, 'train')]
            num_round = 50
            bst = xgb.train(param, dtrain, num_round, evallist, early_stopping_rounds=20)
            # bst = xgb.train(param, dtrain, num_round, evallist, early_stopping_rounds=10)
            bst.save_model('model/'+str(i)+"_"+str(cv_num)+'.model')
            # dump model
            bst.dump_model('dump.raw.txt')
            # dump model with feature map
            # bst.dump_model('dump.raw.txt', 'featmap.txt')
            bst = xgb.Booster({'nthread': 4})  # init model
            bst.load_model('model/'+str(i)+"_"+str(cv_num)+'.model')  # load data
            cv_num = cv_num+1
            ypred = bst.predict(dtest)
            y_pred = [round(value) for value in ypred]
            print(accuracy_score(y_valid, y_pred))
            print(roc_auc_score(y_valid, y_pred))
            pre_scores.append(precision_score(y_valid, y_pred, average='micro'))
            ac_scores.append(accuracy_score(y_valid, y_pred))
            re_scores.append(recall_score(y_valid, y_pred))
            f1_scores.append(f1_score(y_valid, y_pred))
            auc_scores.append(roc_auc_score(y_valid, y_pred))
            roc_scores.append(precision_recall_curve(y_valid, y_pred))

        ac_score = np.mean(ac_scores)
        re_score = np.mean(re_scores)
        pre_score = np.mean(pre_scores)
        auc_score = np.mean(auc_scores)
        # roc_score = np.mean(roc_scores)
        f_score = np.mean(f1_scores)
        ac_final.append(ac_score)
        re_final.append(re_score)
        pre_final.append(pre_score)
        f1_final.append(f_score)
        auc_final.append(auc_score)
        # print("accuracy",ac_score)
        # print("precision",pre_score)
        # print("recall",re_score)
        # print("f1",f_score)
        # print("auc",auc_score)
    ac_final = np.mean(ac_final)
    re_final = np.mean(re_final)
    pre_final = np.mean(pre_final)
    auc_final = np.mean(auc_final)
    # roc_score = np.mean(roc_scores)
    f1_final = np.mean(f1_final)
    print("accuracy",ac_final)
    print("precision",pre_final)
    print("recall",re_final)
    print("f1",f1_final)
    print("auc",auc_final)


if __name__ == "__main__":
    df = pd.read_excel("FDG_features_LNI.xlsx", sheet_name="pet_unvif_RFECV")
    names = pd.read_excel("FDG_features_LNI.xlsx", sheet_name="pet_unvif_RFECV", nrows=0)
    new_df = removing_features_with_low_variance(df, names, y_name="high_load", threshold=1.3)
    save_data(new_df,"FDG_features.xlsx","pet_remove_low_variance")
    print(len(df.columns))
