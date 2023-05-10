from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, fbeta_score
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

def train_test_spiltting(data,test_size=0.2,random_state=0):
    '''
    data: input raw data ; type: pandas dataframe
    test_size: size of training data ; type: float; default: 0.2
    random_state: random state ; type: int; default: 0
    '''
    '''
    train_data: training data ; type: pandas dataframe
    test_data: testing data ; type: pandas dataframe
    '''
    return train_data,test_data
    
def train_val_spiltting(train_data,val_size=0.2,random_state=0):
    '''
    train_data: input raw data ; type: pandas dataframe
    val_size: size of validation data ; type: float; default: 0.2
    random_state: random state ; type: int; default: 0
    '''
    '''
    train_data: training data ; type: pandas dataframe
    val_data: validation data ; type: pandas dataframe
    '''
    return train_data,val_data

    

def standardize(training_data,validation_data,testing_data,method='what method you use'):
    '''
    training_data: input training data(data after spiltting) ; type: pandas dataframe
    validation_data: input training data(data after spiltting) ; type: pandas dataframe
    testing_data: input training data(data after spiltting) ; type: pandas dataframe
    method: input method you use to standardize data ; type: string
            1. standardize the data
            2. minmax the data
    '''

    '''
    training_data: training data after standardize ; type: pandas dataframe
    validation_data: validation data after standardize ; type: pandas dataframe
    testing_data: testing data after standardize ; type: pandas dataframe
    '''
    return training_data,validation_data,testing_data
   
def feature_selection(X, y, method='raw', model=SVC(kernel='rbf', C=10 ),  n_feature=30 , random_state=0):
    '''
    Input:
    X: input raw data include all feature; type: pandas dataframe
    y: input target: type: pandas dataframe
    method: Method of feature selection, can be one of the following:
            - 'basic_filter_approach': removes constant and duplicate features
            - 'mutual_information_approach': selects features with high mutual information scores
            - 'correlation_filter_approach': removes highly correlated features
            - 'variance_inflation_factor': removes features with high VIF scores
            - 'Sequential Feature Selection': selects features using SFS algorithm
            - 'Exhaustive Feature Selection': selects features using EFS algorithm
            - 'Treebased_embedded_approach': select feature using Tree base algorithm
            - 'raw' : select all feature
    model: Model use in 'Sequential Feature Selection' and 'Exhaustive Feature Selection'; default:SVC(kernel='rbf', C=10)
    n_feature: number of features ; default: 30
    random_state: random state ; type: int; default: 0

    
    Returns:
    pandas dataframe with selected features, X_new & y

    '''
    
    npX = np.array(X)

    if method == 'basic_filter_approach':
        
        threshold = 0 # threshold: float, threshold value for obtain constant feature
        drop_feature = []
        
        vt = VarianceThreshold(threshold=threshold)
        vt.fit(X)
    
        #obtain constant feature
        constant_columns = vt.get_support()
        for i,feature in enumerate(X.columns):
            if constant_columns[i] == False:
                drop_feature.append(feature)
        print('Constant Feature : {}'.format(drop_feature))
        
        #obtain duplicate feature
        X_T = X.T 
        duplicate_feature = X_T[X_T.duplicated()].index.values
        print('Duplicated Features={}'.format(duplicate_feature))
        
        for feature in duplicate_feature:
            drop_feature.append(feature)
            
        # drop feature
        X_new = X.drop(drop_feature, axis=1)
       
        
    elif method == 'mutual_information_approach':
        
        threshold = 0.001 # threshold: float, threshold value for mutual information score
        drop_feature = []
        
        #obtain feature score
        score = mutual_info_classif(X, y, random_state=random_state)
        for i, feature in enumerate(X.columns):
            print('{}={}'.format(feature, score[i]))
            
        #drop feature
        feature_score = dict(zip(X.columns, score))
        sorted_feature_score = sorted(feature_score.items(), key=lambda x: x[1], reverse=True)
        low_score_features = [f[0] for f in sorted_feature_score if f[1] < threshold]
        X_new = X.drop(low_score_features, axis=1)
        
        
    elif method == 'correlation_filter_approach':
        ''''
        無法列印多feature的correlation matrix 建議不要用，
        用variance_inflation_factor可以得到類似效果
        '''
        corr_method = ['pearson', 'spearman', 'kendall']
        f_size = len(X.columns)
        h_str = '\t'
        for s1 in X.columns:
            h_str = h_str + s1 + '\t'
        for (corr_idx,corr_str) in enumerate(corr_method):
            r_str, p_str = '', ''
            for i1 in range(0, f_size):
                r_str = r_str + X.columns[i1] + '\t'
                p_str = p_str + X.columns[i1] + '\t'
                for i2 in range(0, f_size):
                    if corr_idx==0:  
                        res = scipy.stats.pearsonr(npX[:,i1], npX[:,i2])
                    if corr_idx==1:  
                        res = scipy.stats.spearmanr(npX[:,i1], npX[:,i2])        
                    if corr_idx==2:  
                        res = scipy.stats.kendalltau(npX[:,i1], npX[:,i2])
                    r_str = r_str + '{:.4f}\t'.format(res[0])
                    p_str = p_str + '{:.4f}\t'.format(res[1])
                    r_str = r_str + '\n'
                    p_str = p_str + '\n'

                print('-'*80, '\n{}\'s correlation'.format(corr_str))
                print(h_str,'\n',r_str)

                print('\n{}\'s prob for testing non-correlatio'.format(corr_str))
                print(h_str,'\n',p_str)
        
        
    elif method == 'variance_inflation_factor':
        
        threshold = 10
        f_size = len(X.columns)
        X = X.assign(const=1)
        record = 0
        for i in range(f_size):
            r_str = ''
            for i1 in range(0, f_size - 3 + 1):
                r_str = r_str + X.columns[i1] \
                        + '\t{:.4f}\t'.format(variance_inflation_factor(X.values,i1)) + '\n'
            print(r_str)
            if variance_inflation_factor(X.values,record) > threshold:
                X.drop([X.columns[record]], axis=1, inplace=True)
                f_size -= 1
            else:
                record += 1
        X_new = X.drop(['const'], axis=1)
        
        
    elif method == 'Sequential Feature Selection':
        
        model = model
        sfs = SFS(model, forward=True, cv=5, floating=False, k_features = n_feature,
                scoring='recall_weighted', verbose=0, n_jobs=-1)
        sfs.fit(X, y, custom_feature_names=X.columns.values)
        print('Best score achieved:{}, Feature\'s names: {}\n'.format(sfs.k_score_, sfs.k_feature_names_))
        for i1 in sfs.subsets_:
            print('{}\n{}\n'.format(i1, sfs.subsets_[i1]))
        
        # drop feature
        X_new = X[sfs.k_feature_names_]
            
            
    elif method == 'Exhaustive Feature Selection':
        '''跑很久'''
        
        model = model
        efs = EFS(model, cv=5, min_features=50, max_features=60, scoring='recall_weighted', n_jobs=-1)
        efs.fit(X, y, custom_feature_names=X.columns.values)
        print('Best score achieved:{}, Feature\'s names: {}\n'.format(efs.best_score_, efs.best_feature_names_))
    
        #drop feature
        X_new = X[sfs.k_feature_names_]      

        
    elif method == 'Treebased_embedded_approach':
        threshold = 0.001 # threshold: float, threshold value for Treebased_embedded_approach - feature_importances
        model = ExtraTreesClassifier(n_estimators=50)
        model.fit(X, y)
        for i, imp in enumerate(model.feature_importances_):
            print('{} = {}'.format(X.columns[i], imp)) 
            
        #drop feature
        feature_importances = model.feature_importances_
        drop_columns = X.columns[feature_importances < threshold]
        X_new = X.drop(drop_columns, axis=1)
        
    
    elif method == 'raw':
        X_new = X
    
    return X_new, y

def imblance_data(training_data,validation_data,testing_data,method='what method you use'):
    '''
    training_data: input training data(data after feature selection) ; type: pandas dataframe
    validation_data: input training data(data after feature selection) ; type: pandas dataframe
    testing_data: input training data(data after feature selection) ; type: pandas dataframe
    method: 
            1. over sampling
            2. under sampling
            3. SMOTE
            4. ADASYN
    '''
    '''
    training_data: training data after imblance ; type: pandas dataframe
    validation_data: validation data after imblance ; type: pandas dataframe
    testing_data: testing data after imblance ; type: pandas dataframe
    '''
    return training_data,validation_data,testing_data



def evaluation(y_test, y_pred):
    '''
    to return metrics score
    
    input:
        y_test: input true label; type: pandas dataframe
        y_pred: input prediction; type: pandas dataframe
        
    output:
        evaluation result; type: tuple
    '''
    ac = round(accuracy_score(y_test, y_pred),4)
    f1 = round(f1_score(y_test, y_pred),4)
    pre = round(precision_score(y_test, y_pred),4)
    rec = round(recall_score(y_test, y_pred),4)
    auc =round(roc_auc_score(y_test, y_pred),4)
    f_beta = round(fbeta_score(y_test, y_pred, beta=3),4)
    
    return ac, f1, pre, rec, auc, f_beta

def basic_ml(using_model = dict, X_train, y_train, X_test, y_test ):
    '''
    to return evaluate dataframe
    
    input:
        using_model: input using model; type: dictionary
        x_train: input x_train; type: numpy.ndarray or dataframe
        y_train: input y_train; type: numpy.ndarray or dataframe
        x_test: input x_test; type: numpy.ndarray  or dataframe
        y_test: input y_test; type: numpy.ndarray  or dataframe
    
    output:
        evaluate dataframe
    '''
    score = []
    for i in using_model:
        model = using_model[i]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        score.append([i]+list(evaluation(y_test, y_pred)))
    return pd.DataFrame(data = score, columns = ['model', 'accuracy', 'f1_score', 'precision', 'recall', 'auc', 'f_beta'])


class Classifier(object):
    def __init__(self,clf, X_train, y_train,X_vaild,y_vaild, X_test, y_test):
        '''
        clf: input classifier ; type: sklearn classifier
            XGBoost, LightGBM, CatBoost, RandomForest,SVM, KNN
        X_train: input training data ; type: pandas dataframe
        y_train: input training label ; type: pandas dataframe
        X_vaild: input validation data ; type: pandas dataframe
        y_vaild: input validation label ; type: pandas dataframe
        X_test: input testing data ; type: pandas dataframe
        y_test: input testing label ; type: pandas dataframe
        '''
        self.clf=clf
        self.X_train=X_train
        self.y_train=y_train
        self.X_vaild=X_vaild
        self.y_vaild=y_vaild
        self.X_test=X_test
        self.y_test=y_test
    def search_best_params(self):
        '''
        search best params
        '''
        return self.best_params
    def objective(self,trial,params):
        '''
        objective function
        '''
        return self.best_score
    
    def train(self):
        '''
        train model
        '''
        return self.clf
    def predict(self):
        '''
        predict result
        '''
        return self.y_pred
    def evaluate(self):
        '''
        evaluate model
        '''
        return self.evaluation_result
    def save_model(self):
        '''
        save model
        '''
        return self.model
    def main(self):
        '''
        main function
        '''
        self.best_params=self.search_best_params()
        self.clf=self.train()
        self.y_pred=self.predict()
        self.evaluation_result=self.evaluate()
        self.model=self.save_model()
def deep_learning_model():
    '''
    deep learning model,
    how to do please think by yourself
    
    '''
    

if __name__ == '__main__':
    '''
    main function
    write your code here
    ex:
    using_model={'DecisionTreeClassifier':DecisionTreeClassifier()}
    '''
    '''
    using_model={'DecisionTreeClassifier':DecisionTreeClassifier()}
    for i in using_model:
        print(i)
        evaluate_classifier(using_model[i],X_train, y_train, X_test, y_test)
        print('------------------')
    '''
   