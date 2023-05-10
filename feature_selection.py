import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn.ensemble import ExtraTreesClassifier


import scipy.stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC
from sklearn.svm import SVR

from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS

def feature_selection(X, y, method = 'raw', model = SVC(kernel = 'rbf', C = 10), n_feature = 30, random_state = 0):
    
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
        
       
        threshold = 0  # threshold: float, threshold value for obtain constant feature
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
        X_new = X.drop(drop_feature, axis=1 )
        
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
        return X_new, y
        
    elif method == 'correlation_filter_approach':
        ''''
        無法列印多feature的correlation matrix 建議不要用，
        用variance_inflation_factor可以得到類似效果
        使用pearson穢語VIF一樣
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
        return X_new, y
        
        
    elif method == 'Sequential_Feature_Selection':
        
        sfs = SFS(model, forward=True, cv=5, floating=False, k_features = n_feature,
                scoring='recall_weighted', verbose=0, n_jobs=-1)
        sfs.fit(X, y, custom_feature_names=X.columns.values)
        print('Best score achieved:{}, Feature\'s names: {}\n'.format(sfs.k_score_, sfs.k_feature_names_))
        for i1 in sfs.subsets_:
            print('{}\n{}\n'.format(i1, sfs.subsets_[i1]))
        
        # drop feature
        X_new = X[sfs.k_feature_names_]
        return X_new, y 
            
            
    elif method == 'Exhaustive_Feature_Selection':
        '''跑很久'''
        
        efs = EFS(model, cv=5, min_features=50, max_features=60, scoring='recall_weighted', n_jobs=-1)
        efs.fit(X, y, custom_feature_names=X.columns.values)
        print('Best score achieved:{}, Feature\'s names: {}\n'.format(efs.best_score_, efs.best_feature_names_))
    
        #drop feature
        X_new = X[sfs.k_feature_names_]      
        return X_new, y
        
    elif method == 'Treebased_embedded_approach':
        threshold = 0.001 # threshold: float, threshold value for Treebased_embedded_approach - feature_importances
        try:
            model.fit(X, y)
            for i, imp in enumerate(model.feature_importances_):
                print('{} = {}'.format(X.columns[i], imp)) 

            #drop feature
            feature_importances = model.feature_importances_
            drop_columns = X.columns[feature_importances < threshold]
            X_new = X.drop(drop_columns, axis=1)
        except:
            print('please use tree_base model')
            pass
        
        return X_new, y
    
    elif method == 'raw':
        X_new = X
        return X_new, y
    else:
        print('error; Please use. correct method')
            
            
df = pd.read_csv('data.csv (1).zip')
df.head()
X = df.drop(['Bankrupt?'], axis=1)
y = df['Bankrupt?']
#X, y = feature_selection(X, y, method='basic_filter_approach')
X, y = feature_selection(X, y, method='mutual_information_approach' )
X, y = feature_selection(X, y, method='variance_inflation_factor')
#X, y = feature_selection(X, y, 'Sequential Feature Selection')
#X, y = feature_selection(X, y, 'Treebased_embedded_approach')
X.head()
