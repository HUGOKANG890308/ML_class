import datetime
import structure as s
today=datetime.date.today()
# some config
random_state = 0
test_size = 0.2
k_fold_num = 5
fs_model = s.SVC(kernel='rbf', C=10 )
fs_feature_num = 30
n_trials = 100

feature_selection_mtehod = {'VIF': 'variance_inflation_factor', 
                            'VIF + SFS': 'Sequential Feature Selection', 
                            'Whole_Feature': 'raw'
                            }

imbalance_method = {'ROS': 1, 
                    'RUS': 2, 
                    'SMOTE': 3, 
                    'ADASYN': 4, 
                    'SMOTEENN': 5, 
                    'Without_balance': 6
                    }

# read X, Y
df = s.pd.read_csv('data.csv')
X = df.drop(['Bankrupt?'], axis = 1)
Y = df['Bankrupt?']

# train, test split
X_train, x_test, Y_train, y_test = s.train_test_split(X, Y, test_size = test_size, random_state = random_state)
x_train, x_val, y_train, y_val = s.train_test_split(X_train, Y_train, test_size = test_size, random_state = random_state)
x_train, x_val, x_test = s.standardize(x_train, x_val, x_test, 'min_max')
x_train = s.pd.DataFrame(x_train, columns = X.columns)
x_val = s.pd.DataFrame(x_val, columns = X.columns)
x_test = s.pd.DataFrame(x_test, columns = X.columns)
x_train_select, y_train_select = s.feature_selection(x_train, y_train, 'variance_inflation_factor', fs_model, fs_feature_num, random_state)
x_test_select = x_test[x_train_select.columns] ; x_val_select = x_val[x_train_select.columns]
x_train_select, y_train_select = s.imblance_data(x_train_select, y_train_select, 1, random_state)
def objective(trial):       
    max_depth = trial.suggest_int("max_depth", 2, 128)
    min_child_weight = trial.suggest_int("min_child_weight", 2, 128)
    gamma = trial.suggest_int("gamma", 2, 128)
    subsample = trial.suggest_discrete_uniform('subsample', 0.5, 1, 0.1)
    colsample_bytree = trial.suggest_discrete_uniform(
        'colsample_bytree', 0.5, 1, 0.1)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
    clf=s.XGBClassifier(max_depth=max_depth, min_child_weight=min_child_weight, gamma=gamma, subsample=subsample,
            colsample_bytree=colsample_bytree, learning_rate=learning_rate,
            random_state=0)#, tree_method='gpu_hist' if torch.cuda.is_available() else 'auto')

    clf.fit(x_train_select, y_train_select )
    y_pred = clf.predict(x_val_select)
    scores =s.fbeta_score(y_val, y_pred, beta=3)
    return scores

study = s.optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=n_trials)
s.optuna.visualization.plot_optimization_history(study)
s.optuna.visualization.plot_param_importances(study)
s.optuna.visualization.plot_intermediate_values(study)