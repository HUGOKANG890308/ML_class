from structure import *
df= pd.read_csv('data.csv')
df = pd.read_csv('data.csv')
X = df.drop(['Bankrupt?'], axis = 1)
Y = df['Bankrupt?']
print(f'X is {X}, X.shape is {X.shape}\n\n')
print(f'Y is {Y}, Y.shape is {Y.shape}\n\n')

Test_size,Validation_size = 0.2,0.2
Random_state = 0
n_trials=10
X_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = Test_size, 
                                                    random_state = Random_state)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, 
                                                                test_size = Validation_size, 
                                                                 random_state = Random_state)
imblance_way=[ 'ROS', 'RUS', 'SMOTE', 'ADASYN', 'SMOTETomek', 'raw']
feature_selection=['variance_inflation_factor']
turn=[0,1]


turn,imblance_way_name,feature_selection_name=[],[],[]
for j in imblance_way:
    X_train, y_train = imblance_data(X_train, y_train, j, Random_state)
    for k in feature_selection:
        for l in turn:
            imblance_way_name.append(j)
            feature_selection_name.append(k)
            turn.append(l)
            if l==0:
                basic_ml(using_model={'xgb':XGBClassifier(random_state=Random_state),'rf':RandomForestClassifier(random_state=Random_state),
                                    'SVM':SVC(random_state=Random_state)},
                         x_train=X_train,y_train=y_train,x_test=x_test,y_test=y_test)
            if l==1:
                basic_ml(using_model2={'xgb': XGBClassifier(**study(method='xgb', n_trials=n_trials)),'rf':RandomForestClassifier(**study(method='rf', n_trials=n_trials))
                                    ,'SVM':SVC(**study(method='SVM', n_trials=n_trials))},
                         x_train=X_train,y_train=y_train,x_test=x_test,y_test=y_test)
            