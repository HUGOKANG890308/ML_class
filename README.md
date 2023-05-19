# 1 程式架構
    記得每個方法不要寫死!!! 
    要可以透過function變動,也不要一個function 只有一個功能把自己搞死
# 2 禁止事項
    1.亂取變數名稱 
    2.亂取function名稱 
    3.一直覆蓋類似東西,ex:x_train=x_train1 
    4 function 要有註解,解釋input/output內容與格式 
    5.print 要有意義,不要寫死 善正print(f'') 
# 3 資料處理
## 3.1 切分資料
    1.切成train test vaild
## 3.2 標準化
    1. min max
## 3.3 feature selection
    1.VIF
    2.VIF＋SFS
    3.raw data
## 3.4 不平衡資料處理
    1. oversampling
    2. udersampling 
    3. raw data
# 4 通用程式 
    1. evalution --> 評分方式,將結果輸出成DataFrame
    2. basic_ml  --> ML 基礎五步驟

# 5 modeling
## 5.1 ML_model
    這邊不要寫死,要檢測是否有gpu可以用,有的話就切換成gpu模式
    1. SVM
    2. XGboost
    3. random forest
## 5.3 default model 結果
    1.vaild data result and test data result
    2.這邊也要試試看原始資料沒有透過不平衡處理結果
    3.從這邊選擇幾個model 開始做hyper parmater turning
## 5.4 hyper parmater turning
    TPE

# 6 deep_learning_model
    1.NN

