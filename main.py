import pandas as pd
from XFM import *
from Lgb import LGBTrain
from datetime import datetime
from sklearn.metrics import accuracy_score,log_loss,roc_auc_score

if __name__ == '__main__':
    trainData = 'data/train.csv'  # 请换为自己文件的路径
    testData = 'data/test.csv'

    train = pd.read_csv(trainData)
    test = pd.read_csv(testData)
    test_id = test['id']
    data = [train,test]
    del train,test

    category_features = ['province','carrier','devtype','nnt','advert_id','f_channel']
    numeric_features = ['creative_tp_dnf_mean', 'creative_tp_dnf_count', 'creative_tp_dnf_sum',
                       'app_cate_id_mean', 'app_cate_id_count', 'app_cate_id_sum',
                       'f_channel_mean', 'app_id_mean', 'inner_slot_id_mean']
    label = 'click'

    dataTrain, dataTest, labelTrain, labelTest  = preprocessData(data,category_features,numeric_features,label)

    print("Train Data Shape :",dataTrain.shape)
    date_startTrain = datetime.now()

    print("开始训练")
    w_0, w, v = SGD_FM(mat(dataTrain), labelTrain, 5, 1000, early_stop = 80)
    date_endTrain = datetime.now()
    print("训练用时为：%s" % (date_endTrain - date_startTrain))

    ####test
    print("开始测试XFM")
    pred_train = predictXFM(mat(dataTrain), w_0, w, v)
    print("train AUC =", roc_auc_score(labelTrain, pred_train))
    print("train logloss =", log_loss(labelTrain, pred_train))
    #print("train accuracy =", accuracy_score(labelTrain, [0 for i in pred_train if i < 0.5 else 1]))

    pred_test = predictXFM(mat(dataTest), w_0, w, v)
    print("test AUC =", roc_auc_score(labelTest, pred_test))
    print("test logloss =", log_loss(labelTest, pred_test))
    # print("test accuracy =", accuracy_score(labelTest, pred_test))

    print("开始测试LGB")
    pred_lgb = LGBTrain(data,category_features,numeric_features,label)
    print("lgb AUC =",roc_auc_score(labelTest, pred_lgb))
    print("lgb logloss =",log_loss(labelTest, pred_lgb))
    # print("lgb accuracy =",accuracy_score(test[label], pred_lgb))

