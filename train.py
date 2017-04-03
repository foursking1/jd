__author__ = 'foursking'
from gen_user_feat import make_train_set
from sklearn.model_selection import train_test_split
import xgboost as xgb
from gen_user_feat import report

def logistic_regression():
    train_start_date = '2016-02-01'
    train_end_date = '2016-03-01'
    test_start_date = '2016-03-01'
    test_end_date = '2016-03-05'
    user_index, training_data, label = make_train_set(train_start_date, train_end_date, test_start_date, test_end_date)


def xgboost_cv():
    train_start_date = '2016-02-01'
    train_end_date = '2016-03-01'
    test_start_date = '2016-03-01'
    test_end_date = '2016-03-06'

    sub_start_date = '2016-02-05'
    sub_end_date = '2016-03-05'
    sub_test_start_date = '2016-03-05'
    sub_test_end_date = '2016-03-10'

    user_index, training_data, label = make_train_set(train_start_date, train_end_date, test_start_date, test_end_date)
    X_train, X_test, y_train, y_test = train_test_split(training_data, label, test_size=0.2, random_state=0)
    dtrain=xgb.DMatrix(X_train, label=y_train)
    dtest=xgb.DMatrix(X_test, label=y_test)
    param = {'max_depth': 10, 'eta': 0.05, 'silent': 1, 'objective': 'binary:logistic'}
    num_round = 4000
    param['nthread'] = 4
    #param['eval_metric'] = "auc"
    plst = param.items()
    plst += [('eval_metric', 'logloss')]
    evallist = [(dtest, 'eval'), (dtrain, 'train')]
    bst=xgb.train( plst, dtrain, num_round, evallist)

    sub_user_index, sub_trainning_date, sub_label = make_train_set(sub_start_date, sub_end_date,
                                                                   sub_test_start_date, sub_test_end_date)
    test = xgb.DMatrix(sub_trainning_date)
    y = xgb.predict(test)

    pred = sub_user_index.copy()
    y_true = sub_user_index.copy()
    pred['label'] = y
    y_true['label'] = label
    report(pred, y_true)


if __name__ == '__main__':
    xgboost_cv()
