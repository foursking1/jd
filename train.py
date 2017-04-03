__author__ = 'foursking'
from gen_user_feat import make_train_set


def logistic_regression():
    train_start_date = '2016-02-01'
    train_end_date = '2016-03-01'
    test_start_date = '2016-03-01'
    test_end_date = '2016-03-05'
    user_index, training_data, label = make_train_set(train_start_date, train_end_date, test_start_date, test_end_date)



if __name__ == '__main__':
    logistic_regression()