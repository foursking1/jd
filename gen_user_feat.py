#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import time
from datetime import datetime
from datetime import timedelta
import pandas as pd
import pickle
import os
import math
import numpy as np

action_1_path = "./data/JData_Action_201602.csv"
action_2_path = "./data/JData_Action_201603.csv"
action_3_path = "./data/JData_Action_201604.csv"
comment_path = "./data/JData_Comment.csv"
product_path = "./data/JData_Product.csv"
user_path = "./data/JData_User.csv"

comment_date = ["2016-02-01", "2016-02-08", "2016-02-15", "2016-02-22", "2016-02-29", "2016-03-07", "2016-03-14",
                "2016-03-21", "2016-03-28",
                "2016-04-04", "2016-04-11", "2016-04-15"]


def convert_age(age_str):
    if age_str == u'-1':
        return 0
    elif age_str == u'15岁以下':
        return 1
    elif age_str == u'16-25岁':
        return 2
    elif age_str == u'26-35岁':
        return 3
    elif age_str == u'36-45岁':
        return 4
    elif age_str == u'46-55岁':
        return 5
    elif age_str == u'56岁以上':
        return 6
    else:
        return -1

def get_basic_user_feat():
    dump_path = './cache/basic_user.pkl'
    if os.path.exists(dump_path):
        user = pickle.load(open(dump_path))
    else:
        user = pd.read_csv(user_path, encoding='gbk')
        user['age'] = user['age'].map(convert_age)
        age_df = pd.get_dummies(user["age"], prefix="age")
        sex_df = pd.get_dummies(user["sex"], prefix="sex")
        user_lv_df = pd.get_dummies(user["user_lv_cd"], prefix="user_lv_cd")
        user = pd.concat([user['user_id'], age_df, sex_df, user_lv_df], axis=1)
        pickle.dump(user, open(dump_path, 'w'))
    return user


def get_basic_product_feat():
    dump_path = './cache/basic_product.pkl'
    if os.path.exists(dump_path):
        product = pickle.load(open(dump_path))
    else:
        product = pd.read_csv(product_path)
        attr1_df = pd.get_dummies(product["attr1"], prefix="attr1")
        attr2_df = pd.get_dummies(product["attr2"], prefix="attr2")
        attr3_df = pd.get_dummies(product["attr3"], prefix="attr3")
        product = pd.concat([product[['sku_id', 'cate', 'brand']], attr1_df, attr2_df, attr3_df], axis=1)
        pickle.dump(product, open(dump_path, 'w'))
    return product


def get_actions_1():
    action = pd.read_csv(action_1_path)
    return action

def get_actions_2():
    action2 = pd.read_csv(action_2_path)
    return action2

def get_actions_3():
    action3 = pd.read_csv(action_3_path)
    return action3


def get_actions(start_date, end_date):
    """

    :param start_date:
    :param end_date:
    :return: actions: pd.Dataframe
    """
    dump_path = './cache/all_action_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path))
    else:
        action_1 = get_actions_1()
        action_2 = get_actions_2()
        action_3 = get_actions_3()
        actions = pd.concat([action_1, action_2, action_3]) # type: pd.DataFrame
        actions = actions[(actions.time >= start_date) & (actions.time < end_date)]
        pickle.dump(actions, open(dump_path, 'w'))
    return actions


def get_action_feat(start_date, end_date):
    dump_path = './cache/action_accumulate_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path))
    else:
        actions = get_actions(start_date, end_date)
        actions = actions[['user_id', 'sku_id', 'type']]
        df = pd.get_dummies(actions['type'], prefix='%s-%s-action' % (start_date, end_date))
        actions = pd.concat([actions, df], axis=1)  # type: pd.DataFrame
        actions = actions.groupby(['user_id', 'sku_id'], as_index=False).sum()
        del actions['type']
        pickle.dump(actions, open(dump_path, 'w'))
    return actions


def get_accumulate_action_feat(start_date, end_date):
    dump_path = './cache/action_accumulate_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path))
    else:
        actions = get_actions(start_date, end_date)
        df = pd.get_dummies(actions['type'], prefix='action')
        actions = pd.concat([actions, df], axis=1) # type: pd.DataFrame
        #近期行为按时间衰减
        actions['weights'] = actions['time'].map(lambda x: datetime.strptime(end_date, '%Y-%m-%d') - datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
        #actions['weights'] = time.strptime(end_date, '%Y-%m-%d') - actions['datetime']
        actions['weights'] = actions['weights'].map(lambda x: math.exp(-x.days))
        print actions.head(10)
        actions['action_1'] = actions['action_1'] * actions['weights']
        actions['action_2'] = actions['action_2'] * actions['weights']
        actions['action_3'] = actions['action_3'] * actions['weights']
        actions['action_4'] = actions['action_4'] * actions['weights']
        actions['action_5'] = actions['action_5'] * actions['weights']
        actions['action_6'] = actions['action_6'] * actions['weights']
        del actions['model_id']
        del actions['type']
        del actions['time']
        del actions['datetime']
        del actions['weights']
        actions = actions.groupby(['user_id', 'sku_id', 'cate', 'brand'], as_index=False).sum()
        pickle.dump(actions, open(dump_path, 'w'))
    return actions


def get_comments_product_feat(start_date, end_date):
    dump_path = './cache/comments_accumulate_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        comments = pickle.load(open(dump_path))
    else:
        comments = pd.read_csv(comment_path)
        comment_date_end = end_date
        comment_date_begin = comment_date[0]
        for date in reversed(comment_date):
            if date < comment_date_end:
                comment_date_begin = date
                break
        comments = comments[(comments.dt >= comment_date_begin) & (comments.dt < comment_date_end)]
        df = pd.get_dummies(comments['comment_num'], prefix='comment_num')
        comments = pd.concat([comments, df], axis=1) # type: pd.DataFrame
        #del comments['dt']
        #del comments['comment_num']
        comments = comments[['sku_id', 'has_bad_comment', 'bad_comment_rate', 'comment_num_1', 'comment_num_2', 'comment_num_3', 'comment_num_4']]
        pickle.dump(comments, open(dump_path, 'w'))
    return comments


def get_accumulate_user_feat(start_date, end_date):
    feature = ['user_id', 'user_action_1_ratio', 'user_action_2_ratio', 'user_action_3_ratio',
               'user_action_5_ratio', 'user_action_6_ratio']
    dump_path = './cache/user_feat_accumulate_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path))
    else:
        actions = get_actions(start_date, end_date)
        df = pd.get_dummies(actions['type'], prefix='action')
        actions = pd.concat([actions['user_id'], df], axis=1)
        actions = actions.groupby(['user_id'], as_index=False).sum()
        actions['user_action_1_ratio'] = actions['action_4'] / actions['action_1']
        actions['user_action_2_ratio'] = actions['action_4'] / actions['action_2']
        actions['user_action_3_ratio'] = actions['action_4'] / actions['action_3']
        actions['user_action_5_ratio'] = actions['action_4'] / actions['action_5']
        actions['user_action_6_ratio'] = actions['action_4'] / actions['action_6']
        actions = actions[feature]
        pickle.dump(actions, open(dump_path, 'w'))
    return actions


def get_accumulate_product_feat(start_date, end_date):
    feature = ['sku_id', 'product_action_1_ratio', 'product_action_2_ratio', 'product_action_3_ratio',
               'product_action_5_ratio', 'product_action_6_ratio']
    dump_path = './cache/product_feat_accumulate_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path))
    else:
        actions = get_actions(start_date, end_date)
        df = pd.get_dummies(actions['type'], prefix='action')
        actions = pd.concat([actions['sku_id'], df], axis=1)
        actions = actions.groupby(['sku_id'], as_index=False).sum()
        actions['product_action_1_ratio'] = actions['action_4'] / actions['action_1']
        actions['product_action_2_ratio'] = actions['action_4'] / actions['action_2']
        actions['product_action_3_ratio'] = actions['action_4'] / actions['action_3']
        actions['product_action_5_ratio'] = actions['action_4'] / actions['action_5']
        actions['product_action_6_ratio'] = actions['action_4'] / actions['action_6']
        actions = actions[feature]
        pickle.dump(actions, open(dump_path, 'w'))
    return actions


def get_labels(start_date, end_date):
    dump_path = './cache/labels_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path))
    else:
        actions = get_actions(start_date, end_date)
        actions = actions[actions['type'] == 4]
        actions = actions.groupby(['user_id', 'sku_id'], as_index=False).sum()
        actions['label'] = 1
        actions = actions[['user_id', 'sku_id', 'label']]
        pickle.dump(actions, open(dump_path, 'w'))
    return actions


def make_test_set(train_start_date, train_end_date):
    dump_path = './cache/test_set_%s_%s.pkl' % (train_start_date, train_end_date)
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path))
    else:
        start_days = "2016-02-01"
        user = get_basic_user_feat()
        product = get_basic_product_feat()
        user_acc = get_accumulate_user_feat(start_days, train_end_date)
        product_acc = get_accumulate_product_feat(start_days, train_end_date)
        comment_acc = get_comments_product_feat(train_start_date, train_end_date)
        #labels = get_labels(test_start_date, test_end_date)

        # generate 时间窗口
        # actions = get_accumulate_action_feat(train_start_date, train_end_date)
        actions = None
        for i in (1, 2, 3, 5, 7, 10, 15, 21, 30):
            start_days = datetime.strptime(train_end_date, '%Y-%m-%d') - timedelta(days=i)
            start_days = start_days.strftime('%Y-%m-%d')
            if actions is None:
                actions = get_action_feat(start_days, train_end_date)
            else:
                actions = pd.merge(actions, get_action_feat(start_days, train_end_date), how='left',
                                   on=['user_id', 'sku_id'])

        actions = pd.merge(actions, user, how='left', on='user_id')
        actions = pd.merge(actions, user_acc, how='left', on='user_id')
        actions = pd.merge(actions, product, how='left', on='sku_id')
        actions = pd.merge(actions, product_acc, how='left', on='sku_id')
        actions = pd.merge(actions, comment_acc, how='left', on='sku_id')
        #actions = pd.merge(actions, labels, how='left', on=['user_id', 'sku_id'])
        actions = actions.fillna(0)
        actions = actions[actions['cate'] == 8]

    users = actions[['user_id', 'sku_id']].copy()
    del actions['user_id']
    del actions['sku_id']
    return users, actions

def make_train_set(train_start_date, train_end_date, test_start_date, test_end_date, days=30):
    dump_path = './cache/train_set_%s_%s_%s_%s.pkl' % (train_start_date, train_end_date, test_start_date, test_end_date)
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path))
    else:
        start_days = "2016-02-01"
        user = get_basic_user_feat()
        product = get_basic_product_feat()
        user_acc = get_accumulate_user_feat(start_days, train_end_date)
        product_acc = get_accumulate_product_feat(start_days, train_end_date)
        comment_acc = get_comments_product_feat(train_start_date, train_end_date)
        labels = get_labels(test_start_date, test_end_date)

        # generate 时间窗口
        # actions = get_accumulate_action_feat(train_start_date, train_end_date)
        actions = None
        for i in (1, 2, 3, 5, 7, 10, 15, 21, 30):
            start_days = datetime.strptime(train_end_date, '%Y-%m-%d') - timedelta(days=i)
            start_days = start_days.strftime('%Y-%m-%d')
            if actions is None:
                actions = get_action_feat(start_days, train_end_date)
            else:
                actions = pd.merge(actions, get_action_feat(start_days, train_end_date), how='left',
                                   on=['user_id', 'sku_id'])

        actions = pd.merge(actions, user, how='left', on='user_id')
        actions = pd.merge(actions, user_acc, how='left', on='user_id')
        actions = pd.merge(actions, product, how='left', on='sku_id')
        actions = pd.merge(actions, product_acc, how='left', on='sku_id')
        actions = pd.merge(actions, comment_acc, how='left', on='sku_id')
        actions = pd.merge(actions, labels, how='left', on=['user_id', 'sku_id'])
        actions = actions.fillna(0)

    users = actions[['user_id', 'sku_id']].copy()
    labels = actions['label'].copy()
    del actions['user_id']
    del actions['sku_id']
    del actions['label']

    return users, actions, labels


def report(pred, label):

    actions = label
    result = pred

    # 所有用户商品对
    all_user_item_pair = actions['user_id'].map(str) + '-' + actions['sku_id'].map(str)
    all_user_item_pair = np.array(all_user_item_pair)
    # 所有购买用户
    all_user_set = actions['user_id'].unique()

    # 所有品类中预测购买的用户
    all_user_test_set = result['user_id'].unique()
    all_user_test_item_pair = result['user_id'].map(str) + '-' + result['sku_id'].map(str)
    all_user_test_item_pair = np.array(all_user_test_item_pair)

    # 计算所有用户购买评价指标
    pos, neg = 0,0
    for user_id in all_user_test_set:
        if user_id in all_user_set:
            pos += 1
        else:
            neg += 1
    all_user_acc = 1.0 * pos / ( pos + neg)
    all_user_recall = 1.0 * pos / len(all_user_set)
    print '所有用户中预测购买用户的准确率为 ' + str(all_user_acc)
    print '所有用户中预测购买用户的召回率' + str(all_user_recall)

    pos, neg = 0, 0
    for user_item_pair in all_user_test_item_pair:
        if user_item_pair in all_user_item_pair:
            pos += 1
        else:
            neg += 1
    all_item_acc = 1.0 * pos / ( pos + neg)
    all_item_recall = 1.0 * pos / len(all_user_item_pair)
    print '所有用户中预测购买商品的准确率为 ' + str(all_item_acc)
    print '所有用户中预测购买商品的召回率' + str(all_item_recall)
    F11 = 6.0 * all_user_recall * all_user_acc / (5.0 * all_user_recall + all_user_acc)
    F12 = 5.0 * all_item_acc * all_item_recall / (2.0 * all_item_recall + 3 * all_item_acc)
    score = 0.4 * F11 + 0.6 * F12
    print 'F11=' + str(F11)
    print 'F12=' + str(F12)
    print 'score=' + str(score)

if __name__ == '__main__':
    train_start_date = '2016-02-01'
    train_end_date = '2016-03-01'
    test_start_date = '2016-03-01'
    test_end_date = '2016-03-05'
    user, action, label = make_train_set(train_start_date, train_end_date, test_start_date, test_end_date)
    print user.head(10)
    print action.head(10)




