#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import pandas as pd
import pickle

action_1_path = "./data/JData_Action_201602.csv"
action_2_path = "./data/JData_Action_201603.csv"
action_2_extra_path = "./data/JData_Action_201603_extra.csv"
action_3_path = "./data/JData_Action_201604.csv"
comment_path = "./data/JData_Comment(修正版).csv"
product_path = "./data/JData_Product.csv"
user_path = "./data/JData_User.csv"

def convert_age(age_str):
    if age_str == 'age_-1':
        return 0
    elif age_str == 'age_15岁以下':
        return 1
    elif age_str == 'age_16-25岁':
        return 2
    elif age_str == 'age_26-35岁':
        return 3
    elif age_str == 'age_36-45岁':
        return 4
    elif age_str == 'age_46-55岁':
        return 5
    elif age_str == 'age_56岁以上':
        return 6
    else:
        return -1


def get_basic_user_feat():
    user = pd.read_csv(user_path, encoding='gbk')
    user['age'] = user['age'].map(convert_age)
    age_df = pd.get_dummies(user["age"], prefix="age")
    sex_df = pd.get_dummies(user["sex"], prefix="sex")
    user_lv_df = pd.get_dummies(user["user_lv_cd"], prefix="user_lv_cd")
    user = pd.concat([user, age_df, sex_df, user_lv_df], axis=1)
    pickle.dump(user, open('./cache/basic_user.pkl', 'w'))
    return user


def get_basic_product_feat():
    product = pd.read_csv(product_path)
    attr1_df = pd.get_dummies(product["attr1"], prefix="attr1")
    attr2_df = pd.get_dummies(product["attr2"], prefix="attr2")
    attr3_df = pd.get_dummies(product["attr3"], prefix="attr3")
    product = pd.concat([product, attr1_df, attr2_df, attr3_df], axis=1)
    pickle.dump(product, open('./cache/basic_product.pkl', 'w'))
    return product

def get_actions():
    pass

def make_train_set(train_start_date, train_end_date, test_start_date, test_end_date, use_dump = True):

    if use_dump == True:
        user = pickle.load(open('./cache/basic_user.pkl'))
        product = pickle.load(open('./cache/basic_product.pkl'))
    else:
        user = get_basic_product_feat()
        product = get_basic_product_feat()









