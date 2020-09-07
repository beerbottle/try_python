#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 31 22:28:19 2018

@author: zhaikun
"""
import PlotModel as pm
import pickle
import pandas as pd

data = pd.read_csv('data_model.csv', encoding='utf-8')

f = open('lr.pkl', 'rb')
lr = pickle.load(f)
f.close()

f = open('lr_features.pkl', 'rb')
allFeatures = pickle.load(f)
f.close()

f = open('WOE_dict.pkl', 'rb')
WOE_dict = pickle.load(f)
f.close()

f = open('merge_bin_dict.pkl', 'rb')
merge_bin_dict = pickle.load(f)
f.close()

f = open('by_encoding_dict.pkl', 'rb')
br_encoding_dict = pickle.load(f)
f.close()

f = open('continous_merged_dict.pkl', 'rb')
continous_merged_dict = pickle.load(f)
f.close()


'''两种方法计算样本分数，验证可解释性'''

y_value = lr.predict_proba(data[allFeatures])[:, 1]
prob_score = pm.prob2score(y_value, odds=30)  # 概率转换的样本分数

logist_score = pm.logist2score(lr, data, odds=30)  # 逻辑回归方程表达式计算出的样本分数

prob_score == logist_score


'''创建评分卡'''
var = [i.replace('_WOE', '')
       for i in allFeatures if i.find('_WOE') > 0]  # 找到建模变量

b = lr.intercept_  # 截距

coe = lr.coef_[0]  # 系数

for i in range(len(allFeatures)):
    locals()['coe'+'_'+str(var[i])] = coe[i]

PDO = 20
odds = 30
B = PDO/np.log(2)
A = 500 - B*np.log(odds)  # 假设好坏比为30:1的时候，对应的评分为500分

s = (A-B*b)/data.shape[0]


for i in var:
    score = []
    for j in list(WOE_dict[i].keys()):
        locals()['score'+str(j)] = int(-(B * locals()
                                         ['coe'+'_'+str(i)] * WOE_dict[i][j]) + s)
        score.append(locals()['score'+str(j)])
    locals()['temp'+str(i)] = pd.DataFrame(columns=['取值', '分数'])
    locals()['temp'+str(i)]['取值'] = list(WOE_dict[i].keys())
    locals()['temp'+str(i)]['分数'] = score
    locals()['temp'+str(i)]['变量'] = i
    if i.find('_Bin') > 0:
        k = i.replace('_WOE', '').replace('_Bin', '')
        locals()['temp'+str(i)]['变量'] = k
        locals()['temp'+str(i)]['取值'] = str(continous_merged_dict[k])

temp_list = list('temp'+str(i) for i in var)
score_card = pd.DataFrame(columns=['取值', '分数', '变量'])
for i in temp_list:
    temp = locals()[i]
    score_card = score_card.append(temp, ignore_index=True)
score_card = score_card[['变量', '取值', '分数']]


'''保存评分卡文件'''
score_card.to_excel('score_card.xlsx', encoding='utf-8', index=False)
