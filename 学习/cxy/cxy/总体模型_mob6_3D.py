# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 11:25:07 2019

@author: aigo
"""
#####################################################################################################################################
import os
import platform
import pandas as pd
if platform.system()=='Windows':
    home=r'G:\\'
    software='python'
    code_dir=os.path.join(home,'RiskModelTools' ) 
else:
    home= '/home/hcfk/suntao_file'
    software='python'
    code_dir=os.path.join('/home/hcfk','RiskModelTools' )

model_dir=os.path.join(home,'data','线上业务数据','线上业务数据_model')
import sys
#__file__=r'/home/hcfk/suntao_file/业务分析/风控模型/总体模型_mob6_3D/总体模型_mob6_3D.py'
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(code_dir)
import  config_producer
from publictools.derived.merge_ds     import  merge_ds
from publictools.derived.replace_na   import  replace_na
import MissingValue 
import VarNumeric  
#from cleankit.DataClean        import DataClean    as dc
#from toolkit.ParallelRun       import ParallelRun  as prun
################################################################
cutoff_date=pd.to_datetime('2018-08-01')
repay_=pd.read_pickle(os.path.join(model_dir,'repay_ds_with_flag.pkl')).reset_index(drop=True)
repay_['intotime']=pd.to_datetime(repay_['intotime'])
repay_ =repay_.loc[(repay_[config_producer.configs['model']['y_name']].isin([1,0,-1]))&(repay_['intotime']>=cutoff_date) ,
                   [
                   'intoid',
                   'flag1_minage2_overduemonth2', 'flag2_minage2_overduemonths2',
                   'flag3_minage2_overduemonths2', 'flag4_minage2_overduemonths2',
                   'flag5_minage2_overduemonths2', 'flag6_minage2_overduedays3',
                   'flag7_minage2_overduedays3', 'flag1_minage3_overduemonth2',
                   'flag2_minage3_overduemonths2', 'flag3_minage3_overduemonths2',
                   'flag4_minage3_overduemonths2', 'flag5_minage3_overduemonths2',
                   'flag6_minage3_overduedays3', 'flag7_minage3_overduedays3',
                   'flag1_minage4_overduemonth2', 'flag2_minage4_overduemonths2',
                   'flag3_minage4_overduemonths2', 'flag4_minage4_overduemonths2',
                   'flag5_minage4_overduemonths2', 'flag6_minage4_overduedays3',
                   'flag7_minage4_overduedays3', 'flag1_minage5_overduemonth2',
                   'flag2_minage5_overduemonths2', 'flag3_minage5_overduemonths2',
                   'flag4_minage5_overduemonths2', 'flag5_minage5_overduemonths2',
                   'flag6_minage5_overduedays3', 'flag7_minage5_overduedays3',
                   'flag1_minage6_overduemonth2', 'flag2_minage6_overduemonths2',
                   'flag3_minage6_overduemonths2', 'flag4_minage6_overduemonths2',
                   'flag5_minage6_overduemonths2', 'flag6_minage6_overduedays3',
                   'flag7_minage6_overduedays3',                  
               ]]
################################################################
application_ds =pd.read_pickle(os.path.join(model_dir,'application_ds.pkl'))	
application_ds =application_ds.loc[application_ds['intoid'].isin(repay_['intoid'])].drop_duplicates(subset=['intoid'],keep='last').reset_index(drop=True)

kx_ds          =pd.read_pickle(os.path.join(model_dir,'kx_ds.pkl'))	
kx_ds          =kx_ds.loc[kx_ds['intoid'].isin(repay_['intoid'])].drop_duplicates(subset=['intoid'],keep='last').reset_index(drop=True)

td_ds          =pd.read_pickle(os.path.join(model_dir,'td_ds.pkl'))	
td_ds          =td_ds.loc[td_ds['intoid'].isin(repay_['intoid'])].drop_duplicates(subset=['intoid'],keep='last').reset_index(drop=True)

br_ds          =pd.read_pickle(os.path.join(model_dir,'br_ds.pkl'))
br_ds          =br_ds.loc[br_ds['intoid'].isin(repay_['intoid'])].drop_duplicates(subset=['intoid'],keep='last').reset_index(drop=True)

br2_ds         =pd.read_pickle(os.path.join(model_dir,'br2_ds.pkl'))	
br2_ds         =br2_ds.loc[br2_ds['intoid'].isin(repay_['intoid'])].drop_duplicates(subset=['intoid'],keep='last').reset_index(drop=True)

#ks_score 如果缺失
tx_ds          =pd.read_pickle(os.path.join(model_dir,'tx_ds.pkl'))	
tx_ds          =tx_ds.loc[tx_ds['intoid'].isin(repay_['intoid'])].drop_duplicates(subset=['intoid'],keep='last').reset_index(drop=True)
#tx_ds.loc[(tx_ds['tx_risk_score'].isnull())|(tx_ds['tx_risk_score']==-99)].shape

xy_ds          =pd.read_pickle(os.path.join(model_dir,'xy_ds.pkl'))	
xy_ds          =xy_ds.loc[xy_ds['intoid'].isin(repay_['intoid'])].drop_duplicates(subset=['intoid'],keep='last').reset_index(drop=True)

#缺失比较高
rh_ds          =pd.read_pickle(os.path.join(model_dir,'rh_ds.pkl'))	
rh_ds          =rh_ds.loc[rh_ds['intoid'].isin(repay_['intoid'])].drop_duplicates(subset=['intoid'],keep='last').reset_index(drop=True)

kl_ds          =pd.read_pickle(os.path.join(model_dir,'kl_ds.pkl'))	
kl_ds          =kl_ds.loc[kl_ds['intoid'].isin(repay_['intoid'])].drop_duplicates(subset=['intoid'],keep='last').reset_index(drop=True)

br3_ds         =pd.read_pickle(os.path.join(model_dir,'br3_ds.pkl'))	
br3_ds         =br3_ds.loc[br3_ds['intoid'].isin(repay_['intoid'])].drop_duplicates(subset=['intoid'],keep='last').reset_index(drop=True)

creditcard_ds  =pd.read_pickle(os.path.join(model_dir,'creditcard_ds.pkl'))	
creditcard_ds  =creditcard_ds.loc[creditcard_ds['intoid'].isin(repay_['intoid'])].drop_duplicates(subset=['intoid'],keep='last').reset_index(drop=True)

nine1_ds       =pd.read_pickle(os.path.join(model_dir,'nine1_ds.pkl'))	
nine1_ds       =nine1_ds.loc[nine1_ds['intoid'].isin(repay_['intoid'])].drop_duplicates(subset=['intoid'],keep='last').reset_index(drop=True)
#业务停止
qh_ds          =pd.read_pickle(os.path.join(model_dir,'qh_ds.pkl'))	
qh_ds          =qh_ds.loc[qh_ds['intoid'].isin(repay_['intoid'])].drop_duplicates(subset=['intoid'],keep='last').reset_index(drop=True)
tj_ds          =pd.read_pickle(os.path.join(model_dir,'tj_ds.pkl'))	
tj_ds          =tj_ds.loc[tj_ds['intoid'].isin(repay_['intoid'])].drop_duplicates(subset=['intoid'],keep='last').reset_index(drop=True)
#####################################################################################
final_ds= merge_ds(
        ds_list=[application_ds,kx_ds,td_ds,br_ds,br2_ds,tx_ds,xy_ds], 
        on_var=['intoid'], 
        how='outer')
final_ds2=replace_na(data=final_ds,varlist=final_ds.columns,num_na=[-99,-98],char_na=['-99'])
#####################################################################################
final_ds3=pd.merge(final_ds2,repay_,on='intoid',how='inner')
#填补缺失值
final_ds4,missing_meta=MissingValue.fitmissing(indata=final_ds3,imputemethod='IMPUTE')
#####################################################################################
var_dict={
'education':{
        'value':['2003', '2004', '2005', '2006', '2002', '2007', '2001'],
        'onehot':True,
        'convert':True,
        'badrate_map':True,
        } ,     
'unittype':{
        'value':['2304', '2305', '2308', '2301'],
        'onehot':True,
        'convert':True,
        'badrate_map':True,
        }, 
'into_device':{ 
        'value':['IOS', '安卓'],
        'onehot':True,
        'convert':False,
        'badrate_map':True,
        }, 
'into_source':{
        'value':['APP-IOS', 'APP-安卓'],
        'onehot':True,
        'convert':False,
        'badrate_map':True,
        },         
'maritalstatus':{
        'value':['2201', '2202', '2203', '2204', '2205'],
        'onehot':True,
        'convert':True,
        'badrate_map':True,
        },          
'housestatus':{
        'value':['2103', '2107', '2104', '2105', '2101' ],
        'onehot':True,
        'convert':True,
        'badrate_map':True,
        }, 
'term':{
        'value':['36', '12', '24'],
        'onehot':True,
        'convert':True,
        'badrate_map':True,
        }, 
'uses':{
        'value':['2411', '2412',  '2413','2414', '2415','2416','2417','2418','2419', '2420','2421','2422', '2423',  ],
        'onehot':True,
        'convert':True,
        'badrate_map':True,
        }, 
'kx_risklevel':{
        'value':['0','1', '2', '3'],
        'onehot':True,
        'convert':True,
        'badrate_map':True,
        }, 
}        

final_ds5,badrate_meta,onehot_ds=VarNumeric.char2numeric(
            indata=final_ds4,
            var_dict=var_dict,
            yname=config_producer.configs['model']['y_name'],
            idvar='intoid',
            
)
missing_meta=missing_meta.append(
        pd.merge(missing_meta.drop('VAR',axis=1),onehot_ds,on='originalVAR',how='inner'),
        ignore_index=True)
meta_total=pd.merge(missing_meta,badrate_meta[['VAR','map_rate']],on='VAR',how='outer')
meta_total['map_rate']=meta_total['map_rate'].fillna('')
#####################################################################################
#拆分
final_=final_ds5.sort_values(['intotime']).reset_index(drop=True)
split_num=int(len(final_)*0.8)
traindata=final_.head(split_num).reset_index(drop=True)
testdata=final_.loc[split_num:].reset_index(drop=True)

traindata.to_pickle( os.path.join(config_producer.root,'traindata.pkl'))
testdata.to_pickle(  os.path.join(config_producer.root,'testdata.pkl'))
meta_total.to_pickle(os.path.join(config_producer.root,'meta_total.pkl'))
###############################################################################################################
os.system(r'{} {} {}'.format(software,os.path.join(code_dir,'files','run_machinelearning.py'), config_producer.root ))
###############################################################################################################
#nohup  python  /home/hcfk/suntao_file/业务分析/风控模型/线上原有字段/线上原有字段验证模型.py > /home/hcfk/suntao_file/业务分析/风控模型/线上原有字段/nohup_myout.file 2>&1 &
#python /home/hcfk/suntao_file/业务分析/风控模型/总体模型_mob6_3D/总体模型_mob6_3D.py
#python G:\业务分析\风控模型\线上原有字段\线上原有字段验证模型.py
#cd /home/hcfk/suntao_file/业务分析/风控模型/线上原有字段/
#cd /home/hcfk/modelTools
#vim /home/hcfk/modelTools/graphkit/Graphing.py
