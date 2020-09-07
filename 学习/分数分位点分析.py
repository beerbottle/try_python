# -*- coding: utf-8 -*-
"""
Created on Tue Jan  6 14:23:49 2015

@author: SYSTEM
"""

import libkit
import numpy as np
import pandas as pd
import time         
import gc
    


#################################################
#####参数设置
#################################################

#历史分数观测时间段
detaildatelist=['2019-12-22','2020-02-01']
#额度观测时间段
amountdetaildatelist=['2019-12-22','2020-02-01']
api_list=['wacai','xianjinbaika','kaniu','360','bairong','jiedianqian','daikuandaohang','51','haodai','xinyongka','boluodai','xinyongguanjia','geinihua']

detailutmcode='ourself'


quantilelist=[0,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.11,0.12,0.13,0.14,0.15,0.16,0.17,0.18,0.19,0.2,0.21,0.22,0.23,0.24,0.25,0.26,0.27,0.28,0.29,0.3,0.31,0.32,0.33,0.34,0.35,0.36,0.37,0.38,0.39,0.4,0.41,0.42,0.43,0.44,0.45,0.46,0.47,0.48,0.49,0.5,0.51,0.52,0.53,0.54,0.55,0.56,0.57,0.58,0.59,0.6,0.61,0.62,0.63,0.64,0.65,0.66,0.67,0.68,0.69,0.7,0.71,0.72,0.73,0.74,0.75,0.76,0.77,0.78,0.79,0.8,0.81,0.82,0.83,0.84,0.85,0.86,0.87,0.88,0.89,0.9,0.91,0.92,0.93,0.94,0.95,0.96,0.97,0.98,0.99,1
]


#################################################
#####数据获取及加工
#################################################
starttime=time.time()

da=libkit.data.DBlink('da')
##first input
info='*' 
info=['da_db_id','intoId','applytime','intoTime','product','utmcode', 'intochannelname', 'vendor','attribution','mobile','whiteSign','duration','earliestCallTime',
      'h1_month_call_num','h1_month_bill_amount','tele_avg_amt_6m','IN_INFOSTR','income','id_match','name_match','birthday','code_phone']
tablename='jsd_execute_input_application'
searchdate={'intoTime':['2020-01-01']}
searchcondition={}
#searchcondition={'utmcode':['wacai','xianjinbaika','kaniu','360','bairong','jiedianqian','daikuandaohang','51','haodai','xinyongka','boluodai','xinyongguanjia','ourself']}
sqlsentence=libkit.data.sqlQuery(info,tablename,searchdate=searchdate,searchcondition=searchcondition)
zeroIntpc=libkit.data.getBySql(sqlsentence,conn=da)

'''
##first input three
info='*'
info=['da_db_id','xykwy_bankKeyInput','xykwy_minBillAgoMonths','xykwy_continuityBillNum','xykwy_newestCreditCny']
tablename='jsd_execute_input_three'
searchcondition={'da_db_id':zeroIntpc.da_db_id.tolist()}
sqlsentence=libkit.data.sqlQuery(info,tablename,searchcondition=searchcondition)
zeroIntpcThree=libkit.data.getBySql(sqlsentence,conn=da)
'''
##first output
info='*'
info=['da_db_id','output_dclnreason as pre_output_dclnreason','output_otherreason as pre_output_otherreason']
tablename='jsd_execute_output_java'
searchcondition={'da_db_id':zeroIntpc.da_db_id.tolist()}
sqlsentence=libkit.data.sqlQuery(info,tablename,searchcondition=searchcondition)
zeroIntpcOutput=libkit.data.getBySql(sqlsentence,conn=da)
##last input
info='*'
info=['da_db_id','intoId','term as input_term','reg_channelid','reg_channelname']
tablename='jsd_input_app_online'
searchcondition={'intoid':zeroIntpc.intoId.tolist()}
sqlsentence=libkit.data.sqlQuery(info,tablename,searchcondition=searchcondition)
zeroFinal=libkit.data.getBySql(sqlsentence,conn=da)
'''
##last phone
info='*'
info=['da_db_id','tele_top10_list_cnt_3m','phone_list_cnt']
tablename='jsd_input_app_phone'
searchcondition={'da_db_id':zeroFinal.da_db_id.tolist()}
sqlsentence=libkit.data.sqlQuery(info,tablename,searchcondition=searchcondition)
zeroFinalPhone=libkit.data.getBySql(sqlsentence,conn=da)
'''
##last output
info='*'
info=['da_db_id','output_amount','output_dclnreason as final_output_dclnreason','output_cgrade','output_otherreason as final_output_otherreason','Score_Test','Score_Test_2','hint_8','output_term','hint_19']
tablename='jsd_output_java'
searchcondition={'da_db_id':zeroFinal.da_db_id.tolist()}
sqlsentence=libkit.data.sqlQuery(info,tablename,searchcondition=searchcondition)
zeroFinalOutput=libkit.data.getBySql(sqlsentence,conn=da)
da.close()

eagle=libkit.data.DBlink('eagle')
##riskcon input,between first and last
info='*'
info=['id','into_id','into_bus_id','into_time','into_state','amount','term as into_term']
tablename='eagle_jsd_intopieces'
searchcondition={'into_id':zeroIntpc.intoId.tolist()}
sqlsentence=libkit.data.sqlQuery(info,tablename,searchcondition=searchcondition)
zeroRiskcon=libkit.data.getBySql(sqlsentence,conn=eagle)
##riskcon output,between first and last
info='*'
info=['ip_id','review_result','revive_time','amount','term','interest_rate','rate','review_state','result_reason', 'refuse_main_reason','refuse_child_reason','refuse_type','term as out_term']
tablename='eagle_jsd_riskcon'
searchcondition={'ip_id':zeroRiskcon.id.tolist()}
sqlsentence=libkit.data.sqlQuery(info,tablename,searchcondition=searchcondition)
zeroRiskconOutput=libkit.data.getBySql(sqlsentence,conn=eagle)
eagle.close()
endtime=time.time()
print('the procedure for the database has been run for %s second!'%(endtime-starttime))

##some concat of the dataframe
tempFirst=pd.merge(zeroIntpc,zeroIntpcOutput,how='left',on='da_db_id')
#tempLast=pd.merge(zeroFinal,zeroFinalPhone,how='left',on='da_db_id')
tempLast=pd.merge(zeroFinal,zeroFinalOutput,how='left',on='da_db_id')
zeroRiskcon.rename(columns={'id':'ip_id'},inplace=True)
tempRisk=pd.merge(zeroRiskcon,zeroRiskconOutput,how='left',on='ip_id')
tempRisk.rename(columns={'into_id':'intoId','amount_x':'askamount','amount_y':'agreeamount'},inplace=True)
tempUnion=pd.merge(tempFirst,tempRisk,how='left',on='intoId')
tempUnion=pd.merge(tempUnion,tempLast,how='left',on='intoId')
tempUnion=tempUnion[~tempUnion.duplicated('intoId','last')].copy()
needTitle=['intoId','da_db_id_x','da_db_id_y','product','into_bus_id','applytime','intoTime','utmcode','intochannelname', 'vendor','attribution','mobile','into_state','revive_time','review_result',
           'result_reason','review_state','refuse_main_reason','refuse_child_reason','refuse_type','pre_output_dclnreason','pre_output_otherreason','final_output_dclnreason','final_output_otherreason','Score_Test',
           'hint_8','hint_19','Score_Test_2','output_cgrade','rate','askamount','output_amount','agreeamount','input_term','output_term','duration','earliestCallTime','h1_month_call_num',
           'h1_month_bill_amount','IN_INFOSTR','into_term','out_term','income','id_match','name_match',
           'birthday','code_phone','reg_channelid','reg_channelname']
zeroUnion=tempUnion[needTitle].copy()
#in case of some abnormal event,like self consistent check
zeroUnion.loc[(zeroUnion.review_result.isnull())&(zeroUnion.review_state.notnull())]=None
zeroUnion.loc[(zeroUnion.review_result.isnull())&(zeroUnion.refuse_main_reason.notnull())]=None
zeroUnion.loc[(zeroUnion.review_result=='通过')&(zeroUnion.refuse_main_reason.notnull())]=None


#zeroUnion['age']=zeroUnion.apply(lambda x:getAge(x.birthday,x.intoTime),axis=1)
zeroUnion['utmcodeNew']=zeroUnion['utmcode'].map(lambda x:x if x in api_list else 'ourself')
zeroUnion['utmcodePlus']=zeroUnion.apply(lambda x:x.utmcodeNew if x['product']=='1001' else '%s-公积金'%(x.utmcodeNew) if x['product']=='1002' else '%s-社保'%(x.utmcodeNew) if x['product']=='1004' else '%s-%s'%(x.utmcodeNew,x['product']) ,axis=1)



#del tempUnion,tempFirst,tempLast,tempRisk
#del zeroIntpc,zeroIntpcOutput,zeroFinal,zeroFinalOutput,zeroRiskcon,zeroRiskconOutput
#gc.collect()

zeroUnion['hint_19']=zeroUnion['hint_19'].astype("float")

zeroUnion['Score_Test_2']=zeroUnion['Score_Test_2'].astype("float")

#api分数分位点
tempframe=zeroUnion[(zeroUnion.intoTime.between(*detaildatelist))&(zeroUnion.hint_19.isna()==False)&(zeroUnion['product'].isin(['1001','1002','1004']))].copy()

utmcode_list=tempframe['utmcodePlus'].dropna().unique().tolist()

result_api=pd.DataFrame(index=quantilelist)

for i in utmcode_list:
    result_api[i] = tempframe[tempframe['utmcodePlus']==i].hint_19.quantile(quantilelist)

#自有渠道分数分位点
tempframe=zeroUnion[(zeroUnion.intoTime.between(*detaildatelist))&(zeroUnion.Score_Test_2.isna()==False)&(zeroUnion['product'].isin(['1001','1002','1004'])&(zeroUnion.Score_Test_2!=0))].copy()

utmcode_list=tempframe['intochannelname'].dropna().unique().tolist()

result_nonapi=pd.DataFrame(index=quantilelist)

for i in utmcode_list:
    result_nonapi[i] = tempframe[tempframe['intochannelname']==i].Score_Test_2.quantile(quantilelist)

result_nonapi['Total'] = tempframe.Score_Test_2.quantile(quantilelist)



























