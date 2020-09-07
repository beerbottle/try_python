import libkit
import sqlalchemy
import numpy as np
import pandas as pd
import time 
from datetime import datetime
from dateutil.relativedelta import relativedelta
import gc
#################################################
#####数据获取及加工
#################################################
#参数设置

api_list=['wacai','xianjinbaika','kaniu','360','bairong','jiedianqian','daikuandaohang','51','haodai','xinyongka','boluodai','xinyongguanjia','geinihua']

loan_date = ['2019-11-01','2020-01-01']


###################################################################################
#11、12月放款
###################################################################################




starttime=time.time()



db_info = {'user':'gedai_query',
           'password':'Dj83_Q2e',
           'host':'10.100.22.31',
           'database':'gedaifx'}
engine=sqlalchemy.create_engine('mysql+pymysql://%(user)s:%(password)s@%(host)s:3307/%(database)s?charset=utf8' % db_info,encoding='utf-8')

'''
db_info1 = {'user':'fk_dev_sel',
           'password':"qazwsx!@#'",
           'host':'10.100.22.92',
           'database':'da'}
engine1=sqlalchemy.create_engine('mysql+pymysql://%(user)s:%(password)s@%(host)s:3306/%(database)s?charset=utf8' % db_info1,encoding='utf-8')
'''


#当前最新数据
sql0='''
select intpc_id, reg_loan_dt, 	repay_day,	contr_amt, cont_prin_int, 	loan_amt,	repay_mth,	repay_beg_dt,	account_age,	cont_overdue_pird,	overdue_pird_num,	remain_prin_int_amt,	clear_flag,	clear_dt,	data_dt from agg_speed_withhold_aggregation 
		where 
			  data_dt= DATE_SUB(date(now()), INTERVAL 1 DAY) and reg_loan_dt BETWEEN '{0}' and '{1}'
            
'''.format(loan_date[0],loan_date[1])

#mob1_1
sql_mob1_1='''
select intpc_id, reg_loan_dt, 	repay_day,	contr_amt, cont_prin_int, 	loan_amt,	repay_mth,	repay_beg_dt,	account_age,	cont_overdue_pird,	overdue_pird_num,	remain_prin_int_amt,	clear_flag,	clear_dt,	data_dt from agg_speed_withhold_aggregation 
		where account_age=1 
			  and data_dt= repay_beg_dt
			  and YEAR(data_dt) >= 2019  and reg_loan_dt BETWEEN '{0}' and '{1}'
            
'''.format(loan_date[0],loan_date[1])

#mob1_3
sql_mob1_3='''
select intpc_id, reg_loan_dt, 	repay_day,	contr_amt, cont_prin_int, 	loan_amt,	repay_mth,	repay_beg_dt,	account_age,	cont_overdue_pird,	overdue_pird_num,	remain_prin_int_amt,	clear_flag,	clear_dt,	data_dt from agg_speed_withhold_aggregation 
		where account_age=1 
			  and data_dt= DATE_ADD(repay_beg_dt, INTERVAL 3 DAY)
			  and YEAR(data_dt) >= 2019  and reg_loan_dt BETWEEN '{0}' and '{1}'
            
'''.format(loan_date[0],loan_date[1])

#mob1_7
sql_mob1_7='''
select intpc_id, reg_loan_dt, 	repay_day,	contr_amt, cont_prin_int, 	loan_amt,	repay_mth,	repay_beg_dt,	account_age,	cont_overdue_pird,	overdue_pird_num,	remain_prin_int_amt,	clear_flag,	clear_dt,	data_dt from agg_speed_withhold_aggregation 
		where account_age=1 
			  and data_dt= DATE_ADD(repay_beg_dt, INTERVAL 7 DAY)
			  and YEAR(data_dt) >= 2019  and reg_loan_dt BETWEEN '{0}' and '{1}'
            
'''.format(loan_date[0],loan_date[1])



#mob1_15
sql_mob1_15='''
select intpc_id, reg_loan_dt, 	repay_day,	contr_amt, cont_prin_int, 	loan_amt,	repay_mth,	repay_beg_dt,	account_age,	cont_overdue_pird,	overdue_pird_num,	remain_prin_int_amt,	clear_flag,	clear_dt,	data_dt from agg_speed_withhold_aggregation 
		where account_age=1 
			  and data_dt= DATE_ADD(repay_beg_dt, INTERVAL 15 DAY)
			  and YEAR(data_dt) >= 2019  and reg_loan_dt BETWEEN '{0}' and '{1}'
            
'''.format(loan_date[0],loan_date[1])


#mob1_30
sql_mob1_30='''
select intpc_id, reg_loan_dt, 	repay_day,	contr_amt, cont_prin_int, 	loan_amt,	repay_mth,	repay_beg_dt,	account_age,	cont_overdue_pird,	overdue_pird_num,	remain_prin_int_amt,	clear_flag,	clear_dt,	data_dt from agg_speed_withhold_aggregation 
		where data_dt= DATE_ADD(repay_beg_dt, INTERVAL 30 DAY)
			  and YEAR(data_dt) >= 2019  and reg_loan_dt BETWEEN '{0}' and '{1}'
            
'''.format(loan_date[0],loan_date[1])


df_now=pd.read_sql(sql0,engine)
df_mob1_1=pd.read_sql(sql_mob1_1,engine)
df_mob1_3=pd.read_sql(sql_mob1_3,engine)
df_mob1_7=pd.read_sql(sql_mob1_7,engine)

df_mob1_15=pd.read_sql(sql_mob1_15,engine)

df_mob1_30=pd.read_sql(sql_mob1_30,engine)



endtime1=time.time()
print('the procedure for the database has been run for %s second!'%(endtime1-starttime))

#进件数据

da=libkit.data.DBlink('da')
info='*' 
info=['intoId','intoTime','utmcode','intochannelname','reg_channelid','product']
tablename='jsd_execute_input_application'
searchdate={'intoTime':['2019-01-01']}
searchcondition={'intoId':df_now.intpc_id.tolist()}
#searchcondition={'utmcode':['wacai','xianjinbaika','kaniu','360','bairong','jiedianqian','daikuandaohang','51','haodai','xinyongka','boluodai','xinyongguanjia','ourself']}
sqlsentence=libkit.data.sqlQuery(info,tablename,searchdate=searchdate,searchcondition=searchcondition)
df_intoid=libkit.data.getBySql(sqlsentence,conn=da)
da.close()


engine.dispose()
#engine1.dispose()

endtime2=time.time()
print('the procedure for the 2nd database has been run for %s second!'%(endtime2-endtime1))


df_mob1_1['df_mob1_1_overdue']=df_mob1_1[['overdue_pird_num']].apply(lambda x: 1 if x['overdue_pird_num'] >= 1 else 0, axis = 1)
df_mob1_3['df_mob1_3_overdue']=df_mob1_3[['overdue_pird_num']].apply(lambda x: 1 if x['overdue_pird_num'] >= 1 else 0, axis = 1)
df_mob1_7['df_mob1_7_overdue']=df_mob1_7[['overdue_pird_num']].apply(lambda x: 1 if x['overdue_pird_num'] >= 1 else 0, axis = 1)
df_mob1_15['df_mob1_15_overdue']=df_mob1_15[['overdue_pird_num']].apply(lambda x: 1 if x['overdue_pird_num'] >= 1 else 0, axis = 1)
df_mob1_30['df_mob1_30_overdue']=df_mob1_30[['account_age','overdue_pird_num']].apply(lambda x: 1 if (x['account_age'] == '1' and x['overdue_pird_num'] >= 1) or (x['account_age'] == '2' and x['overdue_pird_num'] >= 2) else 0, axis = 1)


df_mob1_1['overdue_amt_mob1_1'] = df_mob1_1['remain_prin_int_amt']
df_mob1_3['overdue_amt_mob1_3'] = df_mob1_3['remain_prin_int_amt']
df_mob1_7['overdue_amt_mob1_7'] = df_mob1_7['remain_prin_int_amt']
df_mob1_15['overdue_amt_mob1_15'] = df_mob1_15['remain_prin_int_amt']
df_mob1_30['overdue_amt_mob1_30'] = df_mob1_30['remain_prin_int_amt']


data_temp = pd.merge(df_now,df_mob1_1[['intpc_id','df_mob1_1_overdue','overdue_amt_mob1_1']], on = 'intpc_id', how = 'left')
data_temp = pd.merge(data_temp,df_mob1_3[['intpc_id','df_mob1_3_overdue','overdue_amt_mob1_3']], on = 'intpc_id', how = 'left')
data_temp = pd.merge(data_temp,df_mob1_7[['intpc_id','df_mob1_7_overdue','overdue_amt_mob1_7']], on = 'intpc_id', how = 'left')
data_temp = pd.merge(data_temp,df_mob1_15[['intpc_id','df_mob1_15_overdue','overdue_amt_mob1_15']], on = 'intpc_id', how = 'left')
data_temp = pd.merge(data_temp,df_mob1_30[['intpc_id','df_mob1_30_overdue','overdue_amt_mob1_30']], on = 'intpc_id', how = 'left')

df_intoid.rename(columns={'intoId':'intpc_id'},inplace=True)



data_total = pd.merge(data_temp,df_intoid, on = 'intpc_id', how = 'left')

data_total['utmcodeNew']=data_total['utmcode'].map(lambda x:x if x in api_list else 'ourself')
data_total['utmcodePlus']=data_total.apply(lambda x:x.utmcodeNew if x['product']=='1001' else '%s-%s'%(x['utmcodeNew'],x['product']),axis=1)
data_total['is_api']=data_total['utmcode'].map(lambda x:'API' if x in api_list else 'nonAPI')

data_total['intodate']=data_total.apply(lambda x:x.intoTime[:10],axis=1)
data_total['into_month']=data_total['intodate'].astype(str)
data_total['into_month']=data_total.apply(lambda x:x.into_month[:7],axis=1)

data_total['loan_month']=data_total['reg_loan_dt'].astype(str)
data_total['loan_month']=data_total.apply(lambda x:x.loan_month[:7],axis=1)

data_total['mob1_1_base_flag']=data_total.apply(lambda x:1 if (datetime.now() + relativedelta(years=5)  - pd.to_datetime(x['repay_beg_dt'])).days>=1 else 0, axis=1)
data_total['mob1_3_base_flag']=data_total.apply(lambda x:1 if (datetime.now() + relativedelta(years=5)  - pd.to_datetime(x['repay_beg_dt'])).days>3 else 0, axis=1)
data_total['mob1_7_base_flag']=data_total.apply(lambda x:1 if (datetime.now() + relativedelta(years=5)  - pd.to_datetime(x['repay_beg_dt'])).days>7 else 0, axis=1)
data_total['mob1_15_base_flag']=data_total.apply(lambda x:1 if (datetime.now() + relativedelta(years=5)  - pd.to_datetime(x['repay_beg_dt'])).days>15 else 0, axis=1)
data_total['mob1_30_base_flag']=data_total.apply(lambda x:1 if (datetime.now() + relativedelta(years=5)  - pd.to_datetime(x['repay_beg_dt'])).days>30 else 0, axis=1)

#数据加工完毕
#del df_intoid,df_now,df_mob1_7,df_mob2_7,df_mob3_7,df_mob4_7,df_mob5_7,df_mob6_7,df_mob7_7,df_mob8_7,df_mob9_7,df_mob10_7,df_mob11_7,df_mob12_7,data_temp
#gc.collect()


endtime3=time.time()
print('the procedure for data_total calculation has been run for %s second!'%(endtime3-endtime2))

###################################################################################
###按金额，API，loan_dt
###################################################################################
data_temp=data_total[data_total['is_api']=='API']
result_total=pd.DataFrame()

result_total['loan_count']=data_temp.groupby(['utmcode', 'reg_loan_dt'])['intpc_id'].count()
result_total['loan_amount']=data_temp.groupby(['utmcode', 'reg_loan_dt'])['contr_amt'].sum()
result_total['mob1_1_overdue_sum']=data_temp.loc[data_temp.df_mob1_1_overdue==1].groupby(['utmcode', 'reg_loan_dt'])['overdue_amt_mob1_1'].sum()
result_total['mob1_3_overdue_sum']=data_temp.loc[data_temp.df_mob1_3_overdue==1].groupby(['utmcode', 'reg_loan_dt'])['overdue_amt_mob1_3'].sum()
result_total['mob1_7_overdue_sum']=data_temp.loc[data_temp.df_mob1_7_overdue==1].groupby(['utmcode', 'reg_loan_dt'])['overdue_amt_mob1_7'].sum()
result_total['mob1_15_overdue_sum']=data_temp.loc[data_temp.df_mob1_15_overdue==1].groupby(['utmcode', 'reg_loan_dt'])['overdue_amt_mob1_15'].sum()
result_total['mob1_30_overdue_sum']=data_temp.loc[data_temp.df_mob1_30_overdue==1].groupby(['utmcode', 'reg_loan_dt'])['overdue_amt_mob1_30'].sum()


result_total['mob1_1_base_sum']=data_temp.loc[data_temp.mob1_1_base_flag==1].groupby(['utmcode', 'reg_loan_dt'])['cont_prin_int'].sum()
result_total['mob1_3_base_sum']=data_temp.loc[data_temp.mob1_3_base_flag==1].groupby(['utmcode', 'reg_loan_dt'])['cont_prin_int'].sum()
result_total['mob1_7_base_sum']=data_temp.loc[data_temp.mob1_7_base_flag==1].groupby(['utmcode', 'reg_loan_dt'])['cont_prin_int'].sum()
result_total['mob1_15_base_sum']=data_temp.loc[data_temp.mob1_15_base_flag==1].groupby(['utmcode', 'reg_loan_dt'])['cont_prin_int'].sum()
result_total['mob1_30_base_sum']=data_temp.loc[data_temp.mob1_30_base_flag==1].groupby(['utmcode', 'reg_loan_dt'])['cont_prin_int'].sum()

result_total['mob1_1_pct']=result_total.apply(lambda x:x['mob1_1_overdue_sum']/x['mob1_1_base_sum'], axis = 1)
result_total['mob1_3_pct']=result_total.apply(lambda x:x['mob1_3_overdue_sum']/x['mob1_3_base_sum'], axis = 1)
result_total['mob1_7_pct']=result_total.apply(lambda x:x['mob1_7_overdue_sum']/x['mob1_7_base_sum'], axis = 1)
result_total['mob1_15_pct']=result_total.apply(lambda x:x['mob1_15_overdue_sum']/x['mob1_15_base_sum'], axis = 1)
result_total['mob1_30_pct']=result_total.apply(lambda x:x['mob1_30_overdue_sum']/x['mob1_30_base_sum'], axis = 1)


result_total = result_total.reset_index()

result_output = result_total[['utmcode','reg_loan_dt','loan_count','loan_amount','mob1_1_pct','mob1_3_pct','mob1_7_pct','mob1_15_pct','mob1_30_pct']].copy()

writer=pd.ExcelWriter("E:\AnalystPersonal\chenzihan\每日监控\贷后监控.xlsx")

result_output.to_excel(excel_writer = writer, sheet_name = 'loan_dt_API', float_format='%.4f')


###################################################################################
###按金额，API，into_dt
###################################################################################
data_temp=data_total[data_total['is_api']=='API']
result_total=pd.DataFrame()

result_total['loan_count']=data_temp.groupby(['utmcode', 'intodate'])['intpc_id'].count()
result_total['loan_amount']=data_temp.groupby(['utmcode', 'intodate'])['contr_amt'].sum()
result_total['mob1_1_overdue_sum']=data_temp.loc[data_temp.df_mob1_1_overdue==1].groupby(['utmcode', 'intodate'])['overdue_amt_mob1_1'].sum()
result_total['mob1_3_overdue_sum']=data_temp.loc[data_temp.df_mob1_3_overdue==1].groupby(['utmcode', 'intodate'])['overdue_amt_mob1_3'].sum()
result_total['mob1_7_overdue_sum']=data_temp.loc[data_temp.df_mob1_7_overdue==1].groupby(['utmcode', 'intodate'])['overdue_amt_mob1_7'].sum()
result_total['mob1_15_overdue_sum']=data_temp.loc[data_temp.df_mob1_15_overdue==1].groupby(['utmcode', 'intodate'])['overdue_amt_mob1_15'].sum()
result_total['mob1_30_overdue_sum']=data_temp.loc[data_temp.df_mob1_30_overdue==1].groupby(['utmcode', 'intodate'])['overdue_amt_mob1_30'].sum()


result_total['mob1_1_base_sum']=data_temp.loc[data_temp.mob1_1_base_flag==1].groupby(['utmcode', 'intodate'])['cont_prin_int'].sum()
result_total['mob1_3_base_sum']=data_temp.loc[data_temp.mob1_3_base_flag==1].groupby(['utmcode', 'intodate'])['cont_prin_int'].sum()
result_total['mob1_7_base_sum']=data_temp.loc[data_temp.mob1_7_base_flag==1].groupby(['utmcode', 'intodate'])['cont_prin_int'].sum()
result_total['mob1_15_base_sum']=data_temp.loc[data_temp.mob1_15_base_flag==1].groupby(['utmcode', 'intodate'])['cont_prin_int'].sum()
result_total['mob1_30_base_sum']=data_temp.loc[data_temp.mob1_30_base_flag==1].groupby(['utmcode', 'intodate'])['cont_prin_int'].sum()

result_total['mob1_1_pct']=result_total.apply(lambda x:x['mob1_1_overdue_sum']/x['mob1_1_base_sum'], axis = 1)
result_total['mob1_3_pct']=result_total.apply(lambda x:x['mob1_3_overdue_sum']/x['mob1_3_base_sum'], axis = 1)
result_total['mob1_7_pct']=result_total.apply(lambda x:x['mob1_7_overdue_sum']/x['mob1_7_base_sum'], axis = 1)
result_total['mob1_15_pct']=result_total.apply(lambda x:x['mob1_15_overdue_sum']/x['mob1_15_base_sum'], axis = 1)
result_total['mob1_30_pct']=result_total.apply(lambda x:x['mob1_30_overdue_sum']/x['mob1_30_base_sum'], axis = 1)


result_total = result_total.reset_index()

result_output = result_total[['utmcode','intodate','loan_count','loan_amount','mob1_1_pct','mob1_3_pct','mob1_7_pct','mob1_15_pct','mob1_30_pct']].copy()


result_output.to_excel(excel_writer = writer, sheet_name = 'into_dt_API', float_format='%.4f')

###################################################################################
###按金额，nonAPI，loan_dt
###################################################################################
data_temp=data_total[data_total['is_api']=='nonAPI']
result_total=pd.DataFrame()

result_total['loan_count']=data_temp.groupby(['intochannelname', 'reg_loan_dt'])['intpc_id'].count()
result_total['loan_amount']=data_temp.groupby(['intochannelname', 'reg_loan_dt'])['contr_amt'].sum()
result_total['mob1_1_overdue_sum']=data_temp.loc[data_temp.df_mob1_1_overdue==1].groupby(['intochannelname', 'reg_loan_dt'])['overdue_amt_mob1_1'].sum()
result_total['mob1_3_overdue_sum']=data_temp.loc[data_temp.df_mob1_3_overdue==1].groupby(['intochannelname', 'reg_loan_dt'])['overdue_amt_mob1_3'].sum()
result_total['mob1_7_overdue_sum']=data_temp.loc[data_temp.df_mob1_7_overdue==1].groupby(['intochannelname', 'reg_loan_dt'])['overdue_amt_mob1_7'].sum()
result_total['mob1_15_overdue_sum']=data_temp.loc[data_temp.df_mob1_15_overdue==1].groupby(['intochannelname', 'reg_loan_dt'])['overdue_amt_mob1_15'].sum()
result_total['mob1_30_overdue_sum']=data_temp.loc[data_temp.df_mob1_30_overdue==1].groupby(['intochannelname', 'reg_loan_dt'])['overdue_amt_mob1_30'].sum()


result_total['mob1_1_base_sum']=data_temp.loc[data_temp.mob1_1_base_flag==1].groupby(['intochannelname', 'reg_loan_dt'])['cont_prin_int'].sum()
result_total['mob1_3_base_sum']=data_temp.loc[data_temp.mob1_3_base_flag==1].groupby(['intochannelname', 'reg_loan_dt'])['cont_prin_int'].sum()
result_total['mob1_7_base_sum']=data_temp.loc[data_temp.mob1_7_base_flag==1].groupby(['intochannelname', 'reg_loan_dt'])['cont_prin_int'].sum()
result_total['mob1_15_base_sum']=data_temp.loc[data_temp.mob1_15_base_flag==1].groupby(['intochannelname', 'reg_loan_dt'])['cont_prin_int'].sum()
result_total['mob1_30_base_sum']=data_temp.loc[data_temp.mob1_30_base_flag==1].groupby(['intochannelname', 'reg_loan_dt'])['cont_prin_int'].sum()

result_total['mob1_1_pct']=result_total.apply(lambda x:x['mob1_1_overdue_sum']/x['mob1_1_base_sum'], axis = 1)
result_total['mob1_3_pct']=result_total.apply(lambda x:x['mob1_3_overdue_sum']/x['mob1_3_base_sum'], axis = 1)
result_total['mob1_7_pct']=result_total.apply(lambda x:x['mob1_7_overdue_sum']/x['mob1_7_base_sum'], axis = 1)
result_total['mob1_15_pct']=result_total.apply(lambda x:x['mob1_15_overdue_sum']/x['mob1_15_base_sum'], axis = 1)
result_total['mob1_30_pct']=result_total.apply(lambda x:x['mob1_30_overdue_sum']/x['mob1_30_base_sum'], axis = 1)


result_total = result_total.reset_index()

result_output = result_total[['intochannelname','reg_loan_dt','loan_count','loan_amount','mob1_1_pct','mob1_3_pct','mob1_7_pct','mob1_15_pct','mob1_30_pct']].copy()


result_output.to_excel(excel_writer = writer, sheet_name = 'loan_dt_nonAPI', float_format='%.4f')

###################################################################################
###按金额，nonAPI，into_dt
###################################################################################
data_temp=data_total[data_total['is_api']=='nonAPI']
result_total=pd.DataFrame()

result_total['loan_count']=data_temp.groupby(['intochannelname', 'intodate'])['intpc_id'].count()
result_total['loan_amount']=data_temp.groupby(['intochannelname', 'intodate'])['contr_amt'].sum()
result_total['mob1_1_overdue_sum']=data_temp.loc[data_temp.df_mob1_1_overdue==1].groupby(['intochannelname', 'intodate'])['overdue_amt_mob1_1'].sum()
result_total['mob1_3_overdue_sum']=data_temp.loc[data_temp.df_mob1_3_overdue==1].groupby(['intochannelname', 'intodate'])['overdue_amt_mob1_3'].sum()
result_total['mob1_7_overdue_sum']=data_temp.loc[data_temp.df_mob1_7_overdue==1].groupby(['intochannelname', 'intodate'])['overdue_amt_mob1_7'].sum()
result_total['mob1_15_overdue_sum']=data_temp.loc[data_temp.df_mob1_15_overdue==1].groupby(['intochannelname', 'intodate'])['overdue_amt_mob1_15'].sum()
result_total['mob1_30_overdue_sum']=data_temp.loc[data_temp.df_mob1_30_overdue==1].groupby(['intochannelname', 'intodate'])['overdue_amt_mob1_30'].sum()


result_total['mob1_1_base_sum']=data_temp.loc[data_temp.mob1_1_base_flag==1].groupby(['intochannelname', 'intodate'])['cont_prin_int'].sum()
result_total['mob1_3_base_sum']=data_temp.loc[data_temp.mob1_3_base_flag==1].groupby(['intochannelname', 'intodate'])['cont_prin_int'].sum()
result_total['mob1_7_base_sum']=data_temp.loc[data_temp.mob1_7_base_flag==1].groupby(['intochannelname', 'intodate'])['cont_prin_int'].sum()
result_total['mob1_15_base_sum']=data_temp.loc[data_temp.mob1_15_base_flag==1].groupby(['intochannelname', 'intodate'])['cont_prin_int'].sum()
result_total['mob1_30_base_sum']=data_temp.loc[data_temp.mob1_30_base_flag==1].groupby(['intochannelname', 'intodate'])['cont_prin_int'].sum()

result_total['mob1_1_pct']=result_total.apply(lambda x:x['mob1_1_overdue_sum']/x['mob1_1_base_sum'], axis = 1)
result_total['mob1_3_pct']=result_total.apply(lambda x:x['mob1_3_overdue_sum']/x['mob1_3_base_sum'], axis = 1)
result_total['mob1_7_pct']=result_total.apply(lambda x:x['mob1_7_overdue_sum']/x['mob1_7_base_sum'], axis = 1)
result_total['mob1_15_pct']=result_total.apply(lambda x:x['mob1_15_overdue_sum']/x['mob1_15_base_sum'], axis = 1)
result_total['mob1_30_pct']=result_total.apply(lambda x:x['mob1_30_overdue_sum']/x['mob1_30_base_sum'], axis = 1)


result_total = result_total.reset_index()

result_output = result_total[['intochannelname','intodate','loan_count','loan_amount','mob1_1_pct','mob1_3_pct','mob1_7_pct','mob1_15_pct','mob1_30_pct']].copy()


result_output.to_excel(excel_writer = writer, sheet_name = 'into_dt_nonAPI', float_format='%.4f')

###################################################################################
###按金额，summary, loan_dt， daily
###################################################################################

result_total=pd.DataFrame()

result_total['loan_count']=data_total.groupby(['is_api', 'reg_loan_dt'])['intpc_id'].count()
result_total['loan_amount']=data_total.groupby(['is_api', 'reg_loan_dt'])['contr_amt'].sum()
result_total['mob1_1_overdue_sum']=data_total.loc[data_total.df_mob1_1_overdue==1].groupby(['is_api', 'reg_loan_dt'])['overdue_amt_mob1_1'].sum()
result_total['mob1_3_overdue_sum']=data_total.loc[data_total.df_mob1_3_overdue==1].groupby(['is_api', 'reg_loan_dt'])['overdue_amt_mob1_3'].sum()
result_total['mob1_7_overdue_sum']=data_total.loc[data_total.df_mob1_7_overdue==1].groupby(['is_api', 'reg_loan_dt'])['overdue_amt_mob1_7'].sum()
result_total['mob1_15_overdue_sum']=data_total.loc[data_total.df_mob1_15_overdue==1].groupby(['is_api', 'reg_loan_dt'])['overdue_amt_mob1_15'].sum()
result_total['mob1_30_overdue_sum']=data_total.loc[data_total.df_mob1_30_overdue==1].groupby(['is_api', 'reg_loan_dt'])['overdue_amt_mob1_30'].sum()


result_total['mob1_1_base_sum']=data_total.loc[data_total.mob1_1_base_flag==1].groupby(['is_api', 'reg_loan_dt'])['cont_prin_int'].sum()
result_total['mob1_3_base_sum']=data_total.loc[data_total.mob1_3_base_flag==1].groupby(['is_api', 'reg_loan_dt'])['cont_prin_int'].sum()
result_total['mob1_7_base_sum']=data_total.loc[data_total.mob1_7_base_flag==1].groupby(['is_api', 'reg_loan_dt'])['cont_prin_int'].sum()
result_total['mob1_15_base_sum']=data_total.loc[data_total.mob1_15_base_flag==1].groupby(['is_api', 'reg_loan_dt'])['cont_prin_int'].sum()
result_total['mob1_30_base_sum']=data_total.loc[data_total.mob1_30_base_flag==1].groupby(['is_api', 'reg_loan_dt'])['cont_prin_int'].sum()

result_total['mob1_1_pct']=result_total.apply(lambda x:x['mob1_1_overdue_sum']/x['mob1_1_base_sum'], axis = 1)
result_total['mob1_3_pct']=result_total.apply(lambda x:x['mob1_3_overdue_sum']/x['mob1_3_base_sum'], axis = 1)
result_total['mob1_7_pct']=result_total.apply(lambda x:x['mob1_7_overdue_sum']/x['mob1_7_base_sum'], axis = 1)
result_total['mob1_15_pct']=result_total.apply(lambda x:x['mob1_15_overdue_sum']/x['mob1_15_base_sum'], axis = 1)
result_total['mob1_30_pct']=result_total.apply(lambda x:x['mob1_30_overdue_sum']/x['mob1_30_base_sum'], axis = 1)


result_total = result_total.reset_index()

result_output = result_total[['is_api','reg_loan_dt','loan_count','loan_amount','mob1_1_pct','mob1_3_pct','mob1_7_pct','mob1_15_pct','mob1_30_pct']].copy()


result_output.to_excel(excel_writer = writer, sheet_name = 'loan_dt_summary_daily', float_format='%.4f')




###################################################################################
###按金额，summary, into_dt，daily
###################################################################################

result_total=pd.DataFrame()

result_total['loan_count']=data_total.groupby(['is_api', 'intodate'])['intpc_id'].count()
result_total['loan_amount']=data_total.groupby(['is_api', 'intodate'])['contr_amt'].sum()
result_total['mob1_1_overdue_sum']=data_total.loc[data_total.df_mob1_1_overdue==1].groupby(['is_api', 'intodate'])['overdue_amt_mob1_1'].sum()
result_total['mob1_3_overdue_sum']=data_total.loc[data_total.df_mob1_3_overdue==1].groupby(['is_api', 'intodate'])['overdue_amt_mob1_3'].sum()
result_total['mob1_7_overdue_sum']=data_total.loc[data_total.df_mob1_7_overdue==1].groupby(['is_api', 'intodate'])['overdue_amt_mob1_7'].sum()
result_total['mob1_15_overdue_sum']=data_total.loc[data_total.df_mob1_15_overdue==1].groupby(['is_api', 'intodate'])['overdue_amt_mob1_15'].sum()
result_total['mob1_30_overdue_sum']=data_total.loc[data_total.df_mob1_30_overdue==1].groupby(['is_api', 'intodate'])['overdue_amt_mob1_30'].sum()


result_total['mob1_1_base_sum']=data_total.loc[data_total.mob1_1_base_flag==1].groupby(['is_api', 'intodate'])['cont_prin_int'].sum()
result_total['mob1_3_base_sum']=data_total.loc[data_total.mob1_3_base_flag==1].groupby(['is_api', 'intodate'])['cont_prin_int'].sum()
result_total['mob1_7_base_sum']=data_total.loc[data_total.mob1_7_base_flag==1].groupby(['is_api', 'intodate'])['cont_prin_int'].sum()
result_total['mob1_15_base_sum']=data_total.loc[data_total.mob1_15_base_flag==1].groupby(['is_api', 'intodate'])['cont_prin_int'].sum()
result_total['mob1_30_base_sum']=data_total.loc[data_total.mob1_30_base_flag==1].groupby(['is_api', 'intodate'])['cont_prin_int'].sum()

result_total['mob1_1_pct']=result_total.apply(lambda x:x['mob1_1_overdue_sum']/x['mob1_1_base_sum'], axis = 1)
result_total['mob1_3_pct']=result_total.apply(lambda x:x['mob1_3_overdue_sum']/x['mob1_3_base_sum'], axis = 1)
result_total['mob1_7_pct']=result_total.apply(lambda x:x['mob1_7_overdue_sum']/x['mob1_7_base_sum'], axis = 1)
result_total['mob1_15_pct']=result_total.apply(lambda x:x['mob1_15_overdue_sum']/x['mob1_15_base_sum'], axis = 1)
result_total['mob1_30_pct']=result_total.apply(lambda x:x['mob1_30_overdue_sum']/x['mob1_30_base_sum'], axis = 1)


result_total = result_total.reset_index()

result_output = result_total[['is_api','intodate','loan_count','loan_amount','mob1_1_pct','mob1_3_pct','mob1_7_pct','mob1_15_pct','mob1_30_pct']].copy()


result_output.to_excel(excel_writer = writer, sheet_name = 'into_dt_summary_daily', float_format='%.4f')

###################################################################################
###按金额，summary, loan_dt， monthly
###################################################################################

result_total=pd.DataFrame()

result_total['loan_count']=data_total.groupby(['is_api', 'loan_month'])['intpc_id'].count()
result_total['loan_amount']=data_total.groupby(['is_api', 'loan_month'])['contr_amt'].sum()
result_total['mob1_1_overdue_sum']=data_total.loc[data_total.df_mob1_1_overdue==1].groupby(['is_api', 'loan_month'])['overdue_amt_mob1_1'].sum()
result_total['mob1_3_overdue_sum']=data_total.loc[data_total.df_mob1_3_overdue==1].groupby(['is_api', 'loan_month'])['overdue_amt_mob1_3'].sum()
result_total['mob1_7_overdue_sum']=data_total.loc[data_total.df_mob1_7_overdue==1].groupby(['is_api', 'loan_month'])['overdue_amt_mob1_7'].sum()
result_total['mob1_15_overdue_sum']=data_total.loc[data_total.df_mob1_15_overdue==1].groupby(['is_api', 'loan_month'])['overdue_amt_mob1_15'].sum()
result_total['mob1_30_overdue_sum']=data_total.loc[data_total.df_mob1_30_overdue==1].groupby(['is_api', 'loan_month'])['overdue_amt_mob1_30'].sum()


result_total['mob1_1_base_sum']=data_total.loc[data_total.mob1_1_base_flag==1].groupby(['is_api', 'loan_month'])['cont_prin_int'].sum()
result_total['mob1_3_base_sum']=data_total.loc[data_total.mob1_3_base_flag==1].groupby(['is_api', 'loan_month'])['cont_prin_int'].sum()
result_total['mob1_7_base_sum']=data_total.loc[data_total.mob1_7_base_flag==1].groupby(['is_api', 'loan_month'])['cont_prin_int'].sum()
result_total['mob1_15_base_sum']=data_total.loc[data_total.mob1_15_base_flag==1].groupby(['is_api', 'loan_month'])['cont_prin_int'].sum()
result_total['mob1_30_base_sum']=data_total.loc[data_total.mob1_30_base_flag==1].groupby(['is_api', 'loan_month'])['cont_prin_int'].sum()

result_total['mob1_1_pct']=result_total.apply(lambda x:x['mob1_1_overdue_sum']/x['mob1_1_base_sum'], axis = 1)
result_total['mob1_3_pct']=result_total.apply(lambda x:x['mob1_3_overdue_sum']/x['mob1_3_base_sum'], axis = 1)
result_total['mob1_7_pct']=result_total.apply(lambda x:x['mob1_7_overdue_sum']/x['mob1_7_base_sum'], axis = 1)
result_total['mob1_15_pct']=result_total.apply(lambda x:x['mob1_15_overdue_sum']/x['mob1_15_base_sum'], axis = 1)
result_total['mob1_30_pct']=result_total.apply(lambda x:x['mob1_30_overdue_sum']/x['mob1_30_base_sum'], axis = 1)


result_total = result_total.reset_index()

result_output = result_total[['is_api','loan_month','loan_count','loan_amount','mob1_1_pct','mob1_3_pct','mob1_7_pct','mob1_15_pct','mob1_30_pct']].copy()


result_output.to_excel(excel_writer = writer, sheet_name = 'loan_dt_summary_monthly', float_format='%.4f')




###################################################################################
###按金额，summary, into_dt，monthly
###################################################################################

result_total=pd.DataFrame()

result_total['loan_count']=data_total.groupby(['is_api', 'into_month'])['intpc_id'].count()
result_total['loan_amount']=data_total.groupby(['is_api', 'into_month'])['contr_amt'].sum()
result_total['mob1_1_overdue_sum']=data_total.loc[data_total.df_mob1_1_overdue==1].groupby(['is_api', 'into_month'])['overdue_amt_mob1_1'].sum()
result_total['mob1_3_overdue_sum']=data_total.loc[data_total.df_mob1_3_overdue==1].groupby(['is_api', 'into_month'])['overdue_amt_mob1_3'].sum()
result_total['mob1_7_overdue_sum']=data_total.loc[data_total.df_mob1_7_overdue==1].groupby(['is_api', 'into_month'])['overdue_amt_mob1_7'].sum()
result_total['mob1_15_overdue_sum']=data_total.loc[data_total.df_mob1_15_overdue==1].groupby(['is_api', 'into_month'])['overdue_amt_mob1_15'].sum()
result_total['mob1_30_overdue_sum']=data_total.loc[data_total.df_mob1_30_overdue==1].groupby(['is_api', 'into_month'])['overdue_amt_mob1_30'].sum()


result_total['mob1_1_base_sum']=data_total.loc[data_total.mob1_1_base_flag==1].groupby(['is_api', 'into_month'])['cont_prin_int'].sum()
result_total['mob1_3_base_sum']=data_total.loc[data_total.mob1_3_base_flag==1].groupby(['is_api', 'into_month'])['cont_prin_int'].sum()
result_total['mob1_7_base_sum']=data_total.loc[data_total.mob1_7_base_flag==1].groupby(['is_api', 'into_month'])['cont_prin_int'].sum()
result_total['mob1_15_base_sum']=data_total.loc[data_total.mob1_15_base_flag==1].groupby(['is_api', 'into_month'])['cont_prin_int'].sum()
result_total['mob1_30_base_sum']=data_total.loc[data_total.mob1_30_base_flag==1].groupby(['is_api', 'into_month'])['cont_prin_int'].sum()

result_total['mob1_1_pct']=result_total.apply(lambda x:x['mob1_1_overdue_sum']/x['mob1_1_base_sum'], axis = 1)
result_total['mob1_3_pct']=result_total.apply(lambda x:x['mob1_3_overdue_sum']/x['mob1_3_base_sum'], axis = 1)
result_total['mob1_7_pct']=result_total.apply(lambda x:x['mob1_7_overdue_sum']/x['mob1_7_base_sum'], axis = 1)
result_total['mob1_15_pct']=result_total.apply(lambda x:x['mob1_15_overdue_sum']/x['mob1_15_base_sum'], axis = 1)
result_total['mob1_30_pct']=result_total.apply(lambda x:x['mob1_30_overdue_sum']/x['mob1_30_base_sum'], axis = 1)


result_total = result_total.reset_index()

result_output = result_total[['is_api','into_month','loan_count','loan_amount','mob1_1_pct','mob1_3_pct','mob1_7_pct','mob1_15_pct','mob1_30_pct']].copy()


result_output.to_excel(excel_writer = writer, sheet_name = 'into_dt_summary_monthly', float_format='%.4f')

writer.save()
writer.close()



a=data_total[(data_total.utmcode=='bairong')&(data_total.reg_loan_dt.astype(str)=='2019-11-05')]














