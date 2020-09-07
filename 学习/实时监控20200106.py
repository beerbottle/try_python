# -*- coding: utf-8 -*-
"""
Created on Fri Jan  2 16:03:10 2015

@author: SYSTEM
"""

    # -*- coding: utf-8 -*-
import libkit
import numpy as np
import pandas as pd
import time         
import gc
    


#################################################
#####参数设置
#################################################
#近期审批情况时间
showdatelist=['2020-01-13','2020-01-14'] #审批通过率及拒贷原因分布参数
#历史审批流程观测时间段
detaildatelist=['2020-01-01','2020-02-01']
#额度观测时间段
amountdetaildatelist=['2020-01-01','2020-02-01']
api_list=['wacai','xianjinbaika','kaniu','360','bairong','jiedianqian','daikuandaohang','51','haodai','xinyongka','boluodai','xinyongguanjia','geinihua']

detailutmcode='ourself'

'''
quantilelist=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.75,0.8,0.85,0.9,0.95,0.98]
quantilelist=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.75,0.8,0.85,0.9,0.95,0.98]
'''

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
##first input three
info='*'
info=['da_db_id','xykwy_bankKeyInput','xykwy_minBillAgoMonths','xykwy_continuityBillNum','xykwy_newestCreditCny']
tablename='jsd_execute_input_three'
searchcondition={'da_db_id':zeroIntpc.da_db_id.tolist()}
sqlsentence=libkit.data.sqlQuery(info,tablename,searchcondition=searchcondition)
zeroIntpcThree=libkit.data.getBySql(sqlsentence,conn=da)
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
##last phone
info='*'
info=['da_db_id','tele_top10_list_cnt_3m','phone_list_cnt']
tablename='jsd_input_app_phone'
searchcondition={'da_db_id':zeroFinal.da_db_id.tolist()}
sqlsentence=libkit.data.sqlQuery(info,tablename,searchcondition=searchcondition)
zeroFinalPhone=libkit.data.getBySql(sqlsentence,conn=da)
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
tempFirst=pd.merge(zeroIntpc,zeroIntpcThree,how='left',on='da_db_id')
tempFirst=pd.merge(tempFirst,zeroIntpcOutput,how='left',on='da_db_id')
tempLast=pd.merge(zeroFinal,zeroFinalPhone,how='left',on='da_db_id')
tempLast=pd.merge(tempLast,zeroFinalOutput,how='left',on='da_db_id')
zeroRiskcon.rename(columns={'id':'ip_id'},inplace=True)
tempRisk=pd.merge(zeroRiskcon,zeroRiskconOutput,how='left',on='ip_id')
tempRisk.rename(columns={'into_id':'intoId','amount_x':'askamount','amount_y':'agreeamount'},inplace=True)
tempUnion=pd.merge(tempFirst,tempRisk,how='left',on='intoId')
tempUnion=pd.merge(tempUnion,tempLast,how='left',on='intoId')
tempUnion=tempUnion[~tempUnion.duplicated('intoId','last')].copy()
needTitle=['intoId','da_db_id_x','da_db_id_y','product','into_bus_id','applytime','intoTime','utmcode','intochannelname', 'vendor','attribution','mobile','into_state','revive_time','review_result',
           'result_reason','review_state','refuse_main_reason','refuse_child_reason','refuse_type','pre_output_dclnreason','pre_output_otherreason','final_output_dclnreason','final_output_otherreason','Score_Test',
           'hint_8','hint_19','output_cgrade','rate','askamount','output_amount','agreeamount','input_term','output_term','duration','earliestCallTime','h1_month_call_num',
           'h1_month_bill_amount','tele_avg_amt_6m','IN_INFOSTR','into_term','out_term','xykwy_bankKeyInput','xykwy_minBillAgoMonths','income','id_match','name_match',
           'birthday','code_phone','xykwy_continuityBillNum','xykwy_newestCreditCny','tele_top10_list_cnt_3m','phone_list_cnt','reg_channelid','reg_channelname']
zeroUnion=tempUnion[needTitle].copy()
#in case of some abnormal event,like self consistent check
zeroUnion.loc[(zeroUnion.review_result.isnull())&(zeroUnion.review_state.notnull())]=None
zeroUnion.loc[(zeroUnion.review_result.isnull())&(zeroUnion.refuse_main_reason.notnull())]=None
zeroUnion.loc[(zeroUnion.review_result=='通过')&(zeroUnion.refuse_main_reason.notnull())]=None

'''
def getAge(beg,end):
    age=int(end[:4])-int(beg[:4])
    if end[5:]<beg[5:]:
        age-=1
    return age
'''
#zeroUnion['age']=zeroUnion.apply(lambda x:getAge(x.birthday,x.intoTime),axis=1)
zeroUnion['utmcodeNew']=zeroUnion['utmcode'].map(lambda x:x if x in api_list else 'ourself')
zeroUnion['utmcodePlus']=zeroUnion.apply(lambda x:x.utmcodeNew if x['product']=='1001' else '%s-%s'%(x.utmcodeNew,x['product']),axis=1)
refuselist=zeroUnion['refuse_main_reason'].dropna().unique().tolist()
refuselist.sort()
del tempUnion,tempFirst,tempLast,tempRisk
del zeroIntpc,zeroIntpcOutput,zeroFinal,zeroFinalPhone,zeroFinalOutput,zeroRiskcon,zeroRiskconOutput
gc.collect()


def refuseAnalysis(tempFrame,groupbyname='utmcodeNew'):
    intolist=['益博睿预排除','三方','反欺诈人工','益博睿拒绝','人工信审结束','益博睿', '益博睿直通','人工信审通过','反欺诈待定','信审待定']
    tempStaticFrame=pd.DataFrame(index=refuselist)
    tempAgreeFrame=pd.DataFrame(index=intolist)
    tempAgreeRatioFrame=pd.DataFrame()
    tempFrame['review_state'].fillna('待定',inplace=True)
    for groupname,group in tempFrame.groupby(groupbyname,as_index=False):
        tempStaticFrame[groupname]=group.groupby('refuse_main_reason').intoId.count()
        tempStaticFrame[groupname+'Ratio']=tempStaticFrame[groupname]/len(group)
        tempAgreeFrame[groupname]=group.groupby('review_state').intoId.count()
        tempAgreeFrame[groupname].fillna(0,inplace=True)
        tempAgreeFrame.loc['反欺诈待定',groupname]=len(group[(group.review_state=='待定')&(group.Score_Test.isnull())])
        tempAgreeFrame.loc['信审待定',groupname]=len(group[(group.review_state=='待定')&(group.Score_Test.notnull())])
        tempAgreeFrame.loc['人工信审结束',groupname]=tempAgreeFrame.loc['人工信审结束',groupname]
        tempAgreeFrame.loc['益博睿',groupname]=tempAgreeFrame.loc['益博睿',groupname]+tempAgreeFrame.loc['人工信审结束',groupname]+tempAgreeFrame.loc['信审待定',groupname]
        tempAgreeFrame.loc['益博睿直通',groupname]=len(group[(group.review_result=='通过')&(group.review_state=='益博睿')])
        tempAgreeFrame.loc['益博睿拒绝',groupname]=len(group[(group.review_result=='拒绝')&(group.review_state=='益博睿')])
        tempAgreeFrame.loc['反欺诈人工',groupname]=tempAgreeFrame.loc['反欺诈人工',groupname]+tempAgreeFrame.loc['益博睿',groupname]
        tempAgreeFrame.loc['三方',groupname]=tempAgreeFrame.loc['三方',groupname]+tempAgreeFrame.loc['反欺诈人工',groupname]+tempAgreeFrame.loc['反欺诈待定',groupname]
        tempAgreeFrame.loc['益博睿预排除',groupname]=tempAgreeFrame.loc['益博睿预排除',groupname]+tempAgreeFrame.loc['三方',groupname]
        tempAgreeFrame.loc['人工信审通过',groupname]=len(group[(group.review_result=='通过')&(group.review_state=='人工信审结束')])
        tempAgreeFrame.loc['通过',groupname]=len(group[group.review_result=='通过'])
    tempAgreeRatioFrame['益博睿预排除']=tempAgreeFrame.loc['三方']/tempAgreeFrame.loc['益博睿预排除']
    tempAgreeRatioFrame['三方']=tempAgreeFrame.loc['反欺诈人工']/tempAgreeFrame.loc['三方']
    tempAgreeRatioFrame['反欺诈人工']=tempAgreeFrame.loc['益博睿']/tempAgreeFrame.loc['反欺诈人工']
    tempAgreeRatioFrame['益博睿']=(tempAgreeFrame.loc['益博睿直通']+tempAgreeFrame.loc['人工信审结束']+tempAgreeFrame.loc['信审待定'])/tempAgreeFrame.loc['益博睿']
    tempAgreeRatioFrame['人工信审结束']=tempAgreeFrame.loc['人工信审通过']/tempAgreeFrame.loc['人工信审结束']
#    tempAgreeRatioFrame['待定']=tempAgreeFrame.loc['待定']/tempAgreeFrame.loc['人工信审结束']
    tempAgreeRatioFrame=tempAgreeRatioFrame.T
    return tempStaticFrame,tempAgreeFrame,tempAgreeRatioFrame

#数据获取加工处理完毕

################################################
#审批流监控：API加自有汇总
################################################
##get the individual showframe via the utmcodePlus
tempframe=zeroUnion[(zeroUnion.intoTime.between(*showdatelist))].copy()
#tempframe.replace('51-1004','51-1002',inplace=True)

zeroStaticFrame,zeroAgreeFrame,zeroAgreeRatioFrame=refuseAnalysis(tempframe,'utmcodePlus')

zeroAgreeFrame.rename(columns={'51-1002':'51公积金','51-1004':'51社保'},inplace=True)
zeroAgreeRatioFrame.rename(columns={'51-1002':'51公积金','51-1004':'51社保'},inplace=True)
zeroAgreeFrame['API_Total']=zeroAgreeFrame.loc[:,zeroAgreeFrame.columns!='ourself'].sum(axis=1)
zeroAgreeFrame['Total']=zeroAgreeFrame.sum(axis=1)

#调整列的顺序
temp = zeroAgreeFrame.ourself
column_number=zeroAgreeFrame.shape[1]
zeroAgreeFrame = zeroAgreeFrame.drop('ourself',axis=1)
zeroAgreeFrame.insert(column_number-2,'ourself',temp)
del temp

#调整列的顺序
temp = zeroAgreeRatioFrame.ourself
column_number=zeroAgreeRatioFrame.shape[1]
zeroAgreeRatioFrame = zeroAgreeRatioFrame.drop('ourself',axis=1)
zeroAgreeRatioFrame.insert(column_number-1,'ourself',temp)
del temp

zeroAgreeFrame=zeroAgreeFrame.reindex(['益博睿预排除','三方','反欺诈人工','益博睿','人工信审结束','人工信审通过','益博睿直通','通过','反欺诈待定','信审待定'])

zeroAgreeRatioFrame.loc['益博睿预排除','API_Total'] = zeroAgreeFrame.loc['三方','API_Total']/zeroAgreeFrame.loc['益博睿预排除','API_Total']
zeroAgreeRatioFrame.loc['三方','API_Total'] = zeroAgreeFrame.loc['反欺诈人工','API_Total']/zeroAgreeFrame.loc['三方','API_Total']
zeroAgreeRatioFrame.loc['反欺诈人工','API_Total'] = zeroAgreeFrame.loc['益博睿','API_Total']/zeroAgreeFrame.loc['反欺诈人工','API_Total']
zeroAgreeRatioFrame.loc['益博睿','API_Total'] = (zeroAgreeFrame.loc['人工信审结束','API_Total']+zeroAgreeFrame.loc['信审待定','API_Total']+
zeroAgreeFrame.loc['益博睿直通','API_Total'])/zeroAgreeFrame.loc['益博睿','API_Total']
zeroAgreeRatioFrame.loc['人工信审结束','API_Total'] = zeroAgreeFrame.loc['人工信审通过','API_Total']/zeroAgreeFrame.loc['人工信审结束','API_Total']

zeroAgreeRatioFrame.loc['益博睿预排除','Total'] = zeroAgreeFrame.loc['三方','Total']/zeroAgreeFrame.loc['益博睿预排除','Total']
zeroAgreeRatioFrame.loc['三方','Total'] = zeroAgreeFrame.loc['反欺诈人工','Total']/zeroAgreeFrame.loc['三方','Total']
zeroAgreeRatioFrame.loc['反欺诈人工','Total'] = zeroAgreeFrame.loc['益博睿','Total']/zeroAgreeFrame.loc['反欺诈人工','Total']
zeroAgreeRatioFrame.loc['益博睿','Total'] = (zeroAgreeFrame.loc['人工信审结束','Total']+zeroAgreeFrame.loc['信审待定','Total']+
zeroAgreeFrame.loc['益博睿直通','Total'])/zeroAgreeFrame.loc['益博睿','Total']
zeroAgreeRatioFrame.loc['人工信审结束','Total'] = zeroAgreeFrame.loc['人工信审通过','Total']/zeroAgreeFrame.loc['人工信审结束','Total']

zeroAgreeFrameRelative=pd.DataFrame(index=zeroAgreeFrame.index)

zeroAgreeFrameRelative = zeroAgreeFrame.apply(lambda x: x/x.iloc[0])

#将结果写入Excel
writer=pd.ExcelWriter("E:\AnalystPersonal\chenzihan\每日监控\daily_report_total.xlsx")

zeroAgreeFrame.to_excel(excel_writer = writer, sheet_name = '通过率监控')
zeroAgreeFrameRelative.to_excel(excel_writer = writer, sheet_name = '通过率监控',startrow=12, startcol=0,float_format='%.4f')
zeroAgreeRatioFrame.to_excel(excel_writer = writer, sheet_name = '通过率监控',startrow=0, startcol=10, float_format='%.4f')

writer.save()

writer.close()

#在Python中展示结果
if 'xinyongka' in zeroAgreeFrame.columns:
    zeroShowFrame_xinyongka=pd.DataFrame()
    zeroShowFrame_xinyongka['Number']=zeroAgreeFrame['xinyongka']
    zeroShowFrame_xinyongka['relativeRatio']=zeroAgreeFrame['xinyongka']/zeroAgreeFrame['xinyongka'].iloc[0]
    zeroShowFrame_xinyongka['acceptRatio']=zeroAgreeRatioFrame['xinyongka']
else:
    pass
    
if 'geinihua' in zeroAgreeFrame.columns:
    zeroShowFrame_geinihua=pd.DataFrame()
    zeroShowFrame_geinihua['Number']=zeroAgreeFrame['geinihua']
    zeroShowFrame_geinihua['relativeRatio']=zeroAgreeFrame['geinihua']/zeroAgreeFrame['geinihua'].iloc[0]
    zeroShowFrame_geinihua['acceptRatio']=zeroAgreeRatioFrame['geinihua']
else:
    pass

if 'bairong' in zeroAgreeFrame.columns:
    zeroShowFrame_bairong=pd.DataFrame()
    zeroShowFrame_bairong['Number']=zeroAgreeFrame['bairong']
    zeroShowFrame_bairong['relativeRatio']=zeroAgreeFrame['bairong']/zeroAgreeFrame['bairong'].iloc[0]
    zeroShowFrame_bairong['acceptRatio']=zeroAgreeRatioFrame['bairong']
else:
    pass

if '51社保' in zeroAgreeFrame.columns:
    zeroShowFrame_51sb=pd.DataFrame()
    zeroShowFrame_51sb['Number']=zeroAgreeFrame['51社保']
    zeroShowFrame_51sb['relativeRatio']=zeroAgreeFrame['51社保']/zeroAgreeFrame['51社保'].iloc[0]
    zeroShowFrame_51sb['acceptRatio']=zeroAgreeRatioFrame['51社保']
else:
    pass
    

if '51公积金' in zeroAgreeFrame.columns:
    zeroShowFrame_51gjj=pd.DataFrame()
    zeroShowFrame_51gjj['Number']=zeroAgreeFrame['51公积金']
    zeroShowFrame_51gjj['relativeRatio']=zeroAgreeFrame['51公积金']/zeroAgreeFrame['51公积金'].iloc[0]
    zeroShowFrame_51gjj['acceptRatio']=zeroAgreeRatioFrame['51公积金']
else:
    pass

if 'ourself' in zeroAgreeFrame.columns:
    zeroShowFrame_ourself=pd.DataFrame()
    zeroShowFrame_ourself['Number']=zeroAgreeFrame['ourself']
    zeroShowFrame_ourself['relativeRatio']=zeroAgreeFrame['ourself']/zeroAgreeFrame['ourself'].iloc[0]
    zeroShowFrame_ourself['acceptRatio']=zeroAgreeRatioFrame['ourself']
else:
    pass
    
if 'jiedianqian' in zeroAgreeFrame.columns:
    zeroShowFrame_jiedianqian=pd.DataFrame()
    zeroShowFrame_jiedianqian['Number']=zeroAgreeFrame['jiedianqian']
    zeroShowFrame_jiedianqian['relativeRatio']=zeroAgreeFrame['jiedianqian']/zeroAgreeFrame['jiedianqian'].iloc[0]
    zeroShowFrame_jiedianqian['acceptRatio']=zeroAgreeRatioFrame['jiedianqian']
else:
    pass    


################################################
#审批流监控：自有渠道详细
################################################
##get the individual showframe via the utmcodePlus
tempframe=zeroUnion[(zeroUnion.intoTime.between(*showdatelist))&(zeroUnion.utmcodeNew=='ourself')].copy()
#tempframe.replace('51-1004','51-1002',inplace=True)
zeroStaticFrame,zeroAgreeFrame,zeroAgreeRatioFrame=refuseAnalysis(tempframe,'intochannelname')

zeroAgreeFrame=zeroAgreeFrame.sort_values(by = '益博睿预排除', ascending=False, axis=1)


zeroAgreeFrame['Total']=zeroAgreeFrame.sum(axis=1)


zeroAgreeFrame=zeroAgreeFrame.reindex(['益博睿预排除','三方','反欺诈人工','益博睿','人工信审结束','人工信审通过','益博睿直通','通过','反欺诈待定','信审待定'])

zeroAgreeRatioFrame.loc['益博睿预排除','Total'] = zeroAgreeFrame.loc['三方','Total']/zeroAgreeFrame.loc['益博睿预排除','Total']
zeroAgreeRatioFrame.loc['三方','Total'] = zeroAgreeFrame.loc['反欺诈人工','Total']/zeroAgreeFrame.loc['三方','Total']
zeroAgreeRatioFrame.loc['反欺诈人工','Total'] = zeroAgreeFrame.loc['益博睿','Total']/zeroAgreeFrame.loc['反欺诈人工','Total']
zeroAgreeRatioFrame.loc['益博睿','Total'] = (zeroAgreeFrame.loc['人工信审结束','Total']+zeroAgreeFrame.loc['信审待定','Total']+
zeroAgreeFrame.loc['益博睿直通','Total'])/zeroAgreeFrame.loc['益博睿','Total']
zeroAgreeRatioFrame.loc['人工信审结束','Total'] = zeroAgreeFrame.loc['人工信审通过','Total']/zeroAgreeFrame.loc['人工信审结束','Total']

zeroAgreeFrameRelative=pd.DataFrame(index=zeroAgreeFrame.index)

zeroAgreeFrameRelative = zeroAgreeFrame.apply(lambda x: x/x.iloc[0])

order=zeroAgreeFrame.loc[:,zeroAgreeFrame.columns!='Total'].columns.to_list()

zeroAgreeRatioFrame=zeroAgreeRatioFrame[order]

writer=pd.ExcelWriter("E:\AnalystPersonal\chenzihan\每日监控\daily_report_自有.xlsx")

zeroAgreeFrame.to_excel(excel_writer = writer, sheet_name = '通过率监控')
zeroAgreeFrameRelative.to_excel(excel_writer = writer, sheet_name = '通过率监控',startrow=12, startcol=0,float_format='%.4f')
zeroAgreeRatioFrame.to_excel(excel_writer = writer, sheet_name = '通过率监控',startrow=24, startcol=0,float_format='%.4f')

writer.save()

writer.close()


'''
##get the overview of the point-card
codelist=tempframe.utmcodeNew.unique().tolist()
zeroQuantile=pd.DataFrame()
for i in codelist:
    temp=tempframe[tempframe.utmcodeNew==i].copy()
    zeroQuantile[i]=temp.Score_Test.dropna().astype(float).quantile(quantilelist)
'''

##get the overview of some specific product

########################################
#审批流监控：历史日期
#######################################


tempframe=zeroUnion[(zeroUnion.intoTime.between(*detaildatelist))&(zeroUnion['product'].isin(['1001','1002','1004']))].copy()
    #tempframe=zeroUnion[(zeroUnion.intoTime>='2019-01-31 17:30')&(zeroUnion.utmcodeNew=='wacai')].copy()
tempframe['intodate']=tempframe.apply(lambda x:x.intoTime[:10],axis=1)
datelist=tempframe.intodate.unique().tolist()
_,zeroDateAgree,zeroDateAgreeRatio=refuseAnalysis(tempframe,groupbyname='intodate')
zeroDateAgree['Total']=zeroDateAgree.sum(axis=1)
  
zeroDateAgree=zeroDateAgree.reindex(['益博睿预排除','三方','反欺诈人工','益博睿','人工信审结束','人工信审通过','益博睿直通','通过','反欺诈待定','信审待定'])

zeroDateAgreeRatio.loc['益博睿预排除','Total'] = zeroDateAgree.loc['三方','Total']/zeroDateAgree.loc['益博睿预排除','Total']
zeroDateAgreeRatio.loc['三方','Total'] = zeroDateAgree.loc['反欺诈人工','Total']/zeroDateAgree.loc['三方','Total']
zeroDateAgreeRatio.loc['反欺诈人工','Total'] = zeroDateAgree.loc['益博睿','Total']/zeroDateAgree.loc['反欺诈人工','Total']
zeroDateAgreeRatio.loc['益博睿','Total'] = (zeroDateAgree.loc['人工信审结束','Total']+zeroDateAgree.loc['信审待定','Total']+
										zeroDateAgree.loc['益博睿直通','Total'])/zeroDateAgree.loc['益博睿','Total']
if zeroDateAgree.loc['人工信审结束','Total']==0:
    zeroDateAgreeRatio.loc['人工信审结束','Total']=0
else:
    zeroDateAgreeRatio.loc['人工信审结束','Total'] = zeroDateAgree.loc['人工信审通过','Total']/zeroDateAgree.loc['人工信审结束','Total']

zeroDateFrameRelative=pd.DataFrame(index=zeroDateAgree.index)
zeroDateFrameRelative = zeroDateAgree.apply(lambda x: x/x.iloc[0])
zeroAgreeFrame_temp=pd.concat([zeroDateAgree,zeroDateFrameRelative,zeroDateAgreeRatio], axis=0,sort=False)

writer=pd.ExcelWriter("E:\AnalystPersonal\chenzihan\每日监控\history_report.xlsx")

zeroAgreeFrame_temp.to_excel(excel_writer = writer, sheet_name = 'API+自有',float_format='%.4f')

for i in zeroUnion.utmcodeNew.unique():
    tempframe=zeroUnion[(zeroUnion.intoTime.between(*detaildatelist))&(zeroUnion.utmcodeNew==i)&(zeroUnion['product'].isin(['1001','1002','1004']))].copy()
    #tempframe=zeroUnion[(zeroUnion.intoTime>='2019-01-31 17:30')&(zeroUnion.utmcodeNew=='wacai')].copy()
    tempframe['intodate']=tempframe.apply(lambda x:x.intoTime[:10],axis=1)
    datelist=tempframe.intodate.unique().tolist()
    _,zeroDateAgree,zeroDateAgreeRatio=refuseAnalysis(tempframe,groupbyname='intodate')
    zeroDateAgree['Total']=zeroDateAgree.sum(axis=1)
    zeroDateAgree=zeroDateAgree.reindex(['益博睿预排除','三方','反欺诈人工','益博睿','人工信审结束','人工信审通过','益博睿直通','通过','反欺诈待定','信审待定'])
    
    zeroDateAgreeRatio.loc['益博睿预排除','Total'] = zeroDateAgree.loc['三方','Total']/zeroDateAgree.loc['益博睿预排除','Total']
    zeroDateAgreeRatio.loc['三方','Total'] = zeroDateAgree.loc['反欺诈人工','Total']/zeroDateAgree.loc['三方','Total']
    zeroDateAgreeRatio.loc['反欺诈人工','Total'] = zeroDateAgree.loc['益博睿','Total']/zeroDateAgree.loc['反欺诈人工','Total']
    zeroDateAgreeRatio.loc['益博睿','Total'] = (zeroDateAgree.loc['人工信审结束','Total']+zeroDateAgree.loc['信审待定','Total']+
    										zeroDateAgree.loc['益博睿直通','Total'])/zeroDateAgree.loc['益博睿','Total']
    if zeroDateAgree.loc['人工信审结束','Total']==0:
        zeroDateAgreeRatio.loc['人工信审结束','Total']=0
    else:
        zeroDateAgreeRatio.loc['人工信审结束','Total'] = zeroDateAgree.loc['人工信审通过','Total']/zeroDateAgree.loc['人工信审结束','Total']
    
    zeroDateFrameRelative=pd.DataFrame(index=zeroDateAgree.index)
    zeroDateFrameRelative = zeroDateAgree.apply(lambda x: x/x.iloc[0])
    zeroAgreeFrame_temp=pd.concat([zeroDateAgree,zeroDateFrameRelative,zeroDateAgreeRatio], axis=0,sort=False)
    zeroAgreeFrame_temp.to_excel(excel_writer = writer, sheet_name = i,float_format='%.4f')
    exec('zeroAgreeFrame{}=zeroAgreeFrame.copy()'.format(i))
    
writer.save()
writer.close()

#子涵定制，用自定义渠道名称生成历史数据
tempframe=zeroUnion[(zeroUnion.intoTime.between(*detaildatelist))&(zeroUnion.utmcodeNew==detailutmcode)&(zeroUnion['product'].isin(['1001','1002','1004']))].copy()
#tempframe=zeroUnion[(zeroUnion.intoTime>='2019-01-31 17:30')&(zeroUnion.utmcodeNew=='wacai')].copy()
tempframe['intodate']=tempframe.apply(lambda x:x.intoTime[:10],axis=1)
datelist=tempframe.intodate.unique().tolist()
_,zeroDateAgree,zeroDateAgreeRatio=refuseAnalysis(tempframe,groupbyname='intodate')
zeroDateAgree['Total']=zeroDateAgree.sum(axis=1)


################################
####额度监控(整体)
################################
tempframe=zeroUnion[(zeroUnion.intoTime.between(*amountdetaildatelist))&(zeroUnion['review_result']=='通过')].copy()

tempframe['intodate']=tempframe.apply(lambda x:x.intoTime[:10],axis=1)

tempframe['reviewdatetime']=tempframe['revive_time'].astype(str)
tempframe['reviewdate']=tempframe.apply(lambda x:x.reviewdatetime[:10],axis=1)

tempframe['output_amount'] = pd.to_numeric(tempframe['output_amount'])
tempframe['review_amount'] = pd.to_numeric(tempframe['agreeamount'])



#按进件时间统计平均额度
num_agg = {'intoId':['count'], 'output_amount':['mean'], 'review_amount':['mean']}
amountIntoDate = tempframe.groupby(['utmcodePlus','intodate']).agg(num_agg)
amountReviewDate = tempframe.groupby(['utmcodePlus','reviewdate']).agg(num_agg)

amountIntoDate['amount_gap'] = amountIntoDate.apply(lambda x: x.loc[('output_amount','mean')] - x.loc[('review_amount','mean')],axis=1)


amountReviewDate['amount_gap'] = amountReviewDate.apply(lambda x: x.loc[('output_amount','mean')] - x.loc[('review_amount','mean')],axis=1)

writer=pd.ExcelWriter("E:\AnalystPersonal\chenzihan\每日监控\额度监控.xlsx")

amountIntoDate.to_excel(excel_writer = writer, sheet_name = 'API+自有整体',float_format='%.4f')

amountReviewDate.to_excel(excel_writer = writer, sheet_name = 'API+自有整体', startrow=0, startcol=7,float_format='%.4f')


################################
#####额度监控(自有明细)
################################
tempframe=zeroUnion[(zeroUnion.intoTime.between(*amountdetaildatelist))&(zeroUnion['review_result']=='通过')&(zeroUnion['utmcodeNew']=='ourself')].copy()

tempframe['intodate']=tempframe.apply(lambda x:x.intoTime[:10],axis=1)

tempframe['reviewdatetime']=tempframe['revive_time'].astype(str)
tempframe['reviewdate']=tempframe.apply(lambda x:x.reviewdatetime[:10],axis=1)

tempframe['output_amount'] = pd.to_numeric(tempframe['output_amount'])
tempframe['review_amount'] = pd.to_numeric(tempframe['agreeamount'])


#按进件时间统计平均额度
num_agg = {'intoId':['count'], 'output_amount':['mean'], 'review_amount':['mean']}
amountIntoDate = tempframe.groupby(['intochannelname','intodate']).agg(num_agg)
amountReviewDate = tempframe.groupby(['intochannelname','reviewdate']).agg(num_agg)

amountIntoDate['amount_gap'] = amountIntoDate.apply(lambda x: x.loc[('output_amount','mean')] - x.loc[('review_amount','mean')],axis=1)


amountReviewDate['amount_gap'] = amountReviewDate.apply(lambda x: x.loc[('output_amount','mean')] - x.loc[('review_amount','mean')],axis=1)

amountIntoDate.to_excel(excel_writer = writer, sheet_name = '自有明细',float_format='%.4f')

amountReviewDate.to_excel(excel_writer = writer, sheet_name = '自有明细', startrow=0, startcol=7,float_format='%.4f')

writer.save()
writer.close()


#####################################################
##################拒贷原因分布
#####################################################
#初审
tempframe=zeroUnion[(zeroUnion.intoTime.between(*showdatelist))].copy()
tempframe['intodate']=tempframe.apply(lambda x:x.intoTime[:10],axis=1)
tempframe=tempframe[(tempframe.pre_output_dclnreason!="")].copy()

zeroDateStatic=pd.DataFrame(index=tempframe.pre_output_dclnreason.dropna().unique())
tempDate=pd.DataFrame(index=tempframe.pre_output_dclnreason.dropna().unique())
tempRatio=pd.DataFrame(index=tempframe.pre_output_dclnreason.dropna().unique())
for groupname,group in tempframe.groupby('intodate'):
    tempDate[groupname]=group.groupby('pre_output_dclnreason')['intoId'].count()
    tempRatio[groupname]=group.groupby('pre_output_dclnreason')['intoId'].count()/len(group[group.pre_output_dclnreason.isna()==False])

tempDate=tempDate.sort_values(by = tempDate.columns[tempDate.shape[1]-1], ascending=False, axis=0) 

tempRatio=tempRatio.reindex(tempDate.index)

tempDate['type']='count'
tempDate['reason']=tempDate.index
tempRatio['type']='ratio'
tempRatio['reason']=tempRatio.index
zeroDateStatic=pd.concat([tempDate,tempRatio],axis=0)
zeroDateStatic=zeroDateStatic.set_index(['type','reason'])

writer=pd.ExcelWriter("E:\AnalystPersonal\chenzihan\每日监控\拒贷原因监控.xlsx")

zeroDateStatic.to_excel(excel_writer = writer, sheet_name = '初审_时间段（整体）',float_format='%.4f')

#终审
tempframe=zeroUnion[(zeroUnion.intoTime.between(*showdatelist))].copy()
tempframe['intodate']=tempframe.apply(lambda x:x.intoTime[:10],axis=1)
tempframe=tempframe[(tempframe.final_output_dclnreason!="")].copy()

zeroDateStatic=pd.DataFrame(index=tempframe.final_output_dclnreason.dropna().unique())
tempDate=pd.DataFrame(index=tempframe.final_output_dclnreason.dropna().unique())
tempRatio=pd.DataFrame(index=tempframe.final_output_dclnreason.dropna().unique())
for groupname,group in tempframe.groupby('intodate'):
    tempDate[groupname]=group.groupby('final_output_dclnreason')['intoId'].count()
    tempRatio[groupname]=group.groupby('final_output_dclnreason')['intoId'].count()/len(group[group.final_output_dclnreason.isna()==False])

tempDate=tempDate.sort_values(by = tempDate.columns[tempDate.shape[1]-1], ascending=False, axis=0) 

tempRatio=tempRatio.reindex(tempDate.index)

tempDate['type']='count'
tempDate['reason']=tempDate.index
tempRatio['type']='ratio'
tempRatio['reason']=tempRatio.index
zeroDateStatic=pd.concat([tempDate,tempRatio],axis=0)
zeroDateStatic=zeroDateStatic.set_index(['type','reason'])

#tempframe.to_excel(excel_writer = writer, sheet_name = '终审拒贷明细')

zeroDateStatic.to_excel(excel_writer = writer, sheet_name = '终审_时间段（整体）', startrow=0, startcol=0,float_format='%.4f')


#初审(当天，整体，分渠道)
tempframe=zeroUnion[(zeroUnion.intoTime.between(*showdatelist))].copy()
tempframe['intodate']=tempframe.apply(lambda x:x.intoTime[:10],axis=1)
tempframe=tempframe[(tempframe.pre_output_dclnreason!="")].copy()

zeroDateStatic=pd.DataFrame(index=tempframe.pre_output_dclnreason.dropna().unique())
tempDate=pd.DataFrame(index=tempframe.pre_output_dclnreason.dropna().unique())
tempRatio=pd.DataFrame(index=tempframe.pre_output_dclnreason.dropna().unique())
for groupname,group in tempframe.groupby('utmcodePlus'):
    tempDate[groupname]=group.groupby('pre_output_dclnreason')['intoId'].count()
    tempRatio[groupname]=group.groupby('pre_output_dclnreason')['intoId'].count()/len(group[group.pre_output_dclnreason.isna()==False])

#调整列的顺序
temp = tempDate.ourself
column_number=tempDate.shape[1]
tempDate = tempDate.drop('ourself',axis=1)
tempDate.insert(column_number-1,'ourself',temp)
del temp

#调整列的顺序
temp = tempRatio.ourself
column_number=tempRatio.shape[1]
tempRatio = tempRatio.drop('ourself',axis=1)
tempRatio.insert(column_number-1,'ourself',temp)
del temp

tempDate=tempDate.sort_values(by = tempDate.columns[tempDate.shape[1]-1], ascending=False, axis=0) 

tempRatio=tempRatio.reindex(tempDate.index)   
    
tempDate['type']='count'
tempDate['reason']=tempDate.index
tempRatio['type']='ratio'
tempRatio['reason']=tempRatio.index
zeroDateStatic=pd.concat([tempDate,tempRatio],axis=0)
zeroDateStatic=zeroDateStatic.set_index(['type','reason'])


zeroDateStatic.to_excel(excel_writer = writer, sheet_name = '初审_当天_API+自有整体',float_format='%.4f')



#终审(当天，整体，分渠道)
tempframe=zeroUnion[(zeroUnion.intoTime.between(*showdatelist))].copy()
tempframe['intodate']=tempframe.apply(lambda x:x.intoTime[:10],axis=1)
tempframe=tempframe[(tempframe.final_output_dclnreason!="")].copy()

zeroDateStatic=pd.DataFrame(index=tempframe.final_output_dclnreason.dropna().unique())
tempDate=pd.DataFrame(index=tempframe.final_output_dclnreason.dropna().unique())
tempRatio=pd.DataFrame(index=tempframe.final_output_dclnreason.dropna().unique())
for groupname,group in tempframe.groupby('utmcodePlus'):
    tempDate[groupname]=group.groupby('final_output_dclnreason')['intoId'].count()
    tempRatio[groupname]=group.groupby('final_output_dclnreason')['intoId'].count()/len(group[group.final_output_dclnreason.isna()==False])

#调整列的顺序
temp = tempDate.ourself
column_number=tempDate.shape[1]
tempDate = tempDate.drop('ourself',axis=1)
tempDate.insert(column_number-1,'ourself',temp)
del temp

#调整列的顺序
temp = tempRatio.ourself
column_number=tempRatio.shape[1]
tempRatio = tempRatio.drop('ourself',axis=1)
tempRatio.insert(column_number-1,'ourself',temp)
del temp

tempDate=tempDate.sort_values(by = tempDate.columns[tempDate.shape[1]-1], ascending=False, axis=0) 

tempRatio=tempRatio.reindex(tempDate.index)
    
tempDate['type']='count'
tempDate['reason']=tempDate.index
tempRatio['type']='ratio'
tempRatio['reason']=tempRatio.index
zeroDateStatic=pd.concat([tempDate,tempRatio],axis=0)
zeroDateStatic=zeroDateStatic.set_index(['type','reason'])

zeroDateStatic.to_excel(excel_writer = writer, sheet_name = '终审_当天_API+自有整体', startrow=0, startcol=0,float_format='%.4f')


#初审(当天，自有，分渠道)
tempframe=zeroUnion[(zeroUnion.intoTime.between(*showdatelist))&(zeroUnion.utmcodeNew=='ourself')].copy()
tempframe['intodate']=tempframe.apply(lambda x:x.intoTime[:10],axis=1)
tempframe=tempframe[(tempframe.pre_output_dclnreason!="")].copy()

zeroDateStatic=pd.DataFrame(index=tempframe.pre_output_dclnreason.dropna().unique())
tempDate=pd.DataFrame(index=tempframe.pre_output_dclnreason.dropna().unique())
tempRatio=pd.DataFrame(index=tempframe.pre_output_dclnreason.dropna().unique())
for groupname,group in tempframe.groupby('intochannelname'):
    tempDate[groupname]=group.groupby('pre_output_dclnreason')['intoId'].count()
    tempRatio[groupname]=group.groupby('pre_output_dclnreason')['intoId'].count()/len(group[group.pre_output_dclnreason.isna()==False])

tempDate['Total']=tempDate.sum(axis=1,skipna =True)
tempDate=tempDate.sort_values(by = tempDate.columns[tempDate.shape[1]-1], ascending=False, axis=0) 
tempDate.loc['Total',:]=tempDate.sum(axis=0,skipna =True)

tempDate=tempDate.sort_values(by = tempDate.index[0], ascending=False, axis=1)

 #调整列的顺序
temp = tempDate.Total
column_number=tempDate.shape[1]
tempDate = tempDate.drop('Total',axis=1)
tempDate.insert(column_number-1,'Total',temp)
del temp
 

tempRatio=tempRatio.reindex(tempDate.index) 

order=tempDate.loc[:,tempDate.columns!='Total'].columns.to_list()

tempRatio=tempRatio[order]

tempRatio.loc['Total',:]= 1

tempRatio['Total'] = tempDate['Total'].map(lambda x:x/tempDate.loc['Total','Total'])
  
    
tempDate['type']='count'
tempDate['reason']=tempDate.index
tempRatio['type']='ratio'
tempRatio['reason']=tempRatio.index
zeroDateStatic=pd.concat([tempDate,tempRatio],axis=0)
zeroDateStatic=zeroDateStatic.set_index(['type','reason'])



zeroDateStatic.to_excel(excel_writer = writer, sheet_name = '初审_自有渠道',float_format='%.4f')



#终审
tempframe=zeroUnion[(zeroUnion.intoTime.between(*showdatelist))&(zeroUnion.utmcodeNew=='ourself')].copy()
tempframe['intodate']=tempframe.apply(lambda x:x.intoTime[:10],axis=1)
tempframe=tempframe[(tempframe.final_output_dclnreason!="")].copy()

zeroDateStatic=pd.DataFrame(index=tempframe.final_output_dclnreason.dropna().unique())
tempDate=pd.DataFrame(index=tempframe.final_output_dclnreason.dropna().unique())
tempRatio=pd.DataFrame(index=tempframe.final_output_dclnreason.dropna().unique())
for groupname,group in tempframe.groupby('intochannelname'):
    tempDate[groupname]=group.groupby('final_output_dclnreason')['intoId'].count()
    tempRatio[groupname]=group.groupby('final_output_dclnreason')['intoId'].count()/len(group[group.final_output_dclnreason.isna()==False])


tempDate['Total']=tempDate.sum(axis=1,skipna =True)
tempDate=tempDate.sort_values(by = tempDate.columns[tempDate.shape[1]-1], ascending=False, axis=0) 
tempDate.loc['Total',:]=tempDate.sum(axis=0,skipna =True)

tempDate=tempDate.sort_values(by = tempDate.index[0], ascending=False, axis=1)

 #调整列的顺序
temp = tempDate.Total
column_number=tempDate.shape[1]
tempDate = tempDate.drop('Total',axis=1)
tempDate.insert(column_number-1,'Total',temp)
del temp
 

tempRatio=tempRatio.reindex(tempDate.index) 

order=tempDate.loc[:,tempDate.columns!='Total'].columns.to_list()

tempRatio=tempRatio[order]

tempRatio.loc['Total',:]= 1

tempRatio['Total'] = tempDate['Total'].map(lambda x:x/tempDate.loc['Total','Total'])
    
tempDate['type']='count'
tempDate['reason']=tempDate.index
tempRatio['type']='ratio'
tempRatio['reason']=tempRatio.index
zeroDateStatic=pd.concat([tempDate,tempRatio],axis=0)
zeroDateStatic=zeroDateStatic.set_index(['type','reason'])

zeroDateStatic.to_excel(excel_writer = writer, sheet_name = '终审 _自有渠道', startrow=0, startcol=0,float_format='%.4f')

writer.save()
writer.close()




















