
credit_product_type

import pandas as pd
import numpy as np
dates = pd.date_range('1/1/2000', periods=8)
# 建立一个时间序列，从20000101开始，8个数据
df = pd.DataFrame(np.random.randn(8, 4),index=dates, columns=['A', 'B', 'C', 'D'])
# 随机生成8*4的数组，数据的索引设置为dates,列名为['A', 'B', 'C', 'D']
s = df['A']
# 取出df中名字为”A“的那一列
df.loc[:, ['B', 'A']] = df[['A', 'B']].to_numpy()
# 数据集中的两列数据交换名字
df[:3]
# 取出数据集中的前3行
df.loc['2000-01-01':'2000-01-04']
# 通过索引选择行
df.loc ['2000-01-01':]
# 选择索引“2000-01-01”之后全部的数据

df1 = pd.DataFrame(np.random.randn(6,4),index = list('abcdef'),columns = list('ABCD'))
#生成一个数据集
df1.loc['d':, 'A':'C']
# 选择索引是“d"之后，以及列为”A"到“B"的列
df1.loc['a']
# 选择索引为a的哪一行
df['A'][0] = 111
# 替换数据集中的值

df1.A.loc[lambda s: s > 0]
# 数据集中A列大于0的值取出来
df[df['A'] > 0]
# 筛选数据集中A大于0 的数据

values = ['a', 'b', 1, 3]
df.isin(values)
# 判断df中的值是否在value这个list里面，value可以是list也可以是json

df.where(df < 0, -df)
# 将数据集中的小于0的数据取出来之后，这些值不改变，那些不满足小于0的加上负号
 df_orig.where(df > 0, -df, inplace=True)
# 将数据集中的小于0的数据取出来之后，这些值不改变，那些不满足小于0的加上负号，并且更改到原始数据集中
 df1.where(lambda x: x > 4, lambda x: x + 10)
# 将数据集中的大于4的数据取出来之后，这些值不改变，那些不满足小于0，加上10
 df[(df.a < df.b) & (df.b < df.c)]
# 取出数据集中，b大于a c大于b的行，a b c这里是列名
df.query('(a < b) & (b < c)')
# 取出数据集中，b大于a c大于b的行，a b c这里是列名
df.index.name = 'a'
# 给与数据的索引加上名字
df[df.a.isin(df.b)]
# 筛选出数据集a中有，b中也有的值
 df[~df.a.isin(df.b)]
# 筛选出数据集a中有，b中没有的值
df.query('a not in b')
# 筛选出数据集a中有，b中没有的值
df[df.b.isin(df.a) & (df.c < df.d)]
# 筛选出数据集a中有，b中也有的值，并且d大于c的数据，这里的abcd代表的是列名
df.query('year > 2012 | name == "Frank"')
# 筛选出数据集中year>2012或者name=Frank的数据
df2.drop_duplicates('a')
# 去重，默认保留重复中第一条
df2.drop_duplicates('a', keep='last')
# 去重，保留重复中最后一条
df2.drop_duplicates('a', keep=False)
# 数据集保留a列中没有重复部分
df2.drop_duplicates(['a', 'b'])
# 两列去重



data.set_index('c')
# 令数据集中的c列为索引
data.set_index(['a', 'b'])
# 令数据集中的a,b列为索引
data.set_index('c', drop=False)
# 令数据集中的c列为索引,删掉原先的索引
data.reset_index(drop=True)
# 重置索引,原先的索引删除
data.reset_index(drop=False)
# 重置索引,原先的变成数据集的列





pd.concat(df1, df2, df3)
# 横向合并数据集，相同列部分自动追加数据集后面
pd.concat([df1, df4], axis=1, join='inner')
# 合并之后只取相同部分，合并以索引为主键   
df1.append(df4, sort=False)
# 追加之后不排序，默认是不排序
df1.append([df2, df3])
# 同时追加两个数据集
df1.append(df4, ignore_index=True, sort=False)
# 忽略原先数据集的索引，重新建立索引

pd.merge(left, right, how='inner', on=None, left_on=None, right_on=None,left_index=False, right_index=False, sort=True,
         suffixes=('_x', '_y'), copy=True, indicator=False,validate=None)
pd.merge(left, right, on='key')
# 通过两个数据集共同的列“key"匹配，默认内交
pd.merge(left, right, on=['key1', 'key2'])
# 通过两个数据集共同的列“'key1', 'key2'"匹配，默认内交
pd.merge(left, right, how='left', on=['key1', 'key2'])
# 左交，通过两个数据集共同的列“'key1', 'key2'"匹配
pd.merge(left, right, how='right', on=['key1', 'key2'])
# 右交，通过两个数据集共同的列“'key1', 'key2'"匹配
pd.merge(left, right, how='outer', on=['key1', 'key2'])
# 全连接，通过两个数据集共同的列“'key1', 'key2'"匹配
pd.merge(left, right, left_index=True, right_index=True, how='outer')
# 通过索引全连接
pd.merge(left, right, on='k', suffixes=['_l', '_r'])
# 在链接过程中出现相同列名，可在suffixes定义后缀以区分





df2['one'] == np.nan
# 判断数据集“one”列中为空，反馈布尔序列
df2.dtypes.value_counts（）
# 对数据集中的变量类型做统计
dff.fillna（dff.mean（））
# 给数据集中缺失的值填上平均值
df.dropna（axis = 0）
# 数据集变量中有缺失哪一行删掉
df.dropna（axis = 1）
# 数据集变量中有缺失哪一列删掉
df.replace(1.5, np.nan)
# 数据集中1.5的数值替换成空值
df.isnull()
# 返回数据集值对应位置是否为空
df[df['year'].notnull()]
# 筛选出数据集中year变量不为空的数据
df.isna().sum()
# 计算每一列空值的数量







 df_cat.dtypes
#     查看数据集中个列的类型
df.astype('category')
# 转化数据集中列的类型
dfs.sort_values(by=['A', 'B'])
# 数据集按照A B两列排序
df['A'].value_counts()
# 计算数据集中各个类别的数量
df.groupby("cats").mean()
# 计算数据集中“cats"中每个类别的平均值
df.shape
# 数据集的行列数量
df.columns
# 数据集中的列名
df.name.unique()
# 变量name的唯一值

 df.sum（）
# 数据集求和
 df.min（）
# 数据集最小值
 df.max（）
# 数据集最大值
df.describe（）
# DataFrame的统计摘要，包括四分位数，中位数等
df.mean（）
# 平均值
 df.median（）
# 中间值
df.applymap(np.sqrt)
# 返回数据集全部的数值的平方根

df.head()
# 显示数据集的前5行
df.tail（）
# 显示数据集的后5行

df_renamed = df.rename(columns = {'Id'          : 'TransactionId', 
                                                 'MSSubClass'  : 'BuildingClass',
                                                 'OverallQual' : 'OverallQuality'} 
)
# 数据集重命名


pd.read_csv("nba.csv", index_col ="Name")
# 导入数据集并且定义“name"为索引
pd.to_csv(name_of_the_file_to_save.csv')
# 导出数据集
xlsx = pd.ExcelFile('your_excel_file.xlsx')
df = pd.read_excel(xlsx, 'Sheet 1')
# 导入excel格式数据集



     
df.drop('Country', axis = 1)
# 删除Country这一列
df.drop('Country', axis = 0)
# 删除Country为索引的这一行
del df ['name']
# 删除列‘name'