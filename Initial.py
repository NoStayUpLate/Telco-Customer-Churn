import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import plotly.graph_objs as pltgo
import plotly.offline as pltoff
from sklearn import preprocessing
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, explained_variance_score, \
    median_absolute_error
from sklearn.linear_model import ElasticNetCV,RidgeCV,LassoCV
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import cross_val_score,GridSearchCV
from sklearn import model_selection


# 过滤warings
warnings.filterwarnings('ignore')

# 导入数据
data_train = pd.read_csv("./Data.csv")
y_train = data_train.Churn
# inplace=False，默认该删除操作不改变原数据，而是返回一个执行删除操作后的新dataframe；
# inplace=True，则会直接在原数据上进行删除操作，删除后无法返回。
data_test = data_train.drop(['Churn'],axis = 1,inplace = False)
# 检验一下删除后的 data_train 是否有 21列 数据，data_test 是否有 20列 数据
# print(data_test.info())
# print(data_train.info())

# 利用热力图观察数据之间的相关性
# plt.figure(figsize=(100,100))
# sns.heatmap(train.corr())
# plt.show()

'''
    第一部分：观察数据
'''
# todo 观察有无缺损
# print(data_train.head(10))
# print("data_shape:",data_train.shape)
# print("data_info:",data_train.info())
# 观察发现没有空字符串

font = {
    'family' : 'SimHei',
    'weight' : 'bold'
}
matplotlib.rc("font",**font)

def Percentage(data_label,data_values_list,flag):
    '''
    计算各指标所占百分比
    :param data_label:需要计算的Series
    :param data_values_list: 各指标的数量
    :return: 指标数量和占比的字符串 组成的列表
    '''
    total_value = len(data_label) if flag == 'part' else len(data_train)
    percent_str = []
    for data_value in data_values_list:
        data_value_percent = data_value / total_value
        percent = round(data_value_percent, 3)
        str_total = str(data_value) + "(占比" + str(percent * 100) + "%" + ")"
        percent_str.append(str_total)
    return percent_str

def Churn_yes_bar(data_label,title):
    '''
    观察 流失客户 中的主要因素
    :param data_label:需要绘图的Series
    :param title: 绘制图的名称
    :return: 绘制的图
    '''
    data = data_label[data_train.Churn == "Yes"].value_counts()
    plt.bar(data.index,data.values)
    # 计算百分比
    num_percent_list = Percentage(data_label[data_train.Churn == "Yes"],data.values,'part')
    Indexlen = np.arange(len(data.index))
    # 绘制百分比数据
    for x,y in zip(data.index,Indexlen):
        plt.text(x,data.values[y],s = num_percent_list[y],ha = 'center')
    plt.title(title)
    plt.ylabel("客户数量")
    plt.show()

def Churn_doublebar(data_label,title):
    '''
    若某个因素并非 流失客户 中的因素
    通过这个函数
    对比观察 非流失客户 与 流失客户 的主要因素
    :param data_label: 需要绘图的Series
    :param title: 绘制图的名称
    :return: 绘制的图
    '''
    x = np.arange(len(data_label.value_counts().index))
    data_noChurn = data_label[data_train.Churn == "No"].value_counts()
    data_yesChurn = data_label[data_train.Churn == "Yes"].value_counts()
    y_No = list(data_noChurn.values)
    y_Yes = list(data_yesChurn.values)
    bar_width = 0.35
    x_label = list(data_label.value_counts().index)
    plt.bar(x,y_No,bar_width,align = 'center',label = "未流失客户")
    # 计算百分比
    NoNumPercentlist = Percentage(data_label[data_train.Churn == "No"],data_noChurn.values,'whole')
    YesNumPercentlist = Percentage(data_label[data_train.Churn == "Yes"],data_yesChurn.values,'whole')
    Nolen = np.arange(len(data_noChurn.index))
    Yeslen = np.arange(len(data_noChurn.index))
    # 绘制百分比数据
    for i,j in zip(Nolen,Nolen):
        plt.text(i,y_No[j],s = NoNumPercentlist[j],ha = 'center')
    plt.bar(x + bar_width,y_Yes,bar_width,align = 'center',label = "流失客户")
    for i,j in zip(Yeslen,Yeslen):
        plt.text(i + bar_width,y_Yes[j],s = YesNumPercentlist[j],ha = 'center')
    plt.xlabel(title)
    plt.ylabel("客户数量")
    plt.xticks(x + bar_width/2,x_label)
    plt.legend(loc = 0)
    plt.show()

# todo 观察gender的分布：性别分布均匀，客户流失 性别并不是主要影响因素
# todo 男性：930 女性：939
# todo 观察整体分布，男女比例
# Churn_yes_bar(data_train.gender,"性别")
# Churn_doublebar(data_train.gender,"性别")
# todo SeniorCitizen：老年人可能 客户流失 主要影响因素
# todo 老年人占比：476 / 1869 = 25.5%
# Churn_yes_bar(data_train.SeniorCitizen,"老龄")
# todo PhoneService的有无，可能是 客户流失 主要影响因素
# todo 有：1699 无：170 大多数的 流失客户 有 电话服务
# Churn_yes_bar(data_train.PhoneService,"电话服务")
# todo 有PhoneService的人是否有MultipleLines：人数比列接近，并非主要因素，主要因素为有没有电话服务
# todo ('No', 849) ('Yes', 850)
# Churn_yes_bar(data_train.MultipleLines[data_train.PhoneService == "Yes"],"是否有多条电话服务线路")
# Churn_doublebar(data_train.MultipleLines[data_train.PhoneService == "Yes"],"是否有多条电话服务线路")
# todo Partner 合伙人：接近两倍，可能为主要因素
# todo ('No', 1200) ('Yes', 669)
# Churn_yes_bar(data_train.Partner,"合伙人")
# todo Dependents 家属：可能是主要影响因素
# todo ('No', 1543) ('Yes', 326)
# Churn_yes_bar(data_train.Dependents,"家属")
# todo tenure 大体上来看，客户使用时长 与 客户流失数量 成反比
# todo 需要处理数据，将数据分为
# Churn_yes_bar(data_train.tenure,"使用时长")
# todo InternetService 分为三类，其中fiber optic光纤最为突出 1297 DSL：459，No：113
# Churn_yes_bar(data_train.InternetService,"网络服务")
# todo OnlineSecurity No的比例占了很大一部分，可能为主要因素
# Churn_yes_bar(data_train.OnlineSecurity,"网络安全")
# todo OnlineBackup No的比例占了很大一部分，可能为主要因素
# Churn_yes_bar(data_train.OnlineBackup,"网络备份")
# todo DeviceProtection No的比例占了很大一部分，可能为主要因素
# Churn_yes_bar(data_train.DeviceProtection,"设备保护")
# todo TechSupport No的比例占了很大一部分，可能为主要因素
# Churn_yes_bar(data_train.TechSupport,"技术支持")
# todo StreamingTV 有和没有差距不大 No：942 Yes：814 No interservice:113
# todo 从整体来看，未流失客户 远大于 流失客户，有可能是 促成客户未流失 的原因
# Churn_yes_bar(data_train.StreamingTV,"流媒体电视")
# Churn_doublebar(data_train.StreamingTV,"流媒体电视")
# todo StreamingMovies 有和没有差距不大 No：942 Yes：814 No interservice:113
# todo 从整体来看，未流失客户 远大于 流失客户，有可能是 促成客户未流失 的原因
# Churn_yes_bar(data_train.StreamingMovies,"流媒体电影")
# Churn_doublebar(data_train.StreamingMovies,"流媒体电影")
# todo Contract 每月：1655 每年：166 每两年：48
# Churn_yes_bar(data_train.Contract,"合同")
# todo PaperlessBilling Yes：1400 No：469
# Churn_yes_bar(data_train.PaperlessBilling,"无纸化账单")
# todo PaymentMethod Electronic Check最多：1071
# Churn_yes_bar(data_train.PaymentMethod,"支付方式")
# todo MonthlyCharges 分散众多，主要集中在70-110之间，需要进一步分析
# Churn_yes_bar(data_train.MonthlyCharges,"每月费用")
# todo TotalCharges 几乎持平，非主要因素
# Churn_yes_bar(data_train.TotalCharges,"总共费用")
# Churn_doublebar(data_train.TotalCharges,"总共费用")

# 单独处理tenure
# 将Tenure变量的值转变为分类值
def tenure_lab(data_train):
    if data_train['tenure'] <= 12:
        return 'Tenure_0_12'
    elif (data_train['tenure'] > 12) & (data_train['tenure'] <= 24):
        return 'Tenure_12_24'
    elif (data_train['tenure'] > 24) & (data_train['tenure'] <= 48):
        return 'Tenure_24_48'
    elif (data_train['tenure'] > 48) & (data_train['tenure'] <= 60):
        return 'Tenure_48_60'
    elif data_train['tenure'] > 60:
        return 'Tenure_gt_60'

data_train['tenure_group'] = data_train.apply(lambda data_train: tenure_lab(data_train), axis=1)

# 利用散点图来研究按照tenure分组的每月开支和总开支情况
def plot_tenure_scatter(tenure_group, color):
    tracer = pltgo.Scatter(x = data_train[data_train['tenure_group'] == tenure_group]['MonthlyCharges'],
                        y = data_train[data_train['tenure_group'] == tenure_group]['TotalCharges'],
                        mode = 'markers', marker = dict(line = dict(width = 0.2, color = 'black'),
                                                    size = 4, color = color, symbol = 'diamond-dot'),
                        name = tenure_group, # legend名称
                        opacity = 0.9
                        )
    return tracer

# 利用散点图研究按照churn分组的每月开支和总开支
def plot_churn_scatter(churn, color):
    tracer = pltgo.Scatter(x = data_train[data_train['Churn'] == churn]['MonthlyCharges'],	# 进提取已流失客户数据
                        y = data_train[data_train['Churn'] == churn]['TotalCharges'],	# 进提取已流失客户数据
                        mode = 'markers', marker = dict(line = dict(width = 0.2, color = 'black'),
                                                    size = 4, color = color, symbol = 'diamond-dot'),
                        name = 'Churn'+churn, # legend名称
                        opacity = 0.9)
    return tracer

trace1 = plot_tenure_scatter('Tenure_0_12', '#FF3300')
trace2 = plot_tenure_scatter('Tenure_12_24', '#6666FF')
trace3 = plot_tenure_scatter('Tenure_24_48', '#99FF00')
trace4 = plot_tenure_scatter('Tenure_48_60', '#996600')
trace5 = plot_tenure_scatter('Tenure_gt_60', 'grey')

trace6 = plot_churn_scatter('Yes', 'red')
trace7 = plot_churn_scatter('No', 'blue')

data1 = [trace1, trace2, trace3, trace4, trace5]
data2 = [trace7, trace6]

# 绘制画布
def layout_title(title):
    layout = pltgo.Layout(dict(
        title = title, plot_bgcolor = 'rgb(243, 243, 243)', paper_bgcolor = 'rgb(243, 243, 243)',
        xaxis = dict(gridcolor = 'rgb(255, 255, 255)', title = 'Monthly charges',zerolinewidth = 1, ticklen = 5, gridwidth = 2),
        yaxis = dict(gridcolor = 'rgb(255, 255, 255)', title = 'Total charges',zerolinewidth = 1, ticklen = 5, gridwidth = 2),
        height = 600
    ))
    return layout

layout1 = layout_title('Monthly Charges & Total Charges by Tenure group')
layout2 = layout_title('Monthly Charges & Total Charges by Churn group')

fig1 = pltgo.Figure(data = data1, layout = layout1)
fig2 = pltgo.Figure(data = data2, layout = layout2)

# pltoff.plot(fig1)
# pltoff.plot(fig2)

'''
    结论：
        可能为主要因素的指标：SeniorCitizen、PhoneService(Y/N)、Partner(Y/N)、Dependents(Y/N)、tenure（标准化处理）、InternetService（需要one-hot编码）、
                         OnlineSecurity(Y/N/NIS)、OnlineBackup(Y/N/NIS)、DeviceProtection(Y/N/NIS)、TechSupport(Y/N/NIS)、
                         Contract（需要one-hot编码）、PaperlessBilling(Y/N)、PaymentMethod（需要one-hot编码）、MonthlyCharges（标准化处理）
        可以去掉的指标：gender（去掉）、MultipleLines（去掉）、StreamingTV（不一定可以去掉）、StreamingMovies（不一定可以去掉）、TotalCharges(去掉)
'''

'''
    第二部分：数据清洗和处理
'''

# todo 处理含有 三个维度 得指标，例如：OnlineBackup：Yes、No、No internet service，把No internet service变成No
def NIS_to_No(data):
    '''
    将 No internet service 变成 No 便于后续处理
    :param data: 需要处理的Series
    :return: 处理完成的Series
    '''
    return data.str.replace('No internet service','No')

Changed_OnlineSecurity = NIS_to_No(data_train.OnlineSecurity)
Changed_OnlineBackup = NIS_to_No(data_train.OnlineBackup)
Changed_DeviceProtection = NIS_to_No(data_train.DeviceProtection)
Changed_TechSupport = NIS_to_No(data_train.TechSupport)
Changed_StreamingTV = NIS_to_No(data_train.StreamingTV)
Changed_StreamingMovies = NIS_to_No(data_train.StreamingMovies)

# todo 将有 少量维度 的指标进行one-hot编码
def Get_dummies(data,title):
    '''
    对所有字符串类的数据进行one-hot编码处理
    :param data: 需要处理的Series
    :param title: 编码后用来分类的列名称
    :return: 编码完成的Series
    '''
    return pd.get_dummies(data,prefix = title)

dummies_PhoneService = Get_dummies(data_train.PhoneService,'PhoneService')
dummies_Partner = Get_dummies(data_train.Partner,'Partner')
dummies_Dependents = Get_dummies(data_train.Dependents,'Dependents')
dummies_InternetService = Get_dummies(data_train.InternetService,'InternetService')
dummies_OnlineSecurity = Get_dummies(Changed_OnlineSecurity,'OnlineSecurity')
dummies_OnlineBackup = Get_dummies(Changed_OnlineBackup,'OnlineBackup')
dummies_DeviceProtection = Get_dummies(Changed_DeviceProtection,'DeviceProtection')
dummies_TechSupport = Get_dummies(Changed_TechSupport,'TechSupport')
dummies_Contract = Get_dummies(data_train.Contract,'Contract')
dummies_PaperlessBilling = Get_dummies(data_train.PaperlessBilling,'PaperlessBilling')
dummies_PaymentMethod = Get_dummies(data_train.PaymentMethod,'PaymentMethod')
dummies_StreamingTV = Get_dummies(Changed_StreamingTV,'StreamingTV')
dummies_StreamingMovies = Get_dummies(Changed_StreamingMovies,'StreamingMovies')

df_test = pd.concat([data_test,dummies_PhoneService,dummies_Partner,dummies_Dependents,dummies_InternetService,dummies_OnlineSecurity,
                     dummies_OnlineBackup,dummies_DeviceProtection,dummies_TechSupport,dummies_Contract,dummies_PaperlessBilling,
                     dummies_PaymentMethod,dummies_StreamingTV,dummies_StreamingMovies],axis = 1)

# todo 将有 大量维度 的指标尝试进行标准化处理，降低标准差
# 在新的sklearn库中，处理的数据要是二维数据，当输入的是一维数据的时候，需要用 reshape(-1,1) 将数据二维化
scaler = preprocessing.StandardScaler()
df_test['tenure_scaled'] = scaler.fit_transform(df_test.tenure.values.reshape(-1,1))
# print(df_test['tenure_scaled'].head())
df_test['MonthlyCharges_scaled'] = scaler.fit_transform(df_test.MonthlyCharges.values.reshape(-1,1))
# print(df_test['MonthlyCharges_scaled'].head())

df_test.drop(['gender','customerID','PhoneService','Partner','Dependents','InternetService','OnlineSecurity','OnlineBackup','DeviceProtection',
              'TechSupport','Contract','PaperlessBilling','PaymentMethod','StreamingTV','StreamingMovies',
              'TotalCharges','MultipleLines'],axis = 1,inplace = True)
# print(df_test.info())
df_test1 = df_test

# 利用热力图观察数据之间的相关性
corr_matrix = df_test.corr().abs()
# sns.heatmap(corr_matrix,cmap='spectral')
# plt.show()
# 把相关性较高的指标提出来,相关性高，说明相似，对用户流失得影响也可能相似,提高数据对于训练模型的契合度
# triu将矩阵里面的某几个主对角线的数据清零，在这里是用来取消对称部分的重复数据
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape),k = 1).astype(np.bool))
to_drop = [i for i in upper.columns if any(upper[i] > 0.75)]
# print(to_drop)
# 删掉相似度高的
df_test.drop(to_drop,axis = 1,inplace = True)
# print(df_test.head())

# 将 y_train 变成 0，1 形式表达，进行模型训练
def transfrom_y_train(y_train):
    '''
    对y_train经行处理
    :param y_train:y_train
    :return: 处理后的y_train
    '''
    y0 = y_train.str.replace('Yes', '1')
    y1 = y0.str.replace('No', '0')
    for i in range(len(y1)):
        y1[i] = 1 if y1[i] == '1' else 0
    return y1

y_train = transfrom_y_train(y_train)

# 进行模型训练,因为只有训练集，所以自行分割训练集为训练集和测试集合
# 训练集分割
def Train_split(data_all,ratio_num,y_train):
    '''
    分割训练集为 新训练集 和 测试集
    :param data_all: df_test
    :param ratio_num: 新训练集占原本数据集合的比例：小数形式
    :param y_train: y_train
    :return: 新训练集、测试集、新y_train
    '''
    data_all_len = data_train.shape[0]
    min_len = data_all_len * ratio_num
    while True:
        train_len = int(np.random.uniform(0,data_all_len))
        if train_len > min_len:
            break
    train = data_all[:train_len]
    test = data_all[train_len+1:]
    y_train = y_train[:train_len]
    return train,test,y_train

x_train,x_test,Y_train = Train_split(df_test,0.8,y_train)

# todo 进行模型训练，并且评价回归模型的好坏
def _ApplyLinerAlgo(model,x_train,x_test,y_train):
    model.fit(x_train,y_train)
    y_predict = model.predict(x_train)
    # r2数值越大，模型训练效果越好
    print("r2评价模型好坏：",r2_score(y_train,y_predict))
    print("MSE评价模型好坏：",mean_squared_error(y_train,y_predict))
    print("MAE评价模型好坏：",mean_absolute_error(y_train,y_predict))
    print("EAS评价模型好坏：",explained_variance_score(y_train,y_predict))
    print("MAE评价模型好坏：",median_absolute_error(y_train,y_predict))
    print('\n')
    y_train_pre = np.exp(model.predict(x_test))
    return y_train_pre

# 使用弹性回归模型训练和预测
# ENCV = ElasticNetCV(alphas=[0.0001,0.0005,0.001,0.01,0.1,1,10],l1_ratio= [0.01,0.1,0.5,0.9,0.99],max_iter=20)
# print("ElasticNetCV:\n")
# y_pre_ENCV = _ApplyLinerAlgo(ENCV,x_train,x_test,Y_train)

# 使用岭回归模型训练和预测
# RCV = RidgeCV(alphas=[0.0001,0.0005,0.001,0.01,0.1,1,10])
# print("RidgeCV:\n")
# y_pre_RCV = _ApplyLinerAlgo(RCV,x_train,x_test,Y_train)

# 使用随机森岭回归模型训练和预测
# RFR = RandomForestRegressor()
# print("RandomForestRegressor:\n")
# y_pre_RFR = _ApplyLinerAlgo(RFR,x_train,x_test,Y_train)

# 使用Lasso回归模型训练和预测
# LCV = LassoCV(alphas=[0.0001,0.0005,0.001,0.01,0.1,1,10])
# print("LassoCV:\n")
# y_pre_LCV = _ApplyLinerAlgo(LCV,x_train,x_test,Y_train)

# 使用pipeline融合 回归模型 进行训练和预测
# selectKBest = SelectKBest(k = 'all')
# RFR = RandomForestRegressor(max_depth=50)
# pipeline = make_pipeline(selectKBest,RFR)
# 分割数据集
# split_train,split_test = model_selection.train_test_split(df_test,test_size=0.3,random_state=0)
# split_y_train = y_train[:len(split_train)]
# print("Pipeline:\n")
# y_pre_Pipe = _ApplyLinerAlgo(pipeline,split_train,split_test,split_y_train)

'''
# 建立 分类模型进行预测，使用网格搜索
# 融合两种模型
pipe = Pipeline([('select',SelectKBest(k = 'all')),('classify',RandomForestClassifier(random_state = 10, max_features = 'auto'))])

# 利用 网格搜索 对融合模型得参数进行最优解搜索
param_test = {'classify__n_estimators':list(range(10,50,2)),'classify__max_depth':list(range(20,80,2))}
gsearch = GridSearchCV(estimator=pipe,param_grid=param_test,scoring='roc_auc',cv=10)
# 分割数据集
split_train,split_test = model_selection.train_test_split(df_test,test_size=0.3,random_state=0)
split_y_train = y_train[:len(split_train)]
# 寻找参数最优解
test = split_test.as_matrix()[:,1:]
X = split_train.as_matrix()[:,1:]
y = split_train.as_matrix()[:,0]
# gsearch.fit(X,y)
# print(gsearch.best_params_, gsearch.best_score_)
# {'classify__max_depth': 20, 'classify__n_estimators': 48} 0.7234027336077584

# # 将选出的 n_estimators = 48、max_depth = 20 带入模型中
select = SelectKBest(k = 20)
clf = RandomForestClassifier(random_state = 10, warm_start = True,
                                  n_estimators = 48,
                                  max_depth = 20,
                                  max_features = 'sqrt')
pipeline = make_pipeline(select, clf)
pipeline.fit(X, y)

# 用 交叉验证 对数据进行预测分析
cv_score = cross_val_score(pipeline, X, y, cv= 10)
# 利用 Mean 和 std 对数据预测情况经行判断
print("CV Score : Mean - %.7g | Std - %.7g " % (np.mean(cv_score), np.std(cv_score)))

Prediction = pipeline.predict(test)
'''

