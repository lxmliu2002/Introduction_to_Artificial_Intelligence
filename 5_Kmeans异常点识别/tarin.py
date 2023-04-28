import os
import sklearn
import warnings
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist

warnings.filterwarnings('ignore')

file_dir = './data'
csv_files = os.listdir(file_dir)

# df 作为最后输出的 DataFrame 初始化为空
df = pd.DataFrame()
feature = ['cpc', 'cpm']
df_features = []
for col in feature:
    infix = col + '.csv'
    path = os.path.join(file_dir, infix)
    df_feature = pd.read_csv(path)
    # 将两个特征存储起来用于后续连接
    df_features.append(df_feature)

# 2 张 DataFrame 表按时间连接
df = pd.merge(left=df_features[0], right=df_features[1])
df.head()
#df.info()

# 将 timestamp 列转化为时间类型
df['timestamp'] = pd.to_datetime(df['timestamp'])

# 将 df 数据按时间序列排序，方便数据展示
df = df.sort_values(by='timestamp').reset_index(drop=True)
df.head()

# 按照时间轴绘制 cpc 和 cpm 指标数据
fig, axes = plt.subplots(2, 1)
df.plot(kind='line', x='timestamp', y='cpc', figsize=(20, 10), ax=axes[0], title='cpc', color='g')
df.plot(kind='line', x='timestamp', y='cpm', figsize=(20, 10), ax=axes[1], title='cpm', color='b')
plt.tight_layout();

sns.heatmap(df.corr(), cmap='coolwarm', annot=True);

# 绘制 cpc 和 cpm 的散点图，其中横坐标是 cpc，纵坐标是 cpm
plt.scatter(x=df['cpc'],y=df['cpm'],alpha=0.5);
# 绘制数据集中心点，横坐标是 cpc 的平均值，纵坐标是 cpm 的平均值 
plt.scatter(x=df['cpc'].mean(),y=df['cpm'].mean(),c='red',alpha=0.8);

def simple_distance(data):
    """
    计算当前点（cpc，cpm）到（cpc_mean，cpm_mean）的几何距离（L2 范数）
    :param data: ataDFrame 包含cpc、cpm 列
    :return: Series，每一列cpc、cpm到平均值值点的距离大小
    """
    mean = np.array([data['cpc'].mean(), data['cpm'].mean()])
    distance = []
    for i in range(0, len(data)):
        point = np.array(data.iloc[i, 1:3])
        # 求当前点（cpc，cpm）到平均值点（cpc_mean，cpm_mean）之间的几何距离（L2 范数）
        distance.append(np.linalg.norm(point - mean))
    distance = pd.Series(distance)
    return distance

df['distance'] = simple_distance(df[df.columns])
df.head()

# 按照时间轴绘制 cpc 和 cpm 指标数据
fig,axes = plt.subplots(3,1)
df.plot(kind='line', x = 'timestamp', y = 'cpc',      figsize=(20,10), ax = axes[0], title='cpc')
df.plot(kind='line', x = 'timestamp', y = 'cpm',      figsize=(20,10), ax = axes[1], title='cpm')
df.plot(kind='line', x = 'timestamp', y = 'distance', figsize=(20,10), ax = axes[2], title='distance')
plt.tight_layout()

ratio = 0.005
num_anomaly = int(len(df) * ratio)
threshould = df['distance'].sort_values(ascending=False).reset_index(drop=True)[num_anomaly]
print('阈值距离：' + str(threshould))

df['is_anomaly'] = df['distance'].apply(lambda x: x > threshould)
df.head()

normal = df[df['is_anomaly'] == 0]
anormal = df[df['is_anomaly'] == 1]
plt.scatter(x = normal['cpc'], y = normal['cpm'], c = 'blue', alpha = 0.5)
plt.scatter(x = anormal['cpc'],y = anormal['cpm'],c = 'red', alpha=0.5);

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df[['cpc','cpm']] = scaler.fit_transform(df[['cpc','cpm']])
df.describe()

# 假设异常数据比例
ratio = 0.005
num_anomaly = int(len(df) * ratio)
df['distance2'] = simple_distance(df)
threshould = df['distance2'].sort_values(ascending=False).reset_index(drop=True)[num_anomaly]
print('阈值距离：'+str(threshould))
df['is_anomaly2'] = df['distance2'].apply(lambda x: x > threshould)
normal = df[df['is_anomaly2'] == 0]
anormal = df[df['is_anomaly2'] == 1]
plt.scatter(x=normal['cpc'],y=normal['cpm'],c='blue',alpha=0.5)
plt.scatter(x=anormal['cpc'],y=anormal['cpm'],c='red',alpha=0.5);

a = df.loc[df['is_anomaly2'] == 1, ['timestamp', 'cpc']] 
plt.figure(figsize=(20,10))
plt.plot(df['timestamp'], df['cpc'], color='blue')
plt.scatter(a['timestamp'],a['cpc'], color='red')
plt.show()

# 尝试引入非线性关系
df['cpc X cpm'] = df['cpm'] * df['cpc']
df['cpc / cpm'] = df['cpc'] / df['cpm']
# 尝试获取时间关系
df['hours'] = df['timestamp'].dt.hour
df['daylight'] = ((df['hours'] >= 7) & (df['hours'] <= 22)).astype(int)

df1 = df[['cpc','cpm','cpc X cpm','cpc / cpm','hours','daylight']]
df1.corr()
sns.heatmap(df1.corr(),cmap='coolwarm',annot=True);

from sklearn.decomposition import PCA

#在进行特征变换之前先对各个特征进行标准化
columns = ['cpc', 'cpm', 'cpc X cpm', 'cpc / cpm']
data = df[columns]
scaler = StandardScaler()
data = scaler.fit_transform(data)
data = pd.DataFrame(data, columns=columns)

#通过 n_components 指定需要降低到的维度
n_components = 3
pca = PCA(n_components=n_components)
data = pca.fit_transform(data)
data = pd.DataFrame(data,columns=['Dimension' + str(i+1) for i in range(n_components)])
data.head()

# 特征变换空间（特征矩阵），根据我们指定的n_components = k的值，选择方差最大的 k 个值所对应的的特征向量组成的特征矩阵。
print(pd.DataFrame(pca.components_, columns=['Dimension' + str(i + 1) for i in range(pca.n_features_)]))

# 每个保留特征占所有特征的方差百分比
pca.explained_variance_ratio_

# 保留 n 个特征的方差
pca.explained_variance_

var_explain = pca.explained_variance_ratio_
# 梯形累计和，axis=0，按照行累加。axis=1，按照列累加。axis不给定具体值，就把数组当成一个一维数组
cum_var_explian = np.cumsum(var_explain)

plt.figure(figsize=(10, 5))
plt.bar(range(len(var_explain)), var_explain, alpha=0.3, align='center', label='the interpreted independently variance ')
plt.step(range(len(cum_var_explian)), cum_var_explian, where='mid', label='the cumulative interpretation variance')
plt.ylabel('the explain variance rate')
plt.xlabel('PCA')
plt.legend(loc='best')
plt.show()

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(1,figsize=(6,6))
ax = Axes3D(fig)
ax.scatter(data.iloc[:,0],data.iloc[:,1],data.iloc[:,2],edgecolors='k');

from sklearn.cluster import KMeans

class KMeans():
    """
    Parameters
    ----------
    n_clusters 指定了需要聚类的个数，这个超参数需要自己调整，会影响聚类的效果
    n_init 指定计算次数，算法并不会运行一遍后就返回结果，而是运行多次后返回最好的一次结果，n_init即指明运行的次数
    max_iter 指定单次运行中最大的迭代次数，超过当前迭代次数即停止运行
    """
    def __init__(
                self,
                n_clusters=8,
                n_init=10,
                max_iter=300
                ):

        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.n_init = n_init
  
  
    def fit(self, x):
        """
        用fit方法对数据进行聚类
        param x: 输入数据
        :best_centers: 簇中心点坐标 数据类型: ndarray
        :best_labels: 聚类标签 数据类型: ndarray
        :return: self
        """
   ###################################################################################
          #### 请勿修改该函数的输入输出 ####
   ###################################################################################
          # #
        best_centers=np.array([])
        best_labels=np.array([])
        x=np.array(x)
          
        # 初始化质心
        best_centers=x[np.random.randint(0,x.shape[0],self.n_clusters),:] 
              
        #开始迭代
        for i in range(self.max_iter):
            #1.计算距离矩阵
            distances=cdist(x,best_centers) 
              
            #2.对距离按由近到远排序，选取最近的质心点的类别作为当前点的分类
            c_index=np.argmin(distances,axis=1)  
              
            #3.均值计算，更新质心点坐标
            for i in range(self.n_clusters):
                if i in c_index:  
                    best_centers[i]=np.mean(x[c_index==i],axis=0)  
                      
        each_dist=cdist(x,best_centers)
        each_label=np.argmin(each_dist,axis=1)
        best_labels=each_label
          # #
   ###################################################################################
          ############# 在生成 main 文件时, 请勾选该模块 ############# 
   ###################################################################################
  
        self.cluster_centers_ = best_centers
        self.labels_ = best_labels
        return self

kmeans = KMeans(n_clusters=5, n_init=50, max_iter=800)
kmeans.fit(data)

labels = pd.Series(kmeans.labels_)
fig = plt.figure(1,figsize=(6,6))
ax = Axes3D(fig)
ax.scatter(data.iloc[:,0],data.iloc[:,1],data.iloc[:,2],c=labels.astype('float'),edgecolors='k');

from sklearn.metrics import calinski_harabasz_score,silhouette_score

score1 = calinski_harabasz_score(data,kmeans.labels_)
score2 = silhouette_score(data,kmeans.labels_)

print('calinski_harabasz_score:', score1)
print('silhouette_score:', score2)

# 寻找最佳聚类数目
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score,silhouette_score

'''
n_clusters 指定了需要聚类的个数，这个超参数需要自己调整，会影响聚类的效果
init 指明初始聚类中心点的初始化方式，kmeans++是一种初始化方式，还可以选择为random
n_init 指定计算次数，算法并不会运行一遍后就返回结果，而是运行多次后返回最好的一次结果，n_init即指明运行的次数
max_iter 指定单次运行中最大的迭代次数，超过当前迭代次数即停止运行
'''
score1_list = []
score2_list = []
for i in range(2,10):
    kmeans = KMeans(n_clusters=i,init='k-means++',n_init=10,max_iter=100)
    kmeans.fit(data)
    score1 = round(calinski_harabasz_score(data,kmeans.labels_), 2)
    score2 = round(silhouette_score(data,kmeans.labels_), 2)
    score1_list.append(score1)
    score2_list.append(score2)
    print('聚类数目:%s  calinski_harabasz_score:%-10s  silhouette_score:%-10s'%(i,score1,score2))

def get_distance(data, kmeans, n_features):
    """
    计算距离函数
    :param data: 训练 kmeans 模型的数据
    :param kmeans: 训练好的 kmeans 模型
    :param n_features: 计算距离需要的特征的数量
    :return: 每个点距离自己簇中心的距离
    """
    distance = []
    for i in range(0,len(data)):
        point = np.array(data.iloc[i,:n_features])
        center = kmeans.cluster_centers_[kmeans.labels_[i],:n_features]
        distance.append(np.linalg.norm(point - center))
    distance = pd.Series(distance)
    return distance

from copy import deepcopy
ratio = 0.01
num_anomaly = int(len(data) * ratio)
new_data = deepcopy(data)
new_data['distance'] = get_distance(new_data,kmeans,n_features=len(new_data.columns))
threshould = new_data['distance'].sort_values(ascending=False).reset_index(drop=True)[num_anomaly]
print('阈值距离：'+str(threshould))

# 根据阈值距离大小判断每个点是否是异常值
new_data['is_anomaly'] = new_data['distance'].apply(lambda x: x > threshould)
normal = new_data[new_data['is_anomaly'] == 0]
anormal = new_data[new_data['is_anomaly'] == 1]
new_data.head()

fig = plt.figure(1, figsize=(6, 6))
ax = Axes3D(fig)
ax.scatter(anormal.iloc[:, 0], anormal.iloc[:, 1], anormal.iloc[:, 2], c='red', edgecolors='k')
ax.scatter(normal.iloc[:, 0], normal.iloc[:, 1], normal.iloc[:, 2], c='blue', edgecolors='k');

#保存模型，用于后续的测试打分
from sklearn.externals import joblib
joblib.dump(kmeans, './results/model.pkl')
joblib.dump(scaler, './results/scaler.pkl')
joblib.dump(pca, './results/pca.pkl')

a = df.loc[new_data['is_anomaly'] == 1, ['timestamp', 'cpc']] 
plt.figure(figsize=(20,6))
plt.plot(df['timestamp'], df['cpc'], color='blue')
# 聚类后 cpc 的异常点
plt.scatter(a['timestamp'],a['cpc'], color='red')
plt.show()

a = df.loc[new_data['is_anomaly'] == 1, ['timestamp', 'cpm']] 
plt.figure(figsize=(20,6))
plt.plot(df['timestamp'], df['cpm'], color='blue')
# 聚类后 cpm 的异常点
plt.scatter(a['timestamp'],a['cpm'], color='red')
plt.show()