import os
import sklearn
import numpy as np
import pandas as pd
from copy import deepcopy
from sklearn.cluster import KMeans
from sklearn.externals import joblib
from scipy.spatial.distance import cdist

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
                n_clusters=2,
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


def preprocess_data(df):
    """
    数据处理及特征工程等
    :param df: 读取原始 csv 数据，有 timestamp、cpc、cpm 共 3 列特征
    :return: 处理后的数据, 返回 pca 降维后的特征
    """
    # 请使用joblib函数加载自己训练的 scaler、pca 模型，方便在测试时系统对数据进行相同的变换
    # ====================数据预处理、构造特征等========================

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hours'] = df['timestamp'].dt.hour
    df['daylight'] = ((df['hours'] >= 7) & (df['hours'] <= 22)).astype(int)
    df['cpc X cpm'] = df['cpm'] * df['cpc']
    df['cpc / cpm'] = df['cpc'] / df['cpm']

    # ========================  模型加载  ===========================
    columns = ['cpc', 'cpm', 'cpc X cpm', 'cpc / cpm','daylight']
    data = df[columns]

    # 标准化 + 降维
    scaler = joblib.load('./results/scaler.pkl') #scaler = StandardScaler()
    data = scaler.fit_transform(data)

    pca = joblib.load('./results/pca.pkl') #pca = PCA(n_components=n_components)
    data = pca.fit_transform(data)

    # 将array类型的data变为DataFrame类型，返回DataFrame类型数据
    n_components = data.shape[1]
    data = pd.DataFrame(data,columns=['Dimension' + str(i+1) for i in range(3)])
    return data
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
def get_anomaly(data, kmean, ratio):
    """
    检验出样本中的异常点，并标记为 True 和 False，True 表示是异常点
  
    :param data: preprocess_data 函数返回值，即 pca 降维后的数据，DataFrame 类型
    :param kmean: 通过 joblib 加载的模型对象，或者训练好的 kmeans 模型
    :param ratio: 异常数据占全部数据的百分比,在 0 - 1 之间，float 类型
    :return: data 添加 is_anomaly 列，该列数据是根据阈值距离大小判断每个点是否是异常值，元素值为 False 和 True
    """
      # ====================检验出样本中的异常点========================
    num_anomaly = int(len(data) * ratio)
  
      # 添加中间列 distance
    data['distance'] = get_distance(data[data.columns],kmean,7)
  
      # 计算阈值距离
    threshould = data['distance'].sort_values(ascending=False).reset_index(drop=True)[num_anomaly]
    data['is_anomaly'] = data['distance'].apply(lambda x: x > threshould)
  
    return data
def predict(preprocess_data):
    """
    该函数将被用于测试，请不要修改函数的输入输出，并按照自己的模型返回相关的数据。
    在函数内部加载 kmeans 模型并使用 get_anomaly 得到每个样本点异常值的判断
    :param preprocess_data: preprocess_data函数的返回值，一般是 DataFrame 类型
    :return:is_anomaly:get_anomaly函数的返回值，各个属性应该为（Dimesion1,Dimension2,......数量取决于具体的pca），distance,is_anomaly，请确保这些列存在
            preprocess_data:  即直接返回输入的数据
            kmeans: 通过joblib加载的对象
            ratio:  异常点的比例，ratio <= 0.03   返回非异常点得分将受到惩罚！
    """
      # 异常值所占比率
    ratio = 0.022
      # 加载模型
    kmeans = joblib.load('./results/model.pkl')
      # 获取异常点数据信息
    use_data=preprocess_data[:]
    is_anomaly = get_anomaly(use_data, kmeans, ratio)
  
    return is_anomaly, preprocess_data, kmeans, ratio