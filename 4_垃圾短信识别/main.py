import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

# ---------- 停用词库路径，若有变化请修改 -------------
# stopwords_path = r'scu_stopwords.txt'
stopwords_path = r'chineseStopWords.txt'

# ---------------------------------------------------


def read_stopwords(stopwords_path):
    """
    读取停用词库
    :param stopwords_path: 停用词库的路径
    :return: 停用词列表，如 ['嘿', '很', '乎', '会', '或']
    """
    stopwords = []
    # ----------- 请完成读取停用词的代码 ------------
    with open(stopwords_path, 'r', encoding='utf-8') as f:
        stopwords = f.read()
    stopwords = stopwords.splitlines()
    #----------------------------------------------
    
    return stopwords

# 读取停用词
stopwords = read_stopwords(stopwords_path)
#print(stopwords)

# ----------------- 导入相关的库 -----------------
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import ComplementNB
import pandas as pd
import numpy as np


# 读取数据
data_path = "./datasets/5f9ae242cae5285cd734b91e-momodel/sms_pub.csv"
sms = pd.read_csv(data_path, encoding='utf-8')

# 构建训练集和测试集
from sklearn.model_selection import train_test_split
X = np.array(sms.msg_new)
y = np.array(sms.label)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.1)
print("总共的数据大小", X.shape)
print("训练集数据大小", X_train.shape)
print("测试集数据大小", X_test.shape)

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import ComplementNB

# pipline_list用于传给Pipline作为参数
pipeline_list = [
    # --------------------------- 需要完成的代码 ------------------------------
    
    # ========================== 以下代码仅供参考 =============================
#    ('cv', CountVectorizer(token_pattern=r"(?u)\b\w+\b", stop_words=stopwords)),
#    ('classifier', MultinomialNB())
    # ========================================================================
    ('tf', TfidfVectorizer(stop_words=stopwords)),
    ('ss',StandardScaler(with_mean=False)),
    ('classifier', ComplementNB(alpha=1))
    
    # ------------------------------------------------------------------------
]

# 搭建 pipeline
pipeline = Pipeline(pipeline_list)

# 训练 pipeline
pipeline.fit(X_train, y_train)

# 对测试集的数据集进行预测
y_pred = pipeline.predict(X_test)

# 在测试集上进行评估
from sklearn import metrics
print("在测试集上的混淆矩阵：")
print(metrics.confusion_matrix(y_test, y_pred))
print("在测试集上的分类结果报告：")
print(metrics.classification_report(y_test, y_pred))
print("在测试集上的 f1-score ：")
print(metrics.f1_score(y_test, y_pred))

# 在所有的样本上训练一次，充分利用已有的数据，提高模型的泛化能力
pipeline.fit(X, y)
# 保存训练的模型，请将模型保存在 results 目录下
#from sklearn.externals import joblib
import joblib
pipeline_path = 'results/pipeline.model'
joblib.dump(pipeline, pipeline_path)