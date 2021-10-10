import numpy as np
from Tools.demo.beer import n
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

iris_dataset = load_iris()
# print(iris_dataset) 数据类型  数据属性 type(iris_dataset),dir(iris_dataset) 可视化 iris_df=pd.DataFrame(iris_dataset[
# 'data'],columns=['speal length(cm)','speal width(cm)','petal length','petal width']) iris_df[
# 'Species']=iris_dataset.target print(iris_df) sns.relplot(x='petal width (cm)',y='sepal length (cm)',
# data=iris_df,hue='species')

# 数据划分
X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], test_size=0.2,
                                                    random_state=0)
# print(X_train, X_test, + y_train, y_test)
# 标准化
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

# 模型训练
# 设置邻居数
knn_model = KNeighborsClassifier(n_neighbors=10)
# 构建基于训练集的模型
knn_model.fit(X_train, y_train)

# 一条测试数据
# X_new = np.array([[54, 3.5, 1.4, 0.2]])

# 预测结果
prediction = knn_model.predict(X_test)
print('预测值:\n', prediction)
print("预测值与真实值对比:\n", prediction == y_test)

# 得出测试集X_test测试集的分数
score = knn_model.score(X_test, y_test)
print("预测分数score:{:.2f}".format(score))
