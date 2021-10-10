from flask import Flask
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

app = Flask(__name__)


@app.route('/iris/<string:prediction>', methods=['GET'])
def iris(prediction):
    iris_dataset = load_iris()
    # print(iris_dataset) 数据类型  数据属性 type(iris_dataset),dir(iris_dataset) 可视化 iris_df=pd.DataFrame(iris_dataset[
    # 'data'],columns=['speal length(cm)','speal width(cm)','petal length','petal width']) iris_df[
    # 'Species']=iris_dataset.target print(iris_df) sns.relplot(x='petal width (cm)',y='sepal length (cm)',
    # data=iris_df,hue='species')

    # 数据划分
    X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)
    # print(X_train, X_test, y_train, y_test)

    # 设置邻居数
    knn_model = KNeighborsClassifier(n_neighbors=10)

    # 构建基于训练集的模型
    knn_model.fit(X_train, y_train)

    # 一条测试数据
    # X_new = np.array([[54, 3.5, 1.4, 0.2]])

    # 对X_new预测结果
    y_prediction = knn_model.predict(X_test)
    y_prediction == y_test
    return '预测值 %s' % y_prediction

    # 得出测试集X_test测试集的分数
    # score = knn_model.score(X_test, y_test)
    # return "score:{:.2f}".format(score)


@app.route('/iris_score/<string:score>', methods=['GET'])
def get_iris_score(score):
    iris_dataset = load_iris()
    # print(iris_dataset) 数据类型  数据属性 type(iris_dataset),dir(iris_dataset) 可视化 iris_df=pd.DataFrame(iris_dataset[
    # 'data'],columns=['speal length(cm)','speal width(cm)','petal length','petal width']) iris_df[
    # 'Species']=iris_dataset.target print(iris_df) sns.relplot(x='petal width (cm)',y='sepal length (cm)',
    # data=iris_df,hue='species')

    # 数据划分
    X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], test_size=0.2,
                                                        random_state=0)
    # print(X_train, X_test, y_train, y_test)

    # 设置邻居数
    knn_model = KNeighborsClassifier(n_neighbors=10)

    # 构建基于训练集的模型
    knn_model.fit(X_train, y_train)

    # 一条测试数据
    # X_new = np.array([[54, 3.5, 1.4, 0.2]])

    # 对X_new预测结果
    # prediction = knn_model.predict(X_new)
    # return '预测值 %s' % prediction

    # 得出测试集X_test测试集的分数
    score = knn_model.score(X_test, y_test)
    return "score:{:.2f}".format(score)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
