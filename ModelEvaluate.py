'''
    Model and Evaluate
    date:2019/1/11
'''
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold
import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso, LinearRegression, Ridge, ElasticNet, SGDRegressor, BayesianRidge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA, KernelPCA
from sklearn.svm import SVR, LinearSVR
from GridSearch import grid
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import seaborn as sns

from Ensemble import AverageWeight
from Pipeline import labelenc, skew_dummies, add_feature
from Preprodata import map_values, pro_data

# 计算交叉验证的平均均方误差
def rmse_cv(model,X,y):
    rmse = np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=5))
    return rmse

# 网格搜索 超参数优化
class grid():
    def __init__(self, model):
        self.model = model

    def grid_get(self, X, y, param_grid):
        grid_search = GridSearchCV(self.model, param_grid, cv=5, scoring="neg_mean_squared_error")
        grid_search.fit(X, y)
        print(grid_search.best_params_, np.sqrt(-grid_search.best_score_))
        grid_search.cv_results_['mean_test_score'] = np.sqrt(-grid_search.cv_results_['mean_test_score'])
        print(pd.DataFrame(grid_search.cv_results_)[['params', 'mean_test_score', 'std_test_score']])


if __name__== '__main__':
    full,train,test=pro_data()
    map_values(full)
    # 建立通道
    pipe = Pipeline([
        ('labenc', labelenc()),
        ('skew_dummies', skew_dummies(skew=1)),
    ])
    # save the original data for later use
    full2 = full.copy()
    data_pipe = pipe.fit_transform(full2)
    # print(data_pipe.shape[1])
    # print(data_pipe.head())
    # data_pipe.to_csv('test1.csv')

    # 如果数据中含有异常值，那么使用均值和方差缩放数据的效果并不好。这种情况下，可以使用robust_scale和RobustScaler
    scaler = RobustScaler()

    n_train = train.shape[0]

    X = data_pipe[:n_train]
    test_X = data_pipe[n_train:]
    y = train.SalePrice

    # 调用fit方法，根据已有的训练数据创建一个标准化的转换器
    # 使用上面这个转换器去转换训练数据x,调用transform方法
    X_scaled = scaler.fit(X).transform(X)
    y_log = np.log(train.SalePrice)
    # 用相同的转换器转换test_X
    test_X_scaled = scaler.transform(test_X)

    # 采用Lasso正则化的方法对特征进行选择，权重大的特征更重要，不重要的特征权重为0
    lasso = Lasso(alpha=0.001)
    lasso.fit(X_scaled, y_log)

    FI_lasso = pd.DataFrame({"Feature Importance": lasso.coef_}, index=data_pipe.columns)
    print(FI_lasso.sort_values("Feature Importance", ascending=False))

    # 按特征的重要性排序，画图
    # FI_lasso[FI_lasso["Feature Importance"] != 0].sort_values("Feature Importance").plot(kind="barh", figsize=(15, 25))
    # plt.xticks(rotation=90)
    # plt.show()

    pipe = Pipeline([
        ('labenc', labelenc()),
        ('add_feature', add_feature(additional=2)),
        ('skew_dummies', skew_dummies(skew=1)),
    ])

    full_pipe = pipe.fit_transform(full)

    # PCA降维
    n_train = train.shape[0]
    X = full_pipe[:n_train]
    test_X = full_pipe[n_train:]
    y = train.SalePrice

    X_scaled = scaler.fit(X).transform(X)
    y_log = np.log(train.SalePrice)
    test_X_scaled = scaler.transform(test_X)

    pca = PCA(n_components=410)

    X_scaled = pca.fit_transform(X_scaled)
    test_X_scaled = pca.transform(test_X_scaled)
    print(X_scaled.shape[0])
    print(test_X_scaled.shape[1])

    # 试验13种回归模型，并采用5折交叉验证集对模型进行评估
    # models = [LinearRegression(), Ridge(), Lasso(alpha=0.01, max_iter=10000), RandomForestRegressor(),
    #           GradientBoostingRegressor(), SVR(), LinearSVR(),
    #           ElasticNet(alpha=0.001, max_iter=10000), SGDRegressor(max_iter=1000, tol=1e-3), BayesianRidge(),
    #           KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5),
    #           ExtraTreesRegressor(), XGBRegressor()]
    #
    # names = ["LR", "Ridge", "Lasso", "RF", "GBR", "SVR", "LinSVR", "Ela", "SGD", "Bay", "Ker", "Extra", "Xgb"]
    # for name, model in zip(names, models):
    #     score = rmse_cv(model, X_scaled, y_log)
    #     print("{}: {:.6f}, {:.4f}".format(name, score.mean(), score.std()))

    # 用网格搜索优化超参数
    # grid(Lasso()).grid_get(X_scaled, y_log, {'alpha': [0.0004, 0.0005, 0.0007, 0.0009], 'max_iter': [10000]})
    # grid(Ridge()).grid_get(X_scaled, y_log, {'alpha': [35, 40, 45, 50, 55, 60, 65, 70, 80, 90]})
    # grid(SVR()).grid_get(X_scaled, y_log,
    #                      {'C': [11, 13, 15], 'kernel': ["rbf"], "gamma": [0.0003, 0.0004], "epsilon": [0.008, 0.009]})
    # param_grid = {'alpha': [0.2, 0.3, 0.4], 'kernel': ["polynomial"], 'degree': [3], 'coef0': [0.8, 1]}
    # grid(KernelRidge()).grid_get(X_scaled, y_log, param_grid)
    # grid(ElasticNet()).grid_get(X_scaled, y_log,
    #                             {'alpha': [0.0008, 0.004, 0.005], 'l1_ratio': [0.08, 0.1, 0.3], 'max_iter': [10000]})

    lasso = Lasso(alpha=0.0005, max_iter=10000)
    ridge = Ridge(alpha=60)
    svr = SVR(gamma=0.0004, kernel='rbf', C=13, epsilon=0.009)
    ker = KernelRidge(alpha=0.2, kernel='polynomial', degree=3, coef0=0.8)
    ela = ElasticNet(alpha=0.005, l1_ratio=0.08, max_iter=10000)
    bay = BayesianRidge()

    # assign weights based on their gridsearch score
    w1 = 0.02
    w2 = 0.2
    w3 = 0.25
    w4 = 0.3
    w5 = 0.03
    w6 = 0.2

    #采用加权平均堆叠6个模型
    weight_avg = AverageWeight(mod=[lasso, ridge, svr, ker, ela, bay], weight=[w1, w2, w3, w4, w5, w6])

    score = rmse_cv(weight_avg, X_scaled, y_log)
    print(score.mean())

    # 堆叠2个最好的模型效果更佳
    weight_avg = AverageWeight(mod = [svr,ker],weight=[0.5,0.5])

    score = rmse_cv(weight_avg,X_scaled,y_log)
    print(score.mean())


