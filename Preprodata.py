'''
    kaggle home price predict
    address:https://www.kaggle.com/massquantity/all-you-need-is-pca-lb-0-11421-top-4/notebook
'''


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
# matplotlib inline
plt.style.use('ggplot')

def pro_data():
    train = pd.read_csv('C:/PythonProject/HomePricePredict/train.csv')
    test = pd.read_csv('C:/PythonProject/HomePricePredict/test.csv')
    # print(train)


    # plt.figure(figsize=(15, 8))
    # sns.boxplot(train.YearBuilt,train.SalePrice)
    # plt.show()

    train.drop(train[(train["GrLivArea"] > 4000) & (train["SalePrice"] < 300000)].index, inplace=True)
    # plt.figure(figsize=(12, 6))
    # plt.scatter(x=train.GrLivArea, y=train.SalePrice)
    # plt.xlabel("GrLivArea", fontsize=13)
    # plt.ylabel("SalePrice", fontsize=13)
    # plt.ylim(0, 800000)
    # plt.show()

    full= pd.concat([train,test],ignore_index=True)

    full.drop(['Id'],axis=1,inplace=True)
    print(full.shape[1])
    aa= full.isnull().sum()
    # print(aa[aa > 0].sort_values(ascending=False))
    # print(aa)

    cols1 = ["PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu", "GarageQual", "GarageCond", "GarageFinish",
             "GarageYrBlt", "GarageType", "BsmtExposure", "BsmtCond", "BsmtQual", "BsmtFinType2", "BsmtFinType1",
             "MasVnrType"]
    for col in cols1:
        full[col].fillna("None", inplace=True)


    cols = ["MasVnrArea", "BsmtUnfSF", "TotalBsmtSF", "GarageCars", "BsmtFinSF2", "BsmtFinSF1", "GarageArea"]
    for col in cols:
        full[col].fillna(0, inplace=True)



    cols2 = ["MSZoning", "BsmtFullBath", "BsmtHalfBath", "Utilities", "Functional", "Electrical", "KitchenQual",
             "SaleType", "Exterior1st", "Exterior2nd"]
    for col in cols2:
        full[col].fillna(full[col].mode()[0], inplace=True)



    full['LotFrontage'] = full.groupby(['Neighborhood'])['LotFrontage'].transform(lambda x: x.fillna(x.median()))

    full['LotFrontage'] = full.groupby(['LotArea'])['LotFrontage'].transform(lambda x: x.fillna(x.median()))
    print(full.isnull().sum()[full.isnull().sum() > 0])

    return full,train,test


def map_values(full):

    '''Convert some numerical features into categorical features.
    It's better to use LabelEncoder and get_dummies for these features.
    '''

    NumStr = ["MSSubClass", "BsmtFullBath", "BsmtHalfBath", "HalfBath", "BedroomAbvGr", "KitchenAbvGr", "MoSold",
              "YrSold", "YearBuilt", "YearRemodAdd", "LowQualFinSF", "GarageYrBlt"]
    for col in NumStr:
        full[col] = full[col].astype(str)
    '''

    Now I want to do a long list of value-mapping.
    I was influenced by the insight that we should build as many features as possible and 
    trust the model to choose the right features. 
    So I decided to groupby SalePrice according to one feature and sort it based on mean and median. Here is an example:

    '''
    full["oMSSubClass"] = full.MSSubClass.map({'180': 1,
                                               '30': 2, '45': 2,
                                               '190': 3, '50': 3, '90': 3,
                                               '85': 4, '40': 4, '160': 4,
                                               '70': 5, '20': 5, '75': 5, '80': 5, '150': 5,
                                               '120': 6, '60': 6})

    full["oMSZoning"] = full.MSZoning.map({'C (all)': 1, 'RH': 2, 'RM': 2, 'RL': 3, 'FV': 4})

    full["oNeighborhood"] = full.Neighborhood.map({'MeadowV': 1,
                                                   'IDOTRR': 2, 'BrDale': 2,
                                                   'OldTown': 3, 'Edwards': 3, 'BrkSide': 3,
                                                   'Sawyer': 4, 'Blueste': 4, 'SWISU': 4, 'NAmes': 4,
                                                   'NPkVill': 5, 'Mitchel': 5,
                                                   'SawyerW': 6, 'Gilbert': 6, 'NWAmes': 6,
                                                   'Blmngtn': 7, 'CollgCr': 7, 'ClearCr': 7, 'Crawfor': 7,
                                                   'Veenker': 8, 'Somerst': 8, 'Timber': 8,
                                                   'StoneBr': 9,
                                                   'NoRidge': 10, 'NridgHt': 10})

    full["oCondition1"] = full.Condition1.map({'Artery': 1,
                                               'Feedr': 2, 'RRAe': 2,
                                               'Norm': 3, 'RRAn': 3,
                                               'PosN': 4, 'RRNe': 4,
                                               'PosA': 5, 'RRNn': 5})

    full["oBldgType"] = full.BldgType.map({'2fmCon': 1, 'Duplex': 1, 'Twnhs': 1, '1Fam': 2, 'TwnhsE': 2})

    full["oHouseStyle"] = full.HouseStyle.map({'1.5Unf': 1,
                                               '1.5Fin': 2, '2.5Unf': 2, 'SFoyer': 2,
                                               '1Story': 3, 'SLvl': 3,
                                               '2Story': 4, '2.5Fin': 4})

    full["oExterior1st"] = full.Exterior1st.map({'BrkComm': 1,
                                                 'AsphShn': 2, 'CBlock': 2, 'AsbShng': 2,
                                                 'WdShing': 3, 'Wd Sdng': 3, 'MetalSd': 3, 'Stucco': 3, 'HdBoard': 3,
                                                 'BrkFace': 4, 'Plywood': 4,
                                                 'VinylSd': 5,
                                                 'CemntBd': 6,
                                                 'Stone': 7, 'ImStucc': 7})

    full["oMasVnrType"] = full.MasVnrType.map({'BrkCmn': 1, 'None': 1, 'BrkFace': 2, 'Stone': 3})

    full["oExterQual"] = full.ExterQual.map({'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4})

    full["oFoundation"] = full.Foundation.map({'Slab': 1,
                                               'BrkTil': 2, 'CBlock': 2, 'Stone': 2,
                                               'Wood': 3, 'PConc': 4})

    full["oBsmtQual"] = full.BsmtQual.map({'Fa': 2, 'None': 1, 'TA': 3, 'Gd': 4, 'Ex': 5})

    full["oBsmtExposure"] = full.BsmtExposure.map({'None': 1, 'No': 2, 'Av': 3, 'Mn': 3, 'Gd': 4})

    full["oHeating"] = full.Heating.map({'Floor': 1, 'Grav': 1, 'Wall': 2, 'OthW': 3, 'GasW': 4, 'GasA': 5})

    full["oHeatingQC"] = full.HeatingQC.map({'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5})

    full["oKitchenQual"] = full.KitchenQual.map({'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4})

    full["oFunctional"] = full.Functional.map(
        {'Maj2': 1, 'Maj1': 2, 'Min1': 2, 'Min2': 2, 'Mod': 2, 'Sev': 2, 'Typ': 3})

    full["oFireplaceQu"] = full.FireplaceQu.map({'None': 1, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5})

    full["oGarageType"] = full.GarageType.map({'CarPort': 1, 'None': 1,
                                               'Detchd': 2,
                                               '2Types': 3, 'Basment': 3,
                                               'Attchd': 4, 'BuiltIn': 5})

    full["oGarageFinish"] = full.GarageFinish.map({'None': 1, 'Unf': 2, 'RFn': 3, 'Fin': 4})

    full["oPavedDrive"] = full.PavedDrive.map({'N': 1, 'P': 2, 'Y': 3})

    full["oSaleType"] = full.SaleType.map({'COD': 1, 'ConLD': 1, 'ConLI': 1, 'ConLw': 1, 'Oth': 1, 'WD': 1,
                                           'CWD': 2, 'Con': 3, 'New': 3})

    full["oSaleCondition"] = full.SaleCondition.map(
        {'AdjLand': 1, 'Abnorml': 2, 'Alloca': 2, 'Family': 2, 'Normal': 3, 'Partial': 4})

    print("Done!")

    # full.drop(["LotAreaCut"], axis=1, inplace=True)
    full.drop(['SalePrice'], axis=1, inplace=True)




# if __name__== '__main__':
#     full,train,test=pro_data()
#     map_values(full)
#     # 建立通道
#     pipe = Pipeline([
#         ('labenc', labelenc()),
#         ('skew_dummies', skew_dummies(skew=1)),
#     ])
#     # save the original data for later use
#     full2 = full.copy()
#     data_pipe = pipe.fit_transform(full2)
#     # print(data_pipe.shape[1])
#     # print(data_pipe.head())
#     # data_pipe.to_csv('test1.csv')
#
#     # 如果数据中含有异常值，那么使用均值和方差缩放数据的效果并不好。这种情况下，可以使用robust_scale和RobustScaler
#     scaler = RobustScaler()
#
#     n_train = train.shape[0]
#
#     X = data_pipe[:n_train]
#     test_X = data_pipe[n_train:]
#     y = train.SalePrice
#
#     # 调用fit方法，根据已有的训练数据创建一个标准化的转换器
#     # 使用上面这个转换器去转换训练数据x,调用transform方法
#     X_scaled = scaler.fit(X).transform(X)
#     y_log = np.log(train.SalePrice)
#     # 用相同的转换器转换test_X
#     test_X_scaled = scaler.transform(test_X)
#
#     # 采用Lasso正则化的方法对特征进行选择，权重大的特征更重要，不重要的特征权重为0
#     lasso = Lasso(alpha=0.001)
#     lasso.fit(X_scaled, y_log)
#
#     FI_lasso = pd.DataFrame({"Feature Importance": lasso.coef_}, index=data_pipe.columns)
#     print(FI_lasso.sort_values("Feature Importance", ascending=False))
#
#     # 按特征的重要性排序，画图
#     # FI_lasso[FI_lasso["Feature Importance"] != 0].sort_values("Feature Importance").plot(kind="barh", figsize=(15, 25))
#     # plt.xticks(rotation=90)
#     # plt.show()
#
#     pipe = Pipeline([
#         ('labenc', labelenc()),
#         ('add_feature', add_feature(additional=2)),
#         ('skew_dummies', skew_dummies(skew=1)),
#     ])
#
#     full_pipe = pipe.fit_transform(full)
#
#     # PCA降维
#     n_train = train.shape[0]
#     X = full_pipe[:n_train]
#     test_X = full_pipe[n_train:]
#     y = train.SalePrice
#
#     X_scaled = scaler.fit(X).transform(X)
#     y_log = np.log(train.SalePrice)
#     test_X_scaled = scaler.transform(test_X)
#
#     pca = PCA(n_components=410)
#
#     X_scaled = pca.fit_transform(X_scaled)
#     test_X_scaled = pca.transform(test_X_scaled)
#     print(X_scaled.shape[0])
#     print(test_X_scaled.shape[1])
#
#     models = [LinearRegression(), Ridge(), Lasso(alpha=0.01, max_iter=10000), RandomForestRegressor(),
#               GradientBoostingRegressor(), SVR(), LinearSVR(),
#               ElasticNet(alpha=0.001, max_iter=10000), SGDRegressor(max_iter=1000, tol=1e-3), BayesianRidge(),
#               KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5),
#               ExtraTreesRegressor(), XGBRegressor()]


