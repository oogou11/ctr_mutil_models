import gc

import numpy as np
import pandas as pd
from preprocess import PreprocessData
from dcn_model import DcnModel as DCN
from deep_fm_model import DeepFMModel as DeepFM
from x_deep_fm_model import XDeepFMModel as xDeepFM
from lr_model import LRModel
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


def get_data(data, dense_feats, sparse_feats):
    # 训练集
    dense_x = [data[f].values for f in dense_feats]
    sparse_x = [data[f].values for f in sparse_feats]
    X = dense_x + sparse_x
    label = [data['label'].values]
    return X, label,


def split_data(total_data, dense_feats, sparse_feats):
    label_1_data, label_0_data = total_data[total_data.label == 1], total_data[total_data.label == 0]
    print("======正样本数========", label_1_data.shape)
    print("======负样本数========", label_0_data.shape)
    print("===========数据集============", total_data.shape)
    # 训练集合:测试集合 = 8:2
    train_1, test_1 = train_test_split(label_1_data, test_size=0.2)
    train_0, test_0 = train_test_split(label_0_data, test_size=0.2)

    # 从训练集中抽出5% 作为验证集合
    _, val_1 = train_test_split(train_1, test_size=0.1)
    _, val_0 = train_test_split(train_0, test_size=0.1)

    # 训练集合
    train_data = pd.concat([train_1, train_0], axis=0)
    # 验证集
    valid_data = pd.concat([val_0, val_1], axis=0)
    # 测试集合
    test_data = pd.concat([test_1, test_0], axis=0)

    del total_data
    gc.collect()

    print("==============训练========", train_data.shape)
    print("==============验证========", valid_data.shape)
    print("==============测试========", test_data.shape)

    # 训练集
    train_x, train_label = get_data(train_data, dense_feats, sparse_feats)

    del train_data
    gc.collect()

    # 验证集集
    val_x, val_label = get_data(valid_data, dense_feats, sparse_feats)
    del valid_data
    gc.collect()

    # 测试集
    test_x, test_label = get_data(test_data, dense_feats, sparse_feats)
    del test_data
    gc.collect()

    return train_x, train_label, val_x, val_label, test_x, test_label


def main():
    pre = PreprocessData('criteo_sampled_data.csv')
    dense_feats, sparse_feats, total_data = pre.process()

    train_x, train_label, val_x, val_label, test_x, test_label = \
        split_data(total_data, dense_feats, sparse_feats)

    # DCN模型
    dcn_model = DCN(dense_feats, sparse_feats,
                    total_data,
                    train_x, train_label,
                    val_x, val_label,
                    test_x,
                    test_label, 6)
    dcn_model.train(1, batch_size=256)
    dcn_pre = dcn_model.predict(train_x)
    dcn_model.save()
    temp_feature = pd.DataFrame.from_dict({'a': dcn_pre.T[0]})
    temp_feature.to_csv('data/dcn_train_x.csv', index=False)

    del dcn_pre
    del temp_feature
    gc.collect()

    dcn_pre_test = dcn_model.predict(test_x)
    temp_feature = pd.DataFrame.from_dict({'a': dcn_pre_test.T[0]})
    temp_feature.to_csv('data/dcn_test_x.csv', index=False)

    del dcn_pre_test
    del temp_feature
    del dcn_model
    gc.collect()

    # DeepFM模型
    deep_fm_model = DeepFM(dense_feats, sparse_feats,
                           total_data,
                           train_x, train_label,
                           val_x, val_label,
                           test_x,
                           test_label, 6)
    deep_fm_model.train(5, batch_size=64)
    deep_fm_pre = deep_fm_model.predict(train_x)
    temp_feature = pd.DataFrame.from_dict({'b': deep_fm_pre.T[0]})
    temp_feature.to_csv('data/deep_fm_train_x.csv', index=False)

    del deep_fm_pre
    del temp_feature
    gc.collect()

    deep_fm_pre_test = deep_fm_model.predict(test_x)
    temp_feature = pd.DataFrame.from_dict({'b': deep_fm_pre_test.T[0]})
    temp_feature.to_csv('data/deep_fm_test_x.csv', index=False)
    del deep_fm_pre_test
    del temp_feature
    del deep_fm_model
    gc.collect()

    # xDeepFM模型
    x_deep_fm_model = xDeepFM(dense_feats, sparse_feats,
                              total_data,
                              train_x, train_label,
                              val_x, val_label,
                              test_x,
                              test_label,
                              6)
    x_deep_fm_model.train(5, batch_size=64)
    # 输出测试报告
    x_deep_fm_model.test()
    # #
    # # LR 特征
    x_deep_fm_pre = x_deep_fm_model.predict(train_x)
    temp_feature = pd.DataFrame.from_dict({'c': x_deep_fm_pre.T[0]})
    temp_feature.to_csv('data/x_deep_fm_train_x.csv', index=False)
    del temp_feature
    gc.collect()

    x_deep_fm_pre_test = x_deep_fm_model.predict(test_x)
    temp_feature = pd.DataFrame.from_dict({'c': x_deep_fm_pre_test.T[0]})
    temp_feature.to_csv('data/x_deep_fm_test_x.csv', index=False)
    del temp_feature
    gc.collect()

    temp_feature = pd.DataFrame.from_dict({'label': train_label[0]})
    temp_feature.to_csv('data/train_label.csv', index=False)
    del temp_feature
    gc.collect()

    # 测试结果
    temp_feature = pd.DataFrame.from_dict({'label': test_label[0]})
    temp_feature.to_csv('data/test_label.csv', index=False)
    del temp_feature
    gc.collect()


if __name__ == '__main__':
    main()
