import pandas as pd
from tensorflow.python.keras.models import load_model
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from deepctr.layers import custom_objects
from deepctr.models import DeepFM, DCN, xDeepFM
from flask import Flask, jsonify
from deepctr.feature_column import SparseFeat, DenseFeat, get_feature_names
from sklearn.model_selection import train_test_split

app = Flask(__name__)

data = pd.read_csv('criteo_sampled_data.csv')

sparse_features = ['C' + str(i) for i in range(1, 27)]
dense_features = ['I'+str(i) for i in range(1, 14)]

data[sparse_features] = data[sparse_features].fillna('-1', )
data[dense_features] = data[dense_features].fillna(0,)
target = ['label']

for feat in sparse_features:
    lbe = LabelEncoder()
    data[feat] = lbe.fit_transform(data[feat])

mms = MinMaxScaler(feature_range=(0, 1))
data[dense_features] = mms.fit_transform(data[dense_features])

fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=data[feat].max() + 1,embedding_dim=4)
                       for i,feat in enumerate(sparse_features)] + [DenseFeat(feat, 1,)
                      for feat in dense_features]

dnn_feature_columns = fixlen_feature_columns
linear_feature_columns = fixlen_feature_columns
train, test = train_test_split(data, test_size=0.2)
test = test.iloc[:1, :]
feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)
test_model_input = {name: test[name].values for name in feature_names}

deep_fm_model = DeepFM(linear_feature_columns, dnn_feature_columns)
deep_fm_model.load_weights('models/DeepFM_w.h5')

dcn_model = DCN(linear_feature_columns, dnn_feature_columns)
dcn_model.load_weights('models/DCN_w.h5')

x_deep_fm_model = xDeepFM(linear_feature_columns, dnn_feature_columns)
x_deep_fm_model.load_weights('models/xDeepFM_w.h5')


@app.route('/deep_fm/predict')
def deep_fm_predict():
    pred = deep_fm_model.predict(test_model_input)
    return jsonify({'data': pred.tolist()})


@app.route('/dcn/predict')
def dcn_predict():
    pred = dcn_model.predict(test_model_input)
    return jsonify({'data': pred.tolist()})


@app.route('/xdeep/predict')
def x_deep_fm_predict():
    pred = x_deep_fm_model.predict(test_model_input)
    return jsonify({'data': pred.tolist()})


if __name__ == '__main__':
    app.run(debug=True, port=9091)
