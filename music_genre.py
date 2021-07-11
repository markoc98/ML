import pandas as pd
import sys
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn import svm
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier


def load_data():
    data_path = sys.argv[1]
    df = pd.read_csv(data_path)


    # drop label because that is our prediction - drop filename because it plays no role in prediction, length aswell
    x = df.drop(columns=['label', 'filename', 'length']).copy()
    # normalise data
    min_max_scaler = MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    x = pd.DataFrame(x_scaled, columns=x.columns)

    y = df['label']
    #     splitting into training validation and test dataset - random_state is used so that we use the same groups everytime
    x_train, x_rem, y_train, y_rem = train_test_split(x, y, test_size=0.2, random_state=1)

    #     splitting remaining into validation and test
    x_valid, x_test, y_valid, y_test = train_test_split(x_rem, y_rem, test_size=0.5, random_state=1)

    return x_train, x_valid, x_test, y_train, y_valid, y_test


def test_linear_kernel_svm(x_train, y_train, x_test, y_test):
    lin_svm = svm.SVC(C=100.0, gamma=0.001, kernel='linear')

    lin_svm.fit(x_train, y_train)
    lin_svm_pred = lin_svm.predict(x_test)
    lin_f1 = f1_score(y_test, lin_svm_pred, average='micro')

    print('Linear kernel prediction ' + str(lin_f1))


def test_poly_kernel_svm(x_train, y_train, x_test, y_test):
    poly = svm.SVC(C=1.0, decision_function_shape='ovo', degree=7, kernel='poly')

    poly.fit(x_train, y_train)
    poly_pred = poly.predict(x_test)
    poly_f1 = f1_score(y_test, poly_pred, average='micro')
    print('Polynomial kernel prediction ' + str(poly_f1))


def test_rbf_kernel_svm(x_train, y_train, x_test, y_test):
    rbf = svm.SVC(C=10.0, break_ties=False, cache_size=200, class_weight=None, coef0=0.0,
                  decision_function_shape='ovo', degree=3, gamma=1, kernel='rbf',
                  max_iter=-1, probability=False, random_state=0, shrinking=True,
                  tol=0.001, verbose=False)

    rbf.fit(x_train, y_train)
    rbf_pred = rbf.predict(x_test)
    rbf_f1 = f1_score(y_test, rbf_pred, average='micro')
    print('RBF kernel prediction ' + str(rbf_f1))

def test_xgboost_classifier(x_train, y_train, x_test, y_test):
    xgb = XGBClassifier(use_label_encoder=False)
    le = LabelEncoder()
    le.fit(y_train)
    y_train = le.transform(y_train)
    xgb.fit(x_train, y_train)
    y_pred = xgb.predict(x_test)
    y_pred = le.inverse_transform(y_pred)
    xgb_f1 = f1_score(y_test, y_pred, average='micro')
    print('XGBoost prediction ' + str(xgb_f1))

def grid_search_svm(x_train, y_train):
    c_range = [1.0, 10.0, 100.0, 1000.0]
    gamma_range = [0.001, 0.01, 0.1, 1]
    param_grid =  {'C': c_range, 'gamma': gamma_range, 'kernel': ['linear'], }

    grid = GridSearchCV(svm.SVC(), param_grid, refit=True, verbose = 3)
    grid.fit(x_train, y_train.values)

    print(grid.best_params_)
    print(grid.best_estimator_)

if __name__ == '__main__':
    x_train, x_valid, x_test, y_train, y_valid, y_test = load_data()
    # grid_search_svm(x_train, y_train)
    test_linear_kernel_svm(x_train, y_train, x_test, y_test)
    test_poly_kernel_svm(x_train, y_train, x_test, y_test)
    test_rbf_kernel_svm(x_train, y_train, x_test, y_test)
    test_xgboost_classifier(x_train, y_train, x_test, y_test)
# filename,length,chroma_stft_mean,chroma_stft_var,rms_mean,rms_var,spectral_centroid_mean,spectral_centroid_var,spectral_bandwidth_mean,spectral_bandwidth_var,rolloff_mean,rolloff_var,zero_crossing_rate_mean,zero_crossing_rate_var,harmony_mean,harmony_var,perceptr_mean,perceptr_var,tempo,mfcc1_mean,mfcc1_var,mfcc2_mean,mfcc2_var,mfcc3_mean,mfcc3_var,mfcc4_mean,mfcc4_var,mfcc5_mean,mfcc5_var,mfcc6_mean,mfcc6_var,mfcc7_mean,mfcc7_var,mfcc8_mean,mfcc8_var,mfcc9_mean,mfcc9_var,mfcc10_mean,mfcc10_var,mfcc11_mean,mfcc11_var,mfcc12_mean,mfcc12_var,mfcc13_mean,mfcc13_var,mfcc14_mean,mfcc14_var,mfcc15_mean,mfcc15_var,mfcc16_mean,mfcc16_var,mfcc17_mean,mfcc17_var,mfcc18_mean,mfcc18_var,mfcc19_mean,mfcc19_var,mfcc20_mean,mfcc20_var,label
