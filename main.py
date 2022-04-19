from classes.read_file import read_files
from classes.preProcessing import preProcessing
from classes.find_activities import find_activities
from classes.extractFeatures import extractFeatures
from classes.authentication import authentication
import numpy as np
from sklearn.preprocessing import StandardScaler

def loadData():
    read = read_files(path="/home/mohammad/Work_NC/mywork/NTT/ntt_test/dataset/")
    data = read.load_data()
    return data

def preParedData():
    new_data = preProcessing(loadData())
    prePro_data = new_data.calc_magnitudes_speed()
    return prePro_data

def findActivity():
    activ_ = find_activities(preParedData(), ['magnitude', 'avg_step_speed'], 10, method='None')
    data_cluster = activ_.prepare_data()
    best_k = activ_.max_k(data_cluster)
    return best_k

def creatCluster():
    activ_ = find_activities(preParedData(), ['magnitude', 'avg_step_speed'], 10, method='None')
    data_cluster = activ_.prepare_data()
    cluster_km = activ_.kmeans_cluster(3, data_cluster)
    return cluster_km

def createFeature():
    new_file = extractFeatures(preParedData(), "/home/mohammad/Work_NC/mywork/NTT/ntt_test/dataset/")
    new_file.save_features()

def __loadNewData():
    dataset = np.loadtxt('/home/mohammad/Work_NC/mywork/NTT/ntt_test/dataset/Features.csv',
                         delimiter=",", skiprows=1)
    dataset[np.isnan(dataset)] = 0
    X = dataset[:, 1:]
    y = dataset[:, 0]
    class_names = ['person-1', 'person-2', 'person-3', 'person-4',
                   'person-5', 'person-6', 'person-7', 'person-8',
                   'person-9', 'person-10', 'person-11', 'person-12',
                   'person-13', 'person-14', 'person-15', 'person-16',
                   'person-17', 'person-18', 'person-19', 'person-20',
                   'person-21', 'person-22']
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X = X.reshape((X.shape[0], 10, 6))
    return X ,y

def identify():
    X,y = __loadNewData()
    cnn = authentication(X, y, 50, 32)
    X_train, X_test, y_train, y_test, optModel = cnn.opt_model()
    return X_train, X_test, y_train, y_test, optModel

if __name__ == '__main__':
    loadData()
    preParedData()
    findActivity()
    creatCluster()
    createFeature()
    identify()