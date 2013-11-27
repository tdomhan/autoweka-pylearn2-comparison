from sklearn.datasets import fetch_mldata
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import normalize
from sklearn.datasets import load_svmlight_file

import numpy as np

import pandas

import tempfile

import theano
from pylearn2.models import mlp, maxout
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.training_algorithms import sgd, learning_rule
from pylearn2.termination_criteria import EpochCounter, MonitorBased
from pylearn2 import costs
from pylearn2.costs.mlp.dropout import Dropout
from pylearn2.utils import string_utils

test_data_home = tempfile.mkdtemp()

data_dir = string_utils.preprocess("${PYLEARN2_DATA_PATH}")


class DataSet:
    def __init__(self, X, y, name, X_test=None, y_test=None, norm=False):
        if norm:
            self.X = normalize(X)
        else:
            self.X = X
        self.y = y
        self.name = name
        if not X_test and not y_test:
            self.X, _X_test, self.y, _y_test = train_test_split(self.X, self.y, test_size=0.3)

    def get_cv_performance(self, clf):
        X = self.X
        y = self.y
        scores = cross_validation.cross_val_score(clf, X, y, cv=cross_validation.KFold(n=X.shape[0], n_folds=10))#cv=10)
        print("%s error rate: %0.2f (+/- %0.2f)" % (self.name, 1.-scores.mean(), scores.std() * 2))


def one_hot_encoding(y):
    n_classes = len(np.unique(y))
    print n_classes
    class_to_id = {y: y_id for y_id, y in enumerate(np.unique(y))}
    id_to_class = {y_id: y for y_id, y in enumerate(np.unique(y))}
    Y = np.zeros((len(y), n_classes))
    for row_id, y_class in enumerate(y):
        Y[row_id,class_to_id[y_class]] = 1
    return Y, id_to_class 


class DeepNet(BaseEstimator, ClassifierMixin):
    """
        Simple deep neural network based on pylearn2
        exposing a sklearn interface.
    """
    def __init__(self):
        pass
    
    def fit(self, X, y):
        Y, id_to_class = one_hot_encoding(y)
        X_train, X_valid, Y_train, Y_valid = train_test_split(X,Y,test_size=0.1)

        self.id_to_class = id_to_class
        dataset = DenseDesignMatrix(X=X_train, y=Y_train)
        dataset_valid = DenseDesignMatrix(X=X_valid, y=Y_valid)
        nvis = X.shape[1]
        n_classes = len(np.unique(y))

        h0 = mlp.Sigmoid(layer_name='h0', dim=500, irange=0.05)
        h1 = mlp.Sigmoid(layer_name='h1',dim=1000, sparse_init= 15)
        h2 = mlp.Sigmoid(layer_name='h2',dim=1000, sparse_init= 15)
#        h0 = mlp.RectifiedLinear(layer_name='h0', dim=100, irange=0.05)
#        h1 = mlp.RectifiedLinear(layer_name='h1', dim=200, sparse_init=15)
        #Setup the model
        output = mlp.Softmax(layer_name='y',
                             n_classes=n_classes,
                             irange=.005,
                             max_col_norm=1.9365)
        layers = [h0, h1, output]
        self.model = mlp.MLP(layers, nvis=nvis)
        #Train the model
        term_criterion = MonitorBased(channel_name="valid_y_misclass",N=10,prop_decrease=0.0)
        trainer = sgd.SGD(learning_rate=.01,
                          batch_size=100,
                          init_momentum=.7,
                          monitoring_dataset={'train': dataset, 'valid': dataset_valid},
                          termination_criterion=term_criterion
                          cost=Dropout(input_include_probs={'l1': .8},
                                       input_scales={'l1': 1.}))
                          #                          cost=costs.cost.SumOfCosts(costs=[costs.mlp.Default(),
                          #                                  costs.mlp.WeightDecay(coeffs=[ .00005, .00005, .00005 ])]))
                          
        trainer.setup(self.model, dataset)

        while True:
            trainer.train(dataset=dataset)
            self.model.monitor.report_epoch()
            self.model.monitor()
            if not trainer.continue_learning(self.model):
                break


    def predict(self, X): 
        y_probs = self.model.fprop(theano.shared(X, name='inputs')).eval()
        print y_probs
        class_ids = np.argmax(y_probs, axis=1)
        y_predict = [self.id_to_class[class_id] for class_id in class_ids]
        return  y_predict

#    def score(self, X, y):
#        y_predict = self.predict(X)
#        return accuracy_score(y, y_predict)


def get_wine_dataset():
    wine = pandas.read_csv(os.path.join(data_dir, "WineQuality", "winequality-white.csv"), delimiter=';')
    wine_cols = set(wine.columns)
    wine_cols.remove('quality')
    wine_cols = list(wine_cols)
    return DataSet(wine[wine_cols], wine["quality"], "Wine quality")

def get_dexter_dataset():
    #Note: we merge the train and the valid set, because it seems like this has
    # been done for the autoweka paper as well.
    X, y = load_svmlight_file(os.path.join(data_dir, "Dexter/dexter_all.svmlight"))
    X = np.asarray(X.todense())
    return DataSet(X, y, "Dexter")

def get_dorothea_dataset():
    X = np.zeros((1150,100000))
    i = 0
    folder = os.path.join(data_dir, "Dorothea")
    with open(os.path.join(folder, "dorothea_train.data.txt")) as f:
        for line in f:
            for idx in line.split():
                X[i,int(idx)-1] = 1.
            i+=1
    with open(os.path.join(folder, "dorothea_valid.data.txt")) as f:
        for line in f:
            for idx in line.split():
                X[i,int(idx)-1] = 1.
            i+=1
    y = np.concatenate([np.loadtxt(folder+"dorothea_train.labels.txt"),
                         np.loadtxt(folder+"dorothea_valid.labels.txt")])
    print "read dataset"
    return DataSet(X, y, "Dorothea")


def get_gisette_dataset():
    X1 = np.loadtxt(os.path.join(data_dir, "Gisette", "gisette_train.data"))
    X2 = np.loadtxt(os.path.join(data_dir, "Gisette", "gisette_valid.data"))
    X = np.vstack([X1, X2])
    y1 = np.loadtxt(os.path.join(data_dir, "Gisette", "gisette_train.labels.txt"))
    y2 = np.loadtxt(os.path.join(data_dir, "Gisette", "gisette_valid.labels.txt"))
    y = np.concatenate([y1,y2])
    return DataSet(X, y, "Gisette")


def get_cifar10():
    from pylearn2.datasets.cifar10 import CIFAR10
    cifar = CIFAR10(which_set='train', toronto_prepro=True)
    return DataSet(cifar.X, cifar.y, "CIFAR")

if __name__ == "__main__":
    clf = RandomForestClassifier(n_estimators=10)
    clf = LogisticRegression()
    clf = DeepNet()

#    dataset = get_wine_dataset()
    #dataset = get_dexter_dataset()
    #dataset = get_dorothea_dataset()
    #dataset = get_gisette_dataset()
    dataset = get_cifar10()

    dataset.get_cv_performance(clf)

