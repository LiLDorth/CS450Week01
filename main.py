import numpy
import scipy
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from yellowbrick.classifier import ClassificationReport
from yellowbrick.classifier import ConfusionMatrix


class Model:
    def predict(data):
        prediction = []
        for item in data:
            prediction.append(0)
        return prediction


class HardcodedClassifier:

    def fit(X_Train, Y_Train):
        return Model

def main():
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    iris = datasets.load_iris()
    data_train, data_test, target_train, target_test  = train_test_split(iris.data, iris.target, test_size=45)
    # Show the data (the attributes of each instance)
    #print(iris.data)
    # Show the target values (in numeric format) of each instance
    #print(iris.target)
    # Show the actual target names that correspond to each number
    #print(iris.target_names)

    classifier = HardcodedClassifier
    #classifier = GaussianNB()
    model = classifier.fit(data_train, target_train)
    targets_predicted = model.predict(data_test)
    #print (targets_predicted)

    print(str(accuracy_score(target_test, targets_predicted)*100)+"%")

    #classes = datasets.load_iris().target_names
    #visualizer = ClassificationReport(classifier, classes=classes)
    #visualizer.fit(data_train, target_train)
    #visualizer.score(data_test, target_test)
    #visualizer.poof()

    #cm = ConfusionMatrix(classifier)
    #cm.fit(data_train, target_train)
    #cm.score(data_test, target_test)
    #cm.poof()
if __name__ == '__main__':
    main()

