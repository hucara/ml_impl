from unittest import TestCase
from sklearn import datasets
from supervised.knn_classifier import KnnClassifier

iris = datasets.load_iris()
iris_data = iris.data
iris_labels = iris.target


class TestKnnClassifier(TestCase):
    def test_train(self):
        m = KnnClassifier()
        m.train(iris_data, iris_labels)
        self.assertEqual((m._X == iris_data).all(), True)
        self.assertEqual((m._y == iris_labels).all(), True)

    def test_predict_k1(self):
        m = KnnClassifier(k=1)
        m.train(iris_data, iris_labels)

        self.assertEqual(1, m._k)

        self.assertEqual(m.predict(iris_data[20]), iris_labels[20])
        self.assertEqual(m.predict(iris_data[50]), iris_labels[50])
        self.assertEqual(m.predict(iris_data[60]), iris_labels[60])
        self.assertEqual(m.predict(iris_data[70]), iris_labels[70])

    def test_predict_k5(self):
        m = KnnClassifier(k=5)
        m.train(iris_data, iris_labels)

        self.assertEqual(5, m._k)

        self.assertEqual(m.predict(iris_data[20]), iris_labels[20])
        self.assertEqual(m.predict(iris_data[50]), iris_labels[50])
        self.assertEqual(m.predict(iris_data[60]), iris_labels[60])
        self.assertEqual(m.predict(iris_data[70]), 2)

