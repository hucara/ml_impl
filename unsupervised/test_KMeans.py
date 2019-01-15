from unittest import TestCase
from sklearn import datasets
from unsupervised.kmeans_clustering import KMeans

iris = datasets.load_iris()
iris_data = iris.data
iris_labels = iris.target

class TestKMeans(TestCase):

    def test_train(self):
        m = KMeans()
        m.train(iris_data)
        self.assertEqual((m._X == iris_data).all(), True)
        self.assertNotEqual(m._centroids, [])

    def test_predict(self):
        m = KMeans()
        m.train(iris_data)
        m.predict()
