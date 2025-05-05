from sklearn import tree
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans, DBSCAN
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class MachineLearning:

    def __init__(self):
        self.data = []

    def decision_tree(self,*args):
        """ 
        This function will create a decision Tree.

        Args:

        x - Data
        y - Label data
        predict - Data to predict
        visualize: boolean - If True, an image of the tree is saved
        """
        x = args[0][0].as_py()
        y = args[1].to_pylist()[0]
        predict = args[2][0].as_py()
        visualize = args[3].to_pylist()[0]
    
        df_x = pd.DataFrame(x, columns=[f"c{i+1}" for i in range(len(x[0]))])

        classifier = tree.DecisionTreeClassifier().fit(df_x, y)
        classifier_results = classifier.predict(predict)

        if visualize:
            fig, ax = plt.subplots()
            tree.plot_tree(classifier)
            fig.savefig("./tree.png")
        
        return [classifier_results]

    def linear_regression(self, *args):
        """ 
        This function will create a linear regression.

        Args:

        x - Data
        y - Label data
        predict - Data to predict
        visualize - Visualize data
        """
        x = args[0][0].as_py()
        y = args[1].to_pylist()[0]
        predict = args[2][0].as_py()
    
        df_x = pd.DataFrame(x, columns=[f"c{i+1}" for i in range(len(x[0]))])

        classifier = LinearRegression().fit(df_x, y)
        classifier_results = classifier.predict(predict)

        return [classifier_results]

    def knn_classifier(self, *args):
        """ 
        This function will create a KNN classifier

        Args:

        x - Data
        y - Label data
        predict - Data to predict
        """
        k = args[0].to_pylist()[0]
        y = args[-2].to_pylist()[0]
        predict = args[-1][0].as_py()
       
        df_x = pd.DataFrame(args[1][0].as_py(), columns=[f"l{i+1}" for i in range(3)])
    
        for col in df_x.columns:
            df_x[col] = df_x[col].astype(float)
    
        classifier = KNeighborsClassifier(n_neighbors=k).fit(df_x, y)
        classifier_results = classifier.predict(predict)
        return classifier_results

    def countTotal(self):
        soma = 0
        for val in self.data:
            soma+=len(val)
        print(soma)

    def cluster_kmeans(self, *args):
        k = args[0].to_pylist()[0]
        argsToColumns = [column.to_pylist() for column in args[1:]]

        self.data.append(args[1].chunks[0].buffers())
        print(len(argsToColumns[0]))

        df = pd.DataFrame(np.array(argsToColumns).squeeze().T, columns=[f"l{i+1}" for i in range(len(argsToColumns))])


        cluster = KMeans(k).fit_predict(df)
        print("Cluster called")
        import pyarrow as pa
    
        print(f"ChunkSize:{len(self.data)}")
        if len(self.data)==27:
            self.countTotal()

        return cluster

    def cluster_dbscan(self, x, **kwargs):
        cluster = DBSCAN(**kwargs).fit(x)
        results = cluster.labels_

        return results
