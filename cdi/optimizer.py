import warnings
import numpy as np
import pandas as pd
from kneed import KneeLocator
from sklearn.neighbors import NearestNeighbors
from pymoo.core.problem import ElementwiseProblem

warnings.filterwarnings("ignore", category=RuntimeWarning)

def get_threshold(training_data, n_neighbors=20, figure_name=None, show_fig=False, plot=None):
    """
    It takes in a training data set, computes the kNN distance, sorts the distances, and then finds the
    knee point of the sorted distances."""
    # get training data and compute kNN distance
    nbrs = NearestNeighbors(n_neighbors = n_neighbors).fit(training_data)
    distances, indices = nbrs.kneighbors(training_data)
    distances = distances.flatten()

    # sort and find the knee point
    kneedle = KneeLocator(np.indices(distances.shape)[0], distances, S=1.0, curve="convex", direction="increasing")
    # get the threshold
    threshold = np.sort(distances)[kneedle.knee]
    if show_fig: # plot the knee point
        plot(distances = distances, threshold = threshold, name=figure_name)

    return threshold, nbrs, distances

class DeviceOptimizer():

    def __init__(self) -> None:
        pass

    def ood(self, x, nbrs, threshold):
        """
        > If the maximum distance between the input and its nearest neighbors is greater than twice the
        threshold, then the input is considered out of distribution

        :param x: the data point we want to classify
        :param nbrs: the kNN model
        :param threshold: the threshold for the distance to the nearest neighbor. If the distance is
        greater than 2*threshold, then the point is considered out of distribution
        :return: a boolean value and the distance.
        """
        out_of_distribution = False

        distances, indices = nbrs.kneighbors(x.reshape(1, -1))

        distance = max(distances[0])

        if distance > 2*threshold:
                out_of_distribution = True

        return out_of_distribution, distance


# case D1: maximize salt adsorption capacity
class salt_adsorption_case(ElementwiseProblem):
    def __init__(self, 
                 obj_func, model, scaler,
                 training_data, nbrs, threshold,
                 xl=np.array([
                            0.40, 2.50, 20.00, 4.50, 0.02, 
                            0.00, 0.0, 0.52, 0.00, 0.00]),
                xu = np.array([
                            2, 100, 5000.00, 4482.00, 4.20, 
                            23.71, 1.06, 1.62, 14.33, 32.09])
                ):
        super().__init__(n_var=10, n_obj = 2, n_constr = 2, vtype=float)

        self.model = model
        self.scaler = scaler
        self.obj_func = obj_func
        self.nbrs = nbrs
        self.threshold = threshold
        self.training_data = training_data

        self.xl, self.xu = xl, xu
    
    def _evaluate(self, X, out):

        features = np.asarray([
                                X[0], X[1], X[2], X[3], X[4], 
                                X[5], X[6], X[7], X[8], X[9]])
        # check if the input is out of distribution
        dop = DeviceOptimizer()
        ood, distance = dop.ood(x = np.asarray(features), nbrs = self.nbrs, threshold = self.threshold)
        # objective functions
        f1 = self.obj_func(model=self.model, scaler=self.scaler, X=pd.DataFrame([features], columns=list(self.training_data.columns))) # maximize salt adsorption capacity
        f2 = np.array(distance).reshape(1, )  # minimize distance to the nearest neighbor
        f3 = -f1 # f1 >= 0, -f1 <= 0
        f4 = -f2 # f2 >= 0, -f2 <= 0

        if ood == True:
            f1 = 1000
        
        out["F"] = [-f1, f2]
        out["G"] = [f3, f4]


# case D2: maximize specific capacitance
class specific_capacitance(ElementwiseProblem):
    def __init__(self, 
                 obj_func, model, scaler,
                 training_data, nbrs, threshold,
                 xl=np.array([37.90, 0.38, 0.00, 2.36, 0.10, 0.10, 0.25]),
                 xu=np.array([2276.60, 2.57, 13.40, 35.15, 26.56, 100.00, 6.00])
                 ):
        super().__init__(n_var=7, n_obj = 2, n_constr = 1, vtype=float)

        self.model = model
        self.scaler = scaler
        self.obj_func = obj_func
        self.nbrs = nbrs
        self.threshold = threshold
        self.training_data = training_data
        
        self.xl, self.xu = xl, xu
    
    def _evaluate(self, X, out):

        features = np.asarray([X[0], X[1], X[2], X[3], X[4], X[5], X[6]])

        dop = DeviceOptimizer()
        ood, distance = dop.ood(x = np.asarray(features), nbrs = self.nbrs, threshold = self.threshold)
        # objective functions
        f1 = self.obj_func(model=self.model, scaler=self.scaler, X=pd.DataFrame([features], columns=list(self.training_data.columns))) # maximize specific capacitance
        f2 = np.array(distance).reshape(1, )  # minimize distance to the nearest neighbor
        f3 = f1 * -1 # f1 >= 0, -f1 <= 0
        f4 = -f2 # f2 >= 0, -f2 <= 0

        if ood == True:
            f1 = 1000
 
        out["F"] = [-f1, f2]
        out["G"] = [f3]


