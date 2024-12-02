
import datetime,time
import numpy as np
import multiprocessing
import itertools
from sklearn.feature_selection import mutual_info_regression
import dromi
import dromi.utils as DromiUtils


def process_value(iterables_args,fixed_args):
    """Discretize the frequency of the values in the arrays
    Example:
    >> x = np.array([0, 0, 1, 0, 2, 2, 3, 0, 0, 1, 1, 0, 2, 0, 0])
    #>> x = np.array([0, 0, 1, 0, 2, 2, 3, 0, 0, 1, 1, 0, 3, 0, 0]) #does not work as expected
    >> y = np.array([1, 0, 1, 0, 2, 2, 3, 0, 0, 1, 1, 0, 2, 0, 4])
    >> x.shape
       15
    >> np.histogram2d(x,y,bins=7)
            (array([[6., 1., 0., 0., 0., 0., 1.],
                    [0., 0., 0., 0., 0., 0., 0.],
                    [0., 3., 0., 0., 0., 0., 0.],
                    [0., 0., 0., 0., 0., 0., 0.],
                    [0., 0., 0., 3., 0., 0., 0.],
                    [0., 0., 0., 0., 0., 0., 0.],
                    [0., 0., 0., 0., 0., 1., 0.]]),
            array([0., 0.42857143, 0.85714286, 1.28571429, 1.71428571, 2.14285714, 2.57142857, 3.]),
            array([0., 0.57142857, 1.14285714, 1.71428571, 2.28571429, 2.85714286, 3.42857143, 4.]))

    Explanation:
    The histogram 2d creates 2 arrays of probability bins per variable (x and y) as follows:
        >lower_val_x = 0
        >upper_val_x = 3
        >step_size_x = upper_val_x/nbins
        >bins_array_x = np.arange(lower_val_x,upper_val_x+step_size_x,step_size_x)
        >bins_array_x
            array([0., 0.42857143, 0.85714286, 1.28571429, 1.71428571,2.14285714, 2.57142857,3])
        >lower_val_y=0
        >upper_val_y = 4
        >step_size_y = upper_val_y/nbins
        >bins_array_y = np.arange(lower_val_y,upper_val_y+step_size_y,step_size_y)
        >bins_array_y
            array([0., 0.57142857, 1.14285714, 1.71428571, 2.28571429,2.85714286, 3.42857143,4])

    IN PROGRESS of understanding (IGNORE for now):
    Then, it computes a matrix nbinsxnbins where it places the number of times that a value is found within that bin. This means that in
    the diagonal, we place the minimum frequency of elements within the corresponding bins for each array

    hist2d[0,0] = min(((0 <= x) & (x < 0.42)).sum(),((0 <= y) & (y < 0.57)).sum())

    hist2d[1,1] = min(((0.42 <= x) & (x < 0.85)).sum(),((0.57 <= y) & (y < 1.14)).sum())

    hist2d[?,?] = min(((0 <= x) & (x < 0.42)).sum(),((0.57 <= y) & (y < 1.14)).sum())

    The diagonal can be also checked by computing the 1D histogram values:

    >> hist1d_x = np.histogram(x,bins=7)
    >> hist1d_x
        (array([8, 0, 3, 0, 3, 0, 1]),
         array([0., 0.42857143, 0.85714286, 1.28571429, 1.71428571, 2.14285714, 2.57142857, 3.]))
    >> hist1d_y = np.histogram(y,bins=7)
    >> hist1d_y
        (array([6, 4, 0, 3, 0, 1, 1]),
        array([0., 0.57142857, 1.14285714, 1.71428571, 2.28571429, 2.85714286, 3.42857143, 4.]))

    >> min_freq = np.concatenate([hist1d_x[0][None,:],hist1d_y[0][None,:]],axis=0).min(axis=0)
    >> min_freq
       array([6, 0, 0, 0, 0, 0, 1])


    """

    i,j = iterables_args
    #print("Calculating indexes i {} and j : {}".format(i,j))
    array,bins = fixed_args
    variable_x = array[:,i]
    variable_y = array[:,j]
    hist2d = np.histogram2d(variable_x, variable_y, bins)[0]

    return (hist2d,i,j)


def fill_array(histograms2d,hist2d, i,j):
    """"""
    histograms2d[i,j] = hist2d

    return histograms2d
def calculate_mutual_information(array,bins=None):

    """
    Calculates the normalized Mutual Information applied to continuous random variables

    I(X,Y) = H(X) + H(Y) -H(X,Y)
    i) Entropy of variable X
    H(X) = - \sum_i p(x_i) \log_2 p(x_i)
    ii) Joint entropy
    H(X,Y) = \sum_{x,y} p(x,y) \log_2 p(x,y)
    iii) Normalized Mutual information:
    NMI(X,Y) = \frac{2I(X,Y)}{H(X) + H(Y)}

    :param array : Matrix nxm
    :param str results_dir: Path where to save the results
    :param str name: name of the matrix to save
    NOTES:
        -References:
            Guo, Xue, Hu Zhang, and Tianhai Tian. "Development of stock correlation networks using mutual information and financial big data." PloS one 13.4 (2018): e0195941.
        -Following: https://medium.com/@bass.dolg/computing-mutual-information-matrix-with-python-6ced9169bcb1

    """

    start = time.time()
    ncolumns = array.shape[1]

    i = np.repeat(np.arange(ncolumns),ncolumns).tolist()
    j = np.tile(np.arange(ncolumns), ncolumns).tolist()

    args_iterables = {"i_idx":i,"j_idx":j}
    if bins is not None:
        bins= int((array.shape[0]/5)**.5)
        if bins < 1: #small arrays
            bins = array.shape[0]
    args_fixed = array,bins
    histograms2d = np.zeros((array.shape[0],array.shape[-1],bins,bins),dtype=np.float16)
    print("Done creating histogram")
    with multiprocessing.Pool(multiprocessing.cpu_count() - 1) as pool:
        results = DromiUtils.RunParallel(args_iterables, args_fixed,process_value).outer_loop(pool)
        zipped_results = list(zip(*results))
        hist2d_list = zipped_results[0]
        i_idx = zipped_results[1] #this is just in case we need to keep track of more advanced indexing
        j_idx = zipped_results[2]

    histograms2d_result = list(map(lambda hist2d,i,j: fill_array(histograms2d,hist2d, i,j), hist2d_list,i_idx,j_idx ))[0] #this can be moved inside process_value function, since it is very simple indexing in this case

    probs = histograms2d_result / ncolumns + 1e-100
    joint_entropies = -(probs * np.log2(probs)).sum((2, 3))  #H(x,y)
    entropies = joint_entropies.diagonal() #H(x) , H(y)
    sum_entropies = entropies + entropies.T
    mi_matrix = sum_entropies - joint_entropies

    norm_mi_matrix = mi_matrix * 2 / sum_entropies
    end = time.time()

    print("Overall calculation time {}".format(str(datetime.timedelta(seconds=end - start))))

    return {"mutual_information":mi_matrix,
            "normalized_mutual_information":norm_mi_matrix}



