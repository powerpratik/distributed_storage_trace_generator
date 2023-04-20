from inspect import trace
from os import sep
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
from collections import namedtuple
from scipy.optimize import minimize
import pickle

# Define the candidate distributions
# DISTRIBUTIONS = [
#     st.alpha, st.beta, st.expon, st.gamma, st.lognorm, st.norm, st.weibull_min, st.weibull_max, st.genpareto
# ]
DISTRIBUTIONS = [
    st.alpha,st.beta,st.expon, st.lognorm, st.norm, st.weibull_min, st.weibull_max, st.genpareto
]
# Custom implementation of the hyper-exponential distribution
class hyperexpon(st.rv_continuous): 
    def _pdf(self, x, lambda1, lambda2, p):
        return p * lambda1 * np.exp(-lambda1 * x) + (1 - p) * lambda2 * np.exp(-lambda2 * x)

    def _cdf(self, x, lambda1, lambda2, p):
        return p * (1 - np.exp(-lambda1 * x)) + (1 - p) * (1 - np.exp(-lambda2 * x))

hyperexpon = hyperexpon(name='hyperexpon')

# Load the trace data
def load_trace_data(filename):
    print(filename)
    columns = ['timestamp', 'process_id', 'process_name','lba', 'request_size', 'operation', 'major_no','minor_no','md5']
    trace_data = pd.read_csv(filename, names=columns,sep=' ')
    trace_data= trace_data.drop(['major_no','minor_no','md5'],axis=1)
    trace_data.request_size = trace_data.request_size.apply(lambda x:x*512)
    trace_data['ts_second'] = (trace_data['timestamp'] / 1000000).astype(int)

    return trace_data

# Perform the KS test for all candidate distributions
def fit_distributions(data, distributions):
    results = []
    FittingResult = namedtuple('FittingResult', ['distribution', 'params', 'ks_stat', 'p_value'])

    for distribution in distributions:
        if distribution.name == 'hyperexpon':
            # Custom parameter estimation for the hyper-exponential distribution
            def neg_log_likelihood(params):
                lambda1, lambda2, p = params
                pdf = hyperexpon.pdf(data, lambda1, lambda2, p)
                return -np.sum(np.log(pdf))

            initial_guess = (1, 1, 0.5)
            bounds = [(0, None), (0, None), (0, 1)]
            res = minimize(neg_log_likelihood, initial_guess, bounds=bounds)
            params = res.x

        else:
            params = distribution.fit(data)
        ks_stat, p_value = st.kstest(data, distribution.cdf, args=params)
        results.append(FittingResult(distribution, params, ks_stat, p_value))

    return results

# Find the best-fitting distribution based on p-value and KS statistic
def find_best_fitting_distribution(results):
    best_fit = max(results, key=lambda r: (r.p_value, -r.ks_stat))
    return best_fit


def load_and_show_figure(filename):
    with open(filename, 'rb') as f:
        fig = pickle.load(f)
        plt.show()


def fit_and_plot(metric, metric_name, xlabel,best_fit_filename,subplots_filename):
    results = fit_distributions(metric, DISTRIBUTIONS)
    best_fit = find_best_fitting_distribution(results)

    print(f"{metric_name} - Best fitting distribution: {best_fit.distribution.name}")
    print(f"Parameters: {best_fit.params}")
    print(f"KS Statistic: {best_fit.ks_stat}")
    print(f"P-Value: {best_fit.p_value}")

    # Plot the best-fitting distribution
    # Save the best-fitting distribution
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(metric, bins=50, density=True, alpha=0.5, color='steelblue')
    x = np.linspace(metric.min(), metric.max(), 1000)
    pdf = best_fit.distribution.pdf(x, *best_fit.params)
    ax.plot(x, pdf, 'r-', lw=2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Density')
    ax.set_title(f"Best Fitting Distribution: {best_fit.distribution.name}")
    #Saving using pickle
    with open(best_fit_filename+'.pkl', 'wb') as f:
        pickle.dump(fig, f)
    plt.close(fig)


    # Plot subplots for all other distributions
        
    num_distributions = len(DISTRIBUTIONS)
    num_cols = 2
    num_rows = num_distributions // num_cols + int(num_distributions % num_cols > 0)

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(10, 5 * num_rows))
    fig.subplots_adjust(hspace=0.5, wspace=0.3)

    x = np.linspace(metric.min(), metric.max(), 1000)

    for index, result in enumerate(results):
        row, col = index // num_cols, index % num_cols
        ax = axs[row, col]
        ax.hist(metric, bins=50, density=True, alpha=0.5, color='steelblue')
        pdf = result.distribution.pdf(x, *result.params)
        ax.plot(x, pdf, 'r-', lw=2)
        ax.set_xlabel('X')
        ax.set_ylabel('Density')
        ax.set_title(f"{result.distribution.name}: p={result.p_value:.2e}, KS={result.ks_stat:.2f}")


    with open(subplots_filename+'.pkl', 'wb') as f:
        pickle.dump(fig, f)
    plt.close(fig)

def main():
    DISTRIBUTIONS.append(hyperexpon)
    trace_filename = 'trace_home.csv' 
    distribution_metrics ={}
    trace_data = load_trace_data(trace_filename)
    # print(trace_data.head(10))
    # Calculate inter-arrival times
    # trace_data['timestamp'] = pd.to_datetime(trace_data['timestamp'])
    trace_data = trace_data.sort_values(by='timestamp')
    inter_arrival_times = trace_data['timestamp'].diff().dropna()
    distribution_metrics['IAT']=inter_arrival_times
    # Request Sizes
    request_sizes = trace_data['request_size']
    distribution_metrics['Request_Size']=request_sizes
    
    # SPATIAL LBA DISTRIBUTION
    lba_metric=trace_data['lba']
    distribution_metrics['lba']=lba_metric

    # Operation Type 
    op_type_metric = trace_data['operation'].map({"R":0,"W":1})
    distribution_metrics['rw']= op_type_metric

    for metric in distribution_metrics.keys():
        fit_and_plot(distribution_metrics[metric],metric,metric, str(metric+'_best_fit'),str(metric+'_subplots'))
    

    # LOading pickle object to check, remove later
    # load_and_show_figure('lba_best_fit.pkl')



if __name__ == "__main__":
    main()

