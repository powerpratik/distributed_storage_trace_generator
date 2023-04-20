from inspect import trace
from os import sep
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
from collections import namedtuple
from scipy.optimize import minimize
import pickle
from scipy import stats

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
    '''
    column 0. acess start address in sectors 

	column 1. access size in sectors   note size should be time of 8 because basic block size is 4 KB or 8 sector. But mmc driver add addition sector to some of the size and thus you need to clean it.

	column 2. access size in byte.
	
	column 3. access type & waiting or not: the lowest bit to indicate read or write (0 is read and 1 is write), the third bit represent the request have be waiting or not (4 indicate no wait while 0 indicate has been waiting some time)
		For instance 5 represents that the request is a write request and the request does not wait before been processed, which indicates the queue is empty when the request comes.
	
	column 4. request generate time (the request been generated and inserted into request queue).

	column 5. request process start time (the request been fetched and dequeue by mmc driver from request queue and begin to process)

	column 6. time of request been submitted to hardware (the time that the driver issue the request to the hardware)

	column 7. request finish time (time of the callback function been invoked after request completion)
    '''

    columns = ['lba', 'blocks_no', 'size','type_op', 'req_gen_time', 'req_process_st_time', 'hardware_req_submit_time','req_finish_time']
    trace_data = pd.read_csv(filename, names=columns,sep='\s+')
    trace_data= trace_data.drop(['blocks_no'],axis=1)
    # trace_data.request_size = trace_data.request_size.apply(lambda x:x*512)
    # trace_data['ts_second'] = (trace_data['timestamp'] / 1000000).astype(int)

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

def plot_cdf(metric,metric_name,best_fit,x,results,xlabel):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(metric, bins=50, density=True, alpha=0.5, color='steelblue', cumulative=True, histtype='step')
    x = np.linspace(metric.min(), metric.max(), 1000)
    cdf = best_fit.distribution.cdf(x, *best_fit.params)
    ax.plot(x, cdf, 'r-', lw=2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Cumulative Probability')
    ax.set_title(f"Best Fitting Distribution: {best_fit.distribution.name}")

    # with open(metric_name+'_best_fit_cdf.pkl', 'wb') as f:
    #     pickle.dump(fig, f)
    # plt.show()
    plt.close(fig)




def fit_plot_short(metric, metric_name,xlabel):
    
    results = fit_distributions(metric, DISTRIBUTIONS)
    best_fit = find_best_fitting_distribution(results)

    print(f"{metric_name} - Best fitting distribution: {best_fit.distribution.name}")
    print(f"Parameters: {best_fit.params}")
    print(f"KS Statistic: {best_fit.ks_stat}")
    print(f"P-Value: {best_fit.p_value}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 5))

    # Plot the best-fitting distribution
    ax1.hist(metric, bins=50, density=True, alpha=0.5, color='steelblue')
    x = np.linspace(metric.min(), metric.max(), 50)
    pdf = best_fit.distribution.pdf(x, *best_fit.params)
    plot_cdf(metric,metric_name,best_fit,x,results,xlabel)
    # compute_goodness_of_fit(metric,best_fit)
    ax1.plot(x, pdf, 'r', lw=2)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel('Density')
    ax1.set_title(f"Best Fitting Distribution: {best_fit.distribution.name}")

    # Plot all other distributions
    ax2.hist(metric, bins=50, density=True, alpha=0.5, color='steelblue')
    for result in results:
        pdf = result.distribution.pdf(x, *result.params)
        ax2.plot(x, pdf, lw=2, label=f"{result.distribution.name}: p={result.p_value:.2e}, KS={result.ks_stat:.2f}")

    ax2.legend()
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel('Density')
    ax2.set_title("Comparison of All Distributions")

    plt.tight_layout()
    plt.show()
    # plt.savefig(output_filename, bbox_inches='tight')
    # with open(metric_name+'_best_fit.pkl', 'wb') as f:
    #     pickle.dump(fig, f)
    plt.close(fig)

def compute_cdf(data):
    sorted_data = np.sort(data)
    cdf_values = np.linspace(0, 1, len(sorted_data))
    return sorted_data, cdf_values


def compute_goodness_of_fit(metric,best_fit):
    # JSD COMPUTE
    # print(metric)
    # print(best_fit.distribution)
    # m=0.5* ( np.asarray(metric) + np.asarray(best_fit.distribution))
    # jsd = 0.5*(stats.entropy(metric, m) + stats.entropy(metric, m))
    # R^2 COMPUTE
    slope, intercept, r_value, p_value, std_err = stats.linregress(metric, best_fit.distribution)
    r_squared = r_value**2


    #LOG-LIKELYHOOD COMPUTE
    log_likelihood = np.sum(stats.norm.logpdf(best_fit.distribution, loc=slope * metric + intercept, scale=std_err))

    r_squared_sorted, r_squared_cdf = compute_cdf(r_squared)
    log_likelihood_sorted, log_likelihood_cdf = compute_cdf(log_likelihood)

    plt.figure(figsize=(10, 6))

    plt.plot(r_squared_sorted, r_squared_cdf, label='R^2', marker='o', linestyle='-')
    plt.plot(log_likelihood_sorted, log_likelihood_cdf, label='Log-Likelihood', marker='s', linestyle='-')

    plt.xlabel('Metric Value')
    plt.ylabel('CDF')
    plt.title('Metrics vs CDFs')
    plt.legend()
    plt.grid()

    plt.show()





def main():
    DISTRIBUTIONS.append(hyperexpon)
    trace_filename = 'merged_mobile_traces.txt' 
    distribution_metrics ={}
    trace_data = load_trace_data(trace_filename)
    # print(trace_data.head(10))
    # Calculate inter-arrival times
    # trace_data['timestamp'] = pd.to_datetime(trace_data['timestamp'])
    trace_data = trace_data.sort_values(by='req_gen_time')
    inter_arrival_times = trace_data['req_gen_time'].diff().dropna()
    print(trace_data.req_gen_time)
    print((inter_arrival_times))
    print(min(inter_arrival_times))
    print(max(inter_arrival_times))
    distribution_metrics['IAT']=inter_arrival_times
    # Request Sizes
    request_sizes = trace_data['size']
    distribution_metrics['Request_Size']=request_sizes

    # Service Time
    st_metric= trace_data['req_finish_time']-trace_data['req_process_st_time']
    distribution_metrics['st']=st_metric

    #Response Time
    rt_metric = trace_data['req_finish_time']-trace_data['req_gen_time']
    distribution_metrics['rt']=rt_metric

    for metric in distribution_metrics.keys():
        # fit_and_plot(distribution_metrics[metric],metric,metric, str(metric+'_best_fit'),str(metric+'_subplots'))
        fit_plot_short(distribution_metrics[metric],metric,str(metric+'_best_fit'))
    

    #LOading pickle object to check, remove later
    # load_and_show_figure('lba_best_fit.pkl')



if __name__ == "__main__":
    main()

