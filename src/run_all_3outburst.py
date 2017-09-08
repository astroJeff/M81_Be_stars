import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scipy import stats
from scipy import optimize as opt
from scipy.misc import factorial
import os

import emcee
import corner



def gaussian(t, A, t0, delta_t):
    return A * np.exp(-(t-t0)**2/(2 * delta_t**2) )

def epanechnikov(t, A, t0, delta_t):
    y = A * (1.0 - (t-t0)**2 / delta_t**2)
    return y * (y>0)

def exponential(t, A, t0, delta_t):
    return A * np.exp(-(t-t0)/(delta_t)) * (t>t0)

def poisson(mu, x):
    return mu**x * np.exp(-mu) / factorial(x)

def neg_ln_likelihood(x, data, kernel):
    return -ln_likelihood(x, data, kernel)

def ln_likelihood(x, data, kernel):

    A_1, t0_1, delta_t_1, A_2, t0_2, delta_t_2, A_3, t0_3, delta_t_3 = x

    ln_likelihood = 0.0

    # Iterate through data
    for d in data:

        # Number of observed counts
        counts = d['SRC_CNTS']

        # Contribution to counts from the source
        lambda_source_1 = kernel(d['MJD'], A_1, t0_1, delta_t_1)
        lambda_source_2 = kernel(d['MJD'], A_2, t0_2, delta_t_2)
        lambda_source_3 = kernel(d['MJD'], A_3, t0_3, delta_t_3)

        # Contribution to counts from the background
        lambda_background = d['BKG_CNTS']

        # Calculate the natural log of the poisson probability
#         ln_likelihood += counts * np.log(lambda_source+lambda_background) - len(data) * (lambda_source+lambda_background)
        ln_likelihood += np.log(poisson(lambda_source_1+lambda_source_2+lambda_source_3+lambda_background, counts))

    return ln_likelihood

def ln_posterior(x, data, kernel):

    A_1, t0_1, delta_t_1, A_2, t0_2, delta_t_2, A_3, t0_3, delta_t_3 = x

    # Prior
    if A_1 < 0.01 or A_1 > 100.0: return -np.inf
    if A_2 < 0.01 or A_2 > 100.0: return -np.inf
    if A_3 < 0.01 or A_3 > 100.0: return -np.inf
    if t0_1 < np.min(data['MJD']) or t0_1 > np.max(data['MJD']): return -np.inf
    if t0_2 < np.min(data['MJD']) or t0_2 > np.max(data['MJD']): return -np.inf
    if t0_3 < np.min(data['MJD']) or t0_3 > np.max(data['MJD']): return -np.inf
    if t0_1 > t0_2: return -np.inf
    if t0_2 > t0_3: return -np.inf
    if kernel == 'epanechnikov' or kernel == 'exponential':
        if delta_t_1 < 2.0 or delta_t_1 > 10.0: return -np.inf
        if delta_t_2 < 2.0 or delta_t_2 > 10.0: return -np.inf
        if delta_t_3 < 2.0 or delta_t_3 > 10.0: return -np.inf
    else:
        if delta_t_1 < 2.0 or delta_t_1 > 5.0: return -np.inf
        if delta_t_2 < 2.0 or delta_t_2 > 5.0: return -np.inf
        if delta_t_3 < 2.0 or delta_t_3 > 5.0: return -np.inf

    # Likelihood
    ll = ln_likelihood(x, data, kernel)

    return ll

def run_emcee(data, nwalkers, p0):

    print("Running Gaussian model...")
    sampler_gaussian = emcee.EnsembleSampler(nwalkers=nwalkers, dim=9, lnpostfn=ln_posterior, args=[data, gaussian])
    pos,prob,state = sampler_gaussian.run_mcmc(p0, N=2500)
    print("... finished Gaussian model.")

    print("Running Epanechnikov model...")
    sampler_epan = emcee.EnsembleSampler(nwalkers=nwalkers, dim=9, lnpostfn=ln_posterior, args=[data, epanechnikov])
    pos,prob,state = sampler_epan.run_mcmc(p0, N=2500)
    print("... finished Epanechnikov model.")

    print("Running exponential model...")
    sampler_exp = emcee.EnsembleSampler(nwalkers=nwalkers, dim=9, lnpostfn=ln_posterior, args=[data, exponential])
    pos,prob,state = sampler_exp.run_mcmc(p0, N=2500)
    print("... finished exponential model.")

    return sampler_gaussian, sampler_epan, sampler_exp


def plot_walkers(filename, sampler_gaussian, sampler_epan, sampler_exp):

    fig, ax = plt.subplots(9, 3, figsize=(15,18))

    # Gaussian Model
    n_chains, length, n_var = sampler_gaussian.chain.shape
    ax[0,0].set_title('Gaussian Model')
    for i in range(n_var):
        for j in range(n_chains):
            ax[i,0].plot(sampler_gaussian.chain[j,:,i], color='k', alpha=0.1)

    # Epanechnikov Model
    n_chains, length, n_var = sampler_epan.chain.shape
    ax[0,1].set_title('Epanechnikov Model')
    for i in range(n_var):
        for j in range(n_chains):
            ax[i,1].plot(sampler_epan.chain[j,:,i], color='k', alpha=0.1)

    # Exponential Model
    n_chains, length, n_var = sampler_exp.chain.shape
    ax[0,2].set_title('Exponential Model')
    for i in range(n_var):
        for j in range(n_chains):
            ax[i,2].plot(sampler_exp.chain[j,:,i], color='k', alpha=0.1)


    plt.tight_layout()
#     plt.show()
    plt.savefig(filename, rasterized=True)


def plot_light_curve(filename, data, flat_chain_gaussian_good, flat_chain_epan_good, flat_chain_exp_good):

    fig, ax = plt.subplots(3, 1, figsize=(5, 8), sharex=True)


    times = np.linspace(np.min(data['MJD']), np.max(data['MJD']), 100)


    # Gaussian
    for idx in np.random.randint(len(flat_chain_gaussian_good.T[0]), size=200):
        A_1, t0_1, delta_t_1, A_2, t0_2, delta_t_2, A_3, t0_3, delta_t_3 = flat_chain_gaussian_good[idx]
        ax[0].plot(times, gaussian(times, A_1, t0_1, delta_t_1)+gaussian(times, A_2, t0_2, delta_t_2)+gaussian(times, A_3, t0_3, delta_t_3),
                   color='k', alpha=0.05, linewidth=1.0)
    ax[0].set_title("Gaussian Model")

    # Epanechnikov
    for idx in np.random.randint(len(flat_chain_epan_good.T[0]), size=200):
        A_1, t0_1, delta_t_1, A_2, t0_2, delta_t_2, A_3, t0_3, delta_t_3 = flat_chain_epan_good[idx]
        ax[1].plot(times, epanechnikov(times, A_1, t0_1, delta_t_1)+epanechnikov(times, A_2, t0_2, delta_t_2)+epanechnikov(times, A_3, t0_3, delta_t_3),
                   color='k', alpha=0.05, linewidth=1.0)
    ax[1].set_title("Epanechnikov Model")

    # Exponential
    for idx in np.random.randint(len(flat_chain_exp_good.T[0]), size=200):
        A_1, t0_1, delta_t_1, A_2, t0_2, delta_t_2, A_3, t0_3, delta_t_3 = flat_chain_exp_good[idx]
        ax[2].plot(times, exponential(times, A_1, t0_1, delta_t_1)+exponential(times, A_2, t0_2, delta_t_2)+exponential(times, A_3, t0_3, delta_t_3),
                   color='k', alpha=0.05, linewidth=1.0)
    ax[2].set_title("Exponential Model")


    for i in range(3):
        errors = [data['NET_CNTS']-data['ERR_LOW'], data['ERR_HIGH']-data['NET_CNTS']]
        ax[i].errorbar(data['MJD'], data['NET_CNTS'], yerr=errors, fmt='o', ecolor='k')
        # ax[i].scatter(data['MJD'], data['NET_CNTS'])
        ax[i].set_xlabel('MJD')
        ax[i].set_ylabel('Total Counts minus Background')


    plt.tight_layout()
    plt.savefig(filename)




################# RUN MODELS ##################



folder = "../data/"
nwalkers = 32


for filename in os.listdir(folder):
    if filename.startswith("src_") and filename.endswith(".dat"):

        file_root = filename[:-4]

        if file_root != 'src_213' and file_root != 'src_034': continue

        # Load data
        dtype = [("MJD","f8"), ("SRC_CNTS","f8"), ("BKG_CNTS","f8"), ("NET_CNTS","f8"), ("ERR_LOW","f8"),("ERR_HIGH","f8")]
        data = np.genfromtxt(folder+filename, dtype=dtype)
        data = data[~np.isnan(data['SRC_CNTS'])]


        # Initialize walkers
        A_set_1 = np.random.normal(loc = 10.0, scale=0.1, size=nwalkers)
        A_set_2 = np.random.normal(loc = 10.0, scale=0.1, size=nwalkers)
        A_set_3 = np.random.normal(loc = 10.0, scale=0.1, size=nwalkers)
        t0_set_1 = np.random.normal(loc = 0.4 * (np.max(data['MJD']) - np.min(data['MJD'])) + np.min(data['MJD']), scale=1.0, size=nwalkers)
        t0_set_2 = np.random.normal(loc = 0.6 * (np.max(data['MJD']) - np.min(data['MJD'])) + np.min(data['MJD']), scale=1.0, size=nwalkers)
        t0_set_3 = np.random.normal(loc = 0.6 * (np.max(data['MJD']) - np.min(data['MJD'])) + np.min(data['MJD']), scale=1.0, size=nwalkers)
        delta_t_set_1 = np.random.normal(loc = 0.1*(np.max(data['MJD']) - np.min(data['MJD'])), scale=0.1, size=nwalkers)
        delta_t_set_2 = np.random.normal(loc = 0.1*(np.max(data['MJD']) - np.min(data['MJD'])), scale=0.1, size=nwalkers)
        delta_t_set_3 = np.random.normal(loc = 0.1*(np.max(data['MJD']) - np.min(data['MJD'])), scale=0.1, size=nwalkers)
        p0 = np.array([A_set_1, t0_set_1, delta_t_set_1, A_set_2, t0_set_2, delta_t_set_2, A_set_3, t0_set_3, delta_t_set_3]).T

        # Run emcee
        sampler_gaussian, sampler_epan, sampler_exp = run_emcee(data, nwalkers, p0)

        # Create plot of walker evolution
        plot_walkers("../figures/"+file_root+"_chains_3outburst.jpg", sampler_gaussian, sampler_epan, sampler_exp)

        # Remove burn-in
        chain_gaussian_good = sampler_gaussian.chain[:,500:,:]
        n_chains, length, n_var = chain_gaussian_good.shape
        flat_chain_gaussian_good = chain_gaussian_good.reshape((n_chains*length, n_var))

        chain_epan_good = sampler_epan.chain[:,500:,:]
        n_chains, length, n_var = chain_epan_good.shape
        flat_chain_epan_good = chain_epan_good.reshape((n_chains*length, n_var))

        chain_exp_good = sampler_exp.chain[:,500:,:]
        n_chains, length, n_var = chain_exp_good.shape
        flat_chain_exp_good = chain_exp_good.reshape((n_chains*length, n_var))

        # Create corner plots
        corner.corner(flat_chain_gaussian_good)
        plt.tight_layout()
        plt.savefig("../figures/"+file_root+"_gaussian_corner_3outburst.pdf")
        corner.corner(flat_chain_epan_good)
        plt.tight_layout()
        plt.savefig("../figures/"+file_root+"_epanechnikov_corner_3outburst.pdf")
        corner.corner(flat_chain_exp_good)
        plt.tight_layout()
        plt.savefig("../figures/"+file_root+"_exponential_corner_3outburst.pdf")

        # Plot posterior samples
        plot_light_curve("../figures/"+file_root+"_light_curve_3outburst.pdf", data, flat_chain_gaussian_good, flat_chain_epan_good, flat_chain_exp_good)
