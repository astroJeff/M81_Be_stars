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

    A, t0, delta_t = x

    ln_likelihood = 0.0

    # Iterate through data
    for d in data:

        # Number of observed counts
        counts = d['SRC_CNTS']

        # Contribution to counts from the source
        lambda_source = kernel(d['MJD'], A, t0, delta_t)

        # Contribution to counts from the background
        lambda_background = d['BKG_CNTS']

        # Calculate the natural log of the poisson probability
#         ln_likelihood += counts * np.log(lambda_source+lambda_background) - len(data) * (lambda_source+lambda_background)
        ln_likelihood += np.log(poisson(lambda_source+lambda_background, counts))

    return ln_likelihood

def ln_posterior(x, data, kernel):

    A, t0, delta_t = x

    # Prior
    if A < 0.01 or A > 100.0: return -np.inf
    if t0 < np.min(data['MJD']) or t0 > np.max(data['MJD']): return -np.inf
    if delta_t < 2.0 or delta_t > 20.0: return -np.inf

    # Likelihood
    ll = ln_likelihood(x, data, kernel)

    return ll

def run_emcee(data, nwalkers, p0):

    print("Running Gaussian model...")
    sampler_gaussian = emcee.EnsembleSampler(nwalkers=nwalkers, dim=3, lnpostfn=ln_posterior, args=[data, gaussian])
    pos,prob,state = sampler_gaussian.run_mcmc(p0, N=1200)
    print("... finished Gaussian model.")

    print("Running Epanechnikov model...")
    sampler_epan = emcee.EnsembleSampler(nwalkers=nwalkers, dim=3, lnpostfn=ln_posterior, args=[data, epanechnikov])
    pos,prob,state = sampler_epan.run_mcmc(p0, N=1200)
    print("... finished Epanechnikov model.")

    print("Running exponential model...")
    sampler_exp = emcee.EnsembleSampler(nwalkers=nwalkers, dim=3, lnpostfn=ln_posterior, args=[data, exponential])
    pos,prob,state = sampler_exp.run_mcmc(p0, N=1200)
    print("... finished exponential model.")

    return sampler_gaussian, sampler_epan, sampler_exp


def plot_walkers(filename, sampler_gaussian, sampler_epan, sampler_exp):

    fig, ax = plt.subplots(3, 3, figsize=(15,6))

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
    for idx in np.random.randint(len(flat_chain_gaussian_good.T[0]), size=100):
        A, t0, delta_t = flat_chain_gaussian_good[idx]
        ax[0].plot(times, gaussian(times, A, t0, delta_t), color='k', alpha=0.05)
    ax[0].set_title("Gaussian Model")

    # Epanechnikov
    for idx in np.random.randint(len(flat_chain_epan_good.T[0]), size=100):
        A, t0, delta_t = flat_chain_epan_good[idx]
        ax[1].plot(times, epanechnikov(times, A, t0, delta_t), color='k', alpha=0.05)
    ax[1].set_title("Epanechnikov Model")

    # Exponential
    for idx in np.random.randint(len(flat_chain_exp_good.T[0]), size=100):
        A, t0, delta_t = flat_chain_exp_good[idx]
        ax[2].plot(times, exponential(times, A, t0, delta_t), color='k', alpha=0.05)
    ax[2].set_title("Gaussian Model")


    for i in range(3):
        ax[i].scatter(data['MJD'], data['NET_CNTS'])
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

        # To only get the first system
        if int(file_root[-2:]) > 15: break


        # Load data
        dtype = [("MJD","f8"), ("SRC_CNTS","f8"), ("BKG_CNTS","f8"), ("NET_CNTS","f8"), ("ERR_LOW","f8"),("ERR_HIG","f8")]
        data = np.genfromtxt(folder+filename, dtype=dtype)


        # Initialize walkers
        A_set = np.random.normal(loc = 10.0, scale=0.1, size=nwalkers)
        t0_set = np.random.normal(loc = 0.5 * (np.max(data['MJD']) - np.min(data['MJD'])) + np.min(data['MJD']), scale=1.0, size=nwalkers)
        delta_t_set = np.random.normal(loc = 0.1*(np.max(data['MJD']) - np.min(data['MJD'])), scale=0.1, size=nwalkers)
        p0 = np.array([A_set, t0_set, delta_t_set]).T

        # Run emcee
        sampler_gaussian, sampler_epan, sampler_exp = run_emcee(data, nwalkers, p0)

        # Create plot of walker evolution
        plot_walkers("../figures/"+file_root+"_chains.jpg", sampler_gaussian, sampler_epan, sampler_exp)

        # Remove burn-in
        chain_gaussian_good = sampler_gaussian.chain[:,200:,:]
        n_chains, length, n_var = chain_gaussian_good.shape
        flat_chain_gaussian_good = chain_gaussian_good.reshape((n_chains*length, n_var))

        chain_epan_good = sampler_epan.chain[:,200:,:]
        n_chains, length, n_var = chain_epan_good.shape
        flat_chain_epan_good = chain_epan_good.reshape((n_chains*length, n_var))

        chain_exp_good = sampler_exp.chain[:,200:,:]
        n_chains, length, n_var = chain_exp_good.shape
        flat_chain_exp_good = chain_exp_good.reshape((n_chains*length, n_var))

        # Create corner plots
        corner.corner(flat_chain_gaussian_good)
        plt.tight_layout()
        plt.savefig("../figures/"+file_root+"_gaussian_corner.pdf")
        corner.corner(flat_chain_epan_good)
        plt.tight_layout()
        plt.savefig("../figures/"+file_root+"_epanechnikov_corner.pdf")
        corner.corner(flat_chain_exp_good)
        plt.tight_layout()
        plt.savefig("../figures/"+file_root+"_exponential_corner.pdf")

        # Plot posterior samples
        plot_light_curve("../figures/"+file_root+"_light_curve.pdf", data, flat_chain_gaussian_good, flat_chain_epan_good, flat_chain_exp_good)
