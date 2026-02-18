import os, sys
from pathlib import Path
# Force JAX to ignore TPU/GPU backends in this environment.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
import numpy as np
import scipy.optimize as sp_opt
import jax
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")
jnp = jax.numpy
jsp = jax.scipy
random = jax.random
grad = jax.grad
jit = jax.jit
vmap = jax.vmap
import pickle
import tensorflow_probability.substrates.jax as tfp
tfpd = tfp.distributions
from quadax import quadgk, quadcc
import matplotlib.pyplot as plt
from utils import save_nested_sampler_results, load_nested_sampler_results, import_bh_data
from utils import logskewnormal_logpdf, skewnormal_cdf, logskewnormal_mean

DEBUG = True

@jit
def logskewnormal_system_residuals(mean, sigma, shape, x, deltax_low, deltax_high, quantile_low=0.34, quantile_high=0.34):
    mean_from_model = logskewnormal_mean(mean=mean, sigma=sigma, shape=shape)[0]
    logx = jnp.log(x)
    logx_high = jnp.log(x + deltax_high)
    logx_low = jnp.log(x - deltax_low)
    low_prob_mass = skewnormal_cdf(logx, mean=mean, sigma=sigma, shape=shape)-skewnormal_cdf(logx_low, mean=mean, sigma=sigma, shape=shape)
    up_prob_mass = skewnormal_cdf(logx_high, mean=mean, sigma=sigma, shape=shape)-skewnormal_cdf(logx, mean=mean, sigma=sigma, shape=shape)
    
    return jnp.array([
        mean_from_model - x,
        low_prob_mass - quantile_low,
        up_prob_mass - quantile_high,
    ])


if __name__ == "__main__":
    bh_data = import_bh_data("data/bh_table_1.txt")
    print(f"Loaded columns: {list(bh_data.keys())}")
    if DEBUG:
        for key in bh_data.keys():
            try:
                print(f"{key} type: {bh_data[key][0].dtype}")
            except:
                print(f"{key} type: {type(bh_data[key][0])}")

    M = bh_data["M"]
    sigma_gc = bh_data["sigma_gc"]
    M_equiv_err = 0.5*(bh_data["dM_low"]+bh_data["dM_high"])
    sigma_gc_equiv_err = 0.5*(bh_data["sigma_gc_low"]+bh_data["sigma_gc_high"])
    N_bh = len(M)

    exit()