import os
from pathlib import Path
import pickle
# Force JAX to ignore TPU/GPU backends in this environment.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")
jnp = jax.numpy
jsp = jax.scipy
random = jax.random
grad = jax.grad
jit = jax.jit
vmap = jax.vmap
from jaxns import Prior, Model, NestedSampler
import tensorflow_probability.substrates.jax as tfp
tfpd = tfp.distributions
import matplotlib.pyplot as plt
from mbh_jaxns_normal import save_nested_sampler_results, load_nested_sampler_results, linear_correlation_exp, import_bh_data

DEBUG = False

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

    @jit
    def log_likelihood_normal(a, b, true_sigma_gc):
        M_max = 1.0e+12
        return jnp.sum(tfpd.Normal(linear_correlation_exp(true_sigma_gc, a, b), M_equiv_err).log_prob(M)+
        tfpd.Normal(true_sigma_gc, sigma_gc_equiv_err).log_prob(sigma_gc)) - N_bh * jnp.log(M_max)
    
    def prior_model_normal():
        a = yield Prior(tfpd.Uniform(-10.0, 10.0), name="a")
        b = yield Prior(tfpd.Uniform(-5.0, 5.0), name="b")
        true_sigma_gc = yield Prior(tfpd.Sample(tfpd.Uniform(1.0e-14, 5.0e+3), sample_shape=(N_bh,)), name="true_sigma_gc")
        return a, b, true_sigma_gc
    
    model = Model(prior_model_normal, log_likelihood_normal)
    model.sanity_check(random.PRNGKey(0), S=10)

    bh_ns = load_nested_sampler_results("results/gaussian_ns_results.pkl")
    ns = NestedSampler(model, s=1000, k=model.U_ndims, num_live_points=model.U_ndims*100000)
    ns.plot_cornerplot(bh_ns, save_name='results/gaussian_full_corner.png')

    exit()
