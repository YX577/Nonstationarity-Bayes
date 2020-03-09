# from IPython.core.pylabtools import figsize
import numpy as np
from matplotlib import pyplot as plt
import pymc3 as pm
import theano.tensor as tt
import pandas as pd
import scipy.stats as ss
from pymc3.distributions.dist_math import bound

#%%
"""
count_data = np.loadtxt("data/txtdata.csv")
n_count_data = len(count_data)
"""
def le_dados_sugar_creek():
    df = pd.read_csv('../data/sugar_creek_data.csv', sep=';')
    dataset=pd.Series(df.discharge.values, index=df.year)
    dataset=dataset.apply(lambda x: float(x.replace(',','.')) if isinstance(x,str) else x)
    dataset=dataset.map(lambda x: x*0.0283168465925) #conversão de sistema de unidades para m³/s
    return dataset[:-8] #-8 leva em consideração dados até 2009 (análises de Villarini)
dataset=le_dados_sugar_creek()
n_count_data = len(dataset)

#%%
#    calibration_data=annual_max[:int(3*len(annual_max)/4)]
calibration_data = dataset
#    calibration_data = pd.Series(annual_max['58790000'].values, index=annual_max.index)
locm = calibration_data.mean()
locs = calibration_data.std() / (np.sqrt(len(calibration_data)))
scalem = calibration_data.std()
scales = calibration_data.std() / (np.sqrt(2 * (len(calibration_data) - 1)))
with pm.Model() as model:
    # Priors for unknown model parameters
    c1 = pm.Beta('c1', alpha=6,
                beta=9)  # c=x-0,5: transformation in gev_logp is required due to Beta domain between 0 and 1
    loc1 = pm.Normal('loc1', mu=locm, sd=locs)
    scale1 = pm.Normal('scale1', mu=scalem, sd=scales)

    c2 = pm.Beta('c2', alpha=6,
                beta=9)  # c=x-0,5: transformation in gev_logp is required due to Beta domain between 0 and 1
    loc2 = pm.Normal('loc2', mu=locm, sd=locs)
    scale2 = pm.Normal('scale2', mu=scalem, sd=scales)

    # Likelihood (sampling distribution) of observations | Since GEV is not implemented in pymc a custom log likelihood function was created
    def gev_logp(value):
        scaled = (value - loc_) / scale_
        logp = -(tt.log(scale_)
                 + (((c_ - 0.5) + 1) / (c_ - 0.5) * tt.log1p((c_ - 0.5) * scaled)
                    + (1 + (c_ - 0.5) * scaled) ** (-1 / (c_ - 0.5))))
        bound1 = loc_ - scale_ / (c_ - 0.5)
        bounds = tt.switch((c_ - 0.5) > 0, value > bound1, value < bound1)
        return bound(logp, bounds, c_ != 0)


    tau = pm.DiscreteUniform("tau", lower=0, upper=n_count_data - 1)
    idx = np.arange(n_count_data)  # Index
    c_ = pm.math.switch(tau > idx, c1, c2)
    loc_ = pm.math.switch(tau > idx, loc1, loc2)
    scale_ = pm.math.switch(tau > idx, scale1, scale2)
    gev = pm.DensityDist('gev', gev_logp, observed=calibration_data)

    trace = pm.sample(1000, chains=1, cores=2, progressbar=True)

#%%
loc1_samples = trace['loc1']
loc2_samples = trace['loc2']
scale1_samples = trace['scale1']
scale2_samples = trace['scale2']
c1_samples = trace['c1']
c2_samples = trace['c2']
tau_samples = trace['tau']
# pm.traceplot(trace)
# geweke_plot = pm.geweke(trace, 0.05, 0.5, 20)
# gelman_and_rubin = pm.diagnostics.gelman_rubin(trace)
posterior = pm.trace_to_dataframe(trace)
summary = pm.summary(trace)

#%%
# figsize(12.5, 10)
# histogram of the samples:

ax = plt.subplot(311)
ax.set_autoscaley_on(False)

plt.hist(lambda_1_samples, histtype='stepfilled', bins=30, alpha=0.85,
         label="posterior of $\lambda_1$", color="#A60628", normed=True)
plt.legend(loc="upper left")
plt.title(r"""Posterior distributions of the variables
    $\lambda_1,\;\lambda_2,\;\tau$""")
plt.xlim([15, 30])
plt.xlabel("$\lambda_1$ value")

ax = plt.subplot(312)
ax.set_autoscaley_on(False)
plt.hist(lambda_2_samples, histtype='stepfilled', bins=30, alpha=0.85,
         label="posterior of $\lambda_2$", color="#7A68A6", density=True)
plt.legend(loc="upper left")
plt.xlim([15, 30])
plt.xlabel("$\lambda_2$ value")

plt.subplot(313)
w = 1.0 / tau_samples.shape[0] * np.ones_like(tau_samples)
plt.hist(tau_samples, bins=n_count_data, alpha=1,
         label=r"posterior of $\tau$",
         color="#467821", weights=w, rwidth=2.)
plt.xticks(np.arange(n_count_data))

plt.legend(loc="upper left")
plt.ylim([0, .75])
plt.xlim([35, len(count_data)-20])
plt.xlabel(r"$\tau$ (in days)")
plt.ylabel("probability")
plt.show()
#%%
# figsize(12.5, 5)
# tau_samples, lambda_1_samples, lambda_2_samples contain
# N samples from the corresponding posterior distribution
N = tau_samples.shape[0]
expected_texts_per_day = np.zeros(n_count_data)
for day in range(0, n_count_data):
    # ix is a bool index of all tau samples corresponding to
    # the switchpoint occurring prior to value of 'day'
    ix = day < tau_samples
    # Each posterior sample corresponds to a value for tau.
    # for each day, that value of tau indicates whether we're "before"
    # (in the lambda1 "regime") or
    #  "after" (in the lambda2 "regime") the switchpoint.
    # by taking the posterior sample of lambda1/2 accordingly, we can average
    # over all samples to get an expected value for lambda on that day.
    # As explained, the "message count" random variable is Poisson distributed,
    # and therefore lambda (the poisson parameter) is the expected value of
    # "message count".
    expected_texts_per_day[day] = (lambda_1_samples[ix].sum()
                                   + lambda_2_samples[~ix].sum()) / N


plt.plot(range(n_count_data), expected_texts_per_day, lw=4, color="#E24A33",
         label="expected number of text-messages received")
plt.xlim(0, n_count_data)
plt.xlabel("Day")
plt.ylabel("Expected # text-messages")
plt.title("Expected number of text-messages received")
plt.ylim(0, 60)
plt.bar(np.arange(len(count_data)), count_data, color="#348ABD", alpha=0.65,
        label="observed texts per day")

plt.legend(loc="upper left")
plt.show()
