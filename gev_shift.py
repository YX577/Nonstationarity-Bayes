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
    df = pd.read_csv('data/sugar_creek_data.csv', sep=';')
    dataset=pd.Series(df.discharge.values, index=df.year, name='discharge')
    dataset=dataset.apply(lambda x: float(x.replace(',','.')) if isinstance(x,str) else x)
    dataset=dataset.map(lambda x: x*0.0283168465925) #conversão de sistema de unidades para m³/s
    return dataset[:-8] #-8 leva em consideração dados até 2009 (análises de Villarini)
dataset=le_dados_sugar_creek()
n_count_data = len(dataset)

# Likelihood (sampling distribution) of observations | Since GEV is not implemented in pymc a custom log likelihood function was created

def gev_shift_1(dataset):
    
    locm = dataset.mean()
    locs = dataset.std() / (np.sqrt(len(dataset)))
    scalem = dataset.std()
    scales = dataset.std() / (np.sqrt(2 * (len(dataset) - 1)))
    with pm.Model() as model:
        # Priors for unknown model parameters
        c1 = pm.Beta('c1', alpha=6,
                    beta=9)  # c=x-0,5: transformation in gev_logp is required due to Beta domain between 0 and 1
        loc1 = pm.Normal('loc1', mu=locm, sd=locs)
        scale1 = pm.Normal('scale1', mu=scalem, sd=scales)
    
        c2 = pm.Beta('c2', alpha=6, beta=9)
        loc2 = pm.Normal('loc2', mu=locm, sd=locs)
        scale2 = pm.Normal('scale2', mu=scalem, sd=scales)
          
        def gev_logp(value):
            scaled = (value - loc_) / scale_
            logp = -(tt.log(scale_)
                     + (((c_ - 0.5) + 1) / (c_ - 0.5) * tt.log1p((c_ - 0.5) * scaled)
                        + (1 + (c_ - 0.5) * scaled) ** (-1 / (c_ - 0.5))))
            bound1 = loc_ - scale_ / (c_ - 0.5)
            bounds = tt.switch((c_ - 0.5) > 0, value > bound1, value < bound1)
            return bound(logp, bounds, c_ != 0)
    
        tau = pm.DiscreteUniform("tau", lower=0, upper=n_count_data - 1)
        idx = np.arange(n_count_data)
        c_ = pm.math.switch(tau > idx, c1, c2)
        loc_ = pm.math.switch(tau > idx, loc1, loc2)
        scale_ = pm.math.switch(tau > idx, scale1, scale2)
        gev = pm.DensityDist('gev', gev_logp, observed=dataset)
        trace = pm.sample(1000, chains=1, progressbar=True)
        
       
    # geweke_plot = pm.geweke(trace, 0.05, 0.5, 20)
    # gelman_and_rubin = pm.diagnostics.gelman_rubin(trace)
    posterior = pm.trace_to_dataframe(trace)
    summary = pm.summary(trace)
    loc1_samples = trace['loc1']
    loc2_samples = trace['loc2']
    scale1_samples = trace['scale1']
    scale2_samples = trace['scale2']
    c1_samples = trace['c1']
    c2_samples = trace['c2']
    tau_samples = trace['tau']
    return summary
    
#summary = gev_shift_1(dataset)

def gev_shift_2(dataset):
    locm = dataset.mean()
    locs = dataset.std() / (np.sqrt(len(dataset)))
    scalem = dataset.std()
    scales = dataset.std() / (np.sqrt(2 * (len(dataset) - 1)))
    with pm.Model() as model:
        # Priors for unknown model parameters
        c1 = pm.Beta('c1', alpha=6,
                    beta=9)  # c=x-0,5: transformation in gev_logp is required due to Beta domain between 0 and 1
        loc1 = pm.Normal('loc1', mu=locm, sd=locs)
        scale1 = pm.Normal('scale1', mu=scalem, sd=scales)
    
        c2 = pm.Beta('c2', alpha=6, beta=9)
        loc2 = pm.Normal('loc2', mu=locm, sd=locs)
        scale2 = pm.Normal('scale2', mu=scalem, sd=scales)
        
        c3 = pm.Beta('c3', alpha=6, beta=9)
        loc3 = pm.Normal('loc3', mu=locm, sd=locs)
        scale3 = pm.Normal('scale3', mu=scalem, sd=scales)
        
        def gev_logp(value):
            scaled = (value - loc_) / scale_
            logp = -(tt.log(scale_)
                     + (((c_ - 0.5) + 1) / (c_ - 0.5) * tt.log1p((c_ - 0.5) * scaled)
                        + (1 + (c_ - 0.5) * scaled) ** (-1 / (c_ - 0.5))))
            bound1 = loc_ - scale_ / (c_ - 0.5)
            bounds = tt.switch((c_ - 0.5) > 0, value > bound1, value < bound1)
            return bound(logp, bounds, c_ != 0)
        
        tau1 = pm.DiscreteUniform("tau1", lower=0, upper=n_count_data - 1)
        tau2 = pm.DiscreteUniform("tau2", lower=0, upper=n_count_data - 1)
        idx = np.arange(n_count_data)
              
        c_ = pm.math.switch(tau2>=idx, pm.math.switch(tau1>=idx, c1, c2), c3)
        loc_ = pm.math.switch(tau2>=idx, pm.math.switch(tau1>=idx, loc1, loc2), loc3)
        scale_ = pm.math.switch(tau2>=idx, pm.math.switch(tau1>=idx, scale1, scale2), scale3)
        gev = pm.DensityDist('gev', gev_logp, observed=dataset)
        trace = pm.sample(1000, chains=1, progressbar=True)
#        posterior = pm.trace_to_dataframe(trace)
        summary = pm.summary(trace)
#        geweke_plot = pm.geweke(trace, 0.05, 0.5, 20)
        return summary

summary= gev_shift_2(dataset)
