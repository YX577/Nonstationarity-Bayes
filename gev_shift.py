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

def gev0_shift_1(dataset):
    
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
    return summary, posterior
    


def gev0_shift_2(dataset):
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
        
        
        tau1 = pm.DiscreteUniform("tau1", lower=0, upper=n_count_data - 2)
        tau2 = pm.DiscreteUniform("tau2", lower=tau1+1, upper=n_count_data - 1)
        idx = np.arange(n_count_data)
              
        c_ = pm.math.switch(tau2>=idx, pm.math.switch(tau1>=idx, c1, c2), c3)
        loc_ = pm.math.switch(tau2>=idx, pm.math.switch(tau1>=idx, loc1, loc2), loc3)
        scale_ = pm.math.switch(tau2>=idx, pm.math.switch(tau1>=idx, scale1, scale2), scale3)
        gev = pm.DensityDist('gev', gev_logp, observed=dataset)
        trace = pm.sample(2000, chains=1, progressbar=True)
        posterior = pm.trace_to_dataframe(trace)
        summary = pm.summary(trace)
#        geweke_plot = pm.geweke(trace, 0.05, 0.5, 20)
        return summary, posterior
    
def gev1_shift_1(dataset, alfa=0.95):
    import scipy.stats as ss
    
    serie_lista=list(dataset)
    ano_resetado=pd.Series(list(range(len(dataset))))
    slope, intercept, r_value, pvalue_regress, std_err=ss.linregress(ano_resetado,dataset.values)
    
    #First regression (loc hiperparameters)
    y_regress=ano_resetado.map(lambda x: x*slope+intercept)
    e=dataset.reset_index(drop=True)-y_regress
    se=(e**2).sum()/(len(dataset)-2)
    xi_menos_xbarra=(dataset.map(lambda x: (x-dataset.mean())**2))
    sb=np.sqrt(se/xi_menos_xbarra.sum())
    sa=np.sqrt(se*(1/len(dataset)+dataset.mean()**2/xi_menos_xbarra.sum()))
    cia=abs(slope+ss.t.ppf(1-alfa/2,len(dataset)-2)*sa)
    cib=abs(intercept+ss.t.ppf(1-alfa/2,len(dataset)-2)*sb)
    
    #Scale hiperparameters
    
    scalem=dataset.std()
    scales=dataset.std()/(np.sqrt(2*(len(dataset)-1)))
    
    #Time covariant
    t=range(len(serie_lista))
    
    
    with pm.Model() as model:
        # Priors for unknown model parameters
        c_1 = pm.Beta('c_1', alpha=6, beta=9) #c=x-0,5: transformation in gev_logp is required due to Beta domain [0,1]
        c_2 = pm.Beta('c_2', alpha=6, beta=9)
        
        m1_1 = pm.Normal('m1_1', mu=slope, sd=2*cia)
        m1_2 = pm.Normal('m1_2', mu=slope, sd=2*cia)
        
        m2_1 = pm.Normal('m2_1', mu=intercept, sd=2*cib)
        m2_2 = pm.Normal('m2_2', mu=intercept, sd=2*cib)
        
        scale_1 = pm.Normal('scale_1', mu=scalem, sd=scales)
        scale_2 = pm.Normal('scale_2', mu=scalem, sd=scales)
        
        def gev_logp(value, t):
            loc_=m1_*t+m2_
            scaled = (value - loc_) / scale_
            logp = -(tt.log(scale_)
                     + (((c_-0.5) + 1) / (c_-0.5) * tt.log1p((c_-0.5) * scaled)
                     + (1 + (c_-0.5) * scaled) ** (-1/(c_-0.5))))
            bound1 = loc_ - scale_/(c_-0.5)
            bounds = tt.switch((c_-0.5) > 0, value > bound1, value < bound1)
            return bound(logp, bounds, c_ != 0)
    
        tau = pm.DiscreteUniform("tau", lower=0, upper=n_count_data - 1)
        idx = np.arange(n_count_data)
        c_ = pm.math.switch(tau > idx, c_1, c_2)
        m1_ = pm.math.switch(tau > idx, m1_1, m1_2)
        m2_ = pm.math.switch(tau > idx, m2_1, m2_2)
        scale_ = pm.math.switch(tau > idx, scale_1, scale_2)
        gev = pm.DensityDist('gev', gev_logp, observed={'value': serie_lista, 't': t})
        trace = pm.sample(1000, chains=1, progressbar=True)
        
       
    # geweke_plot = pm.geweke(trace, 0.05, 0.5, 20)
    # gelman_and_rubin = pm.diagnostics.gelman_rubin(trace)
    posterior = pm.trace_to_dataframe(trace)
    summary = pm.summary(trace)
    return summary, posterior

def gev1_shift_2(dataset, alfa=0.95):
    import scipy.stats as ss
    
    serie_lista=list(dataset)
    ano_resetado=pd.Series(list(range(len(dataset))))
    slope, intercept, r_value, pvalue_regress, std_err=ss.linregress(ano_resetado,dataset.values)
    
    #First regression (loc hiperparameters)
    y_regress=ano_resetado.map(lambda x: x*slope+intercept)
    e=dataset.reset_index(drop=True)-y_regress
    se=(e**2).sum()/(len(dataset)-2)
    xi_menos_xbarra=(dataset.map(lambda x: (x-dataset.mean())**2))
    sb=np.sqrt(se/xi_menos_xbarra.sum())
    sa=np.sqrt(se*(1/len(dataset)+dataset.mean()**2/xi_menos_xbarra.sum()))
    cia=abs(slope+ss.t.ppf(1-alfa/2,len(dataset)-2)*sa)
    cib=abs(intercept+ss.t.ppf(1-alfa/2,len(dataset)-2)*sb)
    
    #Scale hiperparameters
    
    scalem=dataset.std()
    scales=dataset.std()/(np.sqrt(2*(len(dataset)-1)))
    
    #Time covariant
    t=range(len(serie_lista))
    
    
    with pm.Model() as model:
        # Priors for unknown model parameters
        c_1 = pm.Beta('c_1', alpha=6, beta=9) #c=x-0,5: transformation in gev_logp is required due to Beta domain [0,1]
        c_2 = pm.Beta('c_2', alpha=6, beta=9)
        c_3 = pm.Beta('c_3', alpha=6, beta=9)
        
        m1_1 = pm.Normal('m1_1', mu=slope, sd=2*cia)
        m1_2 = pm.Normal('m1_2', mu=slope, sd=2*cia)
        m1_3 = pm.Normal('m1_3', mu=slope, sd=2*cia)
        
        m2_1 = pm.Normal('m2_1', mu=intercept, sd=2*cib)
        m2_2 = pm.Normal('m2_2', mu=intercept, sd=2*cib)
        m2_3 = pm.Normal('m2_3', mu=intercept, sd=2*cib)
        
        scale_1 = pm.Normal('scale_1', mu=scalem, sd=scales)
        scale_2 = pm.Normal('scale_2', mu=scalem, sd=scales)
        scale_3 = pm.Normal('scale_3', mu=scalem, sd=scales)
        
        def gev_logp(value, t):
            loc_=m1_*t+m2_
            scaled = (value - loc_) / scale_
            logp = -(tt.log(scale_)
                     + (((c_-0.5) + 1) / (c_-0.5) * tt.log1p((c_-0.5) * scaled)
                     + (1 + (c_-0.5) * scaled) ** (-1/(c_-0.5))))
            bound1 = loc_ - scale_/(c_-0.5)
            bounds = tt.switch((c_-0.5) > 0, value > bound1, value < bound1)
            return bound(logp, bounds, c_ != 0)
        
        
        tau1 = pm.DiscreteUniform("tau1", lower=0, upper=n_count_data - 2)
        tau2 = pm.DiscreteUniform("tau2", lower=tau1+1, upper=n_count_data - 1)
        idx = np.arange(n_count_data)
              
        c_ = pm.math.switch(tau2>=idx, pm.math.switch(tau1>=idx, c_1, c_2), c_3)
        m1_ = pm.math.switch(tau2>=idx, pm.math.switch(tau1>=idx, m1_1, m1_2), m1_3)
        m2_ = pm.math.switch(tau2>=idx, pm.math.switch(tau1>=idx, m2_1, m2_2), m2_3)
        scale_ = pm.math.switch(tau2>=idx, pm.math.switch(tau1>=idx, scale_1, scale_2), scale_3)
        gev = pm.DensityDist('gev', gev_logp, observed={'value': serie_lista, 't': t})
        trace = pm.sample(2000, chains=1, progressbar=True)

    # geweke_plot = pm.geweke(trace, 0.05, 0.5, 20)
    # gelman_and_rubin = pm.diagnostics.gelman_rubin(trace)
    posterior = pm.trace_to_dataframe(trace)
    summary = pm.summary(trace)
    return summary, posterior

# summary, posterior = gev0_shift_1(dataset)
summary, posterior= gev0_shift_2(dataset)
# summary, posterior= gev1_shift_1(dataset)
# summary, posterior= gev1_shift_2(dataset)
