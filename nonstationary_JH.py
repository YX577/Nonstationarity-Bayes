# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 16:37:24 2018

@author: Yan
"""
import pandas as pd
import numpy as np
import scipy.stats as ss
from datetime import datetime
import calendar
from pymc3.distributions.dist_math import bound
import pymc3 as pm
import theano.tensor as tt
import matplotlib.pyplot as plt

## nothing to add. just a comment to test rebase function on github
def le_dados_sugar_creek():
    df = pd.read_csv('data/sugar_creek_data.csv', sep=';')
    dataset=pd.Series(df.discharge.values, index=df.year)
    dataset=dataset.apply(lambda x: float(x.replace(',','.')) if isinstance(x,str) else x)
    dataset=dataset.map(lambda x: x*0.0283168465925) #conversão de sistema de unidades para m³/s
    return dataset[:-8] #[-8] leva em consideração dados até 2009 (análises de Villarini) [63:-8] caso queira contar 1988 pra frente

def le_dados_ana_antigo():
    lista_series_mensais=[]
    with open ('Clube_de_regatas.TXT','r') as file:
        for linha in file.readlines():
            if(linha.startswith("\n") or linha.startswith("/")):
                continue
            s=linha.replace(',','.').split(";")
            data_linha=datetime.strptime(s[2],'%d/%m/%Y')
            dias_no_mes=calendar.monthrange(data_linha.year,data_linha.month)
            rng=pd.date_range(data_linha,periods=dias_no_mes[1], freq='D')
            cons=[s[1] for i in range (dias_no_mes[1])]
            arrays=[rng,cons]
            tuples=list(zip(*arrays))
            index=pd.MultiIndex.from_tuples(tuples,names=['Data','Consistencia'])
            serie_linha=pd.Series(s[16:16+dias_no_mes[1]],index=index, name=s[0])
            lista_series_mensais.append(serie_linha)
    serie_completa=pd.concat(lista_series_mensais)
    serie_completa=pd.to_numeric(serie_completa, errors='coerce', downcast='float')
    serie_completa.sort_index(level=['Data','Consistencia'], inplace=True)
    definicao_de_duplicatas=serie_completa.reset_index(level=1, drop=True).index.duplicated(keep='last')
    dados_sem_duplicatas=serie_completa[~definicao_de_duplicatas]
    serie_com_index_unico=dados_sem_duplicatas.reset_index(level=1, drop=True)
    return serie_com_index_unico

def le_dados_quebec():
    df=pd.read_csv('little_sugar_creek.csv', sep=';', error_bad_lines=False)
    df['Date']=df['Date'].apply(lambda x: datetime.strptime(x, '%d/%m/%Y'))
    dataset=pd.Series(df.discharge.values, index=df.Date)
    dataset=dataset.apply(lambda x: float(x.replace(',','.')) if isinstance(x,str) else x)
    return dataset

def linear_regression(max_data):
    max_data=max_data.reset_index()
    max_data['Data']=max_data['Data'].map(lambda x: x.year)
    slope, intercept, r_value, p_value, std_err = ss.linregress(max_data['Data'],max_data['61834000'])
    return slope, intercept, r_value, p_value, std_err
#    return slope, intercept, r_value, p_value, std_err

def plot_regressions(max_data):
    max_data_plot=max_data
    test=max_data[:int(3*len(max_data_plot)/4)]
    slope, intercept, r_value, p_value, std_err=linear_regression(test)
    max_data=max_data.reset_index()
    max_data['Data']=max_data['Data'].map(lambda x: x.year)
    dates=pd.Series(max_data['Data'], name='61834000')
    y_axis=dates.map(lambda x: x*slope+intercept)
    
#    calibration_data=max_data_plot[:int(3*len(max_data_plot)/4)]
    dates_calibration=dates[:int(3*len(max_data_plot)/4)]
#    plt.scatter(dates_calibration,calibration_data.values, color='black', s=20)
#    validation_data=max_data_plot[int(3*len(max_data_plot)/4):]
    dates_validation=dates[int(3*len(max_data_plot)/4):]
#    plt.scatter(dates_validation,validation_data.values, color='gray', s=20)
#    
#    plt.plot(dates,y_axis)
    plt.xlabel('Anos')
    plt.ylabel('Vazão máxima (m³/s)')
    
    second_series=pd.Series(np.abs(max_data_plot.values-y_axis.values), index=max_data_plot.index, name='61834000')
    slope2, intercept2, r_value2, p_value2, std_err2=linear_regression(second_series)
    y_axis2=dates.map(lambda x: x*slope2+intercept2)
    
    calibration_data=second_series[:int(3*len(second_series)/4)]
    plt.scatter(dates_calibration,calibration_data.values, color='black', s=20)
    validation_data=second_series[int(3*len(second_series)/4):]
    plt.scatter(dates_validation,validation_data.values, color='gray', s=20)
    plt.plot(dates,y_axis2)
    return 

def stationary_posterior(annual_max):
#    calibration_data=annual_max[:int(3*len(annual_max)/4)]
    calibration_data=annual_max
#    calibration_data = pd.Series(annual_max['58790000'].values, index=annual_max.index)
    locm=calibration_data.mean()
    locs=calibration_data.std()/(np.sqrt(len(calibration_data)))
    scalem=calibration_data.std()
    scales=calibration_data.std()/(np.sqrt(2*(len(calibration_data)-1)))
    with pm.Model() as model:
        # Priors for unknown model parameters
        c = pm.Beta('c', alpha=6, beta=9) #c=x-0,5: transformation in gev_logp is required due to Beta domain between 0 and 1
        loc = pm.Normal('loc', mu=locm, sd=locs)
        scale = pm.Normal('scale', mu=scalem, sd=scales)
        
        # Likelihood (sampling distribution) of observations | Since GEV is not implemented in pymc a custom log likelihood function was created
        def gev_logp(value):
            scaled = (value - loc) / scale
            logp = -(tt.log(scale)
                     + (((c-0.5) + 1) / (c-0.5) * tt.log1p((c-0.5) * scaled)
                     + (1 + (c-0.5) * scaled) ** (-1/(c-0.5))))
            bound1 = loc - scale/(c-0.5)
            bounds = tt.switch((c-0.5) > 0, value > bound1, value < bound1)
            return bound(logp, bounds, c != 0)
        gev = pm.DensityDist('gev', gev_logp, observed=calibration_data)
#        step = pm.Metropolis()
        trace = pm.sample(5000, chains=2, cores=1, progressbar=True)
    pm.traceplot(trace)
    # geweke_plot=pm.geweke(trace, 0.05, 0.5, 20)
    # gelman_and_rubin=pm.diagnostics.gelman_rubin(trace)
    waic = pm.waic(trace)
    posterior=pm.trace_to_dataframe(trace)
    summary=pm.summary(trace)
    return posterior, summary, waic

def gevtr(calibration_data):
    
    #Preparing input data
    serie_lista=list(calibration_data)
#    ano_lista=calibration_data.index.year.tolist()
    ano_lista=calibration_data.index.tolist()
    ano_resetado=pd.Series(list(range(len(calibration_data))))

    #First regression

    slope, intercept, r_value, pvalue_regress, std_err=ss.linregress(ano_resetado,calibration_data.values)
    numerator=ano_resetado.map(lambda x: x*slope)

    #Second regression and calculation of detrended series
    
    dists=[]
    for i in range(len(serie_lista)):
        dists.append(abs(serie_lista[i]-(slope*ano_resetado[i]+intercept)))    
    slope2, intercept2, r_value2, pvalue_regress2, std_err2=ss.linregress(ano_resetado,dists)
    data_reset_index=pd.Series(calibration_data.values)
    denominator=ano_resetado.map(lambda x: x*slope2+intercept2)
    detrended_series=(data_reset_index-numerator)/denominator

    locm=detrended_series.mean()
    locs=detrended_series.std()/(np.sqrt(len(calibration_data)))
    scalem=detrended_series.std()
    scales=detrended_series.std()/(np.sqrt(2*(len(calibration_data)-1)))
    with pm.Model() as model:
        # Priors for unknown model parameters
        c = pm.Beta('c', alpha=6, beta=9) #c=x-0,5: transformation in gev_logp is required due to Beta domain is between 0 and 1
        loc = pm.Normal('loc', mu=locm, sd=locs)
        scale = pm.Normal('scale', mu=scalem, sd=scales)
        
        # Likelihood (sampling distribution) of observations | Since GEV is not implemented in pymc a custom log likelihood function was created
        def gev_logp(value):
            scaled = (value - loc) / scale
            logp = -(tt.log(scale)
                     + (((c-0.5) + 1) / (c-0.5) * tt.log1p((c-0.5) * scaled)
                     + (1 + (c-0.5) * scaled) ** (-1/(c-0.5))))
            bound1 = loc - scale/(c-0.5)
            bounds = tt.switch((c-0.5) > 0, value > bound1, value < bound1)
            return bound(logp, bounds, c != 0)
        gev = pm.DensityDist('gev', gev_logp, observed=detrended_series)
        trace = pm.sample(5000, chains=3, cores=1, progressbar=True)
    pm.traceplot(trace)
    # geweke_plot=pm.geweke(trace, 0.05, 0.5, 20)
    # gelman_and_rubin=pm.diagnostics.gelman_rubin(trace)
    posterior=pm.trace_to_dataframe(trace)
    summary=pm.summary(trace)
    return posterior, summary

def gev1(annual_max, alfa=0.95):
    calibration_data=annual_max
    serie_lista=list(calibration_data)
    ano_resetado=pd.Series(list(range(len(calibration_data))))

    #First regression (loc hiperparameters)

    slope, intercept, r_value, pvalue_regress, std_err=ss.linregress(ano_resetado,calibration_data.values)
    y_regress=ano_resetado.map(lambda x: x*slope+intercept)
    e=calibration_data.reset_index(drop=True)-y_regress
    se=(e**2).sum()/(len(calibration_data)-2)
    xi_menos_xbarra=(calibration_data.map(lambda x: (x-calibration_data.mean())**2))
    sb=np.sqrt(se/xi_menos_xbarra.sum())
    sa=np.sqrt(se*(1/len(calibration_data)+calibration_data.mean()**2/xi_menos_xbarra.sum()))
    cia=abs(slope+ss.t.ppf(1-alfa/2,len(calibration_data)-2)*sa)
    cib=abs(intercept+ss.t.ppf(1-alfa/2,len(calibration_data)-2)*sb)
    
    #Scale hiperparameters
    
    scalem=calibration_data.std()
    scales=calibration_data.std()/(np.sqrt(2*(len(calibration_data)-1)))
    
    #Time covariant
    t=range(len(serie_lista))
    
    with pm.Model() as model:
        # Priors for unknown model parameters
        c = pm.Beta('c', alpha=6, beta=9) #c=x-0,5: transformation in gev_logp is required due to Beta domain [0,1]
        m1 = pm.Normal('m1', mu=slope, sd=2*cia)
        m2 = pm.Normal('m2', mu=intercept, sd=2*cib)
        scale = pm.Normal('scale', mu=scalem, sd=scales)

        # Likelihood (sampling distribution) of observations | Since GEV is not implemented in pymc a custom log likelihood function was created
        def gev_logp(value, t):
            loc=m1*t+m2
            scaled = (value - loc) / scale
            logp = -(tt.log(scale)
                     + (((c-0.5) + 1) / (c-0.5) * tt.log1p((c-0.5) * scaled)
                     + (1 + (c-0.5) * scaled) ** (-1/(c-0.5))))
            bound1 = loc - scale/(c-0.5)
            bounds = tt.switch((c-0.5) > 0, value > bound1, value < bound1)
            return bound(logp, bounds, c != 0)
        gev = pm.DensityDist('gev', gev_logp, observed={'value': serie_lista, 't': t})
        trace = pm.sample(5000, chains=3, cores=1, progressbar=True)
    pm.traceplot(trace)
    # geweke_plot=pm.geweke(trace, 0.05, 0.5, 20)
    # gelman_and_rubin=pm.diagnostics.gelman_rubin(trace)
    posterior=pm.trace_to_dataframe(trace)
    summary=pm.summary(trace)
    return posterior, summary

def gev2(serie_maximas_anuais, alfa=0.05):
#    #Time covariant
    t=pd.Series(range(len(serie_maximas_anuais)))
    t2=t**2
#    
#    #Preparing input data
    import statsmodels.formula.api as sm
    calibration_data=serie_maximas_anuais.reset_index(name="values")
    data=calibration_data["values"]
    listofseries=[data, t, t2]
    data_for_regression=pd.concat(listofseries, axis=1, ignore_index=True)
    data_for_regression.columns=['Q','t','t2']
    result=sm.ols(formula="Q ~ t2 + t", data=data_for_regression).fit()
    regress_params=result.params
    regress_cis=result.conf_int(alpha=0.05)
    regress_cis.columns=['inf', 'sup']
    cis=np.absolute(regress_cis['sup']-regress_cis['inf'])
    #First regression (loc hiperparameters)
    
    calibration_data=serie_maximas_anuais
    serie_lista=list(calibration_data)

    #Scale hiperparameters
    
    scalem=calibration_data.std()
    scales=calibration_data.std()/(np.sqrt(2*(len(calibration_data)-1)))
    
    with pm.Model() as model:
        # Priors for unknown model parameters
        c = pm.Beta('c', alpha=6, beta=9) #c=x-0,5: transformation in gev_logp is required due to Beta domain is between 0 and 1
        m1 = pm.Normal('m1', mu=regress_params["t2"], sd=cis["t2"]) #t2
        m2 = pm.Normal('m2', mu=regress_params["t"], sd=cis["t"]) #t
        m3 = pm.Normal('m3', mu=regress_params["Intercept"], sd=cis["Intercept"]) #intercept
        scale = pm.Normal('scale', mu=scalem, sd=scales)

        # Likelihood (sampling distribution) of observations | Since GEV is not implemented in pymc a custom log likelihood function was created
        def gev_logp(value, t, t2):
            loc=m1*t2+m2*t+m3
#            scale=tt.log(tt.exp(scale1))
            scaled = (value - loc) / scale
            logp = -(tt.log(scale)
                     + (((c-0.5) + 1) / (c-0.5) * tt.log1p((c-0.5) * scaled)
                     + (1 + (c-0.5) * scaled) ** (-1/(c-0.5))))
            bound1 = loc - scale/(c-0.5)
            bounds = tt.switch((c-0.5) > 0, value > bound1, value < bound1)
            return bound(logp, bounds, c != 0)
        gev = pm.DensityDist('gev', gev_logp, observed={'value': serie_lista, 't': t, 't2': t2})
        trace = pm.sample(5000, chains=1, cores=1, progressbar=True)
    pm.traceplot(trace)
    # geweke_plot=pm.geweke(trace, 0.05, 0.5, 20)
    # gelman_and_rubin=pm.diagnostics.gelman_rubin(trace)
    posterior=pm.trace_to_dataframe(trace)
    summary=pm.summary(trace)
    return posterior, summary

def gev11(serie_maximas_anuais, alfa=0.05):
    #Preparing input data
    calibration_data=serie_maximas_anuais
#    calibration_data=annual_max[:int(3*len(annual_max)/4)]
#    calibration_data=serie_maximas_anuais
    serie_lista=list(calibration_data)
#    ano_lista=calibration_data.index.year.tolist()
    ano_resetado=pd.Series(list(range(len(calibration_data))))

    #First regression

    slope, intercept, r_value, pvalue_regress, std_err=ss.linregress(ano_resetado,calibration_data.values)
    y_regress=ano_resetado.map(lambda x: x*slope+intercept)
    e=calibration_data.reset_index(drop=True)-y_regress
    se=(e**2).sum()/(len(calibration_data)-2)
    xi_menos_xbarra=(calibration_data.map(lambda x: (x-calibration_data.mean())**2))
    sb=np.sqrt(se/xi_menos_xbarra.sum())
    sa=np.sqrt(se*(1/len(calibration_data)+calibration_data.mean()**2/xi_menos_xbarra.sum()))
    cia=abs(slope+ss.t.ppf(1-alfa/2,len(calibration_data)-2)*sa)
    cib=abs(intercept+ss.t.ppf(1-alfa/2,len(calibration_data)-2)*sb)
    
    #Second regression
    
    dists=[]
    for i in range(len(serie_lista)):
        dists.append(abs(serie_lista[i]-(slope*ano_resetado[i]+intercept)))    
    slope2, intercept2, r_value2, pvalue_regress2, std_err2=ss.linregress(ano_resetado,dists)
    dists=pd.Series(dists)
    y_regress2=ano_resetado.map(lambda x: x*slope2+intercept2)
    e2=dists.reset_index(drop=True)-y_regress2
    se2=(e2**2).sum()/(len(dists)-2)
    xi_menos_xbarra2=(dists.map(lambda x: (x-dists.mean())**2))
    sb2=np.sqrt(se2/xi_menos_xbarra.sum())
    sa2=np.sqrt(se2*(1/len(dists)+dists.mean()**2/xi_menos_xbarra2.sum()))
    cia2=abs(slope2+ss.t.ppf(1-alfa/2,len(dists)-2)*sa2)
    cib2=abs(intercept2+ss.t.ppf(1-alfa/2,len(dists)-2)*sb2)
    t=range(len(serie_lista))
    with pm.Model() as model:
        # Priors for unknown model parameters
        c = pm.Beta('c', alpha=6, beta=9) #c=x-0,5: transformation in gev_logp is required due to Beta domain is between 0 and 1
        m1 = pm.Normal('m1', mu=slope, sd=2*cia)
        m2 = pm.Normal('m2', mu=intercept, sd=2*cib)
        a1 = pm.Normal('a1', mu=slope2, sd=2*cia2)
        a2 = pm.Normal('a2', mu=intercept2, sd=2*cib2)
        
        # Likelihood (sampling distribution) of observations | Since GEV is not implemented in pymc a custom log likelihood function was created
        def gev_logp(value, t):
            loc=m1*t+m2
            scale=a1*t+a2
            scaled = (value - loc) / scale
            logp = -(tt.log(scale)
                     + (((c-0.5) + 1) / (c-0.5) * tt.log1p((c-0.5) * scaled)
                     + (1 + (c-0.5) * scaled) ** (-1/(c-0.5))))
            bound1 = loc - scale/(c-0.5)
            bounds = tt.switch((c-0.5) > 0, value > bound1, value < bound1)
            return bound(logp, bounds, c != 0)
        gev = pm.DensityDist('gev', gev_logp, observed={'value': serie_lista, 't': t})
        trace = pm.sample(5000, chains=1, cores=1, progressbar=True)
    pm.traceplot(trace)
    # geweke_plot=pm.geweke(trace, 0.05, 0.5, 20)
    # gelman_and_rubin=pm.diagnostics.gelman_rubin(trace)
    posterior=pm.trace_to_dataframe(trace)
    summary=pm.summary(trace)
    return posterior, summary

if __name__=="__main__":
    dataset=le_dados_sugar_creek()
#    dataset=le_dados_analise()
#    dataset=le_dados_quebec()
    posterior, summary, waic=stationary_posterior(dataset)
    # posterior, summary=gevtr(dataset)
#    posterior, summary, geweke_plot, gelman_and_rubin_diag=gev11(dataset)
    # posterior, summary=gev1(dataset)
    # posterior, summary=gev2(dataset)
    # posterior, summary=gev11(dataset)
#    likelihood=likelihood_calculation(serie_maximas_anuais)
#    params=grafico_gev11(serie_maximas_anuais, sorted_series, df_posteriori)