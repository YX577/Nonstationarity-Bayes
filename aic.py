# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 10:13:58 2019

@author: laixalmeida
"""

import pandas as pd
import math
import scipy.stats as ss
import numpy as np

xls=pd.ExcelFile('summary_sugar_creek.xlsx')
gev0=pd.read_excel(xls,'GEV_0',header=0,index_col=0)
gevtr=pd.read_excel(xls,'GEVr',header=0,index_col=0)
gev1=pd.read_excel(xls,'GEV1',header=0,index_col=0)
gev2=pd.read_excel(xls,'GEV2',header=0,index_col=0)
gev11=pd.read_excel(xls,'GEV11',header=0,index_col=0)

def le_dados_sugar_creek():
    df = pd.read_csv('sugar_creek_data.csv', sep=';')
#    df['year']=df['datetime'].apply(lambda x: datetime.strptime(x, '%d/%m/%Y'))
    dataset=pd.Series(df.discharge.values, index=df.year)
    dataset=dataset.apply(lambda x: float(x.replace(',','.')) if isinstance(x,str) else x)
    dataset=dataset.map(lambda x: x*0.0283168465925)
    return dataset[-8:]

def regression(dataset):
    serie_lista=list(dataset)
    ano_resetado=pd.Series(list(range(len(dataset))))

    #First regression
    slope, intercept, r_value, pvalue_regress, std_err=ss.linregress(ano_resetado,dataset.values)
    numerator=ano_resetado.map(lambda x: x*slope)
    
    #Second regression and calculation of detrended series
    
    dists=[]
    for i in range(len(serie_lista)):
        dists.append(abs(serie_lista[i]-(slope*ano_resetado[i]+intercept)))    
    slope2, intercept2, r_value2, pvalue_regress2, std_err2=ss.linregress(ano_resetado,dists)
    data_reset_index=pd.Series(dataset.values)
    denominator=ano_resetado.map(lambda x: x*slope2+intercept2)
    detrended_series=(data_reset_index-numerator)/denominator
    return detrended_series, numerator, denominator

def likelihood_gev0 (dataset, loc, scale, c):
    likelihood=[]
    likelihood = dataset.map(lambda x: ss.genextreme.pdf(x, -c, loc, scale))
    return np.prod(likelihood)

def likelihood_gevtr (detrended, loc, scale, c):
    likelihood=[]
    likelihood = detrended.map(lambda x: ss.genextreme.pdf(x, -c, loc, scale))
    return np.prod(likelihood)

def likelihood_gev1 (dataset, m1, m2, scale, c):
    likelihood=[]
    for t in range(len(dataset)):
        likelihood.append(ss.genextreme.pdf(dataset.iloc[t], -c, m1*t+m2, scale))
    return np.prod(likelihood)

def likelihood_gev2 (dataset, m1, m2, m3, scale, c):
    likelihood=[]
    for t in range(len(dataset)):
        likelihood.append(ss.genextreme.pdf(dataset.iloc[t], -c, m1*t**2+m2*t+m3, scale))
    return np.prod(likelihood)

def likelihood_gev11 (dataset, m1, m2, a1, a2, c):
    likelihood=[]
    for t in range(len(dataset)):
        likelihood.append(ss.genextreme.pdf(dataset.iloc[t], -c, m1*t+m2, a1*t+a2))
    return np.prod(likelihood)

def aic (k,l):
    aic=2*k-2*math.log(l)
    return aic

def quantile_gev0(q):
    q_gev0 = ss.genextreme.ppf(q, -gev0.loc['Mean']['shape'], gev0.loc['Mean']['loc'], gev0.loc['Mean']['scale'])
    q_min_gev0 = ss.genextreme.ppf(q, -gev0.loc['HPD 2.5']['shape'], gev0.loc['HPD 2.5']['loc'], gev0.loc['HPD 2.5']['scale'])
    q_max_gev0 = ss.genextreme.ppf(q, -gev0.loc['HPD 97.5']['shape'], gev0.loc['HPD 97.5']['loc'], gev0.loc['HPD 97.5']['scale'])
    return q_gev0, q_min_gev0, q_max_gev0

def quantile_gev1(q, t):
    q_gev1 = ss.genextreme.ppf(q, -gev1.loc['Mean']['shape'], gev1.loc['Mean']['m1']*t+gev1.loc['Mean']['m2'], gev1.loc['Mean']['scale'])
    q_min_gev1 = ss.genextreme.ppf(q, -gev1.loc['HPD 2.5']['shape'], gev1.loc['HPD 2.5']['m1']*t+gev1.loc['Mean']['m2'], gev1.loc['HPD 2.5']['scale'])
    q_max_gev1 = ss.genextreme.ppf(q, -gev1.loc['HPD 97.5']['shape'], gev1.loc['HPD 97.5']['m1']*t+gev1.loc['Mean']['m2'], gev1.loc['HPD 97.5']['scale'])
    return q_gev1, q_min_gev1, q_max_gev1

def quantile_gev2(q, t):
    q_gev2 = ss.genextreme.ppf(q, -gev2.loc['Mean']['shape'], gev2.loc['Mean']['m1']*t**2+gev2.loc['Mean']['m2']*t+gev2.loc['Mean']['m3'], gev2.loc['Mean']['scale'])
    q_min_gev2 = ss.genextreme.ppf(q, -gev2.loc['HPD 2.5']['shape'], gev2.loc['HPD 2.5']['m1']*t**2+gev2.loc['Mean']['m2']*t+gev2.loc['Mean']['m3'], gev2.loc['HPD 2.5']['scale'])
    q_max_gev2 = ss.genextreme.ppf(q, -gev2.loc['HPD 97.5']['shape'], gev2.loc['HPD 97.5']['m1']*t**2+gev2.loc['Mean']['m2']*t+gev2.loc['Mean']['m3'], gev2.loc['HPD 97.5']['scale'])
    return q_gev2, q_min_gev2, q_max_gev2

def quantile_gev11(q, t):
    q_gev11 = ss.genextreme.ppf(q, -gev11.loc['Mean']['shape'], gev11.loc['Mean']['m1']*t+gev11.loc['Mean']['m2'], gev11.loc['Mean']['a1']*t+gev11.loc['Mean']['a2'])
    q_min_gev11 = ss.genextreme.ppf(q, -gev11.loc['HPD 2.5']['shape'], gev11.loc['HPD 2.5']['m1']*t+gev11.loc['Mean']['m2'], gev11.loc['HPD 2.5']['a1']*t+gev11.loc['Mean']['a2'])
    q_max_gev11 = ss.genextreme.ppf(q, -gev11.loc['HPD 97.5']['shape'], gev11.loc['HPD 97.5']['m1']*t+gev11.loc['Mean']['m2'], gev11.loc['HPD 97.5']['a1']*t+gev11.loc['Mean']['a2'])
    return q_gev11, q_min_gev11, q_max_gev11

def quantile_gevtr(q, t):
    detrended, numerator, denominator = regression(dataset)
    
    q_gevtr = ss.genextreme.ppf(q, -gevtr.loc['Mean']['shape'], gevtr.loc['Mean']['loc'], gevtr.loc['Mean']['scale'])
    q_min_gevtr = ss.genextreme.ppf(q, -gevtr.loc['HPD 2.5']['shape'], gevtr.loc['HPD 2.5']['loc'], gevtr.loc['HPD 2.5']['scale'])
    q_max_gevtr = ss.genextreme.ppf(q, -gevtr.loc['HPD 97.5']['shape'], gevtr.loc['HPD 97.5']['loc'], gevtr.loc['HPD 97.5']['scale'])
    
    q_gevtr=q_gevtr*denominator+numerator
    q_min_gevtr=q_min_gevtr*denominator+numerator
    q_max_gevtr=q_max_gevtr*denominator+numerator
    return q_gevtr, q_min_gevtr, q_max_gevtr

if __name__=="__main__":
    dataset = le_dados_sugar_creek()
    detrended, numerator, denominator = regression(dataset)
    likelihood_gev0 = likelihood_gev0(dataset, gev0.loc['Mean']['loc'], gev0.loc['Mean']['scale'], gev0.loc['Mean']['shape'])
    likelihood_gevtr = likelihood_gevtr(detrended, gevtr.loc['Mean']['loc'], gevtr.loc['Mean']['scale'], gevtr.loc['Mean']['shape'])
    likelihood_gev1 = likelihood_gev1(dataset, gev1.loc['Mean']['m1'], gev1.loc['Mean']['m2'], gev1.loc['Mean']['scale'], gev1.loc['Mean']['shape'])
    likelihood_gev2 = likelihood_gev2(dataset, gev2.loc['Mean']['m1'], gev2.loc['Mean']['m2'], gev2.loc['Mean']['m3'], gev2.loc['Mean']['scale'], gev2.loc['Mean']['shape'])
    likelihood_gev11 = likelihood_gev11(dataset, gev11.loc['Mean']['m1'], gev11.loc['Mean']['m2'], gev11.loc['Mean']['a1'], gev11.loc['Mean']['a2'], gev11.loc['Mean']['shape'])
    aic_gev0 = aic(3, likelihood_gev0)
    aic_gevtr = aic(3, likelihood_gevtr)
    aic_gev1 = aic(4, likelihood_gev1)
    aic_gev2 = aic(5, likelihood_gev2)
    aic_gev11 = aic(5, likelihood_gev11)
#    q_gev0, q_gev0_min, q_gev0_max = quantile_gev0(0.99)
#    q_gev1, q_gev1_min, q_gev1_max = quantile_gev1(0.5, len(dataset))
#    q_gev2, q_gev2_min, q_gev2_max = quantile_gev2(0.99, len(dataset))
#    q_gev11, q_gev11_min, q_gev11_max = quantile_gev11(0.99, len(dataset))
#    q_gevtr, q_gevtr_min, q_gevtr_max = quantile_gevtr(0.99, len(dataset))
    
#modelo1
#params1=pd.read_csv('finalmente.csv')
#likelihoods1=pd.read_csv('likelihoods.csv', header=None)
#k1 = len(params1.columns)-1
#l1 = likelihoods1[1].max()
#n=len(serie_maximas_anuais)
# 
#def bic (k,l,n):
#    bic= math.log(n)*k - 2*math.log(l)
#    return bic
#        
#
#aic1=aic(k1,l1)
#bic1=bic(k1,l1,n)