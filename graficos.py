# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 15:16:20 2019

@author: laixalmeida
"""

import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as ss
import numpy as np
from datetime import datetime

#dados
df = pd.read_csv('data/sugar_creek_data.csv', sep=';', header=0)
# dataset = df['discharge'].apply(lambda x: x*0.0283168465925) #conversão de sistema de unidades para m³/s
dataset = pd.Series(df['discharge'])
dataset.index=df['year']
dataset = dataset.map(lambda x: x*0.0283168465925)
dataset = dataset[61:-8] #[61:-8] se ano > 1986
ano_resetado=pd.Series(list(range(len(dataset))))

xls=pd.ExcelFile('summary_sugar_creek86.xlsx')
GEV0 = pd.read_excel(xls,'GEV0',header=0,index_col=0)
GEVr=pd.read_excel(xls,'GEVr',header=0,index_col=0)
GEV1=pd.read_excel(xls,'GEV1',header=0,index_col=0)
GEV2=pd.read_excel(xls,'GEV2',header=0,index_col=0)
GEV11=pd.read_excel(xls,'GEV11',header=0,index_col=0)

probability=[0.5,0.98,0.99]
ax=[]
fig, ax = plt.subplots(4,3, figsize=(7.48,9.45), sharey=True)
#plt.style.use('seaborn-white')
#fig = plt.figure()

for i in range(0,len(probability)):
    #GEV0
    q_gev0_mean= ss.genextreme.ppf(probability[i], -GEV0.iloc[0]['shape'], GEV0.iloc[0]['loc'],GEV0.iloc[0]['scale'])
    q_gev0_low= ss.genextreme.ppf(probability[i], -GEV0.iloc[3]['shape'], GEV0.iloc[3]['loc'],GEV0.iloc[3]['scale'])
    q_gev0_high= ss.genextreme.ppf(probability[i], -GEV0.iloc[4]['shape'], GEV0.iloc[4]['loc'],GEV0.iloc[4]['scale'])

    quantiles_gev0_mean = [q_gev0_mean]*len(range(len(dataset)))
    quantiles_gev0_low = [q_gev0_low]*len(range(len(dataset)))
    quantiles_gev0_high = [q_gev0_high]*len(range(len(dataset)))

    #plot
    ax[0,i].scatter(dataset.index[:85],dataset.values[:85], color='black', s=5)
    ax[0,i].scatter(dataset.index[85:],dataset.values[85:], color='blue', s=5)
    ax[0,i].plot(dataset.index, quantiles_gev0_mean, color='#2F4F4F')
    ax[0,i].fill_between(dataset.index, quantiles_gev0_high, quantiles_gev0_low, color='grey', alpha=0.1)
    ax[0,i].set_ylim([0,1500])
    ax[0,i].tick_params(axis='both',labelsize=8)
    
    
    if i==0:
        ax[0,i].set_title("RP = 2 years",fontsize=10)
        ax[0,i].set_ylabel('Discharge (m3/s)', fontsize=10)             
        ax[1,i].set_ylabel('Discharge (m3/s)', fontsize=10)
        ax[2,i].set_ylabel('Discharge (m3/s)', fontsize=10)
        ax[3,i].set_ylabel('Discharge (m3/s)', fontsize=10)
        #ax[4,i].set_ylabel('Discharge (m³/s)', fontsize=10)       
       
    elif i==1:
        ax[0,i].set_title("RP = 50 years", fontsize=10)
    else:
        ax[0,i].set_title("RP = 100 years", fontsize=10)
        ax[0,i].yaxis.set_label_position("right")
        ax[0,i].set_ylabel('GEVr', fontsize=10)
        ax[1,i].yaxis.set_label_position("right")
        ax[1,i].set_ylabel('GEV1', fontsize=10)
        ax[2,i].yaxis.set_label_position("right")
        ax[2,i].set_ylabel('GEV2',fontsize=10)
        ax[3,i].yaxis.set_label_position("right")
        ax[3,i].set_ylabel('GEV11', fontsize=10)
       # ax[4,i].yaxis.set_label_position("right")
       # ax[4,i].set_ylabel('GEV11', fontsize=10)
        

    #GEVr
    
    q_gevr_mean= ss.genextreme.ppf(probability[i], -GEVr.iloc[0]['shape'], GEVr.iloc[0]['loc'], GEVr.iloc[0]['scale'])
    q_gevr_low= ss.genextreme.ppf(probability[i], -GEVr.iloc[3]['shape'], GEVr.iloc[3]['loc'], GEVr.iloc[3]['scale'])
    q_gevr_high= ss.genextreme.ppf(probability[i], -GEVr.iloc[4]['shape'], GEVr.iloc[4]['loc'], GEVr.iloc[4]['scale'])

    quantiles_gevr_mean = [q_gevr_mean]*len(range(len(dataset)))
    quantiles_gevr_low = [q_gevr_low]*len(range(len(dataset)))
    quantiles_gevr_high = [q_gevr_high]*len(range(len(dataset)))

    detrends=[quantiles_gevr_mean, quantiles_gevr_low,quantiles_gevr_high]   
    retrend_series=[]
    for l in range(0,len(detrends)):
        serie_lista=dataset.values
        ano_lista=dataset.index.tolist()
        ano_resetado=pd.Series(list(range(len(dataset))))

        #First regression

        slope, intercept, r_value, pvalue_regress, std_err=ss.linregress(ano_resetado,serie_lista)
        numerator=ano_resetado.map(lambda x: x*slope)

        #Second regression 
    
        dists=[]
        for j in range(len(serie_lista)):
            dists.append(abs(serie_lista[j]-(slope*ano_resetado[j]+intercept)))    
        slope2, intercept2, r_value2, pvalue_regress2, std_err2=ss.linregress(ano_resetado,dists)
        data_reset_index=pd.Series(detrends[l])
        denominator=ano_resetado.map(lambda x: x*slope2+intercept2)
        retrend_series.append(data_reset_index*denominator + numerator)
    

    #plot
    ax[0,i].scatter(dataset.index[:85],dataset.values[:85], color='black', s=5)
    ax[0,i].scatter(dataset.index[85:],dataset.values[85:], color='blue', s=5)
    ax[0,i].plot(ano_lista, retrend_series[0], color='blue')
    ax[0,i].fill_between(ano_lista, retrend_series[1], retrend_series[2], color='blue', alpha=0.1)
    ax[0,i].plot(dataset.index, quantiles_gev0_mean, color='gray')
    ax[0,i].fill_between(dataset.index, quantiles_gev0_high, quantiles_gev0_low, color='grey', alpha=0.1)
    #ax[0,i].set_ylim([0,2000])
    ax[0,i].tick_params(axis='both',labelsize=8)
        
#    #GEV1
    q_gev1_mean=[]
    q_gev1_low=[]
    q_gev1_high=[]
    
    for t in ano_resetado:
        GEV1['loc']=GEV1.m1 * t + GEV1.m2
               
        q_gev1_mean.append(ss.genextreme.ppf(probability[i], -GEV1.iloc[0]['shape'], GEV1.iloc[0]['loc'],GEV1.iloc[0]['scale']))
        q_gev1_low.append(ss.genextreme.ppf(probability[i], -GEV1.iloc[3]['shape'], GEV1.iloc[3]['loc'],GEV1.iloc[3]['scale']))
        q_gev1_high.append(ss.genextreme.ppf(probability[i], -GEV1.iloc[4]['shape'], GEV1.iloc[4]['loc'],GEV1.iloc[4]['scale']))
#
#       #plot
    ax[1,i].scatter(dataset.index[:85],dataset.values[:85], color='black', s=5)
    ax[1,i].scatter(dataset.index[85:],dataset.values[85:], color='blue', s=5)
    ax[1,i].plot(dataset.index, q_gev1_mean, color='blue')
    ax[1,i].fill_between(dataset.index, q_gev1_high, q_gev1_low, color='blue', alpha=0.1)
    ax[1,i].plot(dataset.index, quantiles_gev0_mean, color='gray')
    ax[1,i].fill_between(dataset.index, quantiles_gev0_high, quantiles_gev0_low, color='grey', alpha=0.1)
    #ax[1,i].set_ylim([0,2000])
    ax[1,i].tick_params(axis='both',labelsize=8)
#        
#    #GEV2
    q_gev2_mean=[]
    q_gev2_low=[]
    q_gev2_high=[]
    
    for t in ano_resetado:
        GEV2['loc']=GEV2.m1 * t**2 + GEV2.m2 * t + GEV2.m3
               
        q_gev2_mean.append(ss.genextreme.ppf(probability[i], -GEV2.iloc[0]['shape'], GEV2.iloc[0]['loc'],GEV2.iloc[0]['scale']))
        q_gev2_low.append(ss.genextreme.ppf(probability[i], -GEV2.iloc[3]['shape'], GEV2.iloc[3]['loc'],GEV2.iloc[3]['scale']))
        q_gev2_high.append(ss.genextreme.ppf(probability[i], -GEV2.iloc[4]['shape'], GEV2.iloc[4]['loc'],GEV2.iloc[4]['scale']))
#
#       #plot
    ax[2,i].scatter(dataset.index[:85],dataset.values[:85], color='black', s=5)
    ax[2,i].scatter(dataset.index[85:],dataset.values[85:], color='blue', s=5)
    ax[2,i].plot(dataset.index, q_gev2_mean, color='blue')
    ax[2,i].fill_between(dataset.index, q_gev2_high, q_gev2_low, color='blue', alpha=0.1)
    ax[2,i].plot(dataset.index, quantiles_gev0_mean, color='gray')
    ax[2,i].fill_between(dataset.index, quantiles_gev0_high, quantiles_gev0_low, color='grey', alpha=0.1)
    #ax[2,i].set_ylim([0,2000])
    ax[2,i].tick_params(axis='both',labelsize=8)

#        
#    #GEV11
    q_gev11_mean=[]
    q_gev11_low=[]
    q_gev11_high=[]
    
    for t in ano_resetado:
        GEV11['loc']=GEV11.m1 * t + GEV11.m2 
        GEV11['scale'] = GEV11.a1 * t + GEV11.a2 
               
        q_gev11_mean.append(ss.genextreme.ppf(probability[i], -GEV11.iloc[0]['shape'], GEV11.iloc[0]['loc'],GEV11.iloc[0]['scale']))
        q_gev11_low.append(ss.genextreme.ppf(probability[i], -GEV11.iloc[3]['shape'], GEV11.iloc[3]['loc'],GEV11.iloc[3]['scale']))
        q_gev11_high.append(ss.genextreme.ppf(probability[i], -GEV11.iloc[4]['shape'], GEV11.iloc[4]['loc'],GEV11.iloc[4]['scale']))
#
#       #plot
    ax[3,i].scatter(dataset.index[:85],dataset.values[:85], color='black', s=5)
    ax[3,i].scatter(dataset.index[85:],dataset.values[85:], color='blue', s=5)
    ax[3,i].plot(dataset.index, q_gev11_mean, color='blue')
    ax[3,i].fill_between(dataset.index, q_gev11_high, q_gev11_low, color='blue', alpha=0.1)
    ax[3,i].plot(dataset.index, quantiles_gev0_mean, color='gray')
    ax[3,i].fill_between(dataset.index, quantiles_gev0_high, quantiles_gev0_low, color='grey', alpha=0.1)
    #ax[3,i].set_ylim([0,2000])
    ax[3,i].set_xlabel('Year')
    ax[3,i].tick_params(axis='both',labelsize=8)
    

fig = plt.savefig('graphs_02PC009.pdf',format='pdf')


#t fixo e evolução ao longo do tempo de retorno
#tr = np.arange(2,100)
#t=85 #equivalente a 2009
#
##GEV11
#q_gev11_mean=[]
#q_gev11_low=[]
#q_gev11_high=[]
#q_gev0_mean=[]
#q_gev0_low=[]
#q_gev0_high=[]
#for i in range(0,len(tr)):
#    prob = 1 - 1/tr[i]
#    GEV11['loc'] = GEV11.m1 * t + GEV11.m2
#    GEV11['scale'] = GEV11.a1 * t + GEV11.a2    
#    q_gev11_mean.append(ss.genextreme.ppf(prob, -GEV11.iloc[0]['shape'], GEV11.iloc[0]['loc'],GEV11.iloc[0]['scale']))
#    q_gev11_low.append(ss.genextreme.ppf(prob, -GEV11.iloc[3]['shape'], GEV11.iloc[3]['loc'],GEV11.iloc[3]['scale']))
#    q_gev11_high.append(ss.genextreme.ppf(prob, -GEV11.iloc[4]['shape'], GEV11.iloc[4]['loc'],GEV11.iloc[4]['scale']))
#
#    q_gev0_mean.append(ss.genextreme.ppf(prob, -GEV0.iloc[0]['shape'], GEV0.iloc[0]['loc'],GEV0.iloc[0]['scale']))
#    q_gev0_low.append(ss.genextreme.ppf(prob, -GEV0.iloc[3]['shape'], GEV0.iloc[3]['loc'],GEV0.iloc[3]['scale']))
#    q_gev0_high.append(ss.genextreme.ppf(prob, -GEV0.iloc[4]['shape'], GEV0.iloc[4]['loc'],GEV0.iloc[4]['scale']))
#
#    
##plot
#plt.figure()
#plt.plot(tr, q_gev11_mean, color='#2F4F4F')
#plt.plot(tr, q_gev0_mean, color='blue')
#plt.fill_between(tr, q_gev11_high, q_gev11_low, color='grey', alpha=0.1)
#plt.fill_between(tr, q_gev0_high, q_gev0_low, color='blue', alpha=0.1)
#axes = plt.gca()
#axes.set_ylim([0,4000])
#plt.title('Ano 2009, GEV11')
#
#GEV2
#q_gev2_mean=[]
#q_gev2_low=[]
#q_gev2_high=[]
#
#for i in range(0,len(tr)):
#    prob = 1 - 1/tr[i]
#    GEV2['loc']=GEV2.m1 * t**2 + GEV2.m2 * t + GEV2.m3
#           
#    q_gev2_mean.append(ss.genextreme.ppf(prob, -GEV2.iloc[0]['shape'], GEV2.iloc[0]['loc'],GEV2.iloc[0]['scale']))
#    q_gev2_low.append(ss.genextreme.ppf(prob, -GEV2.iloc[3]['shape'], GEV2.iloc[3]['loc'],GEV2.iloc[3]['scale']))
#    q_gev2_high.append(ss.genextreme.ppf(prob, -GEV2.iloc[4]['shape'], GEV2.iloc[4]['loc'],GEV2.iloc[4]['scale']))
#
##plot
#plt.figure()
#plt.plot(tr, q_gev2_mean, color='#2F4F4F')
#plt.plot(tr, q_gev0_mean, color='blue')
#plt.fill_between(tr, q_gev2_high, q_gev2_low, color='grey', alpha=0.1)
#plt.fill_between(tr, q_gev0_high, q_gev0_low, color='blue', alpha=0.1)
#axes = plt.gca()
#axes.set_ylim([0,4000])
#plt.title('Ano 2009, GEV2')
#
#GEV1
#q_gev1_mean=[]
#q_gev1_low=[]
#q_gev1_high=[]
#
#for i in range(0,len(tr)):
#   prob = 1 - 1/tr[i]
#   GEV1['loc'] = GEV1.m1 * t + GEV1.m2
#   q_gev1_mean.append(ss.genextreme.ppf(prob, -GEV11.iloc[0]['shape'], GEV11.iloc[0]['loc'],GEV11.iloc[0]['scale']))
#   q_gev1_low.append(ss.genextreme.ppf(prob, -GEV11.iloc[3]['shape'], GEV11.iloc[3]['loc'],GEV11.iloc[3]['scale']))
#   q_gev1_high.append(ss.genextreme.ppf(prob, -GEV11.iloc[4]['shape'], GEV11.iloc[4]['loc'],GEV11.iloc[4]['scale']))
#   
##plot
#plt.figure()
#plt.plot(tr, q_gev1_mean, color='#2F4F4F')
#plt.plot(tr, q_gev0_mean, color='blue')
#plt.fill_between(tr, q_gev1_high, q_gev1_low, color='grey', alpha=0.1)
#plt.fill_between(tr, q_gev0_high, q_gev0_low, color='blue', alpha=0.1)
#axes = plt.gca()
#axes.set_ylim([0,4000])
#plt.title('Ano 2009, GEV1')
#
##GEVr
#q_gevr_mean=[]
#q_gevr_low=[]
#q_gevr_high=[]
#
#for i in range(0,len(tr)):
#   prob = 1 - 1/tr[i]
#   q_gevr_mean.append(ss.genextreme.ppf(prob, -GEVr.iloc[0]['shape'], GEVr.iloc[0]['loc'], GEVr.iloc[0]['scale']))
#   q_gevr_low.append(ss.genextreme.ppf(prob, -GEVr.iloc[3]['shape'], GEVr.iloc[3]['loc'], GEVr.iloc[3]['scale']))
#   q_gevr_high.append(ss.genextreme.ppf(prob, -GEVr.iloc[4]['shape'], GEVr.iloc[4]['loc'], GEVr.iloc[4]['scale']))
#
#detrends=[q_gevr_mean, q_gevr_low,q_gevr_high]   
#retrend_series=[]
#for l in range(0,len(detrends)):
#    serie_lista=dataset.values
#    ano_lista=dataset.index.tolist()
#    ano_resetado=pd.Series(list(range(len(dataset))))
#
#    #First regression
#
#    slope, intercept, r_value, pvalue_regress, std_err=ss.linregress(ano_resetado,serie_lista)
#    numerator=ano_resetado.map(lambda x: x*slope)
#
#    #Second regression 
#
#    dists=[]
#    for j in range(len(serie_lista)):
#        dists.append(abs(serie_lista[j]-(slope*ano_resetado[j]+intercept)))    
#    slope2, intercept2, r_value2, pvalue_regress2, std_err2=ss.linregress(ano_resetado,dists)
#    data_reset_index=pd.Series(detrends[l])
#    denominator=ano_resetado.map(lambda x: x*slope2+intercept2)
#    retrend_series.append(data_reset_index*denominator + numerator)
#
##plot
#plt.figure()
#plt.plot(tr, retrend_series[0], color='#2F4F4F')
#plt.plot(tr, q_gev0_mean, color='blue')
#plt.fill_between(tr, retrend_series[1], retrend_series[2], color='grey', alpha=0.1)
#plt.fill_between(tr, q_gev0_high, q_gev0_low, color='blue', alpha=0.1)
#axes = plt.gca()
#axes.set_ylim([0,4000])
#plt.title('Ano 2009, GEVr')
   
