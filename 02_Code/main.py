# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 18:47:05 2020

@author: diego
"""
# All this work is bases on: https://github.com/dlainfiesta/covid-19/blob/master/Realtime%20R0.ipynb

#%% Setting 

import os

print(os.getcwd())
path= 'C:\\Users\\diego\\Desktop\\DL\\Covid_Guatemala\\Rt_Guatemala_m1'
os.chdir(path)
print(os.getcwd())

path_input= path+'\\01_Input\\'
path_code= path+'\\02_Code\\'
path_output= path+'\\03_Output\\'

#%% Importing packages

import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
from matplotlib.dates import date2num, num2date
from matplotlib import dates as mdates
from matplotlib import ticker
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from scipy import stats as sps
from scipy.interpolate import interp1d

from datetime import datetime

#%% Loading data Guatemala

url = 'https://covid.ourworldindata.org/data/owid-covid-data.csv'
df = pd.read_csv(url, index_col=0,parse_dates=[0], usecols=['location' ,'date', 'new_cases'])

country= 'Guatemala'
guatemala_raw= df[df.index==country]
guatemala_raw['date']= pd.to_datetime(guatemala_raw['date'], format='%Y-%m-%d')
guatemala_raw= guatemala_raw.set_index('date', drop= 'True')

confirmed = guatemala_raw.new_cases.dropna()
confirmed.tail()

state_name = 'Guatemala'

def prepare_cases(cases, cutoff=25):
    new_cases = cases.copy()

    smoothed = new_cases.rolling(7,
        win_type='gaussian',
        min_periods=1,
        center=True).mean(std=2).round()
    
    idx_start = np.searchsorted(smoothed, cutoff)
    
    smoothed = smoothed.iloc[idx_start:]
    original = new_cases.loc[smoothed.index]
    
    return original, smoothed

original, smoothed = prepare_cases(confirmed)

update= original.tail(1).index[0].strftime('%d-%m-%Y')
update_string= original.tail(1).index[0].strftime('%d_%m_%Y')

fig, ax= plt.subplots(figsize=(5,3))
original.plot(title=f"{state_name} Nuevos Casos por Día act. "+update,
               c='k',
               linestyle=':',
               alpha=.5,
               label='Nuevos casos confirmados',
               legend=True,
             figsize=(500/72, 300/72))

ax = smoothed.plot(label='Nuevos casos confirmados-Suavizada',
                   legend=True)

ax.get_figure().set_facecolor('w')

ax.yaxis.set_ticks(np.arange(0, 600, 50))

ax.grid(color='lightgrey', linestyle='-', linewidth= 0.5);
ax.legend();
fig.tight_layout()
fig.savefig(path_output+update_string+'_'+'Guatemala_nuevos_casos_suavizada.jpg', dpi=500)

#%% Getting posteriors

GAMMA = 1/7
# We create an array for every possible value of Rt
R_T_MAX = 12
r_t_range = np.linspace(0, R_T_MAX, R_T_MAX*100+1)


def get_posteriors(sr, sigma=0.25):

    # (1) Calculate Lambda
    lam = sr[:-1].values * np.exp(GAMMA * (r_t_range[:, None] - 1))

    
    # (2) Calculate each day's likelihood
    likelihoods = pd.DataFrame(
        data = sps.poisson.pmf(sr[1:].values, lam),
        index = r_t_range,
        columns = sr.index[1:])
    
    # (3) Create the Gaussian Matrix
    process_matrix = sps.norm(loc=r_t_range,
                              scale=sigma
                             ).pdf(r_t_range[:, None]) 

    # (3a) Normalize all rows to sum to 1
    process_matrix /= process_matrix.sum(axis=0)
    
    # (4) Calculate the initial prior
    #prior0 = sps.gamma(a=4).pdf(r_t_range)
    prior0 = np.ones_like(r_t_range)/len(r_t_range)
    prior0 /= prior0.sum()

    # Create a DataFrame that will hold our posteriors for each day
    # Insert our prior as the first posterior.
    posteriors = pd.DataFrame(
        index=r_t_range,
        columns=sr.index,
        data={sr.index[0]: prior0}
    )
    
    # We said we'd keep track of the sum of the log of the probability
    # of the data for maximum likelihood calculation.
    log_likelihood = 0.0

    # (5) Iteratively apply Bayes' rule
    for previous_day, current_day in zip(sr.index[:-1], sr.index[1:]):

        #(5a) Calculate the new prior
        current_prior = process_matrix @ posteriors[previous_day]
        
        #(5b) Calculate the numerator of Bayes' Rule: P(k|R_t)P(R_t)
        numerator = likelihoods[current_day] * current_prior
        
        #(5c) Calcluate the denominator of Bayes' Rule P(k)
        denominator = np.sum(numerator)
        
        # Execute full Bayes' Rule
        posteriors[current_day] = numerator/denominator
        
        # Add to the running sum of log likelihoods
        log_likelihood += np.log(denominator)
    
    return posteriors, log_likelihood

# We fix sigma,this value will be optimized later
posteriors, log_likelihood = get_posteriors(smoothed, sigma=.15)


fig,ax = plt.subplots(figsize=(10,6))
posteriors.plot(title=f'{state_name} - Posterior diaria para $R_t$ - 15/06/2020',
                ax= ax,
           legend=False, 
           lw=1,
           c='k',
           alpha=.3,
           xlim=(0.4,6))
ax.set_xlabel('$R_t$');
fig.tight_layout()
fig.savefig(path_output+update_string+'_'+'Posterior_Guatemala.jpg', dpi=500)
plt.show()


def highest_density_interval(pmf, p=.9, debug=False):
    # If we pass a DataFrame, just call this recursively on the columns
    if(isinstance(pmf, pd.DataFrame)):
        return pd.DataFrame([highest_density_interval(pmf[col], p=p) for col in pmf],
                            index=pmf.columns)
    
    cumsum = np.cumsum(pmf.values)
    
    # N x N matrix of total probability mass for each low, high
    total_p = cumsum - cumsum[:, None]
    
    # Return all indices with total_p > p
    lows, highs = (total_p > p).nonzero()
    
    # Find the smallest range (highest density)
    best = (highs - lows).argmin()
    
    low = pmf.index[lows[best]]
    high = pmf.index[highs[best]]
    
    return pd.Series([low, high],
                     index=[f'Low_{p*100:.0f}',
                            f'High_{p*100:.0f}'])


# Note that this takes a while to execute - it's not the most efficient algorithm
hdis = highest_density_interval(posteriors, p=.9)
most_likely = posteriors.idxmax().rename('ML')

# Look into why you shift -1
result = pd.concat([most_likely, hdis], axis=1)

# Calculating parameters for the day
cent= result.tail(1)['ML'].values[0]
up_lim= result.tail(1)['High_90'].values[0]
lw_lim= result.tail(1)['Low_90'].values[0]


def plot_rt(result, ax, state_name):
    
    ax.set_title(f"{state_name}")
    
    # Colors
    ABOVE = [1,0,0]
    MIDDLE = [1,1,1]
    BELOW = [0,0,0]
    cmap = ListedColormap(np.r_[
        np.linspace(BELOW,MIDDLE,25),
        np.linspace(MIDDLE,ABOVE,25)
    ])
    color_mapped = lambda y: np.clip(y, .5, 1.5)-.5
    
    index = result['ML'].index
    values = result['ML'].values
    
    # Plot dots and line
    ax.plot(index, values, c='k', zorder=1, alpha=.25)
    ax.scatter(index,
               values,
               s=40,
               lw=.5,
               c=cmap(color_mapped(values)),
               edgecolors='k', zorder=2)
    
    # Aesthetically, extrapolate credible interval by 1 day either side
    lowfn = interp1d(date2num(index),
                     result['Low_90'].values,
                     bounds_error=False,
                     fill_value='extrapolate')
    
    highfn = interp1d(date2num(index),
                      result['High_90'].values,
                      bounds_error=False,
                      fill_value='extrapolate')
    
    extended = pd.date_range(start=pd.Timestamp('2020-03-01'),
                             end=index[-1]+pd.Timedelta(days=1))
    
    ax.fill_between(extended,
                    lowfn(date2num(extended)),
                    highfn(date2num(extended)),
                    color='k',
                    alpha=.1,
                    lw=0,
                    zorder=3)

    ax.axhline(1.0, c='k', lw=1.5, label='$R_t=1.0$', alpha=.25);
    
    # Formatting
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax.xaxis.set_minor_locator(mdates.DayLocator())
    
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.1f}"))
    ax.yaxis.tick_right()
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.margins(0)
    ax.grid(which='major', axis='y', c='k', alpha=.1, zorder=-2)
    ax.margins(0)
    ax.set_ylim(0.0, 3.0)
    ax.yaxis.set_ticks(np.arange(0, 3.2, 0.20))
    ax.set_xlim(pd.Timestamp('2020-04-14'), result.index[-1]+pd.Timedelta(days=1))
    fig.set_facecolor('w')
    
fig, ax = plt.subplots(figsize=(600/72, 400/72))
plot_rt(result, ax, state_name)
ax.set_title(f'Número de reproducción en tiempo real $R_t$ para {state_name}')
ax.xaxis.set_major_locator(mdates.WeekdayLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
plt.annotate("Actualizada hasta el "+update, (pd.Timestamp('2020-04-17'), 2.5), fontsize= 12, color= 'red')
plt.annotate("Rt: "+str(round(cent, 2)), (pd.Timestamp('2020-06-7'), 2.5), fontsize= 14, color= 'red')
fig.tight_layout()
fig.savefig(path_output+update_string+'_'+'R_t.jpg', dpi=500)
plt.show()

# Printing result
if 1 <= lw_lim:
    print('Malas noticias:')
    print('Todo el pronóstico de Rt es creciente')
    print('La Rt es igual a '+str(round(cent, 2)))
    print('La Rt máxima es igual a '+str(round(up_lim, 2))+' y la mínima de '+str(round(up_lim, 2))+' con un 90% de confianza')

else:
    print('Noticias:')
    print('No todo el pronóstico de la Rt es creciente')
    print('La Rt es igual a '+str(round(cent, 2)))
    print('La Rt máxima es igual a '+str(round(up_lim, 2))+' y la mínima de '+str(round(lw_lim, 2))+' con un 90% de confianza')
    print('Existe un '+str(round((1-lw_lim)/(up_lim-lw_lim), 2)*100)+'% de probabilidades que la Rt sea inferior a 1 y '+\
          str(100-round((1-lw_lim)/(up_lim-lw_lim), 2)*100) + '% que sea superior a 1, con un 90% de confianza.')
