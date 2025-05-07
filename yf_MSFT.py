## 
# %%
# Imports
from scipy import stats
import yfinance as yf
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf, q_stat
from arch.unitroot import PhillipsPerron
from statsmodels.tsa.stattools import grangercausalitytests
import statsmodels.api as sm
import yfinance as yf
import pandas as pd
import numpy as np


# %%
# Baixar dados de ações da Microsoft para os últimos 10 anos usando yfinance
msft = yf.download('MSFT',  start='2013-06-30', end='2023-06-30')
msft.reset_index(inplace=True)

# %%
msft.head()

# %% 
msft.columns

# %%
# Realizar análise estatística básica
summary = msft.describe()
summary

# %%
# Salvar sumário estatístico em CSV
summary.to_csv('dados/script1_sumario.csv', header=True, index=False)


#%%
msft.columns

#%%
msft.columns = msft.columns.droplevel('Ticker')

#%%
msft.head()

#%%
msft.dtypes

#%%
msft.columns

#%%
msft.index = msft['Date']

# %%
msft.head()

# %% 
msft.drop('Date', axis=1, inplace=True)

# %%
msft.head()

# %%
mean_close = msft['Close'].mean()
mean_close

# %%


std_close = msft['Close'].std()
std_close


# %%

min_close = msft['Close'].min()
min_close

# %%
max_close = msft['Close'].max()
max_close

# %%
stats = pd.DataFrame(
    { "mean_close" : [mean_close],
      "std_close" : [std_close],
      "min_close" : [min_close],
        "max_close" : [max_close] 
     }
)

# %%
stats

# %%
# Salvar estatísticas adicionais em CSV
stats.to_csv('dados/script1_stats.csv', header=True, index=False)


# %%

close_prices = msft['Close']

# %%
close_prices.head()

# %%
shapiro_test = stats.shapiro(close_prices)
print(f"\nTeste Shapiro-Wilk: W={shapiro_test[0]}, p-value={shapiro_test[1]}")

# %%
# Teste de Normalidade (Anderson-Darling)
ad_test = stats.anderson(close_prices)
print(f"\nTeste Anderson-Darling: Statistic={ad_test.statistic}, Critical Values={ad_test.critical_values}, Significance Level={ad_test.significance_level}")

# %%
# Teste de Homogeneidade de Variância (Levene)
# Assumindo que estamos comparando entre duas partes dos dados
split_index = len(close_prices) // 2
levene_test = stats.levene(close_prices[:split_index], close_prices[split_index:])
print(f"\nTeste de Levene: Statistic={levene_test.statistic}, p-value={levene_test.pvalue}")




# %%
stats_adv = pd.DataFrame(
    { "shapiro-wilk" : [shapiro_test],
      "anderson" : [ad_test],
      "levene" : [levene_test],
     }
)

# %%
stats_adv

# %%
stats_adv.to_csv('dados/script1_stats_adv.csv', header=True, index=False)


#%%
# Teste de Dickey-Fuller Aumentado (ADF)
adf_test = adfuller(close_prices)
print(f"\nTeste ADF: Statistic={adf_test[0]}, p-value={adf_test[1]}, Critical Values={adf_test[4]}")

#%%
# Teste KPSS
kpss_test = kpss(close_prices)
print(f"\nTeste KPSS: Statistic={kpss_test[0]}, p-value={kpss_test[1]}, Critical Values={kpss_test[3]}")

#%%
# Teste de Phillips-Perron
pp_test = PhillipsPerron(close_prices)
pp_test_statistic = pp_test.stat
pp_test_pvalue = pp_test.pvalue
print(f"\nTeste Phillips-Perron: Statistic={pp_test_statistic}, p-value={pp_test_pvalue}")

#%%
# Teste de Ljung-Box
ljung_box_test = q_stat(acf(close_prices, fft=False, nlags=40), len(close_prices))
print(f"\nTeste Ljung-Box: Statistics={ljung_box_test[0]}, p-values={ljung_box_test[1]}")

#%%
# ACF e PACF
acf_vals = acf(close_prices, fft=False, nlags=40)
pacf_vals = pacf(close_prices, nlags=40)

#%%
# Salvar ACF e PACF em CSV
acf_df = pd.DataFrame({'Lag': np.arange(len(acf_vals)), 'ACF': acf_vals})
pacf_df = pd.DataFrame({'Lag': np.arange(len(pacf_vals)), 'PACF': pacf_vals})

#%%
acf_df.to_csv('dados/script2_acf.csv', index=False)
pacf_df.to_csv('dados/script2_pacf.csv', index=False)

# Baixar dados de ações da Microsoft e Apple para os últimos 10 anos usando yfinance
msft = yf.download('MSFT', start='2013-06-30', end='2023-06-30')
aapl = yf.download('AAPL', start='2013-06-30', end='2023-06-30')



#%%
aapl = yf.download('AAPL', start='2013-06-30', end='2023-06-30')

# %%
aapl.head()

# %%
# Manter apenas a coluna de fechamento ajustado
msft_close = msft[['Close']]
aapl_close = aapl[['Close']]


# %%
aapl_close.head()

# %%
# %%
msft_close.head()

#%%
# Combinar os dois dataframes em um único dataframe
combined_df = pd.concat([msft_close, aapl_close], axis=1)
combined_df.columns = ['MSFT_Close', 'AAPL_Close']

# %%
# Remover NaNs
combined_df.dropna(inplace=True)

# %%
combined_df.head()

# %%
# Aplicar o Teste de Causalidade de Granger
# O parâmetro 'maxlag' define o número máximo de lags a serem testados
maxlag = 5
test_result = grangercausalitytests(combined_df[['AAPL_Close', 'MSFT_Close']], maxlag = maxlag)
# %%
