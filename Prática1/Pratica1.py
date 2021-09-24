import pandas as pd
import numpy as np
import matplotlib.pylab as plt
plt.style.use('seaborn-whitegrid')
plt.rc('text', usetex=False)
plt.rc('font', family='times')
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)
plt.rc('font', size=12)

tabela = pd.read_csv('F:\Métodos Estatísticos/educ_figdp_1_Data.csv',na_values=':', usecols=['TIME', 'GEO', 'Value'], engine='python')
#print(tabela)
#print(tabela.head())
#print(tabela.columns)
#print(tabela.index)
#print(tabela.describe())
#print(tabela['Value'])
#print(tabela[10:14])
#print(tabela.ix[90:94,['TIME','GEO']])
#print(tabela[tabela['Value'] > 6.5].tail())
#print(tabela[tabela['Value'].isnull()].head())
#print(tabela.max(axis=0))
#s = tabela['Value'] / 100
#print(s.head())
#s = tabela['Value'].apply(np.sqrt)
#print(s.head())
#s = tabela['Value'].apply(lambda d: d**2)
#print(s.head())
#tabela['ValueNorm'] = tabela['Value'] / tabela['Value'].max()
#print(tabela.tail())
#tabela.drop('ValueNorm',axis=1,inplace=True)
#print(tabela.head())
#tabela = tabela.append({'TIME': 2000, 'Value': 5.00, 'GEO': 'a'}, ignore_index=True)
#print(tabela.tail())
#tabela.drop(max(tabela.index), axis=0, inplace=True)
#print(tabela.tail())
#tabelaDrop = tabela.drop(tabela['Value'].isnull(),axis=0)                  (não funciona)
#print(tabelaDrop.head())
#tabelaDrop = tabela.dropna(how='any', subset=['Value'], axis=0)
#print(tabelaDrop.head())
#tabelaFilled = tabela.fillna(value={'Value': 0})
#print(tabelaFilled.head())
#tabela.sort_values(by='Value', ascending=False, inplace=True)
#print(tabela.head())
#tabela.sort_index(axis=0, ascending=True, inplace=True)
#print(tabela.head())
#group = tabela[['GEO', 'Value']].groupby('GEO').mean()
#print(group.head())
#filtered_data = tabela[tabela['TIME'] > 2005]
#pivedu = pd.pivot_table(filtered_data, values='Value', index=['GEO'], columns=['TIME'])
#print(pivedu.head())
#print(pivedu.ix[['Spain', 'Portugal'], [2006, 2011]])
#pivedu = pivedu.drop(['Euro area (13 countries)',
#                      'Euro area (15 countries)',
#                      'Euro area (17 countries)',
#                      'Euro area (18 countries)',
#                      'European Union (25 countries)',
#                      'European Union (27 countries)',
#                      'European Union (28 countries)'
#                      ], axis=0)
#pivedu = pivedu.rename(
#    index={'Germany (until 1990 former territory of the FRG)': 'Germany'})
#pivedu = pivedu.dropna()
#print(pivedu.rank(ascending=False, method='first').head())
#totalSum = pivedu.sum(axis=1)
#print(totalSum.rank(ascending=False, method='dense').sort_values().head())
#fig = plt.figure(figsize=(12, 5))
#totalSum = pivedu.sum(axis=1).sort_values(ascending=False)
#totalSum.plot(kind='bar', style='b', alpha=0.4, title='Total Values for Country')
#plt.savefig('Totalvalue_Country.png', dpi=300, bbox_inches='tight')
my_colors = ['b', 'r', 'g', 'y', 'm', 'c']
ax = pivedu.plot(kind='barh', stacked=True, color=my_colors, figsize=(12, 6))
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.savefig('Value_Time_Country.png', dpi=300, bbox_inches='tight')