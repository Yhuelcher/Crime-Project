
""" Importing all files"""


import pandas as pd


pd.read_table('Crime_Incidents_2012.csv')
crime_12 = pd.read_table('Crime_Incidents_2012.csv', sep=',')

pd.read_table('Crime_Incidents_2013.csv')
crime_13 = pd.read_table('Crime_Incidents_2013.csv', sep=',')

pd.read_table('Crime_Incidents_2014.csv')
crime_14 = pd.read_table('Crime_Incidents_2014.csv', sep=',')

pd.read_table('weather12.csv')
weather_12 = pd.read_table('weather12.csv', sep=',')

pd.read_table('weather13.csv')
weather_13 = pd.read_table('weather13.csv', sep=',')

pd.read_table('weather14.csv')
weather_14 = pd.read_table('weather14.csv', sep=',')

pd.read_table('crimecombined.csv')
crime = pd.read_table('crimecombined.csv', sep=',')

pd.read_table('Employment.csv')
emp = pd.read_table('Employment.csv', sep=',')

pd.read_table('weather2014date.csv')
weatherdate14 = pd.read_table('weather2014date.csv', sep=',')

pd.read_table('weather2013date.csv')
weatherdate13 = pd.read_table('weather2013date.csv', sep=',')

pd.read_table('weather2012date.csv')
weatherdate12 = pd.read_table('weather2012date.csv', sep=',')

pd.read_table('weather2012datexxx.csv')
weatherdate = pd.read_table('weather2012datexxx.csv', sep=',')

pd.read_table('crimecombinedx.csv')
crimex = pd.read_table('crimecombinedx.csv', sep=',')



crime.to_excel('crime.xls')

"""Exploring the Crime data"""
crime                
type(crime)             
crime.head(10)            
crime.tail(1)           
crime.columns           
crime.dtypes           
crime.shape            


crime.WARD.value_counts()     
crime.OFFENSE.value_counts()  
crime.groupby('year').OFFENSE.count()
crime.groupby('WARD').OFFENSE.count()



crime.groupby('OFFENSE').year.count()

import matplotlib.pyplot as plt
crime.WARD.plot(kind='hist', bins=20, title='Histogram of Ward')
plt.xlabel('WARD')
plt.ylabel('Frequency')

"""Splitting out REPORTDATETIME to have date and time seperate to match with the Weather data""" 

crime = pd.read_csv('crimecombined.csv', parse_dates=[0], infer_datetime_format=True)
temp = pd.DatetimeIndex(crime['REPORTDATETIME'])
crime['Date'] = temp.date
crime['Time'] = temp.time
del crime['REPORTDATETIME']

"""merging tables on date"""

crimecom = crimex.merge(weatherdate12xxx, on='Date')
crimecom.head()


"""since its giving me issues, export and reimport"""
crimecom.to_csv('crimecom.csv')


crime.groupby('year').WARD.plot(kind='hist', bins=20, title='Histogram of Ward')


import pandas as pd
import matplotlib.pyplot as plt
pd.read_table('Crime_Incidents_2012.csv')
crime12 = pd.read_table('Crime_Incidents_2012.csv', sep=',')


crime                
type(crime12)             
crime12.head(10)                       
crime12.columns 
crime12.rename(columns={'REPORTDATETIME':'Date', 'SHIFT':'Shift', 'OFFENSE':'Offense','METHOD':'Method', 'WARD':'Ward'}, inplace=True)          
crime12.dtypes           
crime12.shape   

crime12.groupby('WARD').OFFENSE.count() 

import pandas as pd
pd.read_table('Crime_Incidents_2013.csv')
crime13 = pd.read_table('Crime_Incidents_2013.csv', sep=',')


crime13              
type(crime13)             
crime13.head(10)                       
crime13.columns 
crime13.rename(columns={'REPORTDATETIME':'Date', 'SHIFT':'Shift', 'OFFENSE':'Offense','METHOD':'Method', 'WARD':'Ward'}, inplace=True)          
crime13.dtypes           
crime13.shape   

crime13.groupby('Ward').Offense.count() 

import pandas as pd
pd.read_table('Crime_Incidents_2014.csv')
crime14 = pd.read_table('Crime_Incidents_2014.csv', sep=',')


crime14                
type(crime14)             
crime14.head(10)                       
crime14.columns 
crime14.rename(columns={'REPORTDATETIME':'Date', 'SHIFT':'Shift', 'OFFENSE':'Offense','METHOD':'Method', 'WARD':'Ward'}, inplace=True)          
crime14.dtypes           
crime14.shape   

crime14.groupby('Ward').Offense.count()
 

crime14.Offense.hist(by=crime14.Ward, sharex=True)

crime14.WARD.plot(kind='hist', bins=20, title='Histogram of Ward')
plt.xlabel('WARD')
plt.ylabel('Frequency')

crime.isnull()
crime.isnull().sum() 
crime.WARD.isnull().sum()

crime.groupby ('year').WARD.value_counts().sort_index()
crime.groupby ('WARD', 'SHIFT').year.value_counts().sort_index()
crime.groupby ('year').OFFENSE.value_counts().sort_index()

crime.WARD.plot(kind='hist', bins=20, title='Histogram of Ward')
plt.xlabel('WARD')
plt.ylabel('Frequency')

pd.read_table('Employment.csv')
employment = pd.read_table('Employment.csv', sep=',')

employment                
type(employment)             
employment.head(10)                       
employment.columns
employment.dtypes 
employment.boxplot(column='Unemployment', by='YEAR')
pd.scatter_matrix(employment[['Unemployment', 'Employment', 'LaborForce']])
employment.plot(kind='scatter', x='Unemployment', y='Employment', c='YEAR', colormap='Blues')