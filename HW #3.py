#!/usr/bin/env python
# coding: utf-8

# # Homework #3

# For this assignment, you will learn about two new libraries, pyttsx3 and speech_recognition, and use voice recognition to obtain weather info <br />
# 
# **pyttsx3** - https://pypi.org/project/pyttsx3/ <br />
# **speech_recognition** - https://pypi.org/project/SpeechRecognition/ <br />
# 
# **pyAudio** - PyAudio‑0.2.11‑cp38‑cp38‑win_amd64.whl <br />
# Use following stackoverflow post to troubleshoot if needed 
# https://stackoverflow.com/questions/61290821/error-command-errored-out-with-exit-status-1-while-installing-pyaudio

# First, try to import the two libraries to see if can import the modules, if not, you need to do pip install to have them installed through terminal (MAC) or command line (MS)

# In[34]:


#!pip3 install SpeechRecognition 
#!pip3 install pyttsx3
#!brew install portaudio
#!brew --prefix portaudio
get_ipython().system('pip3 install pyAudio')


# In[1]:


#functions 

import numpy as np
import pandas as pd
import yfinance as yf

def city_clean(cell):
    if '[' in cell:
        cut = cell.index('[')
        return cell[0:cut]
    else:
        return cell
def get_temp(city,state):
    city = city.replace(" ", "%20")
    response = urlopen('http://api.openweathermap.org/data/2.5/weather?q=' + city + ',' + state + ',US&appid=7dc34849d7e8b6fbdcb3f12454c92e88')
    rawWeatherData = response.read().decode("utf-8") ##read into text
    weatherData = json.loads(rawWeatherData)
    Desc = weatherData['weather'][0]['description']
    Temp = round((weatherData['main']['temp'] - 273.15) * 9/5 +32,3)
    Temp = str(Temp)
    feels = round((weatherData['main']['feels_like'] - 273.15) * 9/5 +32,3)
    feels = str(feels)
    ### condense return in 1 shot so no need to call API multiple times
    return [Temp, feels,Desc]


# Try out the test script from ptytsx3 website, see if it runs, did you hear the computer speak?

# In[2]:


import pyttsx3
import speech_recognition as sr


engine = pyttsx3.init()
engine.say("test 1,2,3")
engine.runAndWait()


# Based on website description, see if you can make engine speak in female voice (**hint:** try voices 1, and 10)

# In[125]:


voices = engine.getProperty('voices') 
engine.setProperty('voice', voices[10].id)  
engine.say("Shalashaska")
engine.runAndWait()


# Check your microphone

# In[126]:


for index, name in enumerate(sr.Microphone.list_microphone_names()):
    print(f'{index}, {name}')


# Now let's try listening, try to speak to it, does it provide correct return?
# **Note:** only talk to the computer after "listening" is printed

# In[3]:


##Make sure you install pyaudio first!!

listener = sr.Recognizer()

try:
    with sr.Microphone(device_index = 0) as source:
        print('listening...')
        voice = listener.listen(source, timeout=4) ##timeout after 2 seconds
        command = listener.recognize_google(voice, language = 'us-eng')
        print(command)
except:
    print("Nothing is Captured")


# In[3]:


from urllib.request import urlopen  ##import a function
import json

weather = pd.read_html('https://en.wikipedia.org/wiki/List_of_United_States_cities_by_population')
w_table = weather[4]
s_map_t = pd.read_html('https://simple.wikipedia.org/wiki/U.S._postal_abbreviations')
s_map = s_map_t[1]


A = 'New York City[d]'
cut = A.index('[')
A = A[0:cut]
w_table['city_clean'] = w_table['City'].apply(city_clean)
comb = pd.merge(w_table, s_map, left_on = 'State[c]', right_on = 'State Name', how='left')
comb = comb[['city_clean','State Abbreviation','State Name']]
comb.at[19, 'State Abbreviation'] = 'DC'

response = urlopen('http://api.openweathermap.org/data/2.5/weather?q=paoli,pa,US&appid=7dc34849d7e8b6fbdcb3f12454c92e88')
rawWeatherData = response.read().decode("utf-8") ##read into text
weatherData = json.loads(rawWeatherData)
weatherData['main']['temp'] = round((weatherData['main']['temp'] - 273.15) * 9/5 +32,3)
weatherData['main']['feels_like'] = round((weatherData['main']['feels_like'] - 273.15) * 9/5 +32,3)


# In[ ]:





# Now you can get Python to both listen and talk, let's leverage what we learned in class and have computer tell us weather of a city
# for the first test, let's only worry about the "happy path", don't worry about error for now
# 1. Computer initiate by asking what city you are looking for?
# 2. User provide a city
# 3. Computer ask for state abbreviation
# 4. User provide state abrreviation 
# 5. System reads out weather description, temperature and feels like (all part of json readout) 

# In[6]:


from urllib.request import urlopen  ##import a function
import json


# In[167]:


listener = sr.Recognizer()

try:
    with sr.Microphone(device_index = 0) as source:
        engine.say("Where would you like me to look up?")
        engine.runAndWait()
        print('listening...')
        voice = listener.listen(source, timeout=4) ##timeout after 2 seconds
        city = listener.recognize_google(voice, language = 'us-eng')
        print(city)
except:
    city = NA
    print("Nothing is Captured")
    
    
try:
    with sr.Microphone(device_index = 0) as source:
        engine.say('What state is the city in?')
        engine.runAndWait()
        print('listening...')
        voice2 = listener.listen(source, timeout=8) ##timeout after 2 seconds
        State = listener.recognize_google(voice2, language = 'us-eng')

except:
    State = NA
    print('Nothing captured')
    
    

#S = comb.loc[comb['State Name'] == State]             #useful if they say full state name
#S = S.iloc[0,1]
#S 
print(State, city)

engine.say('The temperature in ' + city + ' ' + State +' is ' + V[0] + ' degrees fahrenheit')
engine.say('The temperature in ' + city + ' ' + State +' feels like ' + V[1] + ' degrees fahrenheit')
engine.say('The weather conditions are: ' + V[2])
engine.runAndWait()  


# In[4]:


city = 'Orlando'
State = 'FL'
def get_temp(city,state):
    city = city.replace(" ", "%20")
    response = urlopen('http://api.openweathermap.org/data/2.5/weather?q=' + city + ',' + state + ',US&appid=7dc34849d7e8b6fbdcb3f12454c92e88')
    rawWeatherData = response.read().decode("utf-8") ##read into text
    weatherData = json.loads(rawWeatherData)
    Desc = weatherData['weather'][0]['description']
    Temp = round((weatherData['main']['temp'] - 273.15) * 9/5 +32,3)
    Temp = str(Temp)
    feels = round((weatherData['main']['feels_like'] - 273.15) * 9/5 +32,3)
    feels = str(feels)
    ### condense return in 1 shot so no need to call API multiple times
    return [Temp, feels,Desc]
V = get_temp(city,State)


# In[160]:


engine.say('The temperature in ' + city + ' ' + State +' is ' + V[0] + ' degrees fahrenheit')
engine.say('The temperature in ' + city + ' ' + State +' feels like ' + V[1] + ' degrees fahrenheit')
engine.say('The weather conditions are: ' + V[2])
engine.runAndWait()        


# In[6]:


engine.endLoop()


# Were there times the voice didn't capture correctly? Convert it into a function
# Please leverage various parameters with speech_recognition to optimize output
# In addition, please write some looping code to loop back for potential errors
# 1. For example, if 'MJ' is captured instead of 'NJ', we should ask user to confirm
# 2. Timeout errors

# In[5]:


states = np.array(comb['State Abbreviation'])
listener = sr.Recognizer()
def Weather_get():
    city, State = 'NA', 'NA'
    while city == 'NA':
        try:
            with sr.Microphone(device_index = 0) as source:
                engine.say("Which City are you looking for?")
                print('listening...')
                engine.runAndWait()
                voice = listener.listen(source, timeout=8) ##timeout after 2 seconds
                city = listener.recognize_google(voice, language = 'us-eng')

                print(city)

        except:
            city = 'NA'
            engine.say('That is not in my database. Please try again')
            engine.runAndWait()
    
    while State == 'NA':
        try:
            with sr.Microphone(device_index = 0) as source:
                engine.say('What state is the city in?')
                print('listening...')
                engine.runAndWait()

                voice2 = listener.listen(source, timeout=6) ##timeout after 2 seconds
                State = listener.recognize_google(voice2, language = 'us-eng')
            for i in states:
                in_data = False
                if State == i:
                    break
            if in_data == False:
                State = 'NA'
            else: 
                State = State
        except:
            State = 'NA'
            engine.say('That is not in my database. Please try again')
            engine.runAndWait()

Weather_get()


# Let's leverage what you learned in class this week and design a quick tool

# In[43]:


import pandas as pd
import numpy as np
from datetime import date

###Symbol column will provide you ticker of all 30 index
ticker = pd.read_html('https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average')
ticker[1]


# In[41]:


###Pull volume of stock for individual ticker
import requests
url_link = 'https://finance.yahoo.com/quote/MMM/key-statistics?p=MMM'
enhanced_link = requests.get(url_link,headers ={'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'})
MMM = pd.read_html(enhanced_link.text)
MMM[0]


# In[161]:


tickers = ticker[1].Symbol        #get all the relevant tickers
ticks = []
today = date.today()
for i in tickers:                 #add tickers to list
    ticks.append(i)
tickers_df = yf.download(ticks, 
                      start='2023-03-28', 
                      end= today, 
                      progress=False, auto_adjust=True)
tickers_df = tickers_df[1:]             #now have all tickers in one dataframe


# In[95]:


#Highest Volume Calc
m = 0
Volume = tickers_df.Volume

for i in ticks:
    if Volume[i][0] > m:
        m = Volume[i][0]
        V = i          #save ticker with highest volume
    else:
        m = m
#Largest Fluc calc
m2 = 0
High = tickers_df.High
Low = tickers_df.Low
for i in ticks:
    h = High[i][0]
    l = Low[i][0]
    fl = (h-l)/l
    if fl > m2:
        m2 = fl
        F = i          #save ticker with highest fluctuation
    else:
        m2=m2
#Max Gain Calc
m3 = 0
m4 = 0
Close = tickers_df.Close
Open = tickers_df.Open
for i in ticks:
    o = Open[i][0]
    c = Close[i][0]
    g = (c-o)/o
    if g > m3:
        m3 = g
        G = i           #Save ticker with highest max gain
    if g < m4:
        m4 = g
        ML = i          #Save ticker with max loss
    
        
    else:
        m3=m3
        m4=m4


# Write a program that will send your self an email on stock summary based on last trading day (You may assume the code is only ran after market is closed)
# 1. Stock Ticker with highest volume
# 2. Stock Ticker with largest fluctuation (High - Low) / Low
# 3. Stock Ticker with max gain (Close-Open) / Open
# 4. Stock Ticket with max loss or min gain (Close-Open) / Open
# 
# 
# Once you have your code finalized, have a copy send to rutgersadvancedpython01@outlook.com

# In[96]:


import smtplib
from os.path import basename
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


# In[156]:


msg = MIMEMultipart()

msg['Subject'] = "HW3 Stock Ticker Data Adv Programming"
msg['From'] = 'APpythondummy@outlook.com'
msg['To'] = 'rutgersadvancedpython01@outlook.com'

###insert password
f = open("/Users/chrisk/Desktop/nice", "r")
password = str(f.read())

text = 'The stock with the highest volume is ' + V + ' at ' + str(m) + '\n' + \
         'The stock with the highest fluctuation is ' + F + ' at ' + str(m2) + '\n' + \
        'The stock with the max gain is ' + G + ' at ' + str(m3) + '\n' + \
        'The stock with the max loss is ' + ML + ' at ' + str(m4) + '\n' + 'THIS EMAIL IS FROM CHRIS KOZAK, netid: CJK188'

html = """\
<html>
  <head></head>
    <body>
       <p>
       """ + text + """\
       </p>   
       <img src="cid:image1" alt="Logo"><br>
    </body>
</html>
"""
part2 = MIMEText(html, 'html')
msg.attach(part2)


# In[157]:


s = smtplib.SMTP(host='smtp-mail.outlook.com', port = 587)
s.starttls()
s.login('APpythondummy@outlook.com',password)
s.sendmail('APpythondummy@outlook.com', 'rutgersadvancedpython01@outlook.com', msg.as_string())


# Below is in case I cannot get the email to run but would still like the message to at least show up. I did get the email to send, but I'd like to keep this here just in case.

# In[151]:


print(text)


# In[ ]:




