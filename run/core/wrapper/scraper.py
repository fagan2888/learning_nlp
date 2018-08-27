#!/usr/local/bin/ python3
import requests
import re
import json

from datetime import datetime
from bs4 import BeautifulSoup as soup

def get_company_name(ticker):
	try:
		endpoint = 'http://dev.markitondemand.com/MODApis/Api/v2/Quote/json?symbol='+ticker
		response = requests.get(endpoint).json()
		company_name = response['Name']
		return company_name
	except:
		return 'N/A'

def get_barrons_news(ticker):
    link = f'https://www.barrons.com/search?keyword={ticker}&numResults=75&sort=date-desc&author=&searchWindow=0&minDate=&maxDate=&source=barrons'
    source = requests.get(link, proxies={'http':'50.207.31.221:80'}).text
    s = soup(source, 'lxml')    
    articles = s.find('div',{"class" : "tab-pane active"})
    lst = []
    for row in articles.findAll('li'):
        date  = ' '.join(row.find('span',{'class' : 'date'}).text.split(' ')[:3])
        date  = datetime.strptime(date,'%b %d, %Y').date()
        news  = row.find('span',{'class' : 'headline'}).text.lower()
        summary = row.find('p',{'class' : 'news__summary hidden'}).text.lower()
        lst.append([date,news + ' ' + summary])
    company_name = get_company_name(ticker).split(' ')[0].lower()
    barrons_news = [[date,text] for date,text in lst if company_name in text]
    return barrons_news

def get_finviz_news(ticker):
    source = requests.get(f'https://finviz.com/quote.ashx?t={ticker}', proxies={'http':'50.207.31.221:80'}).text
    s    = soup(source, 'lxml')
    articles = s.find('table',{ "class" : "fullview-news-outer" })
    x    = [row.text.split('\xa0\xa0') for row in articles.findAll('tr')]
    lst  = "CNBC|American City Business|Investor's Business|Financial|Motley|The Wall Street"
    t    = [[z,re.sub(lst,'',' '.join(y.split(' ')[:-1]).replace("\xa0",' '))] for z,y in x]
    lst  = []
    news = []
    for x,y in t:
        date = x.split(' ')
        if len(date) == 2:
            news_date = datetime.strptime(date[0],'%b-%d-%y').date()
            lst.append(news_date)
            news.append([lst[-1],y.lower()])
        elif len(date) == 1:
            news.append([lst[-1],y.lower()])
    company_name = get_company_name(ticker).split(' ')[0].lower()
    finviz_news = [[date,text] for date,text in news if company_name in text]
    return finviz_news

def get_ticker_news(ticker):
    news = get_barrons_news(ticker) + get_finviz_news(ticker)
    news.sort(key=lambda r:r[0])
    return news