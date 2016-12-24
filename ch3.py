import nltk, re, pprint, feedparser
from nltk import word_tokenize
from urllib import request
from bs4 import BeautifulSoup


#get file from url
url="http://www.gutenberg.org/files/2554/2554.txt"
response=request.urlopen(url)
raw=response.read().decode('utf8')
#get only book content
print(raw.find("PART I"))
print(raw.rfind("End of Project Gutenberg's Crime"))
raw=raw[5338:1157746]

#tokenize the string
tokens=word_tokenize(raw)
text=nltk.Text(tokens)
print(text.collocations())

#access a html document
url="http://www.nltk.org/book/ch03.html"
html=request.urlopen(url).read().decode('utf8')
print(html[:70])

#parse html using beautifulsoup
raw=BeautifulSoup(html).get_text()
print(raw.find('Processing Search Engine Results'))
print(raw.find('if it is reasonable'))
tokens=word_tokenize(raw[18655:19404])
print(tokens)

#blog feed parsing
llog=feedparser.parse("http://languagelog.ldc.upenn.edu/nll/?feed=atom") #its a json
print(llog['feed']['title'])
print(len(llog.entries))
for entry in llog.entries:
    print(entry.title)