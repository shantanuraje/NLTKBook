import nltk, re, feedparser, unicodedata
from nltk import word_tokenize
from urllib import request
from bs4 import BeautifulSoup
from nltk.corpus import words # english vocabulary

##uncomment this block after gutenberg ban lifts
##get file from url
#url="http://www.gutenberg.org/files/2554/2554.txt"
#response=request.urlopen(url)
#raw=response.read().decode('utf8')
##get only book content
#print(raw.find("PART I"))
#print(raw.rfind("End of Project Gutenberg's Crime"))
#raw=raw[5338:1157746]

##tokenize the string
#tokens=word_tokenize(raw)
#text=nltk.Text(tokens)
#print(text.collocations())

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
    
#read local file
tale_of_peter = open('tale_of_peter_rabbit.txt') # r for reading, U for Universal - to ignore different conventions used to mark newlines
print(tale_of_peter.read())
#reading nltk corpora using file open
path = nltk.data.find('corpora/gutenberg/melville-moby_dick.txt')
print(path)
raw=open(path).read()
print(raw)

#path statment causing value error in spyder, check and solve later
#extract encoded text (eg. polish) polish-lat2.txt
path = nltk.data.find('corpora/unicode_samples/polish-lat2.txt')
f=open(path, encoding='latin2')
#causing value error - malformed node or string, ascii codec cant encode character '\u0144' in position xxxx: ordinal not in range
for line in f:
  print(line) #if we want to print \xXX or \uXXXX representations, use print(line.encode('unicode_escape'))


#statements below work but also cause internal console errors
print(ord('ń') )
nacute='\u0144'
print(nacute)
#select all chars from this polish line, outside ascii range, print utf-8 byte sequence, code point integer
print(line.encode('unicode_escape'))
for c in line:
  if ord(c)>127:#ord() returns unicode code point for a 1-char string
      print('{} U+{:04x} {}'.format(c.encode('utf8'), ord(c), unicodedata.name(c)))

lines=open(path, encoding='latin2').readlines()
print(lines)
line=lines[2]
print("Line")
print(line)
print(line.find('zosta\u0142y'))
print(line.encode('unicode_escape'))
m=re.search('\u015b\w*',line)
print(m.group())
print(word_tokenize(line)) #tokenize unicode string, output is unicode list

#regular expressions
wordlist=[w for w in words.words('en')if w.islower()]
#words ending with 'ed'
print([w for w in wordlist if re.search("ed$",w)])
#words having j as 3rd,t as 6th, letters with fixed number of letters in between
print([w for w in wordlist if re.search('^..j..t..$',w)])
print([w for w in wordlist if re.search('..j..t..',w)]) #gives any number of chars before j and after t, but only two characters between j and t
