import nltk, re, feedparser, unicodedata, pprint
from nltk import word_tokenize
from urllib import request
from bs4 import BeautifulSoup
from nltk.corpus import words # english vocabulary
from nltk.corpus import gutenberg, nps_chat, brown

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
print([w for w in wordlist if re.search("^[ghi][mno][jlk][def]$",w)])
#finger twisters - words that can be typed using only certain parts of the t9 keypad
print([w for w in wordlist if re.search("^[ghijklmno]+$",w)]) #only use numbers 4 5 6
print([w for w in wordlist if re.search("^[abcdef]+$",w)]) #only use numbers 2 3
print([w for w in wordlist if re.search("^[defmnowxyz]+$",w)]) #only use numbers 3 6 9
chat_words=sorted(set(w for w in nltk.corpus.nps_chat.words()))
print([w for w in chat_words if re.search("^m+i+n+e+$",w)]) # starting with m,end with e, 1 or more occurences of m i n e
print([w for w in chat_words if re.search("^[ha]+$",w)]) # starting with h or a, one more occurences of h or a
print([w for w in chat_words if re.search("^[^aeiouAEIOU]+$",w)]) # anything but vowels

treebank_data=sorted(set(nltk.corpus.treebank.words()))
print([w for w in treebank_data if re.search('^[0-9]+\.[0-9]+$',w)])
print([w for w in treebank_data if re.search('^[A-Z]+\$$',w)])
print([w for w in treebank_data if re.search('^[0-9]{4}$',w)])
print([w for w in treebank_data if re.search('^[0-9]+-[a-z]{3,5}$', w)])
print([w for w in treebank_data if re.search('^[a-z]{5,}-[a-z]{2,3}-[a-z]{,6}$', w)])
print([w for w in treebank_data if re.search('(ed|ing)$', w)])
#extracting word pieces
#check for all vowels in words, find number of vowels
word = 'supercalifragilisticexpialidocious'
print(re.findall(r'[aeiou]',word))
print(len(re.findall(r'[aeiou]',word)))
#look for all sequences of 2 or more vowels, determine relative frequency
vowel_sequences=nltk.FreqDist(vs for word in treebank_data for vs in re.findall(r'[aeiou]{2,}',word))
print(vowel_sequences.most_common(30))
print([int(n) for n in re.findall(r'([0-9]{4}|[0-9]{2}|[0-9]{2})', '2009-12-31')])
#leave word internal vowels out to make text readable
regexp=r'^[AEIOUaeiou]+|[AEIOUaeiou]+$|[^AEIOUaeiou]'
def compress(word):
    pieces=re.findall(regexp,word)
    return ''.join(pieces)

english_udhr=nltk.corpus.udhr.words('English-Latin1')
print(nltk.tokenwrap(compress(w) for w in english_udhr[:75]))
#conditional frequency distribution with regular expressions
#extract consonant-vowel sequences, since each is a pair we can use cfd
rotokas_words=nltk.corpus.toolbox.words('rotokas.dic')
#have a dict of the words containing the cv sequence
cvs=[(cv,w) for w in rotokas_words for cv in re.findall(r'[ptksvr][aeiou]',w)]
cfd=nltk.ConditionalFreqDist(cvs)
cfd.tabulate()
cv_index=nltk.Index(cvs)
print(cv_index['po'])
print(cv_index['ke'])
#simple stem function
def stem(word):
    for suffix in ['ing', 'ly', 'ed', 'ious', 'ies', 'ive', 'es', 's', 'ment']:
        if word.endswith(suffix):
            return word[:-len(suffix)]
    return word

wordlist_stems=[stem(w) for w in wordlist]
print(wordlist)
#using regex
wordlist_suffixes=[suffix for w in wordlist for suffix in re.findall(r'^.*(ing|ly|ed|ious|ies|ive|es|s|ment)$',w)]
print(nltk.FreqDist(wordlist_suffixes).most_common(20))
raw = """DENNIS: Listen, strange women lying in ponds distributing swords is no basis for a system of government.  Supreme executive power derives from a mandate from the masses, not from some farcical aquatic ceremony."""
raw_tokens=word_tokenize(raw)
raw_stems=[stem(t) for t in raw_tokens]
print(raw_stems)

#searching tokenized text
moby=nltk.Text(gutenberg.words('melville-moby_dick.txt'))
print(moby.findall(r'<a><man>')) #print only a man
print(moby.findall(r'<a>(<.*>)<man>')) #prints words between a and man
chat_words=nltk.Text(nps_chat.words())
print(chat_words.findall(r'<.*><.*><bro>'))
print(chat_words.findall(r'<1.*>{3,}'))
#discover hypernyms in text i.e a and other ys
hobbies_learned=nltk.Text(brown.words(categories=['hobbies','learned']))
print(hobbies_learned.findall(r'<\w*><and><other><\w*s>'))
print(hobbies_learned.findall(r'<\w*><as><\w*>'))

#text normalization
#stemmers - to remove affixes from words, 2 off-the-shelf in nltk 1.PorterStemmer 2.LancasterStemmer
print(raw_tokens)
porter=nltk.PorterStemmer()
lancaster=nltk.LancasterStemmer()
print([porter.stem(w) for w in raw_tokens])
print([lancaster.stem(w) for w in raw_tokens])
#Indexing a Text Using a Stemmer, support search for alternative forms of words
#revise later
class IndexedText(object):
    def __init__(self, stemmer, text):
        self._text = text
        self._stemmer = stemmer
        self._index = nltk.Index((self._stem(word), i)
                                 for (i, word) in enumerate(text))

    def concordance(self, word, width=40):
        key = self._stem(word)
        wc = int(width/4)                # words of context
        for i in self._index[key]:
            lcontext = ' '.join(self._text[i-wc:i])
            rcontext = ' '.join(self._text[i:i+wc])
            ldisplay = '{:>{width}}'.format(lcontext[-width:], width=width)
            rdisplay = '{:{width}}'.format(rcontext[:width], width=width)
            print(ldisplay, rdisplay)

    def _stem(self, word):
        return self._stemmer.stem(word).lower()
grail=nltk.corpus.webtext.words('grail.txt')
text=IndexedText(porter,grail)
text.concordance('lie')
#lemmatizer - making sure resulting word is in the dictionary then remove affixes
wnl=nltk.WordNetLemmatizer()
print([wnl.lemmatize(t) for t in raw_tokens])

#simple approaches to tokenizing text
raw = """'When I'M a Duchess,' she said to herself, (not in a very hopeful tone
though), 'I won't have any pepper in my kitchen AT ALL. Soup does very
well without--Maybe it's always pepper that makes people hot-tempered,'..."""
#easiest approach is to split by ' ', leaves out \n and tabs
print(raw.split(' '))
print(re.split(r'[ \t\n]+',raw)) #notice the space
print(re.split(r'\s+',raw)) #includes any white space character
print(re.split(r'\w+',raw)) #try 'xx'.split('x')
print(re.findall(r'\w+',raw))#why does this happen?
print(re.split(r'\W+',raw)) #complement of \w, all characters other than letters, digits and underscores
print(re.findall(r'\w+|\S\w*',raw)) #first match sequence of word chars, if no match try to match any non-whitespace character(complement of \s) followed by other word characters
print(re.findall(r'\w+([-\']\w+)*',raw)) #permit word internal hyphens and apostrophes, this expression means \w+ followed by zero or more instances of [-']\w+
print(re.findall(r'\w+(?:[-\']\w+)*',raw))
print(re.findall(r"\w+(?:[-']\w+)*|'|[-.(]+|\S\w*", raw)) #[-.(]= causes double hyphen, ellipsis and open parenthesis get tokenized separately

#nltk's regexp tokenizer
#try using nltk.regexp_tokenize(text,pattern)

#segmentation
#sentence segmentation
#compute average number of words per sentence in Brown corpus
print(len(brown.words())/len(brown.sents()))
#Unsupervised Multilingual Sentence Boundary Detection by kiss & Strunk
text=gutenberg.raw('chesterton-thursday.txt')
sents=nltk.sent_tokenize(text)
pprint.pprint(sents[70:90])

#word segmentation when no word boundary exists
text = "doyouseethekittyseethedoggydoyoulikethekittylikethedoggy"
#1's in the segment indicate whether or not a wordbrea appears after the character
seg1 = "0000000000000001000000000010000000000000000100000000000"
seg2 = "0100100100100001001001000010100100010010000100010010000"
def segment(text, segs):
    words = []
    last = 0
    for i in range(len(segs)):
        if segs[i] == '1':
            words.append(text[last:i+1])
            last = i+1
    words.append(text[last:])
    return words

print(segment(text,seg1))
print(segment(text,seg2))
#revisit and read objective function, non deterministic search using simulated annealing
