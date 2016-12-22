import nltk
from nltk.corpus import gutenberg # project gutenberg text
from nltk.corpus import webtext # webtext corpora - discussion forum, pirated of carribean, etc
from nltk.corpus import nps_chat #naval postgraduate school for research
from nltk.corpus import brown #categorized by genre, 500 sources
from nltk.corpus import reuters #training vs test to automatically detect topic, each story covers multiple topics
from nltk.corpus import inaugural # inaugral text corpus (text4 in nltk.book) seperated  into individual speeches
from nltk.corpus import udhr #universal declaration of human rights in > 300 languages
from nltk.corpus import PlaintextCorpusReader #load you own corpus
from nltk.corpus import stopwords # list of stopwords
from nltk.corpus import words # english vocabulary
from nltk.corpus import names # male-female names
from nltk.corpus import cmudict #pronouncing dictionary

print("text_name  |  num_of_chars  |  num_of_words  |  num_of_sentences  |  num_of_vocab_items  |  avg_word_len  |  avg_sent_len  |  lexical_diversity")

for fileid in gutenberg.fileids():
    num_chars = len(gutenberg.raw(fileid))
    num_words = len(gutenberg.words(fileid))
    num_sents = len(gutenberg.sents(fileid))
    num_vocab = len(set(w.lower() for w in gutenberg.words(fileid)))
    avg_word_len = round(num_chars/num_words)
    avg_sent_len = round(num_words/num_sents)
    lexical_diversity = round(num_words/num_vocab)
    print(fileid,"  |  ",num_chars,"  |  ",num_words,"  |  ",num_sents,"  |  ",num_vocab,"  |  ",avg_word_len,"  |  ",avg_sent_len,"  |  ",lexical_diversity)

for fileid in webtext.fileids():
    print(fileid)

brown.categories()
brown.raw("cr09")

#stylistics - systematic differences between genres
# by use of modal verbs - [can could may might must will]
news_text = brown.words(categories='news')
hobbies_text = brown.words(categories='hobbies')
news_text_fdist = nltk.FreqDist(w.lower() for w in news_text)
hobbies_text_fdist = nltk.FreqDist(w.lower() for w in hobbies_text)
modals = ['can','could','may','might','must','will']
for m in modals:
    print(m,":",news_text_fdist[m],"  |  ",hobbies_text_fdist[m])

event_words = ["who","what","when","where","why"]
for m in event_words:
    print(m,":",news_text_fdist[m],"  |  ",hobbies_text_fdist[m])

#conditional frequency distributions
cfd = nltk.ConditionalFreqDist((genre,word)for genre in brown.categories() for word in brown.words(categories=genre))
genres = ['news', 'religion', 'hobbies', 'science_fiction', 'romance', 'humor']
cfd.tabulate(conditions=genres, samples=modals) #conditional frequency distributions with modals
cfd.tabulate(conditions=genres, samples=event_words) #conditional frequency distributions with event_words

#reuters corpus - we can ask for topics covered by multiple documents, titles (first handful words in each document)stored as upper case
reuters.categories()
print(reuters.categories('training/9865'))
print(reuters.categories(['training/9865','training/9880']))

print(reuters.words('training/9865')[:30])

#inaugural text corpus
print(inaugural.fileids())
[fileid[:4] for fileid in inaugural.fileids()]

#cfd for inaugral address speeches for each president showing count of words american and citizen each speech
cfd = nltk.ConditionalFreqDist((target,fileid[:4])for fileid in inaugural.fileids() for w in inaugural.words(fileid) for target in  ['american','citizen'] if w.lower().startswith(target))
cfd.plot()

#corpora in other languages
print(nltk.corpus.cess_esp.words())
print(nltk.corpus.floresta.words())
print(nltk.corpus.indian.words('hindi.pos'))
print(nltk.corpus.udhr.fileids()) #universal declaration of human rights in > 300 languages
print(nltk.corpus.udhr.words('Javanese-Latin1'))

#cfd for udhr
languages = ['Chickasaw', 'English', 'German_Deutsch','Greenlandic_Inuktikut', 'Hungarian_Magyar', 'Ibibio_Efik']
cfd = nltk.ConditionalFreqDist((lang,len(word)) for lang in languages for word in udhr.words(lang + '-Latin1'))
cfd.plot(cumulative = True)

#frequency distributions of letters in a text
raw_text = udhr.raw('Afrikaans-Latin1')
nltk.FreqDist(raw_text).plot()

#loading your own corpus
#for later (need to download a text corpus)

#conditional frequency distributions (theory) 
genre_word = [(genre,word) for genre in ['news','romance'] for word in brown.words(categories=genre)]
print(genre_word[:4])
print(genre_word[-4:])
cfd = nltk.ConditionalFreqDist(genre_word)
print(cfd)
print(cfd.conditions())
print(cfd["news"])
print(cfd["romance"])
print(cfd["news"].most_common(20))

#plotting and tabulation 
#cfd for udhr
languages = ['Chickasaw', 'English', 'German_Deutsch','Greenlandic_Inuktikut', 'Hungarian_Magyar', 'Ibibio_Efik']
cfd = nltk.ConditionalFreqDist((lang,len(word)) for lang in languages for word in udhr.words(lang + '-Latin1'))
cfd.tabulate(conditions = ['English','Chickasaw'], samples = range(15), cumulative=True)
#cfd for brown corpus
days = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
cfd = nltk.ConditionalFreqDist(genre_word)
cfd.tabulate(samples = days)
cfd.plot(samples = days)

#genrating random text with bigrams
sent = ['In', 'the', 'beginning', 'God', 'created', 'the', 'heaven', 'and', 'the', 'earth', '.']
bigramList = list(nltk.bigrams(sent))
print(bigramList)

def genrate_model(cfdist, word, num=15):
    for i in range(num):
        print(word, end=' ')
        word = cfdist[word].max()
        
text=nltk.corpus.genesis.words('english-kjv.txt')
bigrams=nltk.bigrams(text)
cfd=nltk.ConditionalFreqDist(bigrams)
print(cfd['living'])
genrate_model(cfd,'living')

#lexical resources - wordlist with info such as lexical resources, sense definition etc.
#unusual words
def unusual_words(text):
    text_vocab = set(w.lower() for w in text if w.isalpha())
    english_vocab = set(w.lower() for w in nltk.corpus.words.words())
    unusual = text_vocab - english_vocab
    return sorted(unusual)
    
unusual_words(gutenberg.words('austen-sense.txt'))
unusual_words(nps_chat.words())
#stop words such as if the for etc.
print(stopwords.words('english'))
#function to compute what % of words are not is stopwords list
def content_fraction(text):
    stopwords_list = stopwords.words('english')
    content = [w for w in text if w.lower() not in stopwords_list]
    return len(content)/len(text)*100 

content_fraction(reuters.words())
#solving word puzzle
puzzle_letters = nltk.FreqDist('egivrvonl')
obligatory = 'r'
wordlist = words.words()
result = [w for w in wordlist if len(w)>=6 and obligatory in w and nltk.FreqDist(w)<=puzzle_letters]
print(result)
#find names common to both genders
print(names.fileids())
male_names = names.words('male.txt')
female_names = names.words('female.txt')
common_names = [w for w in male_names if w in female_names]
print(common_names)
#cfd against last letters for all names to check well known fact that names ending in letter a are almost always female
cfd = nltk.ConditionalFreqDist((fileid,name[-1]) for fileid in names.fileids() for name in names.words(fileid))
cfd.plot()
#pronouncing dictionary for speech synthesizers - corpus cmu pronoucing dictionary
entries = cmudict.entries()
print(len(entries))
#for entry in entries: #can also use word,pronoun format
#    print(entry)
for word,pron in entries:
    if(len(pron)==3):
        ph1,ph2,ph3 = pron;
        if ph1=='P' and ph3=='T':
            print(word,ph2,end='')
#find words whose pronunciation ends in nicks
syllable = ['N','IH0','K','S']
result = [word for word,pron in entries if pron[-4:]==syllable]
print(result)
#words ending with n whose pronunciation ends with m
result = [w for w,pron in entries if pron[-1]=='M' and w[-1]=='n']
print(result)
#function to extract stress from phones [0 - no stress, 1- primary, 2 - secondary] - to find words having a particular stress pattern
def stress(pron):
    return [char for phone in pron for char in phone if char.isdigit()] # returns digit in phone

result = [w for w,pron in entries if stress(pron) == ['0','1','0','2','0']] # check if stress pattern of word is 01020
print(result)
result = [w for w,pron in entries if stress(pron) == ['0','2','0','1','0']] # check if stress pattern of word is 02010
print(result)
#find all p words consisting of three sounds
p3 = [(pron[0]+'-'+pron[2], word) for word,pron in entries if pron[0]=='P' and len(pron)==3]
cfd = nltk.ConditionalFreqDist(p3)
print(cfd.conditions())
for template in sorted(cfd.conditions()):
    #print(cfd[template]," ",len(cfd[template])) 
    if len(cfd[template])>10:
        words = sorted(cfd[template])
        wordstring = ' '.join(words)
        print(template, wordstring+"...")
#using dictionaries
prondict = cmudict.dict()
print(prondict['file'])
print(prondict['clog'])
prondict['blog'] = [['B', 'L', 'AA1', 'G']] #assign pronunciation data to non existent key (temporary)
print(prondict['blog'])
