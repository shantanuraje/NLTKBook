import nltk
from nltk.corpus import gutenberg # project gutenberg text
from nltk.corpus import webtext # webtext corpora - discussion forum, pirated of carribean, etc
from nltk.corpus import nps_chat #naval postgraduate school for research
from nltk.corpus import brown #categorized by genre, 500 sources
from nltk.corpus import reuters #training vs test to automatically detect topic, each story covers multiple topics
from nltk.corpus import  inaugural # inaugral text corpus (text4 in nltk.book) seperated  into individual speeches
from nltk.corpus import udhr #universal declaration of human rights in > 300 languages

print ("text_name  |  num_of_chars  |  num_of_words  |  num_of_sentences  |  num_of_vocab_items  |  avg_word_len  |  avg_sent_len  |  lexical_diversity")

for fileid in gutenberg.fileids():
    num_chars = len(gutenberg.raw(fileid))
    num_words = len(gutenberg.words(fileid))
    num_sents = len(gutenberg.sents(fileid))
    num_vocab = len(set(w.lower() for w in gutenberg.words(fileid)))
    avg_word_len = round(num_chars/num_words)
    avg_sent_len = round(num_words/num_sents)
    lexical_diversity = round(num_words/num_vocab)
    print (fileid,"  |  ",num_chars,"  |  ",num_words,"  |  ",num_sents,"  |  ",num_vocab,"  |  ",avg_word_len,"  |  ",avg_sent_len,"  |  ",lexical_diversity)

for fileid in webtext.fileids():
    print (fileid)

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
    print (m,":",news_text_fdist[m],"  |  ",hobbies_text_fdist[m])

event_words = ["who","what","when","where","why"]
for m in event_words:
    print (m,":",news_text_fdist[m],"  |  ",hobbies_text_fdist[m])

#conditional frequency distributions
cfd = nltk.ConditionalFreqDist((genre,word)for genre in brown.categories() for word in brown.words(categories=genre))
genres = ['news', 'religion', 'hobbies', 'science_fiction', 'romance', 'humor']
cfd.tabulate(conditions=genres, samples=modals) #conditional frequency distributions with modals
cfd.tabulate(conditions=genres, samples=event_words) #conditional frequency distributions with event_words

#reuters corpus - we can ask for topics covered by multiple documents, titles (first handful words in each document)stored as upper case
reuters.categories()
print (reuters.categories('training/9865'))
print (reuters.categories(['training/9865','training/9880']))

print (reuters.words('training/9865')[:30])

#inaugural text corpus
print (inaugural.fileids())
[fileid[:4] for fileid in inaugural.fileids()]

#cfd for inaugral address speeches for each president showing count of words american and citizen each speech
cfd = nltk.ConditionalFreqDist((target,fileid[:4])for fileid in inaugural.fileids() for w in inaugural.words(fileid) for target in  ['american','citizen'] if w.lower().startswith(target))
cfd.plot()

#corpora in other languages
print (nltk.corpus.cess_esp.words())
print (nltk.corpus.floresta.words())
print (nltk.corpus.indian.words('hindi.pos'))
print (nltk.corpus.udhr.fileids()) #universal declaration of human rights in > 300 languages
print (nltk.corpus.udhr.words('Javanese-Latin1'))

#cfd for udhr
languages = ['Chickasaw', 'English', 'German_Deutsch','Greenlandic_Inuktikut', 'Hungarian_Magyar', 'Ibibio_Efik']
cfd = nltk.ConditionalFreqDist((lang,len(word)) for lang in languages for word in udhr.words(lang + '-Latin1'))
cfd.plot(cumulative = True)

#frequency distributions of letters in a text
raw_text = udhr.raw('Afrikaans-Latin1')
nltk.FreqDist(raw_text).plot()