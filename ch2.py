import nltk
from nltk.corpus import gutenberg # project gutenberg text
from nltk.corpus import webtext # webtext corpora - discussion forum, pirated of carribean, etc
from nltk.corpus import nps_chat
from nltk.corpus import brown

print "text_name  |  num_of_chars  |  num_of_words  |  num_of_sentences  |  num_of_vocab_items  |  avg_word_len  |  avg_sent_len  |  lexical_diversity"
for fileid in gutenberg.fileids():
    num_chars = len(gutenberg.raw(fileid))
    num_words = len(gutenberg.words(fileid))
    num_sents = len(gutenberg.sents(fileid))
    num_vocab = len(set(w.lower() for w in gutenberg.words(fileid)))
    avg_word_len = round(num_chars/num_words)
    avg_sent_len = round(num_words/num_sents)
    lexical_diversity = round(num_words/num_vocab)
    print fileid,"  |  ",num_chars,"  |  ",num_words,"  |  ",num_sents,"  |  ",num_vocab,"  |  ",avg_word_len,"  |  ",avg_sent_len,"  |  ",lexical_diversity

for fileid in webtext.fileids():
    print fileid

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
    print m,":",news_text_fdist[m],"  |  ",hobbies_text_fdist[m]

event_words = ["who","what","when","where","why"]
for m in event_words:
    print m,":",news_text_fdist[m],"  |  ",hobbies_text_fdist[m]

cfd = nltk.ConditionalFreqDist((genre,word)for genre in brown.categories() for word in brown.words(categories=genre))
genres = ['news', 'religion', 'hobbies', 'science_fiction', 'romance', 'humor']
cfd.tabulate(conditions=genres, samples=modals)