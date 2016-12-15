from nltk.corpus import gutenberg # project gutenberg text
from nltk.corpus import webtext # webtext corpora - discussion forum, pirated of carribean, etc

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