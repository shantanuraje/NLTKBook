# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 00:49:44 2016

@author: shant
"""
# import text and material from nltk.book package
from nltk.book import *
#list text and sentences available 
texts()
sents()
# returns <class 'nltk.text.Text'>
print type(text1)
# checking concordance, similar, common contexts of some words from different texts
# Moby Dick by Herman Melville 1851
text1.concordance("very")
text1.concordance("monstrous")
#Chat Corpus
text5.concordance("hello")
text5.concordance("crazy")
text5.concordance("lol")
text5.concordance("lmao")
text5.concordance("shit")
text5.concordance("killer")
text5.concordance("kill")
#The Man Who Was Thursday by G . K . Chesterton 1908
text9.concordance("man")
text9.concordance("artist")
text9.concordance("anarchist")
text9.concordance("bomb")
text9.concordance("realist")
text9.concordance("bradshaw")
text9.concordance("I")
text9.similar("I")
text9.similar("artist")
text9.similar("bomb")
text9.similar("brain")
text9.common_contexts(["artist","bomb"])
text9.common_contexts(["artist","anarchist"])
# checking dispersion plots of few selected words
text5.dispersion_plot(["lol","lmao","weird"])
text9.dispersion_plot(["man","artist","anarchist"])
# text5.generate() does not work

# print len of all texts - 
for j in ["text"+str(i) for i in range(1,10)]:
    print j,":",len(eval(j)) # eval to evaluate "textX" string as input
    #includes words and punctuation symbols i.e tokens

#obtain sorted list of vocab items, set returns distinct items (word types) in set
sortedSetText2 = sorted(set(text2))
#obtain length of sorted set
len(sortedSetText2)
len(text2)
#lexical richness of text,
#need to cast float.either float(b) or a/(b*1.0) (preferred) or from __future__ import division (python3)
lexRichText2 = len(sortedSetText2)/float(len(text2))
print "Lexical richness: Number of distinct words is just", lexRichText2*100,"% of the total number of words"
print "equivalently that each word is used", float(len(text2))/len(sortedSetText2) ,"times on average"
# frequency of a word in a text - is case sensitive
lolCount = text5.count("lol")
print "The word lol appears",lolCount,"times in text5 - Chat corpus"
print "It takes up",lolCount*100/(len(text5)*1.0),"% of the text"

#fuction examples
def lexical_diversity(text):
    lexDiv = len(set(text))*100/(len(text)*1.0)
    avgNoOfTimesPerWord = float(len(text2))/len(sortedSetText2)    
    print "Lexical richness: Number of distinct words is just", lexDiv,"% of the total number of words"
    print "equivalently that each word is used", avgNoOfTimesPerWord ,"times on average"
    return
    
def percentage(count, total) :
    return count*100/total
    
#frequency distributions of words -built in function FreqDist
fdist1 = FreqDist(text3)
print fdist1
fdist1.most_common(50)
fdist1.plot(50)
fdist1.plot(50,cumulative=True)

#hapaxes?
fdist2 = FreqDist(text2)
fdist2.hapaxes()

#words > length x in text words
textSet = set(text6)
long_words = [w for w in textSet if len(w)>10] # no words longer then 13, 44 10 letter words

# frequent long words 
frequent_long_words = [w for w in textSet if len(w)>11 and fdist2[w]>5]

#collocations and bigrams
#bigrams() # bigrams function is not available ??
text6.collocations()

#frequency distrubution of length of words
fworddist = FreqDist([len(w) for w in text6])
fworddist.plot()
fworddist.most_common()
#FreqDist are of type probability.FreqDist with word and counts table
fdist1 |= fdist2 # 
# print words in text2 that have substrin cei or cie 
tricky = sorted(w for w in set(text2) if 'cie' in w or 'cei' in w)
for word in tricky:
    print word