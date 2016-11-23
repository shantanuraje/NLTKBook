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
