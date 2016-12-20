# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 00:56:36 2016

@author: shant
"""
from nltk.book import *

#1 Try using the Python interpreter as a calculator, and typing expressions like 12 / (4 + 1).
print(12/(4+1))

#2 Given an alphabet of 26 letters, there are 26 to the power 10, or 26 ** 10, ten-letter strings we can form. That works out to 141167095653376. How many hundred-letter strings are possible?
print(26**100)

#3 The Python multiplication operation can be applied to lists. What happens when you type ['Monty', 'Python'] * 20, or 3 * sent1?
print(['Monty', 'Python'] * 20)
print(3 * sent1)

#4 Review 1 on computing with language. How many words are there in text2? How many distinct words are there?
print(len(text2))
sortedSetText2 = sorted(set(text2))
#obtain length of sorted set
print(len(sortedSetText2))

#5 Compare the lexical diversity scores for humor and romance fiction in 1.1. Which genre is more lexically diverse?
#Moby Dick by Herman Melville genre adventure 1851
#Sense and Sensibility by Jane Austen genre romance 1811 text2
#Monty Python and the Holy Grail genre comedy text6
#The Man Who Was Thursday by G . K . Chesterton genre thriller 1908
lexDivText2 = len(sorted(set(text2)))/float(len(text2)) *100
lexDivText6 = len(sorted(set(text6)))/float(len(text6)) *100
print("Sense and Sensibility by Jane Austen has scored ",lexDivText2)
print("Monty Python and the Holy Grail ",lexDivText6)

#6 Produce a dispersion plot of the four main protagonists in Sense and Sensibility: Elinor, Marianne, Edward, and Willoughby. What can you observe about the different roles played by the males and females in this novel? Can you identify the couples?
#observe first occurence on character, corresponding simultaneous occurence in text happens in pairs
#elinor and edward, marianne and willoughby
text2.dispersion_plot(["Elinor", "Marianne", "Edward", "Willoughby"])

#7 Find the collocations in text5.
text5.collocations()

#8 Consider the following Python expression: len(set(text4)). State the purpose of this expression. Describe the two steps involved in performing this computation.
#A. identify unique set of words in text, calculate length of set

#9 Review 2 on lists and strings.
# Define a string and assign it to a variable, e.g., my_string = 'My String' (but put something more interesting in the string). Print the contents of this variable in two ways, first by simply typing the variable name and pressing enter, then by using the print statement.
my_string = "Hello world"
print(my_string)
# Try adding the string to itself using my_string + my_string, or multiplying it by a number, e.g., my_string * 3. Notice that the strings are joined together without any spaces. How could you fix this?
#fix it - add a space?
print(my_string + my_string)
print(my_string * 3)

#10 Define a variable my_sent to be a list of words, using the syntax my_sent = ["My", "sent"] (but with your own words, or a favorite saying).
my_sent = ["Hello","world","!"]
# Use ' '.join(my_sent) to convert this into a string.
print(' '.join(my_sent))
# Use split() to split the string back into the list form you had to start with.
print(' '.join(my_sent).split())

#11 Define several variables containing lists of words, e.g., phrase1, phrase2, and so on. Join them together in various combinations (using the plus operator) to form whole sentences. What is the relationship between len(phrase1 + phrase2) and len(phrase1) + len(phrase2)?
# diff is 1st calculates len of sent1+sent2, 2nd calculates len of sent1 + len of sent2
print(len(sent1 + sent2))
print(len(sent1)+len(sent2))

#12 Consider the following two expressions, which have the same value. Which one will typically be more relevant in NLP? Why?
"Monty Python"[6:12]
["Monty", "Python"][1]
#both are relevant, 2nd one is more as it lets you index the whole word in a list

#13 We have seen how to represent a sentence as a list of words, where each word is a sequence of characters. What does sent1[2][2] do? Why? Experiment with other index values.
# refers to 2nd character of 2nd word

#14 The first sentence of text3 is provided to you in the variable sent3. The index of the in sent3 is 1, because sent3[1] gives us 'the'. What are the indexes of the two other occurrences of this word in sent3?
#1,5,8

#15 Review the discussion of conditionals in 4. Find all words in the Chat Corpus (text5) starting with the letter b. Show them in alphabetical order.
bWordsText5 = [word for word in text5 if word.startswith('b') or word.startswith('B')]
sortedSetBWordsText5 = sorted(set(bWordsText5))
print(sortedSetBWordsText5)

#16 Type the expression list(range(10)) at the interpreter prompt. Now try list(range(10, 20)), list(range(10, 20, 2)), and list(range(20, 10, -2)). We will see a variety of uses for this built-in function in later chapters.
print(list(range(10)))
print(list(range(10, 20)))
print(list(range(10, 20, 2)))
print(list(range(20, 10, -2)))

#17 Use text9.index() to find the index of the word sunset. You'll need to insert this word as an argument between the parentheses. By a process of trial and error, find the slice for the complete sentence that contains this word.
print(text9.index("sunset")) #629
print(text9[610:640])
print(text9[620:640])
print(text9[610:650])
print(text9[621:644])

#18 Using list addition, and the set and sorted operations, compute the vocabulary of the sentences sent1 ... sent8.
# print len of all texts -
allSents = []
for j in ["sent"+str(i) for i in range(1,10)]:
    allSents = allSents + eval(j)
sortedSetAllSent = sorted(set(allSents))
sortedSetAllSent = [word for word in sortedSetAllSent if word.isalpha()] #include on words containing alphabets
print(sortedSetAllSent)

#19 What is the difference between the following two lines? Which one will give a larger value? Will this be the case for other texts?
# sorted set of all words in text1
print(len(sorted(set(w.lower() for w in text1))))
# sorted set of all unique words in text1 - returns higher value
print(len(sorted(w.lower() for w in set(text1))))

#20 What is the difference between the following two tests: w.isupper() and not w.islower()?
# .isupper checks if string is upper case, not .islower checks is string is anything other than lower case

#21 Write the slice expression that extracts the last two words of text2.
print(text2[len(text2)-2:])

#22 Find all the four-letter words in the Chat Corpus (text5). With the help of a frequency distribution (FreqDist), show these words in decreasing order of frequency.
freqDistText5 = FreqDist(text5)
text5FourLetterWords = sorted(set([w for w in text5 if w.isalpha() and len(w) == 4]))
print(text5FourLetterWords)
print(freqDistText5)
for sample in [w for w in freqDistText5]:
    if sample not in text5FourLetterWords:
        freqDistText5.pop(sample)
print(freqDistText5)
freqDistText5.plot(50)

#23 Review the discussion of looping with conditions in 4. Use a combination of for and if statements to loop over the words of the movie script for Monty Python and the Holy Grail (text6) and print all the uppercase words, one per line.
for word in text6:
    if word.isupper():
        print(word)

#24 Write expressions for finding all words in text6 that meet the conditions listed below. The result should be in the form of a list of words: ['word1', 'word2', ...].
# Ending in ize
print([w for w in text6 if w.endswith('ize')])
# Containing the letter z
print([w for w in text6 if 'z' in w])
# Containing the sequence of letters pt
print([w for w in text6 if 'pt' in w])
# Having all lowercase letters except for an initial capital (i.e., titlecase)
print([w for w in text6 if w.istitle()])

#25 Define sent to be the list of words ['she', 'sells', 'sea', 'shells', 'by', 'the', 'sea', 'shore']. Now write code to perform the following tasks:
sent = ['she', 'sells', 'sea', 'shells', 'by', 'the', 'sea', 'shore']
# Print all words beginning with sh
print([w for w in sent if w.startswith('sh')])
# Print all words longer than four characters
print([w for w in sent if len(w)>4])

#26 What does the following Python code do? sum(len(w) for w in text1) Can you use it to work out the average word length of a text?
print(sum(len(w) for w in text1)) # sums length of each word
print(sum(len(w) for w in text1)/len(text1))

#27 Define a function called vocab_size(text) that has a single parameter for the text, and which returns the vocabulary size of the text.
def vocab_size(text):
    return len(text)
vocab_size(text3)    

#28 Define a function percent(word, text) that calculates how often a given word occurs in a text, and expresses the result as a percentage.
def percent(word, text):
    return FreqDist(text)[word]/float(len(text))*100
percent("the",text1)

#29 We have been using sets to store vocabularies. Try the following Python expression: set(sent3) < set(text1). Experiment with this using different arguments to set(). What does it do? Can you think of a practical application for this?
#compare length of sets?