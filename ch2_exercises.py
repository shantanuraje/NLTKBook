import nltk
from nltk.corpus import gutenberg
from nltk.corpus import brown
from nltk.corpus import state_union
from nltk.corpus import wordnet as wn
#1 Create a variable phrase containing a list of words. Review the operations described in the previous chapter, including addition, multiplication, indexing, slicing, and sorting.
tempPhrase = ["Create", "a", "variable", "phrase", "containing", "a", "list", "of", "words"]
print(tempPhrase+tempPhrase)
print(tempPhrase*3)
print(tempPhrase[5])
print(tempPhrase[-4:])
print(sorted(w.lower() for w in set(tempPhrase))) #only sort puts capital letters first

#2 Use the corpus module to explore austen-persuasion.txt. How many word tokens does this book have? How many word types?
austen_persuasion = gutenberg.words('austen-persuasion.txt')
print("Number of word tokens = ",len(austen_persuasion))
print("Number of word types = ",len(set(austen_persuasion)))

#3 Use the Brown corpus reader nltk.corpus.brown.words() or the Web text corpus reader nltk.corpus.webtext.words() to access some sample text in two different genres.
print(brown.categories())
news_data=brown.words(categories='news')
religion_data=brown.words(categories='religion')

#4 Read in the texts of the State of the Union addresses, using the state_union corpus reader. Count occurrences of men, women, and people in each document. What has happened to the usage of these words over time?
print(state_union.fileids())
#cfd for inaugral address speeches for each president showing count of words american and citizen each speech
cfd = nltk.ConditionalFreqDist((target,fileid[:4])for fileid in state_union.fileids() for w in state_union.words(fileid) for target in  ['men','women'] if w.lower().startswith(target))
#cfd.plot()

#5 Investigate the holonym-meronym relations for some nouns. Remember that there are three kinds of holonym-meronym relation, so you need to use: member_meronyms(), part_meronyms(), substance_meronyms(), member_holonyms(), part_holonyms(), and substance_holonyms().
house = wn.synsets('house')
print(house)
house = wn.synset('house.n.01')
print(house.lemma_names())
print(house.definition())
print(house.examples())
print(house.member_meronyms())
print(house.part_meronyms())
print(house.substance_meronyms())
print(house.member_holonyms())
print(house.part_holonyms())
print(house.substance_holonyms())

food = wn.synsets('food')
print(food)
food = wn.synset('food.n.01')
print(food.lemma_names())
print(food.definition())
print(food.examples())
print(food.member_meronyms())
print(food.part_meronyms())
print(food.substance_meronyms())
print(food.member_holonyms())
print(food.part_holonyms())
print(food.substance_holonyms())

#6 In the discussion of comparative wordlists, we created an object called translate which you could look up using words in both German and Spanish in order to get corresponding words in English. What problem might arise with this approach? Can you suggest a way to avoid this problem?

#7 According to Strunk and White's Elements of Style, the word however, used at the start of a sentence, means "in whatever way" or "to whatever extent", and not "nevertheless". They give this example of correct usage: However you advise him, he will probably do as he thinks best. (http://www.bartleby.com/141/strunk3.html) Use the concordance tool to study actual usage of this word in the various texts we have been considering. See also the LanguageLog posting "Fossilized prejudices about 'however'" at http://itre.cis.upenn.edu/~myl/languagelog/archives/001913.html

#8 Define a conditional frequency distribution over the Names corpus that allows you to see which initial letters are more frequent for males vs. females (cf. 4.4).

#9 Pick a pair of texts and study the differences between them, in terms of vocabulary, vocabulary richness, genre, etc. Can you find pairs of words which have quite different meanings across the two texts, such as monstrous in Moby Dick and in Sense and Sensibility?

#10 Read the BBC News article: UK's Vicky Pollards 'left behind' http://news.bbc.co.uk/1/hi/education/6173441.stm. The article gives the following statistic about teen language: "the top 20 words used, including yeah, no, but and like, account for around a third of all words." How many word types account for a third of all word tokens, for a variety of text sources? What do you conclude about this statistic? Read more about this on LanguageLog, at http://itre.cis.upenn.edu/~myl/languagelog/archives/003993.html.

#11 Investigate the table of modal distributions and look for other patterns. Try to explain them in terms of your own impressionistic understanding of the different genres. Can you find other closed classes of words that exhibit significant differences across different genres?

#12 The CMU Pronouncing Dictionary contains multiple pronunciations for certain words. How many distinct words does it contain? What fraction of words in this dictionary have more than one possible pronunciation?

#13 What percentage of noun synsets have no hyponyms? You can get all noun synsets using wn.all_synsets('n').

#14 Define a function supergloss(s) that takes a synset s as its argument and returns a string consisting of the concatenation of the definition of s, and the definitions of all the hypernyms and hyponyms of s.

#15 Write a program to find all words that occur at least three times in the Brown Corpus.

#16 Write a program to generate a table of lexical diversity scores (i.e. token/type ratios), as we saw in 1.1. Include the full set of Brown Corpus genres (nltk.corpus.brown.categories()). Which genre has the lowest diversity (greatest number of tokens per type)? Is this what you would have expected?

#17 Write a function that finds the 50 most frequently occurring words of a text that are not stopwords.

#18 Write a program to print the 50 most frequent bigrams (pairs of adjacent words) of a text, omitting bigrams that contain stopwords.

#19 Write a program to create a table of word frequencies by genre, like the one given in 1 for modals. Choose your own words and try to find words whose presence (or absence) is typical of a genre. Discuss your findings.

#20 Write a function word_freq() that takes a word and the name of a section of the Brown Corpus as arguments, and computes the frequency of the word in that section of the corpus.

#21 Write a program to guess the number of syllables contained in a text, making use of the CMU Pronouncing Dictionary.

#22 Define a function hedge(text) which processes a text and produces a new version with the word 'like' between every third word.

#23 Zipf's Law: Let f(w) be the frequency of a word w in free text. Suppose that all the words of a text are ranked according to their frequency, with the most frequent word first. Zipf's law states that the frequency of a word type is inversely proportional to its rank (i.e. f × r = k, for some constant k). For example, the 50th most common word type should occur three times as frequently as the 150th most common word type.Write a function to process a large text and plot word frequency against word rank using pylab.plot. Do you confirm Zipf's law? (Hint: it helps to use a logarithmic scale). What is going on at the extreme ends of the plotted line? Generate random text, e.g., using random.choice("abcdefg "), taking care to include the space character. You will need to import random first. Use the string concatenation operator to accumulate characters into a (very) long string. Then tokenize this string, and generate the Zipf plot as before, and compare the two plots. What do you make of Zipf's Law in the light of this?

#24 Modify the text generation program in 2.2 further, to do the following tasks:Store the n most likely words in a list words then randomly choose a word from the list using random.choice(). (You will need to import random first.)Select a particular genre, such as a section of the Brown Corpus, or a genesis translation, one of the Gutenberg texts, or one of the Web texts. Train the model on this corpus and get it to generate random text. You may have to experiment with different start words. How intelligible is the text? Discuss the strengths and weaknesses of this method of generating random text.Now train your system using two distinct genres and experiment with generating text in the hybrid genre. Discuss your observations.

#25 Define a function find_language() that takes a string as its argument, and returns a list of languages that have that string as a word. Use the udhr corpus and limit your searches to files in the Latin-1 encoding.

#26 What is the branching factor of the noun hypernym hierarchy? I.e. for every noun synset that has hyponyms — or children in the hypernym hierarchy — how many do they have on average? You can get all noun synsets using wn.all_synsets('n').

#27 The polysemy of a word is the number of senses it has. Using WordNet, we can determine that the noun dog has 7 senses with: len(wn.synsets('dog', 'n')). Compute the average polysemy of nouns, verbs, adjectives and adverbs according to WordNet.

#28 Use one of the predefined similarity measures to score the similarity of each of the following pairs of words. Rank the pairs in order of decreasing similarity. How close is your ranking to the order given here, an order that was established experimentally by (Miller & Charles, 1998): car-automobile, gem-jewel, journey-voyage, boy-lad, coast-shore, asylum-madhouse, magician-wizard, midday-noon, furnace-stove, food-fruit, bird-cock, bird-crane, tool-implement, brother-monk, lad-brother, crane-implement, journey-car, monk-oracle, cemetery-woodland, food-rooster, coast-hill, forest-graveyard, shore-woodland, monk-slave, coast-forest, lad-wizard, chord-smile, glass-magician, rooster-voyage, noon-string.
