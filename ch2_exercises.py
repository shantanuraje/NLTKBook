import nltk
import re
from nltk.corpus import gutenberg
from nltk.corpus import brown
from nltk.corpus import state_union
from nltk.corpus import wordnet as wn
from nltk.corpus import swadesh
from nltk.book import text1 as mobydick #mobydick
from nltk.book import text2 as sense_and_sensibility #sense_and_sensibility
from nltk.corpus import names # male-female names
from nltk.corpus import cmudict #pronouncing dictionary
from nltk.corpus import stopwords # list of stopwords
from matplotlib import pyplot
import random
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
translate = dict()
de2en = swadesh.entries(['de','en'])
es2en = swadesh.entries(['es','en'])
translate.update(dict(de2en))
translate.update(dict(es2en))
print(translate)
#one word could have multiple corresponding words or vice versa?
#keep only one in dictionary???

#7 According to Strunk and White's Elements of Style, the word however, used at the start of a sentence, means "in whatever way" or "to whatever extent", and not "nevertheless". They give this example of correct usage: However you advise him, he will probably do as he thinks best. (http://www.bartleby.com/141/strunk3.html) Use the concordance tool to study actual usage of this word in the various texts we have been considering. See also the LanguageLog posting "Fossilized prejudices about 'however'" at http://itre.cis.upenn.edu/~myl/languagelog/archives/001913.html
print(mobydick.concordance('however'))
print(sense_and_sensibility.concordance('however'))

#8 Define a conditional frequency distribution over the Names corpus that allows you to see which initial letters are more frequent for males vs. females (cf. 4.4).
#cfd against last letters for all names to check well known fact that names ending in letter a are almost always female
cfd = nltk.ConditionalFreqDist((fileid,name[1]) for fileid in names.fileids() for name in names.words(fileid))
cfd.plot()

#9 Pick a pair of texts and study the differences between them, in terms of vocabulary, vocabulary richness, genre, etc. Can you find pairs of words which have quite different meanings across the two texts, such as monstrous in Moby Dick and in Sense and Sensibility?
#already have news and religion data from brown corpus
#concordance works on Text objects, so need to instantiate a Text with news and religion data
news_data = nltk.Text(news_data)
religion_data = nltk.Text(religion_data)
#trying to find common words
news_fd = nltk.FreqDist(news_data)
religion_fd = nltk.FreqDist(religion_data)
print(news_data.concordance('state'))
print(religion_data.concordance('state'))

#10 Read the BBC News article: UK's Vicky Pollards 'left behind' http://news.bbc.co.uk/1/hi/education/6173441.stm. The article gives the following statistic about teen language: "the top 20 words used, including yeah, no, but and like, account for around a third of all words." How many word types account for a third of all word tokens, for a variety of text sources? What do you conclude about this statistic? Read more about this on LanguageLog, at http://itre.cis.upenn.edu/~myl/languagelog/archives/003993.html.
fdist1 = nltk.FreqDist(nltk.book.text3)
print(fdist1)
fdist1.most_common(50)
fdist1.plot(50)
fdist1.plot(50,cumulative=True)
#true, most words in text are stop words!!

#11 Investigate the table of modal distributions and look for other patterns. Try to explain them in terms of your own impressionistic understanding of the different genres. Can you find other closed classes of words that exhibit significant differences across different genres?
#conditional frequency distributions
cfd = nltk.ConditionalFreqDist((genre,word)for genre in brown.categories() for word in brown.words(categories=genre))
genres = ['news', 'religion', 'hobbies', 'science_fiction', 'romance', 'humor']
# check distribution of 5 w's 1 h
general_words = ["who", "what", "when", "where", "why", "how"]
#conditional frequency distributions with event_words
cfd.tabulate(conditions=genres, samples=general_words)
# most frequent in new is who, when;religion is who, what;hobbies is who, when,etc.

#12 The CMU Pronouncing Dictionary contains multiple pronunciations for certain words. How many distinct words does it contain? What fraction of words in this dictionary have more than one possible pronunciation?
words = [word for word,pron in cmudict.entries() ]
wordset=set(words)
cmu=cmudict.dict()
print(len(words))
print(len(wordset))
more_than_one_pron=[word for word in wordset if len(cmu.get(word))>1]
print(len(more_than_one_pron)/len(wordset)*100,"% words have more than one pronounciation")

#13 What percentage of noun synsets have no hyponyms? You can get all noun synsets using wn.all_synsets('n').
no_hyp_nouns=[noun for noun in wn.all_synsets('n') if len(noun.hyponyms())==0]
all_noun_words=[noun for noun in wn.all_synsets('n')]
print("Percentage of noun having no hyponyms: ",len(no_hyp_nouns)/len(all_noun_words)*100)
#weird: had to define all_nouns twice as on 2 operation it was blank, mutability maybe

#14 Define a function supergloss(s) that takes a synset s as its argument and returns a string consisting of the concatenation of the definition of s, and the definitions of all the hypernyms and hyponyms of s.
def supergloss(s):
    definitions=s.definition();
    for hypo in s.hyponyms():
        definitions+="\n"+hypo.definition()
    for hyper in s.hypernyms():
        definitions+="\n"+hyper.definition()
    return definitions

definitions = supergloss(wn.synset('car.n.01'))
print(definitions)

#15 Write a program to find all words that occur at least three times in the Brown Corpus.
all_unique_words_brown=set(brown.words())
brown_fd=nltk.FreqDist(brown.words())
atleast_3times=[word for word in all_unique_words_brown if brown_fd[word]>2]
print(atleast_3times)

#16 Write a program to generate a table of lexical diversity scores (i.e. token/type ratios), as we saw in 1.1. Include the full set of Brown Corpus genres (nltk.corpus.brown.categories()). Which genre has the lowest diversity (greatest number of tokens per type)? Is this what you would have expected?
brown_categories=brown.categories()
print("Category, Tokens, Types, Lexical Diversity")
for category in brown_categories:
    category_words = brown.words(categories=category)
    print(category,len(category_words),len(set(category_words)),len(category_words)/(len(set(category_words))*1.0))
#science fiction has least diversity score

#17 Write a function that finds the 50 most frequently occurring words of a text that are not stopwords.
most_freq_50_fd=nltk.FreqDist(brown.words(categories='news'))
#fd that includes stop words
print(most_freq_50_fd.most_common(50))
words=[word for word in most_freq_50_fd]
for word in words:
    if word in stopwords.words('english') or not word.isalpha():
        most_freq_50_fd.pop(word)
#fd that excludes stop words
print(most_freq_50_fd.most_common(50))

#18 Write a program to print the 50 most frequent bigrams (pairs of adjacent words) of a text, omitting bigrams that contain stopwords.
# brown_word_bigrams = nltk.bigrams(brown.words(categories="romance"))
bigrams_without_stopwords = [(a,b) for a,b in nltk.bigrams(brown.words(categories="romance")) if a not in stopwords.words('english') and b not in stopwords.words('english')]
bigrams_without_stopwords_fd = nltk.FreqDist(bigrams_without_stopwords)
print(bigrams_without_stopwords_fd.most_common(50))

#19 Write a program to create a table of word frequencies by genre, like the one given in 1 for modals. Choose your own words and try to find words whose presence (or absence) is typical of a genre. Discuss your findings.
cfd = nltk.ConditionalFreqDist((genre,word)for genre in brown.categories() for word in brown.words(categories=genre))
# check distribution of "love","hate","death","life","marriage","work","children"
general_words = ["love", "hate", "death", "life", "marriage", "work", "children","magic"]
#conditional frequency distributions with event_words
cfd.tabulate(conditions=brown.categories(), samples=general_words)
#conclusion: belles_letters contains alot of refrences to life and work
#learned category has a lot of refrences to work
#lore contains a lot of refrences to life
# religion contains most refrences to life death work and magic

#20 Write a function word_freq() that takes a word and the name of a section of the Brown Corpus as arguments, and computes the frequency of the word in that section of the corpus.
def word_freq(word,category):
    category_text=brown.words(categories=category)
    return sum(1 for wd in category_text if wd==word)

print(word_freq("work","learned"))

#21 Write a program to guess the number of syllables contained in a text, making use of the CMU Pronouncing Dictionary.
# a unit of pronunciation having one vowel sound, with or without surrounding consonants, forming the whole or a part of a word; e.g., there are two syllables in water and three in inferno.
#easiest guess - number of vowels = number of syllables
#previous example
#syllable = ['N','IH0','K','S'] this syllable contains one vowel
cmu_dict = cmudict.entries()
print(len(cmu_dict))
# print(cmu_dict['water']) # contains 2 vowels so two syllables
#get a text
print(len(brown.words(categories='hobbies')))
print(len(set(brown.words(categories='hobbies'))))
brown_hobbies=sorted(set(brown.words(categories='hobbies')))
brown_hobbies_dict_words = [(word,pron) for word,pron in cmu_dict if word in brown_hobbies]
print(len(brown_hobbies_dict_words))
def count_syllables(pron):
    return len([w for w in pron if re.findall("[aeiou]",w.lower())])

no_of_syll_per_word = [count_syllables(pron) for word,pron in brown_hobbies_dict_words]
print("Number of syllables contained in brown corpus hobbies category: ",sum(no_of_syll_per_word))

#22 Define a function hedge(text) which processes a text and produces a new version with the word 'like' between every third word.
def hedge(text):
    # test = 'this is a test sentence to insert like after every third word'.split()
    ids = [index-1 for index in list(range(3,len(text)+1,3))]
    for id in ids:
        text.insert(id,'like')
    return text

test_text=hedge('this is a test sentence to insert like after every third word'.split())
print(test_text)

#23 Zipf's Law: Let f(w) be the frequency of a word w in free text. Suppose that all the words of a text are ranked according to their frequency, with the most frequent word first. Zipf's law states that the frequency of a word type is inversely proportional to its rank (i.e. f × r = k, for some constant k). For example, the 50th most common word type should occur three times as frequently as the 150th most common word type.
# Write a function to process a large text and plot word frequency against word rank using pylab.plot. Do you confirm Zipf's law? (Hint: it helps to use a logarithmic scale). What is going on at the extreme ends of the plotted line?
def zipfs_law(text,n):
    text_fd=nltk.FreqDist(text)
    text_fd_common=text_fd.most_common(n)
    freqs=[y for x,y in text_fd_common]
    ranks=[1/freq for freq in freqs]
    pyplot.plot(ranks,freqs)

zipfs_law(nltk.corpus.gutenberg.words('austen-sense.txt'),50)
# zipfs_law(nltk.corpus.gutenberg.words('austen-sense.txt'),100)
# zipfs_law(nltk.corpus.gutenberg.words('austen-sense.txt'),500)
#looks inversely proportional but is the solution correct??

# Generate random text, e.g., using random.choice("abcdefg "), taking care to include the space character. You will need to import random first. Use the string concatenation operator to accumulate characters into a (very) long string. Then tokenize this string, and generate the Zipf plot as before, and compare the two plots. What do you make of Zipf's Law in the light of this?
random_text=''
for i in range(0,random.randrange(10000,1000000)):
    random_text+=random.choice("abcdefg ")
# print(random_text)
zipfs_law(random_text.split(' '),100)
#yes it is almost inversely proportion

#24 Modify the text generation program in 2.2 further, to do the following tasks:
def generate_model(text, n):
    text_fd=nltk.FreqDist(text)
    text_fd_common=text_fd.most_common(n)
    rand_words = [word for word,index in text_fd_common]
    return rand_words

# Store the n most likely words in a list words then randomly choose a word from the list using random.choice(). (You will need to import random first.)

text = nltk.corpus.genesis.words('english-kjv.txt')
genesis_rand_words = generate_model(text, 100)
print(genesis_rand_words)
print(random.choice(genesis_rand_words))

# Select a particular genre, such as a section of the Brown Corpus, or a genesis translation, one of the Gutenberg texts, or one of the Web texts. Train the model on this corpus and get it to generate random text. You may have to experiment with different start words. How intelligible is the text? Discuss the strengths and weaknesses of this method of generating random text.

brown_romance=brown.words(categories='romance')
brown_romance_rand = generate_model(brown_romance,200)
print(brown_romance_rand,len(brown_romance_rand))
#we could use title case words to be start words.
#sentences have to end in anyone of the punctuation marks
def generate_sentence(text):
    start_words = set(word for word in brown_romance_rand if word.istitle())
    punc_symbols = set(word for word in brown_romance_rand if not word.isalpha() and len(word) == 1)
    other_words = set(brown_romance_rand).difference(punc_symbols)
    other_words = list(other_words)
    start_words = list(start_words)
    punc_symbols = list(punc_symbols)
    limit_1 = random.randrange(1, len(other_words))
    limit_2 = random.randrange(1, len(other_words))

    if limit_1 < limit_2:
        rand_indices = list(range(limit_1, limit_2))
        other_indexes = [random.choice(rand_indices) for id in rand_indices]
        wordlist = [other_words[id] for id in other_indexes]
    else:
        rand_indices = list(range(limit_2, limit_1))
        other_indexes = [random.choice(rand_indices) for id in rand_indices]
        wordlist = [other_words[id] for id in other_indexes]

    wordlist.insert(0,random.choice(start_words))
    wordlist.append(random.choice(punc_symbols))
    print(" ".join(wordlist))

generate_sentence(brown_romance)
#this method generated non sensical text with one title case word at start, one punctuation mark at the end and a random selection of words in between

# Now train your system using two distinct genres and experiment with generating text in the hybrid genre. Discuss your observations.
generate_sentence(brown_romance+brown.words(categories='religion'))

#25 Define a function find_language() that takes a string as its argument, and returns a list of languages that have that string as a word. Use the udhr corpus and limit your searches to files in the Latin-1 encoding.
latin1_files = [f for f in nltk.corpus.udhr.fileids() if re.search(r'Latin1',f)]

def find_language(word,latin1_files):
    return sum([1 for file in latin1_files for w in nltk.corpus.udhr.words(file) if word in w])

word_count_latin1=find_language("human",latin1_files)
print(word_count_latin1)
word_count_latin1=find_language("rights",latin1_files)
print(word_count_latin1)

#26 What is the branching factor of the noun hypernym hierarchy? I.e. for every noun synset that has hyponyms — or children in the hypernym hierarchy — how many do they have on average? You can get all noun synsets using wn.all_synsets('n').
all_synsets=wn.all_synsets('n')
hyper_counts=[len(syn.hypernyms()) for syn in all_synsets]
average_num_hyper=sum(hyper_counts)/len(hyper_counts)
print("branching factor of the noun hypernym hierarchy: ",average_num_hyper)

#27 The polysemy of a word is the number of senses it has. Using WordNet, we can determine that the noun dog has 7 senses with: len(wn.synsets('dog', 'n')). Compute the average polysemy of nouns, verbs, adjectives and adverbs according to WordNet.
#*.[nvas].*
all_synsets = wn.all_synsets()
synsets_per_word = [synst for synst in all_synsets]

#28 Use one of the predefined similarity measures to score the similarity of each of the following pairs of words. Rank the pairs in order of decreasing similarity. How close is your ranking to the order given here, an order that was established experimentally by (Miller & Charles, 1998): car-automobile, gem-jewel, journey-voyage, boy-lad, coast-shore, asylum-madhouse, magician-wizard, midday-noon, furnace-stove, food-fruit, bird-cock, bird-crane, tool-implement, brother-monk, lad-brother, crane-implement, journey-car, monk-oracle, cemetery-woodland, food-rooster, coast-hill, forest-graveyard, shore-woodland, monk-slave, coast-forest, lad-wizard, chord-smile, glass-magician, rooster-voyage, noon-string.
