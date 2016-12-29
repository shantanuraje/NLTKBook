import re
import nltk
import pprint
from timeit import Timer
from nltk import word_tokenize
from nltk.corpus import wordnet as wn
from numpy import arange
from matplotlib import pyplot
import networkx as nx

#convert FreqDist to a sequence
raw='Red lorry, yellow lorry, red lorry, yellow lorry.'
text=word_tokenize(raw)
fdist=nltk.FreqDist(text)
print(sorted(fdist))
for key in fdist:
    print(key+':',fdist[key],end=';')

#rearrange contents of a list
words = ['I', 'turned', 'off', 'the', 'spectroroute']
words[2], words[3], words[4] = words[3], words[4], words[2]
print(words)
tmp = words[2]
words[2] = words[3]
words[3] = words[4]
words[4] = tmp
print(words)

#zip two tuples
words = ['I', 'turned', 'off', 'the', 'spectroroute']
tags = ['noun', 'verb', 'prep', 'det', 'noun']
# @@::''@@""££
print(zip(words, tags))
print(list(zip(words, tags)))
print(list(enumerate(words)))

# cut dataset into training and test 90-10
text = nltk.corpus.nps_chat.words()
cut = int(0.9 * len(text))
training_data, test_data = text[:cut], text[cut:]
print(text == training_data + test_data)
print(len(training_data) / len(test_data))

#combine list, strings, tuples to sort string by len of each word
words = 'I turned off the spectroroute'.split()
wordlens = [(len(word), word) for word in words]
print(wordlens)
wordlens.sort()
print(wordlens)
print(' '.join(w for (_, w) in wordlens)) #underscore is just for convention indicating this value will not be used

#using enumerate to go over key-value pairs
fd = nltk.FreqDist(nltk.corpus.brown.words())
cumulative = 0.0
most_common_words = [word for (word, count) in fd.most_common()]
for rank, word in enumerate(most_common_words):
    cumulative += fd.freq(word)
    print("%3d %6.2f%% %s" % (rank + 1, cumulative * 100, word))
    if cumulative > 0.25:
        break

#easy way to find max len word in string
text=nltk.corpus.gutenberg.words('milton-paradise.txt')
maxlen=max(len(word) for word in text)
print([word for word in text if len(word)==maxlen])

#extract successive overlapping n-grams of a list
sent = ['The', 'dog', 'gave', 'John', 'the', 'newspaper']
n = 3
print([sent[i:i+n] for i in range(len(sent)-n+1)])
#nltk offers bigrams, trigrams, ngrams

#building multidimensional structures using list comprehensions
m, n = 3, 7
array = [[set() for i in range(n)] for j in range(m)]
array[2][5].add('Alice')
pprint.pprint(array)

#checking parameter types - can use type(), better use assert isinstance()
#what is basestring? missing information
# def tag(word):
#     assert isinstance(word, basestring), "argument to tag() must be a string"
#     if word in ['a','all','the']:
#         return 'det'
#     else:
#         return 'noun'

#using functions as arguments
sent = ['Take', 'care', 'of', 'the', 'sense', ',', 'and', 'the',
         'sounds', 'will', 'take', 'care', 'of', 'themselves', '.']
def extract_property(prop):
     return [prop(word) for word in sent]

print(extract_property(len))

def last_letter(word):
     return word[-1]

extract_property(last_letter)

#lambda expressions - inline functions, something we dont need a function for
extract_property(lambda w: w[-1])
#sort by decreasing length
print(sorted(sent))
#cmp is not a function anymore, check later for alternatives
# print(sorted(sent, cmp))
# print(sorted(sent, lambda x,y: cmp(len(x), len(y)) ))

#accumulative functions
#standard way - initialize empty list, compute results, return list
def search1(substring, words):
    result = []
    for word in words:
        if substring in word:
            result.append(word)
    return result
#generator function
def search2(substring, words):
    for word in words:
        if substring in word:
            yield word

for item in search1('zz', nltk.corpus.brown.words()):
    print(item, end="\n")
#preferred method as function only generates data as is required, does not need to allocate additional memory to store the ouput
for item in search2('azz', nltk.corpus.brown.words()):
    print(item, end="\n")
# print()
#another example of using generator functions, note also uses recursion
def permutations(seq):
    if len(seq) <= 1:
        yield seq
    else:
        for perm in permutations(seq[1:]):
            for i in range(len(perm) + 1):
                yield perm[:i] + seq[0:1] + perm[i:]

print(list(permutations(['police', 'fish', 'buffalo'])))

#higher order functions (like map, reduce, etc.)
def is_content_word(word):
     return word.lower() not in ['a', 'of', 'the', 'and', 'will', ',', '.']
#We use this function as the first parameter of filter(), which applies the function to each item in the sequence contained in its second parameter, and only retains the items for which the function returns True.
sent = ['Take', 'care', 'of', 'the', 'sense', ',', 'and', 'the','sounds', 'will', 'take', 'care', 'of', 'themselves', '.']
print(list(filter(is_content_word, sent)))
print([w for w in sent if is_content_word(w)])
#map reduce functions
# with map
lengths = list(map(len, nltk.corpus.brown.sents(categories='news')))
print(sum(lengths) / len(lengths))
#without map
lengths = [len(sent) for sent in nltk.corpus.brown.sents(categories='news')]
print(sum(lengths) / len(lengths))
#we can also provide lambda expressions instead of functions(custom/built-in)

#count number of vowels in a word
# with map
# print(list(map(lambda w: len(filter(lambda c: c.lower() in "aeiou", w)), sent)))
# without map
# print([len(c for c in w if c.lower() in "aeiou") for w in sent])
#not working

#named (keyword) arguments
#mixing unnamed and named arguments : unnamed should precede named
def generic(*args, **kwargs):
    print(args)
    print(kwargs)
#*args - in-place list of arguments; **kwargs - in-place list of keyword arguments
generic(1,"African swallow", monty="python")

# nothing special about *args, can use *song to refer to song[0],[1] and [2]
song = [['four', 'calling', 'birds'],
         ['three', 'French', 'hens'],
         ['two', 'turtle', 'doves']]
print(list(zip(song[0], song[1], song[2])))
print(list(zip(*song)))

#named arguments permit optionality, if we are happy with default value, we dont have to specify that value
def freq_words(file, min=1, num=10):
     text = open(file).read()
     tokens = word_tokenize(text)
     freqdist = nltk.FreqDist(t for t in tokens if len(t) >= min)
     return freqdist.most_common(num)
fw = freq_words('tale_of_peter_rabbit.txt', 4, 10)
print(fw)
fw = freq_words('tale_of_peter_rabbit.txt', min=4, num=10)
print(fw)
fw = freq_words('tale_of_peter_rabbit.txt', num=10, min=4)
print(fw)
#optional aguments can also be used to permit flags, for example:
def freq_words(file, min=1, num=10, verbose=False):
     freqdist = nltk.FreqDist()
     if verbose: print("Opening", file)
     text = open(file).read()
     if verbose: print("Read in %d characters" % len(file))
     for word in word_tokenize(text):
         if len(word) >= min:
             freqdist[word] += 1
             if verbose and freqdist.N() % 100 == 0: print(".", sep="")
     if verbose: print
     return freqdist.most_common(num)

fw = freq_words('tale_of_peter_rabbit.txt', num=10, min=4, verbose=True)
print(fw)
#when opening file, good idea to close them
#if with open('file.txt') as f is used, it will automatically close files

#count size of hypernym hierarchy rooted at a given synset
# do this by finding size of each hyponym of s, then add these together, add 1 for synset itself
#using recursion
def size1(s):
    return 1+sum(size1(child) for child in s.hyponyms())

print(size1(wn.synset('car.n.01')))
print(size1(wn.synset('house.n.01')))
print(size1(wn.synset('tree.n.01')))

#letter trie - data structure used to index a lexicon, one letter at a time
#Building a Letter Trie: A recursive function that builds a nested dictionary structure; each level of nesting contains all words with a given prefix, and a sub-trie containing all possible continuations.
def insert(trie, key, value):
    if key:
        first, rest = key[0], key[1:]
        if first not in trie:
            trie[first] = {}
        insert(trie[first], rest, value)
    else:
        trie['value'] = value

trie = {}
insert(trie, 'chat', 'cat')
insert(trie, 'chien', 'dog')
insert(trie, 'chair', 'flesh')
insert(trie, 'chic', 'stylish')
trie = dict(trie)               # for nicer printing
trie['c']['h']['a']['t']['value']
pprint.pprint(trie, width=40)

#simple text retrieval system for movie reviews corpus. indexing the document collection provides a faster lookup
def raw(file):
    contents = open(file).read()
    contents = re.sub(r'<.*?>', ' ', contents)
    contents = re.sub('\s+', ' ', contents)
    return contents

def snippet(doc, term):
    text = ' '*30 + raw(doc) + ' '*30
    pos = text.index(term)
    return text[pos-30:pos+30]

print("Building Index")
files = nltk.corpus.movie_reviews.abspaths()
idx = nltk.Index((w, f) for f in files for w in raw(f).split())

query = ''
while query != "quit":
    query = input("query> ")     # use raw_input() in Python 2
    if query in idx:
        for doc in idx[query]:
            print(snippet(doc, query))
    else:
        print("Not found")

#sets work faster than lists since they are indexed
vocab_size = 100000
setup_list = "import random; vocab = range(%d)" % vocab_size
setup_set = "import random; vocab = set(range(%d))" % vocab_size
statement = "random.randint(0, %d) in vocab" % (vocab_size * 2)
print(Timer(statement, setup_list).timeit(1000))
print(Timer(statement, setup_set).timeit(1000))
#getting same times for set and list

#chandas shastra - number of ways of combining short and long syllables to create a meter of length n
#REVISIT LATER

#matplotlib
#frequency of particular modal verbs in brown corpus
colors = 'rgbcmyk' # red, green, blue, cyan, magenta, yellow, black
def bar_chart(categories, words, counts):
    "Plot a bar chart showing counts for each word by category"
    ind = arange(len(words))
    width = 1 / (len(categories) + 1)
    bar_groups = []
    for c in range(len(categories)):
        bars = pyplot.bar(ind+c*width, counts[categories[c]], width,
                         color=colors[c % len(colors)])
        bar_groups.append(bars)
    pyplot.xticks(ind+width, words)
    pyplot.legend([b[0] for b in bar_groups], categories, loc='upper left')
    pyplot.ylabel('Frequency')
    pyplot.title('Frequency of Six Modal Verbs by Genre')
    pyplot.show()


genres = ['news', 'religion', 'hobbies', 'government', 'adventure']
modals = ['can', 'could', 'may', 'might', 'must', 'will']
cfdist = nltk.ConditionalFreqDist(
              (genre, word)
              for genre in genres
              for word in nltk.corpus.brown.words(categories=genre)
              if word in modals)

counts = {}
for genre in genres:
     counts[genre] = [cfdist[genre][word] for word in modals]
bar_chart(genres, modals, counts)

#networkx - package for defining and manipulating structures consisting of nodes and edges, known as graphs
#eg. visualize words in wordnet
# def traverse(graph, start, node):
#     graph.depth[node.name] = node.shortest_path_distance(start)
#     for child in node.hyponyms():
#         graph.add_edge(node.name, child.name)
#         traverse(graph, start, child)
#
# def hyponym_graph(start):
#     G = nx.Graph()
#     G.depth = {}
#     traverse(G, start, start)
#     return G
#
# def graph_draw(graph):
#     nx.draw_graphviz(graph,
#          node_size = [16 * graph.degree(n) for n in graph],
#          node_color = [graph.depth[n] for n in graph],
#          with_labels = False)
#     pyplot.show()
#
# dog = wn.synset('dog.n.01')
# graph = hyponym_graph(dog)
# graph_draw(graph)

#example not working - module networkx.drawing has no attribute graphviz_layout
