import nltk
from nltk import word_tokenize
from nltk.corpus import brown

#parts of speech tagger
text=word_tokenize("And now for something completely different")
print(nltk.pos_tag(text))

#getting help about pos tag
nltk.help.upenn_tagset('RB')
nltk.help.upenn_tagset('JJ')

text=word_tokenize("They refuse to permit us to obtain the refuse permit")
print(nltk.pos_tag(text))

#analysis involving: woman (a noun), bought (a verb), over (a preposition), and the (a determiner)
text = nltk.Text(word.lower() for word in nltk.corpus.brown.words())
print(text.similar('woman'))
print(text.similar('bought'))
print(text.similar('over'))
print(text.similar('the'))
#each word class mostly finds words of the same class

#tagged token represented as tuple consisting of token and tag
#can create this tuple using str2tuple
tagged_token = nltk.tag.str2tuple('fly/NN')
print(tagged_token)

#converted string to tagged text
sent = '''The/AT grand/JJ jury/NN commented/VBD on/IN a/AT number/NN of/IN
         other/AP topics/NNS ,/, AMONG/IN them/PPO the/AT Atlanta/NP and/CC
         Fulton/NP-tl County/NN-tl purchasing/VBG departments/NNS which/WDT it/PPS
         said/VBD ``/`` ARE/BER well/QL operated/VBN and/CC follow/VB generally/RB
         accepted/VBN practices/NNS which/WDT inure/VB to/IN the/AT best/JJT
         interest/NN of/IN both/ABX governments/NNS ''/'' ./.  '''
print([nltk.tag.str2tuple(t) for t in sent.split()])
print([nltk.tag.str2tuple(t) for t in nltk.corpus.brown.raw('cr09').split()])
#or
print(nltk.corpus.brown.tagged_words('cr09'))

#use universal tagset to avoid complications
nltk.corpus.treebank.tagged_words(tagset='universal')

#tagged corpus for other languages
print(nltk.corpus.sinica_treebank.tagged_words())
print(nltk.corpus.indian.tagged_words())
print(nltk.corpus.mac_morpho.tagged_words())
print(nltk.corpus.conll2002.tagged_words())
print(nltk.corpus.cess_cat.tagged_words())

#POS-Tagged Data from Four Indian Languages: Bangla, Hindi, Marathi, and Telugu
print(nltk.corpus.indian.raw())

#most common tags in brown corpus
brown_news_tagged=brown.tagged_words(categories='news',tagset='universal')
tag_fd=nltk.FreqDist(tag for word,tag in brown_news_tagged)
print(tag_fd.most_common())
tag_fd.plot(cumulative=True)
print(sum([tag_fd[tag] for tag in ['ADP','ADJ','NOUN','CONJ','X']])/sum([tag_fd[tag] for tag in tag_fd])*100,"%")

#checkout POS concondance tool
#nltk.app.concordance()

#inspect tagged text to see what parts of speech occur before a noun
word_tag_pairs = nltk.bigrams(brown_news_tagged)
noun_preceders = [a[1] for (a,b) in word_tag_pairs if b[1]=='NOUN']
fdist = nltk.FreqDist(noun_preceders)
print(fdist.most_common())

#most common verbs
wsj=nltk.corpus.treebank.tagged_words(tagset='universal')
word_tag_fd=nltk.FreqDist(wsj)
print([wt[0] for (wt,_) in word_tag_fd.most_common() if wt[1]=='VERB'])
#cfd with word as condition and tag as event
cfd1=nltk.ConditionalFreqDist(wsj)
print(cfd1['yield'].most_common())
print(cfd1['cut'].most_common())
#reverse order of pairs so that tags are conditions and words are events
wsj=nltk.corpus.treebank.tagged_words()
cfd2=nltk.ConditionalFreqDist((tag,word) for (word,tag) in wsj)
print(list(cfd2['VBD'])[-20:])#past tense
print(list(cfd2['VBN'])[-20:])#past participle
#only cfd prints generator expression, wrap in list it prints a list
#find words that can both be VBD and VBN
print(set(cfd2['VBD']))
print(set(cfd2['VBN']))
print(sorted(set(cfd2['VBD']).intersection(set(cfd2['VBN']))))

idx1=wsj.index(('kicked','VBD'))
print(wsj[idx1-4:idx1+1])
idx2=wsj.index(('kicked','VBN'))
print(wsj[idx2-4:idx2+1])

#Given the list of past participles produced by list(cfd2['VN']), try to collect a list of all the word-tag pairs that immediately precede items in that list.
#trying cfd2['VBN']
word_tag_pairs = nltk.bigrams(nltk.corpus.treebank.tagged_words())
vbn_preceders = [(a[0],b[0]) for (a,b) in word_tag_pairs if b[1]=='VBN' and b[0] in list(cfd2['VBN'])]
print(vbn_preceders)

#find most frequent nouns of each noun pos type i.e start with NN
def findtags(tag_prefix, tagged_text):
    cfd = nltk.ConditionalFreqDist((tag, word) for (word, tag) in tagged_text
                                  if tag.startswith(tag_prefix))
    return dict((tag, cfd[tag].most_common(5)) for tag in cfd.conditions())

tagdict = findtags('NN', nltk.corpus.brown.tagged_words(categories='news'))
for tag in sorted(tagdict):
    print(tag, tagdict[tag])
#the most important contain $ for possessive nouns, S for plural nouns (since plural nouns typically end in s) and P for proper nouns. In addition, most of the tags have suffix modifiers: -NC for citations, -HL for words in headlines and -TL for titles (a feature of Brown tabs)

#study words that follow 'often'
brown_learned_words = brown.words(categories='learned')
print(sorted(set(b for (a,b) in nltk.bigrams(brown_learned_words) if a.lower()=='often')))
#using tagged words instead of words
brown_learned_tagged = brown.tagged_words(categories='learned',tagset='universal')
tags=[b[1] for (a,b) in nltk.bigrams(brown_learned_tagged) if a[0]=='often']
print(tags)
fd=nltk.FreqDist(tags)
print(fd.tabulate())

#check for sequence of tags eg verb to verb
#Searching for Three-Word Phrases Using POS Tags
def process(sentence):
    for (w1,t1), (w2,t2), (w3,t3) in nltk.trigrams(sentence):
        if (t1.startswith('V') and t2 == 'TO' and t3.startswith('V')):
            print(w1, w2, w3)

for tagged_sent in brown.tagged_sents():
     process(tagged_sent)

#words that are highly ambiguous as to their part of speech tag. Understanding why such words are tagged as they are in each context can help us clarify the distinctions between the tags.
brown_news_tagged = brown.tagged_words(categories='news', tagset='universal')
data = nltk.ConditionalFreqDist((word.lower(), tag) for (word, tag) in brown_news_tagged)
for word in sorted(data.conditions()):
     if len(data[word]) > 3: # words having more than 3 pos tags
         tags = [tag for (tag, _) in data[word].most_common()]
         print(word, ' '.join(tags))

#python dictionary
pos = {}
print(pos)
pos['colorless'] = 'ADJ'
print(pos)
pos['ideas'] = 'N'
pos['sleep'] = 'V'
pos['furiously'] = 'ADV'
print(pos)
print(list(pos))
print(sorted(pos))
print([w for w in pos if w.endswith('s')])
print(list(pos.keys()))
print(list(pos.values()))
print(list(pos.items()))
#ways to define a dictionary
pos = {'colorless': 'ADJ', 'ideas': 'N', 'sleep': 'V', 'furiously': 'ADV'}
pos = dict(colorless='ADJ', ideas='N', sleep='V', furiously='ADV')
#pos = {['ideas', 'blogs', 'adventures']: 'N'} #list objects are unhashable

#defaultdict - creates a key with default value if key is not present in dict
from collections import defaultdict
frequency = defaultdict(int)
frequency['colorless'] = 4
print(frequency['ideas'])
print(frequency)
pos = defaultdict(list)
pos['sleep'] = ['NOUN', 'VERB']
pos['ideas']
print(pos)
pos = defaultdict(lambda: 'NOUN')
pos['colorless'] = 'ADJ'
pos['blog']
print(pos)

#automatically tagging words not present in vocab
alice = nltk.corpus.gutenberg.words('carroll-alice.txt')
vocab = nltk.FreqDist(alice)
v1000 = [word for (word, _) in vocab.most_common(1000)]
mapping = defaultdict(lambda: 'UNK')
for v in v1000:
     mapping[v] = v

alice2 = [mapping[v] for v in alice]
print(alice2[:100])

#incrementally updating dictionary
counts=defaultdict(int)
for (word, tag) in brown.tagged_words(categories='news', tagset='universal'):
     counts[tag] += 1
print(counts['NOUN'])
print(sorted(counts))

from operator import itemgetter
print(sorted(counts.items(), key=itemgetter(1), reverse=True))
print([t for t, c in sorted(counts.items(), key=itemgetter(1), reverse=True)])
#itemgetter - returns a function that can be called on some other sequence object to obtain nth element
pair = ('NP',98009)
print(pair[1])
print(itemgetter(1)(pair))
print(itemgetter(0)(pair))

#index words using their last two letters
last_letters = defaultdict(list)
words=nltk.corpus.words.words('en')
for word in words:
    key=word[-2:]
    last_letters[key].append(word)

print(last_letters['ly'])
print(last_letters['ed'])

#create an anagram dictionary
anagrams = defaultdict(list)
for word in words:
     key = ''.join(sorted(word))
     anagrams[key].append(word)

print(anagrams['aeilnrt'])
print(anagrams['ijno'])

#complex keys and values
pos = defaultdict(lambda: defaultdict(int))
brown_news_tagged = brown.tagged_words(categories='news', tagset='universal')
for ((w1, t1), (w2, t2)) in nltk.bigrams(brown_news_tagged):
    pos[(t1, w2)][t2] += 1

print(pos[('DET', 'right')])

#inverting a dictionary - so keys can be looked up by values
#cumbersome method
counts = defaultdict(int)
for word in nltk.corpus.gutenberg.words('milton-paradise.txt'):
     counts[word] += 1

print([key for (key, value) in counts.items() if value == 32])

#easier to construct a new dictionary
pos = {'colorless': 'ADJ', 'ideas': 'N', 'sleep': 'V', 'furiously': 'ADV'}
pos2 = dict((value, key) for (key, value) in pos.items())
print(pos2['N'])
#invert the dictionary
pos.update({'cats': 'N', 'scratch': 'V', 'peacefully': 'ADV', 'old': 'ADJ'})
pos2 = defaultdict(list)
for key, value in pos.items():
     pos2[value].append(key)

print(pos2['ADV'])
#easier way using nltk Index
pos2 = nltk.Index((value, key) for (key, value) in pos.items())
print(pos2['ADV'])

#automatic tagging of words - tagging a word depends on the word and its context, so we will use sentences instead of words
brown_tagged_sents = brown.tagged_sents(categories='news')
brown_sents=brown.sents(categories='news')

#default tagger - assign same tag(most likely) to each token - on a typical corpus, will only get 1/8 items right
tags = [tag for (word, tag) in brown.tagged_words(categories='news')]
print(nltk.FreqDist(tags).max())
#example
raw = 'I do not like green eggs and ham, I do not like them Sam I am!'
tokens = word_tokenize(raw)
default_tagger = nltk.DefaultTagger('NN')
print(default_tagger.tag(tokens))
#compare result with brown tagged sentences
print(default_tagger.evaluate(brown_tagged_sents))

#regular expression tagger
#assign tags based on matching patterns - eg. word ending 'ed' is past participle of a verb, word ending with 's is a possesive noun
# around 1/5th are correct
patterns = [
     (r'.*ing$', 'VBG'),               # gerunds
     (r'.*ed$', 'VBD'),                # simple past
     (r'.*es$', 'VBZ'),                # 3rd singular present
     (r'.*ould$', 'MD'),               # modals
     (r'.*\'s$', 'NN$'),               # possessive nouns
     (r'.*s$', 'NNS'),                 # plural nouns
     (r'^-?[0-9]+(.[0-9]+)?$', 'CD'),  # cardinal numbers
     (r'.*', 'NN')                     # nouns (default) tag everything else as noun
]
regexp_tagger = nltk.RegexpTagger(patterns)
print(regexp_tagger.tag(brown_sents[3]))
print(regexp_tagger.evaluate(brown_tagged_sents))

#lookup tagger - find most frequent words, store their most likely tag, use this to look up later
fd = nltk.FreqDist(brown.words(categories='news'))
cfd = nltk.ConditionalFreqDist(brown.tagged_words(categories='news'))
most_freq_words = fd.most_common(100)
likely_tags = dict((word, cfd[word].max()) for (word, _) in most_freq_words) #cfd[word].max() select tag with max count
baseline_tagger = nltk.UnigramTagger(model=likely_tags)
baseline_tagger.evaluate(brown_tagged_sents) # efficiency of auto tagging almost 45%
#try on some untagged input
sent = brown.sents(categories='news')[3]
baseline_tagger.tag(sent) #automatically assigned None for words it could not find

#backoff method - if first tagger unable to assign tag, switch to default tagger
baseline_tagger = nltk.UnigramTagger(model=likely_tags, backoff=nltk.DefaultTagger('NN'))
baseline_tagger.tag(sent)

#create and evaluate lookup taggers having a range of sizes
def performance(cfd, wordlist):
    lt = dict((word, cfd[word].max()) for word in wordlist)
    baseline_tagger = nltk.UnigramTagger(model=lt, backoff=nltk.DefaultTagger('NN'))
    return baseline_tagger.evaluate(brown.tagged_sents(categories='news'))

def display():
    import pylab
    word_freqs = nltk.FreqDist(brown.words(categories='news')).most_common()
    words_by_freq = [w for (w, _) in word_freqs]
    cfd = nltk.ConditionalFreqDist(brown.tagged_words(categories='news'))
    sizes = 2 ** pylab.arange(15) # define various sizes
    perfs = [performance(cfd, words_by_freq[:size]) for size in sizes]
    pylab.plot(sizes, perfs, '-bo')
    pylab.title('Lookup Tagger Performance with Varying Model Size')
    pylab.xlabel('Model Size')
    pylab.ylabel('Performance')
    pylab.show()

display()

#n-gram tagging
#unigram tagging behaves just like a lookup tagger, except there is a more convenient technique for setting it up, called training
brown_tagged_sents = brown.tagged_sents(categories='news')
brown_sents = brown.sents(categories='news')
unigram_tagger = nltk.UnigramTagger(brown_tagged_sents) #inspect each tag and store most likely tag of a word
unigram_tagger.tag(brown_sents[2007])
unigram_tagger.evaluate(brown_tagged_sents)
# how does this work?
# training and test data is the same
# we are training the UnigramTagger on brown tagged sentences, and making it tag on brown untagged sentences

# Now that we are training a tagger on some data, we must be careful not to test it on the same data, as we did in the above example. A tagger that simply memorized its training data and made no attempt to construct a general model would get a perfect score
#so lets split our training data into 90-10
size = int(len(brown_tagged_sents) * 0.9)
train_sents = brown_tagged_sents[:size]
test_sents = brown_tagged_sents[size:]
unigram_tagger = nltk.UnigramTagger(train_sents)
unigram_tagger.evaluate(test_sents)

#bigram tagger
bigram_tagger = nltk.BigramTagger(train_sents)
bigram_tagger.tag(brown_sents[2007])
unseen_sent = brown_sents[4203]
bigram_tagger.tag(unseen_sent)
bigram_tagger.evaluate(test_sents)
#bad performance - fails to tag on unseen words

#tradeoff between accuracy and coverage - precision/recall tradeoff
#combining taggers - use more accurate algorithms as we can, but fall back to algorithms with wider coverage when necessary
#eg try bigram tagging, then unigram else back off to default tagger
t0 = nltk.DefaultTagger('NN')
t1 = nltk.UnigramTagger(train_sents, backoff=t0)
t2 = nltk.BigramTagger(train_sents, backoff=t1)
t2.evaluate(test_sents)
#extend above model to use trigram tagger
t3=nltk.TrigramTagger(train_sents,backoff=t2)
t3.evaluate(test_sents)
#if bigram tagger would assign same tag as unigram tagger, bigram tagger discards the training instance
#can also use parameter cutoff to indicate number of contexts required to be seen for tagger to not discard context

#storing taggers - because training every time is not practical
import pickle
from pickle import dump
output = open('t2.pkl', 'wb')
dump(t2, output, -1)
output.close()

input = open('t2.pkl', 'rb')
tagger = pickle.load(input)
input.close()
#use to tag sentence
text = """The board's action shows what free enterprise is up against in our complex maze of regulatory laws ."""
tagger.tag(text.split())

#given a tagger how many cases of ambiguity does it encounter?
cfd = nltk.ConditionalFreqDist(
            ((x[1], y[1], z[0]), z[1])
            for sent in brown_tagged_sents
            for x, y, z in nltk.trigrams(sent))
ambiguous_contexts = [c for c in cfd.conditions() if len(cfd[c]) > 1]
print(sum(cfd[c].N() for c in ambiguous_contexts) / cfd.N())
# Thus, one out of twenty trigrams is ambiguous [EXAMPLES]. Given the current word and the previous two tags, in 5% of cases there is more than one tag that could be legitimately assigned to the current word according to the training data.

#another way to invesitgate performance is to study its mistakes..
#confusion matrix - convenient way of looking for tagging errors
#charts expected(gold standard) tags against actual tags generated by tagger
test_tags = [tag for sent in brown.sents(categories='editorial')
                  for (word, tag) in t2.tag(sent)]
gold_tags = [tag for (word, tag) in brown.tagged_words(categories='editorial')]
print(nltk.ConfusionMatrix(gold_tags, test_tags))

# nltk.tag.brill.demo() #does not work