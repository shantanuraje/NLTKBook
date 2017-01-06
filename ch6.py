import nltk
import math
import random
from nltk.corpus import names
from nltk.classify import apply_features
from nltk.corpus import movie_reviews
from nltk.corpus import brown

#gender classification
# The returned dictionary, known as a feature set, maps from feature names to their values
def gender_features(word):
    return {'last_letter': word[-1], 'first_letter': word[0], 'name_length': len(word)}

gender_features('Shrek')
# Now that we've defined a feature extractor, we need to prepare a list of examples and corresponding class labels.
labeled_names = ([(name, 'male') for name in names.words('male.txt')] + [(name, 'female') for name in names.words('female.txt')])
print(labeled_names)
random.shuffle(labeled_names) #shuffles list
#train classifier
featuresets = [(gender_features(n), gender) for (n, gender) in labeled_names]
train_set, test_set = featuresets[500:], featuresets[:500] #500 to end, first 500
classifier = nltk.NaiveBayesClassifier.train(train_set)
#check classifier output
print(classifier.classify(gender_features('Neo')))
print(classifier.classify(gender_features('Trinity')))
#check classifier accuracy on test set
print(nltk.classify.accuracy(classifier,test_set))
#examine classifier to see which features it found most effective
print(classifier.show_most_informative_features(20)) #shows likelihood ratios
#not feasible to store features of every instance in a list, apply_features returns an object that acts like a list but does not store all the feature sets in memory
train_set = apply_features(gender_features, labeled_names[500:])
test_set = apply_features(gender_features, labeled_names[:500])

#feature extractor that overfits gender featuresdef gender_features2(name):
def gender_features2(name):
    features = {}
    features["first_letter"] = name[0].lower()
    features["last_letter"] = name[-1].lower()
    for letter in 'abcdefghijklmnopqrstuvwxyz':
        features["count({})".format(letter)] = name.lower().count(letter)
        features["has({})".format(letter)] = (letter in name.lower())
    return features
#function has features like count of every character in a-z and boolean value
#leads to overfitting
print(gender_features2('John'))
#test this feature set
featuresets = [(gender_features2(n), gender) for (n, gender) in labeled_names]
train_set, test_set = featuresets[500:], featuresets[:500]
classifier = nltk.NaiveBayesClassifier.train(train_set)
print(nltk.classify.accuracy(classifier, test_set))

#productive method for refining feature set is error analysis
#select a development set (containing corpus for creating the model) - divide it into training and dev-test sets
train_names = labeled_names[1500:] #1500 and up, used to train
devtest_names = labeled_names[500:1500] #500 to 1500, used to perform error analysis
test_names = labeled_names[:500] #first 500, servers in final evaluation of system
#train model using training set, run it on dev-test set
train_set = [(gender_features(n), gender) for (n, gender) in train_names]
devtest_set = [(gender_features(n), gender) for (n, gender) in devtest_names]
test_set = [(gender_features(n), gender) for (n, gender) in test_names]
classifier = nltk.NaiveBayesClassifier.train(train_set)
print(nltk.classify.accuracy(classifier, devtest_set))
# Using the dev-test set, we can generate a list of the errors that the classifier makes when predicting name genders:
errors = []
for (name, tag) in devtest_names:
     guess = classifier.classify(gender_features(name))
     if guess != tag:
         errors.append( (tag, guess, name) )
errors.sort()
for (tag,guess,name) in errors:
    print("correct = ",tag,"| guess = ",guess," | name = ",name)
#adjust feature extractor to include features of 2 letter suffixes
def gender_features3(word):
    return {'suffix1':word[-1:],'suffix2':word[-2:]}
#re-train model
train_set = [(gender_features3(n), gender) for (n, gender) in train_names]
devtest_set = [(gender_features3(n), gender) for (n, gender) in devtest_names]
test_set = [(gender_features3(n), gender) for (n, gender) in test_names]
classifier = nltk.NaiveBayesClassifier.train(train_set)
print(nltk.classify.accuracy(classifier, devtest_set))
#repear error analysis procedure, select different devtest/taining split, keep test set unused until model development is complete

#document classification
#example: movie reviews classification, positive or negative
documents = [(list(movie_reviews.words(fileid)), category) for category in movie_reviews.categories() for fileid in movie_reviews.fileids(category)]
#construct list of most frequent words in entire corpus, define feature extractor that simply checks whether each of these words is present in a given document
all_words = nltk.FreqDist(w.lower() for w in movie_reviews.words())
word_features = list(all_words)[:2000]
#that is 2000 features
def document_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains({})'.format(word)] = (word in document_words)
    return features
# print(document_features(movie_reviews.words('pos/cv957_8737.txt')))
#train classifier
featuresets = [(document_features(d), c) for (d,c) in documents]
train_set, test_set = featuresets[100:], featuresets[:100]
classifier = nltk.NaiveBayesClassifier.train(train_set)
print(nltk.classify.accuracy(classifier, test_set))
print(classifier.show_most_informative_features(20))

#POS tagging
#create a fd. add 1,2,3 letter suffix and increment their counts
suffix_dist=nltk.FreqDist()
for word in brown.words():
    word = word.lower()
    suffix_dist[word[-1:]] += 1
    suffix_dist[word[-2:]] += 1
    suffix_dist[word[-3:]] += 1
# pick most common suffixes
common_suffixes = [suffix for (suffix, count) in suffix_dist.most_common(100)]
print(common_suffixes)
#define feature extractor to check a given word for these suffixes
def pos_features(word):
     features = {}
     for suffix in common_suffixes:
         features['endswith({})'.format(suffix)] = word.lower().endswith(suffix)
     return features
#train a decision tree classifier
#get training set of tagged words format word,tag
tagged_words = brown.tagged_words(categories='news')
#loop through those tagged words and create a dict of features, tag
featuresets = [(pos_features(n), g) for (n,g) in tagged_words]
#define size to split 90-10
size = int(len(featuresets) * 0.1)
# divide feature sent
train_set, test_set = featuresets[size:], featuresets[:size]
# train classifier
classifier = nltk.DecisionTreeClassifier.train(train_set,verbose=True)
# check accuracy
print(nltk.classify.accuracy(classifier, test_set))
# predict
print(classifier.classify(pos_features('cats')))
# print pseudo code of decision tree
print(classifier.pseudocode(depth=4))
#pos classifier that uses context of words by including previous and next words as features
def pos_features2(sentence, i):
    features = {"suffix(1)": sentence[i][-1:],
                "suffix(2)": sentence[i][-2:],
                "suffix(3)": sentence[i][-3:]}
    if i == 0:
        features["prev-word"] = "<START>"
    else:
        features["prev-word"] = sentence[i-1]
    return features
print(pos_features2(brown.sents()[0], 8))#i is which word
tagged_sents = brown.tagged_sents(categories='news')
featuresets = []
for tagged_sent in tagged_sents:
     untagged_sent = nltk.tag.untag(tagged_sent)
     for i, (word, tag) in enumerate(tagged_sent):
         featuresets.append( (pos_features2(untagged_sent, i), tag) )

size = int(len(featuresets) * 0.1)
train_set, test_set = featuresets[size:], featuresets[:size]
classifier = nltk.NaiveBayesClassifier.train(train_set)

print(nltk.classify.accuracy(classifier, test_set))

#sequence classification - eg. in pos tagging a variety of different sequence classifier models can be used to jointly choose pos tags for all the words in a given sentence
#consecutive/greedy sequence classification -find most likely input for 1st input, use it to find best label for next input - like n gram tagger example (bigram)
#we have to augment our feature extractor to take history argument
# def pos_features(sentence, i, history):
#      features = {"suffix(1)": sentence[i][-1:],
#                  "suffix(2)": sentence[i][-2:],
#                  "suffix(3)": sentence[i][-3:]}
#      if i == 0:
#          features["prev-word"] = "<START>"
#          features["prev-tag"] = "<START>"
#      else:
#          features["prev-word"] = sentence[i-1]
#          features["prev-tag"] = history[i-1]
#      return features
#
# class ConsecutivePosTagger(nltk.TaggerI):
#
#     def __init__(self, train_sents):
#         train_set = []
#         for tagged_sent in train_sents:
#             untagged_sent = nltk.tag.untag(tagged_sent)
#             history = []
#             for i, (word, tag) in enumerate(tagged_sent):
#                 featureset = pos_features(untagged_sent, i, history)
#                 train_set.append( (featureset, tag) )
#                 history.append(tag)
#         self.classifier = nltk.NaiveBayesClassifier.train(train_set)
#
#     def tag(self, sentence):
#         history = []
#         for i, word in enumerate(sentence):
#             featureset = pos_features(sentence, i, history)
#             tag = self.classifier.classify(featureset)
#             history.append(tag)
#         return list((sentence, history))
#
# tagged_sents = brown.tagged_sents(categories='news')
# size = int(len(tagged_sents) * 0.1)
# train_sents, test_sents = tagged_sents[size:], tagged_sents[:size]
# tagger = ConsecutivePosTagger(train_sents)
# print(tagger.evaluate(test_sents))
# revisit this function not running
# disadvantage of this approach is that no way to go back fix errors once tag is decided
# using a tranformational strategy instead
# transformational joint classifiers - work by creating an initial assignment of labels for inputs and then iteratively refine assignment to repair inconsistencies - like the brill tagger discusses previously
# another strategy is to assign scores to all possible sequences of pos tags choose the highest score. This approach is taken by Hidden Markov models - similar to consecutive classifiers as they look at both, input and history

#sentence segmentation - classification task for punctuation marks
# sents = nltk.corpus.treebank_raw.sents()
sents = nltk.corpus.treebank_raw.sents('wsj_0199')
tokens = [] #merged list of tokens from individual sentences
boundaries = set() #set containing indexes of all sentence boundary tokens
offset = 0
for sent in sents:
     tokens.extend(sent) #extend list by appending items of other list
     # print(sent,len(sent))
     offset += len(sent) #keeps adding len of sent to value of offset to mark sentence boundary of each token
     # print(offset)
     boundaries.add(offset-1)
# print(tokens)
# print(boundaries)
# print(offset)

#define features of data to be used in order to decide whether punctuation indicates a sentence boundary
def punct_features(tokens, i):
    print(tokens[i],tokens[i+1])
    return {'next-word-capitalized': tokens[i+1][0].isupper(),
             'prev-word': tokens[i-1].lower(),
             'punct': tokens[i],
             'prev-word-is-one-char': len(tokens[i-1]) == 1}

featuresets = [(punct_features(tokens, i), (i in boundaries)) for i in range(1, len(tokens)-1) if tokens[i] in '.?!']
#train and evaluate classifier
size = int(len(featuresets) * 0.1)
train_set, test_set = featuresets[size:], featuresets[:size]
classifier = nltk.NaiveBayesClassifier.train(train_set)
print(nltk.classify.accuracy(classifier, test_set))
#to use this classifier to perform sentence segmentation, we simply check each punctuation mark to see whether it is labelled as a boundary
# def segment_sentences(words):
#     print(words)
#     start = 0
#     sents = []
#     for i, word in enumerate(words):
#         # print(i,word)
#         if word in '.?!' and classifier.classify(punct_features(words, i)) == True:
#             sents.append(words[start:i+1])
#             start = i+1
#     if start < len(words):
#         sents.append(words[start:])
#     return sents
#
# segment_sentences(nltk.corpus.treebank_raw.words('wsj_0197'))
# # [(i,word) for i,word in enumerate(brown.tagged_sents(categories='news'))]
# # does not work revisit later

#identifying dialogue act types - statements, greetings, questions, answers, assertions, and clarifications,etc  - types of speech based actions
#nps chat corpus with > 10000 posts each labeled as one of 15 dialogue types suct as "Statement," "Emotion," "ynQuestion", and "Continuer."
#we can use this data to build a classifier that can identify dialogue act types for new im posts
#1st - extract basic messaging data
posts = nltk.corpus.nps_chat.xml_posts()[:10000]
def dialogue_act_features(post):
     features = {}
     for word in nltk.word_tokenize(post):
         # print(word)
         features['contains({})'.format(word.lower())] = True
     # print("---------------------")
     return features

featuresets = [(dialogue_act_features(post.text), post.get('class')) for post in posts]
size = int(len(featuresets) * 0.1)
train_set, test_set = featuresets[size:], featuresets[:size]
classifier = nltk.NaiveBayesClassifier.train(train_set)
print(nltk.classify.accuracy(classifier, test_set))

#recognizing textual entailment - given a text and hypothesis deciding whether the hypothesis is true or false
# def rte_features(rtepair):
#     extractor = nltk.RTEFeatureExtractor(rtepair) #builds bag of words for both text and hypothesis after throwing away some stopwords, calculates overlap and difference
#     features = {}
#     features['word_overlap'] = len(extractor.overlap('word'))
#     features['word_hyp_extra'] = len(extractor.hyp_extra('word'))
#     features['ne_overlap'] = len(extractor.overlap('ne'))
#     features['ne_hyp_extra'] = len(extractor.hyp_extra('ne'))
#     return features
#
# rtepair = nltk.corpus.rte.pairs(['rte3_dev.xml'])[33]
# extractor = nltk.RTEFeatureExtractor(rtepair)
# print(extractor.text_words)
# print(extractor.hyp_words)
# print(extractor.overlap('word'))
# print(extractor.overlap('ne'))
# print(extractor.hyp_extra('word'))

# for rtepair in nltk.corpus.rte.pairs('rte3_dev.xml'):
#     extractor = nltk.RTEFeatureExtractor(rtepair)
#     # print(extractor.text_tokens)
#     # print(ext.)

#problems with rtepairs revisit later

#distributing training and test sets
file_ids = brown.fileids(categories='news')
size = int(len(file_ids) * 0.1)
train_set = brown.tagged_sents(file_ids[size:])
test_set = brown.tagged_sents(file_ids[:size])
#instead of using data from same file for test and training, use data from one file for training and data from another for testing. (of same genre)

#confusion matrix - in classification problems this can be more informative
#table where each cell[i,j] indicates how often label j was predicted when correct label was i. Diagonal entries indicate correct predictions, other error.
#example using the bigram tagger
# def tag_list(tagged_sents):
#      return [tag for sent in tagged_sents for (word, tag) in sent]
# def apply_tagger(tagger, corpus):
#      return [tagger.tag(nltk.tag.untag(sent)) for sent in corpus]
# data=nltk.corpus.brown.tagged_sents(categories='editorial')
# size=int(len(data)*0.9)
# gold = tag_list(nltk.corpus.brown.tagged_sents(categories='editorial'))
# print(len(gold))
# test = tag_list(apply_tagger(nltk.BigramTagger(data[size:]), data[:size]))
# print(len(test))
# cm = nltk.ConfusionMatrix(gold, test)
# print(cm.pretty_format(sort_by_count=True, show_percents=True, truncate=9))
# #how to work this example out?? in test = tag_list(apply_tagger) what should be the first argument and its training data

#entropy and information gain
# entropy is defined as sum of probability for each label with log of probability for that same label
# if many labels have same value entropy is low, if more wide variety entropy is high
def entropy(labels):
    freqdist = nltk.FreqDist(labels)
    probs = [freqdist.freq(l) for l in freqdist]
    print(probs)
    return -sum(p * math.log(p,2) for p in probs)

print(entropy(['male', 'male', 'male', 'male']))
print(entropy(['male', 'female', 'male', 'male']))
print(entropy(['female', 'male', 'female', 'male']))
print(entropy(['female', 'female', 'male', 'female']))
print(entropy(['female', 'female', 'female', 'female']))
