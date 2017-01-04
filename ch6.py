import nltk
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
pos_features2(brown.sents()[0], 8)#i is which word
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