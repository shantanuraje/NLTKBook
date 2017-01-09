import nltk, re
from nltk.corpus import conll2000
from nltk.corpus import conll2002

#information extraction from structured data such as tables
#information about companies and locations stored in tuples (entity, relation, entity)
locs = [('Omnicom', 'IN', 'New York'),
        ('DDB Needham', 'IN', 'New York'),
        ('Kaplan Thaler Group', 'IN', 'New York'),
        ('BBDO South', 'IN', 'Atlanta'),
        ('Georgia-Pacific', 'IN', 'Atlanta')]
#which organizations operate in atlanta
query = [e1 for (e1, rel, e2) in locs if e2=='Atlanta']
print(query)
#convert unstructured data into structured tables then query it.

#information extraction architecture - raw text -> sentence segmentation -> tokenization -> pos tagging -> entity detection -> relation detection -> relations
#for first three tasks use we can use nltk's default sentence segmenter, word tokenizer, pos tagger
def ie_preprocess(document):
    sentences = nltk.sent_tokenize(document)
    sentences = [nltk.word_tokenize(sent) for sent in sentences]
    sentences = [nltk.pos_tag(sent) for sent in sentences]
#entity detection - look for definite noun phrases and proper names, ignore indefinite nouns and noun chunks
#relation extraction - search for specific patterns between pairs of entities, use those patterns to build tuples recording the relationships between entities

#chunking - segments and labels multi token sequences
#noun phrase chunking - search for chunks corresponding to individual noun phrases
#pos tagging is one of the motivations for np chunking
sentence = [("the", "DT"), ("little", "JJ"), ("yellow", "JJ"),("dog", "NN"), ("barked", "VBD"), ("at", "IN"),  ("the", "DT"), ("cat", "NN")]
grammar = "NP: {<DT>?<JJ>*<NN>}" #np chunk should be formed whenever the chunker finds an optional determiner (DT) followed by any number of adjectives (JJ) and then a nound (NN)
cp = nltk.RegexpParser(grammar)
result = cp.parse(sentence)
print(result)
result.draw()
#rules that make up chunk grammar use tag patterns to describe sequence of tagged words. Tag pattern is a sequence of pos tags delimited using <>
#similar to regex patterns
#slight refinement in 1st pattern - <DT>?<JJ.*>*<NN.*>+
grammar = r"""
  NP: {<DT|PP\$>?<JJ>*<NN>} # chunk determiner/possessive, adjectives and noun
      {<NNP>+}              # chunk sequences of proper nouns
"""
cp = nltk.RegexpParser(grammar)
sentence = [("Rapunzel", "NNP"), ("let", "VBD"), ("down", "RP"),("her", "PP$"), ("long", "JJ"), ("golden", "JJ"), ("hair", "NN")]
print(cp.parse(sentence))
#chunk 2 consecutive nouns - in the case that tag pattern matches overlapping locations.
nouns = [("money", "NN"), ("market", "NN"), ("fund", "NN")]
grammar = "NP: {<NN><NN>}  # Chunk two consecutive nouns"
cp = nltk.RegexpParser(grammar)
print(cp.parse(nouns))
#by removing the third noun we have lost context, instead we can have a more permissive chunk rule NP: {<NN>+}
grammar = r"""
  NP: {<DT|PP\$>?<JJ>*<NN>+} # chunk determiner/possessive, adjectives and noun
      {<NNP>+}              # chunk sequences of proper nouns
"""
cp = nltk.RegexpParser(grammar)
print(cp.parse(nouns))

#exploring text corpora
#using a chunker to extract phrases matching a particular sequence of pos tags
cp = nltk.RegexpParser('CHUNK: {<V.*> <TO> <V.*>}')
brown = nltk.corpus.brown
for sent in brown.tagged_sents()[:100]:
     tree = cp.parse(sent)
     for subtree in tree.subtrees():
         if subtree.label() == 'CHUNK': print(subtree)
#convert this to a function
def find_chunks(chunk_exp,corpus_tagged_sents):
    cp = nltk.RegexpParser(chunk_exp)
    for sent in corpus_tagged_sents[:100]:
        tree = cp.parse(sent)
        for subtree in tree.subtrees():
            if subtree.label() == 'CHUNK': print(subtree)
# find_chunks("CHUNK: {<N.*>{4,}}",brown.tagged_sents()) #not working
find_chunks("CHUNK: {<V.*> <TO> <V.*>}",brown.tagged_sents())

#chinking - removing a sequence of tokens from a text
# 3 possibilities -
sentence = [("the", "DT"), ("little", "JJ"), ("yellow", "JJ"),("dog", "NN"), ("barked", "VBD"), ("at", "IN"),  ("the", "DT"), ("cat", "NN")]
grammar = r""" NP:
                    {<.*>+}          # Chunk everything
                    }<VBD|IN>+{      # Chink sequences of VBD and IN
          """
cp = nltk.RegexpParser(grammar)
print(cp.parse(sentence))

#reading iob format and conll 2000 corpus
text = '''
he PRP B-NP
accepted VBD B-VP
the DT B-NP
position NN I-NP
of IN B-PP
vice NN B-NP
chairman NN I-NP
of IN B-PP
Carlyle NNP B-NP
Group NNP I-NP
, , O
a DT B-NP
merchant NN I-NP
banking NN I-NP
concern NN I-NP
. . O
'''
nltk.chunk.conllstr2tree(text, chunk_types=['NP']).draw()
# nltk.chunk.conllstr2tree(text, chunk_types=['VP']).draw()
# nltk.chunk.conllstr2tree(text, chunk_types=['PP']).draw()
print(conll2000.chunked_sents('train.txt')[99])
print(conll2000.chunked_sents('train.txt', chunk_types=['NP'])[99])

#evaluating chunkers
#for trivial chunk parser that creates no chunks
cp = nltk.RegexpParser("")
test_sents = conll2000.chunked_sents('test.txt', chunk_types=['NP'])
print(cp.evaluate(test_sents))
#now for a naive reg exp chunkner
grammar = r"NP: {<[CDJNP].*>+}"
cp = nltk.RegexpParser(grammar)
print(cp.evaluate(test_sents))

#chunker using a unigram tagger
class UnigramChunker(nltk.ChunkParserI):
    # constructor called when we build a new unigramchunker
    def __init__(self, train_sents):
        train_data = [[(t,c) for w,t,c in nltk.chunk.tree2conlltags(sent)]
                      for sent in train_sents]
        self.tagger = nltk.UnigramTagger(train_data)
    # used to chunk new sentences
    def parse(self, sentence):
        pos_tags = [pos for (word,pos) in sentence]
        tagged_pos_tags = self.tagger.tag(pos_tags)
        chunktags = [chunktag for (pos, chunktag) in tagged_pos_tags]
        conlltags = [(word, pos, chunktag) for ((word,pos),chunktag)
                     in zip(sentence, chunktags)]
        return nltk.chunk.conlltags2tree(conlltags)

test_sents = conll2000.chunked_sents('test.txt', chunk_types=['NP'])
train_sents = conll2000.chunked_sents('train.txt', chunk_types=['NP'])
unigram_chunker = UnigramChunker(train_sents)
print(unigram_chunker.evaluate(test_sents))

postags = sorted(set(pos for sent in train_sents for (word,pos) in sent.leaves()))
print(unigram_chunker.tagger.tag(postags))

#bigram chunker
class BigramChunker(nltk.ChunkParserI):
    # constructor called when we build a new unigramchunker
    def __init__(self, train_sents):
        train_data = [[(t,c) for w,t,c in nltk.chunk.tree2conlltags(sent)]
                      for sent in train_sents]
        self.tagger = nltk.BigramTagger(train_data)
    # used to chunk new sentences
    def parse(self, sentence):
        pos_tags = [pos for (word,pos) in sentence]
        tagged_pos_tags = self.tagger.tag(pos_tags)
        chunktags = [chunktag for (pos, chunktag) in tagged_pos_tags]
        conlltags = [(word, pos, chunktag) for ((word,pos),chunktag)
                     in zip(sentence, chunktags)]
        return nltk.chunk.conlltags2tree(conlltags)

test_sents = conll2000.chunked_sents('test.txt', chunk_types=['NP'])
train_sents = conll2000.chunked_sents('train.txt', chunk_types=['NP'])
bigram_chunker = BigramChunker(train_sents)
print(bigram_chunker.evaluate(test_sents))

postags = sorted(set(pos for sent in train_sents for (word,pos) in sent.leaves()))
print(bigram_chunker.tagger.tag(postags))

#not working - unable to find megam file
#megam - mega model optimization package
#classifier based tagger
#need to use info about content of words in addition to pos tags to maximize chunking performance
# class ConsecutiveNPChunkTagger(nltk.TaggerI):
#
#     def __init__(self, train_sents):
#         train_set = []
#         for tagged_sent in train_sents:
#             untagged_sent = nltk.tag.untag(tagged_sent)
#             history = []
#             for i, (word, tag) in enumerate(tagged_sent):
#                 featureset = npchunk_features(untagged_sent, i, history)
#                 train_set.append( (featureset, tag) )
#                 history.append(tag)
#         self.classifier = nltk.MaxentClassifier.train(train_set, algorithm='megam', trace=0)
#
#     def tag(self, sentence):
#         history = []
#         for i, word in enumerate(sentence):
#             featureset = npchunk_features(sentence, i, history)
#             tag = self.classifier.classify(featureset)
#             history.append(tag)
#         return zip(sentence, history)
#
# class ConsecutiveNPChunker(nltk.ChunkParserI):
#     def __init__(self, train_sents):
#         tagged_sents = [[((w,t),c) for (w,t,c) in
#                          nltk.chunk.tree2conlltags(sent)]
#                         for sent in train_sents]
#         self.tagger = ConsecutiveNPChunkTagger(tagged_sents)
#
#     def parse(self, sentence):
#         tagged_sents = self.tagger.tag(sentence)
#         conlltags = [(w,t,c) for ((w,t),c) in tagged_sents]
#         return nltk.chunk.conlltags2tree(conlltags)
#
#
# def npchunk_features(sentence, i, history):
#     word, pos = sentence[i]
#     return {"pos": pos}
#
# chunker = ConsecutiveNPChunker(train_sents)
# print(chunker.evaluate(test_sents))
# #add a feature for previous pos tag
# def npchunk_features(sentence, i, history):
#      word, pos = sentence[i]
#      if i == 0:
#          prevword, prevpos = "<START>", "<START>"
#      else:
#          prevword, prevpos = sentence[i-1]
#      return {"pos": pos, "prevpos": prevpos}
# chunker = ConsecutiveNPChunker(train_sents)
# print(chunker.evaluate(test_sents))
# #add feature for current word
# def npchunk_features(sentence, i, history):
#      word, pos = sentence[i]
#      if i == 0:
#          prevword, prevpos = "<START>", "<START>"
#      else:
#          prevword, prevpos = sentence[i-1]
#      return {"pos": pos, "word": word, "prevpos": prevpos}
# chunker = ConsecutiveNPChunker(train_sents)
# print(chunker.evaluate(test_sents))
# # extending the feature extractor - includes lookahead features, paired features, and complex contextual features
# def npchunk_features(sentence, i, history):
#      word, pos = sentence[i]
#      if i == 0:
#          prevword, prevpos = "<START>", "<START>"
#      else:
#          prevword, prevpos = sentence[i-1]
#      if i == len(sentence)-1:
#          nextword, nextpos = "<END>", "<END>"
#      else:
#          nextword, nextpos = sentence[i+1]
#      return {"pos": pos,
#              "word": word,
#              "prevpos": prevpos,
#              "nextpos": nextpos, [1]
#              "prevpos+pos": "%s+%s" % (prevpos, pos),  [2]
#              "pos+nextpos": "%s+%s" % (pos, nextpos),
#              "tags-since-dt": tags_since_dt(sentence, i)}  [3]
# #feature - tags-since-dt
# def tags_since_dt(sentence, i):
#      tags = set()
#      for word, pos in sentence[:i]:
#          if pos == 'DT':
#              tags = set()
#          else:
#              tags.add(pos)
#      return '+'.join(sorted(tags))
# chunker = ConsecutiveNPChunker(train_sents)
# print(chunker.evaluate(test_sents))

#building nested structure with cascaded chunkers - by creating multistage chunk grammar containing recursive rules
grammar = r"""
  NP: {<DT|JJ|NN.*>+}          # Chunk sequences of DT, JJ, NN
  PP: {<IN><NP>}               # Chunk prepositions followed by NP
  VP: {<VB.*><NP|PP|CLAUSE>+$} # Chunk verbs and their arguments
  CLAUSE: {<NP><VP>}           # Chunk NP, VP
  """
cp = nltk.RegexpParser(grammar)
sentence = [("Mary", "NN"), ("saw", "VBD"), ("the", "DT"), ("cat", "NN"),
    ("sit", "VB"), ("on", "IN"), ("the", "DT"), ("mat", "NN")]
print(cp.parse(sentence)) #misses the verb phrase from saw
sentence = [("John", "NNP"), ("thinks", "VBZ"), ("Mary", "NN"),("saw", "VBD"), ("the", "DT"), ("cat", "NN"), ("sit", "VB"),("on", "IN"), ("the", "DT"), ("mat", "NN")]
print(cp.parse(sentence)) #fails to identify verb phrase
#solution to these problems - loop over its patterns
cp = nltk.RegexpParser(grammar, loop=2)
print(cp.parse(sentence)) #included the saw phrase
#cascading enables to create deep structures - however creating and debugging is difficult . Cascading can only produce trees of fixed depth (no deeper than the number of stages in cascade), this is insufficient for syntactic analysis

#trees
tree1 = nltk.Tree('NP',['Alice'])
print(tree1)
tree2 = nltk.Tree('NP',['the','rabbit'])
print(tree2)
tree3 = nltk.Tree('VP',['chased',tree2])
tree4 = nltk.Tree('S', [tree1, tree3])
print(tree4)
print(tree4[1].label())
print(tree4.leaves())
print(tree4[1][1][1]) #difficult to read
tree4.draw()

#tree traversal - using a recursive function
def traverse(t):
    try:
        t.label()
    except AttributeError:
        print(t, end=" ")
    else:
        # Now we know that t.node is defined
        print('(', t.label(), end=" ")
        for child in t:
            traverse(child)
        print(')', end=" ")

# t = nltk.Tree('(S (NP Alice) (VP chased (NP the rabbit ) ) )')
traverse(tree4)

#named entity recognition
sent = nltk.corpus.treebank.tagged_sents()[22]
print(nltk.ne_chunk(sent, binary=True)) [1]
print(nltk.ne_chunk(sent))

#relation extraction between named entities
IN = re.compile(r'.*\bin\b(?!\b.+ing)')
for doc in nltk.corpus.ieer.parsed_docs('NYT_19980315'):
     for rel in nltk.sem.extract_rels('ORG', 'LOC', doc, corpus='ieer', pattern = IN):
         print(nltk.sem.rtuple(rel))

vnv = """
  (
  is/V|    # 3rd sing present and
  was/V|   # past forms of the verb zijn ('be')
  werd/V|  # and also present
  wordt/V  # past of worden ('become)
  )
  .*       # followed by anything
  van/Prep # followed by van ('of')
  """
VAN = re.compile(vnv, re.VERBOSE)
for doc in conll2002.chunked_sents('ned.train'):
    for r in nltk.sem.extract_rels('PER', 'ORG', doc,corpus='conll2002', pattern=VAN):
        print(nltk.sem.clause(r, relsym="VAN"))
        # print(rtuple(rel, lcon=True, rcon=True)) #This will show you the actual words that intervene between the two NEs and also their left and right context, within a default 10 - word window.