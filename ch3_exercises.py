import nltk, re
from urllib import request

#1 Define a string s = 'colorless'. Write a Python statement that changes this to "colourless" using only the slice and concatenation operations.
s ='colorless'
print(s[0:4]+'u'+s[4:len(s)])

#2 We can use the slice notation to remove morphological endings on words. For example, 'dogs'[:-1] removes the last character of dogs, leaving dog. Use slice notation to remove the affixes from these words (we've inserted a hyphen to indicate the affix boundary, but omit this from your strings): dish-es, run-ning, nation-ality, un-do, pre-heat.
words = ["dishes", "running", "nationality", "undo", "preheat"]
print(words[0][:-2],words[1][:-4],words[2][:-5],words[3][:-2],words[4][:-4])

#3 We saw how we can generate an IndexError by indexing beyond the end of a string. Is it possible to construct
# an index that goes too far to the left, before the start of the string?
#yes it is, it wraps around

#4 We can specify a "step" size for the slice. The following returns every second character within the slice: monty[6:11:2]. It also works in the reverse direction: monty[10:5:-2] Try these for yourself, then experiment with different step values.
monty="We can specify a 'step' size for the slice."
print(monty[0:len(monty):3])
print(monty[7:len(monty):-2])

#5 What happens if you ask the interpreter to evaluate monty[::-1]? Explain why this is a reasonable result.
print(monty[::-1])
#reverses the string completely

#6 Describe the class of strings matched by the following regular expressions.
# [a-zA-Z]+ - one or more instances of a-z or A-Z
# [A-Z][a-z]* - one of A-Z followed by zero or more instances of a-z
# p[aeiou]{,2}t - p followed by 2 repeats of [aeiou] followed by t
# \d+(\.\d+)? - one or more decimal digit followed by . followed by one or more decimal digits
# ([^aeiou][aeiou][^aeiou])* - zero or more instances of does not start with aeiou followed by aeiou followed by not aeiou
# \w+|[^\w\s]+ - one or more instance of word or does not start with word then any whitespace character
# Test your answers using nltk.re_show().
print(nltk.re_show(r'[a-zA-Z]+',"This matches all the words in a string not numbers 0998"))
print(nltk.re_show(r'[A-Z][a-z]*',"This matches Title Case stuff"))
print(nltk.re_show(r'p[aeiou]{,2}t',"Matches pout peit not parrot post paaaat"))
print(nltk.re_show(r'\d+(\.\d+)?',"Match 98.3232 .98 879 f9889"))
print(nltk.re_show(r'([^aeiou][aeiou][^aeiou])*',"This will match"))
print(nltk.re_show(r'([^aeiou][aeiou][^aeiou])+',"This will match"))
print(nltk.re_show(r'\w+|[^\w\s]+',"98 90734 23 this matches what"))

#7 Write regular expressions to match the following classes of strings:
# A single determiner (assume that a, an, and the are the only determiners).
# An arithmetic expression using integers, addition, and multiplication, such as 2*3+8.
test_string = "This is a ball. This is the tree. Lets eat an apple"
print(nltk.re_show(r'(a\s|an\s|the\s)',test_string))
test_exp="2+3 2*4-6 2+8-1 6+7-8-9*5"
print(nltk.re_show(r'([0-9][\+\-\*]|[0-9])*',test_exp))

#8 Write a utility function that takes a URL as its argument, and returns the contents of the URL, with all HTML markup removed. Use from urllib import request and then request.urlopen('http://nltk.org/').read().decode('utf8') to access the contents of the URL.
url="http://nltk.org"
response=request.urlopen(url)
raw=response.read().decode('utf8')
print(raw)
print(re.search(r'(<.*?>|<\/.*?>)',raw)) # ? makes * non greedy
print(re.findall(r'(<.*?>|<\/.*?>)',raw)) # ? makes * non greedy
# (<.*?>|<\/.*?>)
# (<.*?>|<\/.*?>)(?s)
#re.sub
#9 Save some text into a file corpus.txt. Define a function load(f) that reads from the file named in its sole argument, and returns a string containing the text of the file.
# Use nltk.regexp_tokenize() to create a tokenizer that tokenizes the various kinds of punctuation in this text. Use one multi-line regular expression, with inline comments, using the verbose flag (?x).
# Use nltk.regexp_tokenize() to create a tokenizer that tokenizes the following kinds of expression: monetary amounts; dates; names of people and organizations.
def load(f):
    # with open(f,'r') as text:
    #     text_data=text.read()
    text_data=f
    all_punctations_set = set(nltk.regexp_tokenize(text_data,r'''(?x)    # set flag to allow verbose regexps
                                                          [\.,;:!?\'"-] #captures this group of punctuation marks
                                                          |[\[\]\(\)\{\}\<\>] #captures all types of brackets'''))
    print(all_punctations_set)
    # print(text_data)
    tokenized_text = nltk.regexp_tokenize(text_data,r'''(?x) # set flag to allow verbose regexps
                             \$?\d+(\.\d+)?%? #find monetary values
                            |[0-9]{4} #find just yyyy
                            |[A-Z][a-z]* [A-Z][a-z]* #find names by check 2 consecutive words are title case''')
    print(tokenized_text)

load(' '.join(nltk.corpus.brown.words()))
#10 Rewrite the following loop as a list comprehension:
# >>> sent = ['The', 'dog', 'gave', 'John', 'the', 'newspaper']
# >>> result = []
# >>> for word in sent:
# ...     word_len = (word, len(word))
# ...     result.append(word_len)
# >>> result
# [('The', 3), ('dog', 3), ('gave', 4), ('John', 4), ('the', 3), ('newspaper', 9)]
sent = ['The', 'dog', 'gave', 'John', 'the', 'newspaper']
result =[(word,len(word)) for word in sent]
print(result)

#11 Define a string raw containing a sentence of your own choosing. Now, split raw on some character other than space, such as 's'.
raw="This is a sample string"
print(raw.split('s'))

#12 Write a for loop to print out the characters of a string, one per line.
for ch in raw:
    print(ch,end='\n')

#13 What is the difference between calling split on a string with no argument or with ' ' as the argument, e.g. sent.split() versus sent.split(' ')? What happens when the string being split contains tab characters, consecutive space characters, or a sequence of tabs and spaces? (In IDLE you will need to use '\t' to enter a tab character.)
# calling split on a string with no argument or with ' ' - splits by space by default
# sent.split(' ') - splits by space
# string being split contains tab characters, consecutive space characters - still splits by space, list also contains elements that are tab characters, if consec spcaes then multiple '' are in the list

#14 Create a variable words containing a list of words. Experiment with words.sort() and sorted(words). What is the difference?
raw_words=nltk.corpus.brown.sents()[0]
print(raw_words)
raw_words.sort()
print(raw_words)
raw_words=nltk.corpus.brown.sents()[0]
print(sorted(raw_words))
# .sort sorts the same list, sorted creates a new list

#15 Explore the difference between strings and integers by typing the following at a Python prompt: "3" * 7 and 3 * 7. Try converting between strings and integers using int("3") and str(3).
print("5"*10)
print(5*10)
print(int("5")*10)
print(5*str(10))

#16 Use a text editor to create a file called prog.py containing the single line monty = 'Monty Python'. Next, start up a new session with the Python interpreter, and enter the expression monty at the prompt. You will get an error from the interpreter. Now, try the following (note that you have to leave off the .py part of the filename):
# >>> from prog import monty
# >>> monty
# This time, Python should return with a value. You can also try import prog, in which case Python should be able to evaluate the expression prog.monty at the prompt.
from ch3_ex_16 import monty
print(monty)

#17 What happens when the formatting strings %6s and %-6s are used to display strings that are longer than six characters?
print("Number is : %6s" % 'thisisaword')
print("Number is : %-6s" % 'thisisaword')
print("Number is : %6s" % 'aword')
print("Number is : %-6s" % 'aword')
#both prints successfully printed all the charaters irrespective of len of string

#18 Read in some text from a corpus, tokenize it, and print the list of all wh-word types that occur. (wh-words in English are used in questions, relative clauses and exclamations: who, which, what, and so on.) Print them in order. Are any words duplicated in this list, because of the presence of case distinctions or punctuation?
wh_words=[word for word in nltk.corpus.brown.words() if word.lower()[:2]=='wh' and len(word)<5]
print(sorted(set(wh_words)))
#yes there are duplicates due to difference in case and also other punctuations such as What's

#19 Create a file consisting of words and (made up) frequencies, where each line consists of a word, the space character, and a positive integer, e.g. fuzzy 53. Read the file into a Python list using open(filename).readlines(). Next, break each line into its two fields using split(), and convert the number into an integer using int(). The result should be a list of the form: [['fuzzy', 53], ...].
result=[]
with open('ch3_ex_19.txt','r') as f:
    for line in f.readlines():
        # print(line)
        line=line.split()
        # print(line)
        result.append([str(line[0]),int(line[1])])

print(result)

#20 Write code to access a favorite webpage and extract some text from it. For example, access a weather site and extract the forecast top temperature for your town or city today.
url="https://jobs.uncc.edu/postings/search?&query=&query_v0_posted_at_date=&query_organizational_tier_2_id=any&1976=&2074=&2075=7&commit=Search"
response=request.urlopen(url)
raw=response.read().decode('utf8')
new_jobs = [job[20:len(job)-2] for job in re.findall(r"data\-posting\-title=\".*",raw)]
print(new_jobs)
#extract jobs from jobs.uncc.edu

#21 Write a function unknown() that takes a URL as its argument, and returns a list of unknown words that occur on that webpage. In order to do this, extract all substrings consisting of lowercase letters (using re.findall()) and remove any items from this set that occur in the Words Corpus (nltk.corpus.words). Try to categorize these words manually and discuss your findings.
def unknown(url):
    response=request.urlopen(url)
    raw=response.read().decode('utf8')
    words = [word.lower() for word in re.sub(r'(<.*?>|<\/.*?>)(?s)', '', raw).split() if word.isalpha()]
    words_fd =nltk.FreqDist(words)
    # print(words_fd.most_common(10))
    nltk_words = nltk.corpus.words.words()
    unknown_words = set([word for word in words if word not in nltk_words])
    print(unknown_words)
#result consists of plurals, verb ending in ing, clubbed words like toolkit, treebank, person names
unknown('http://www.nltk.org/book/ch03.html')
# unknown('http://www.nltk.org/book/ch07.html')
# unknown('https://jobs.uncc.edu/postings/search?&query=&query_v0_posted_at_date=&query_organizational_tier_2_id=any&1976=&2074=&2075=7&commit=Search"')

#22 Examine the results of processing the URL http://news.bbc.co.uk/ using the regular expressions suggested above. You will see that there is still a fair amount of non-textual data there, particularly Javascript commands. You may also find that sentence breaks have not been properly preserved. Define further regular expressions that improve the extraction of text from this web page.
response = request.urlopen('http://news.bbc.co.uk/')
raw = response.read().decode('utf8')
print(re.sub(r'(<.*?>|<\/.*?>)(?s)', '', raw))

#23 Are you able to write a regular expression to tokenize text in such a way that the word don't is tokenized into do and n't? Explain why this regular expression won't work: «n't|\w+».

#24 Try to write code to convert text into hAck3r, using regular expressions and substitution, where e → 3, i → 1, o → 0, l → |, s → 5, . → 5w33t!, ate → 8. Normalize the text to lowercase before converting it. Add more substitutions of your own. Now try to map s to two different values: $ for word-initial s, and 5 for word-internal s.

#25 Pig Latin is a simple transformation of English text. Each word of the text is converted as follows: move any consonant (or consonant cluster) that appears at the start of the word to the end, then append ay, e.g. string → ingstray, idle → idleay. http://en.wikipedia.org/wiki/Pig_Latin
# Write a function to convert a word to Pig Latin.
# Write code that converts text, instead of individual words.
# Extend it further to preserve capitalization, to keep qu together (i.e. so that quiet becomes ietquay), and to detect when y is used as a consonant (e.g. yellow) vs a vowel (e.g. style).

#26 Download some text from a language that has vowel harmony (e.g. Hungarian), extract the vowel sequences of words, and create a vowel bigram table.

#27 Python's random module includes a function choice() which randomly chooses an item from a sequence, e.g. choice("aehh ") will produce one of four possible characters, with the letter h being twice as frequent as the others. Write a generator expression that produces a sequence of 500 randomly chosen letters drawn from the string "aehh ", and put this expression inside a call to the ''.join() function, to concatenate them into one long string. You should get a result that looks like uncontrolled sneezing or maniacal laughter: he haha ee heheeh eha. Use split() and join() again to normalize the whitespace in this string.

#28 Consider the numeric expressions in the following sentence from the MedLine Corpus: The corresponding free cortisol fractions in these sera were 4.53 +/- 0.15% and 8.16 +/- 0.23%, respectively. Should we say that the numeric expression 4.53 +/- 0.15% is three words? Or should we say that it's a single compound word? Or should we say that it is actually nine words, since it's read "four point five three, plus or minus zero point fifteen percent"? Or should we say that it's not a "real" word at all, since it wouldn't appear in any dictionary? Discuss these different possibilities. Can you think of application domains that motivate at least two of these answers?

#29 Readability measures are used to score the reading difficulty of a text, for the purposes of selecting texts of appropriate difficulty for language learners. Let us define μw to be the average number of letters per word, and μs to be the average number of words per sentence, in a given text. The Automated Readability Index (ARI) of the text is defined to be: 4.71 μw + 0.5 μs - 21.43. Compute the ARI score for various sections of the Brown Corpus, including section f (lore) and j (learned). Make use of the fact that nltk.corpus.brown.words() produces a sequence of words, while nltk.corpus.brown.sents() produces a sequence of sentences.

#30 Use the Porter Stemmer to normalize some tokenized text, calling the stemmer on each word. Do the same thing with the Lancaster Stemmer and see if you observe any differences.

#31 Define the variable saying to contain the list ['After', 'all', 'is', 'said', 'and', 'done', ',', 'more', 'is', 'said', 'than', 'done', '.']. Process this list using a for loop, and store the length of each word in a new list lengths. Hint: begin by assigning the empty list to lengths, using lengths = []. Then each time through the loop, use append() to add another length value to the list. Now do the same thing using a list comprehension.

#32 Define a variable silly to contain the string: 'newly formed bland ideas are inexpressible in an infuriating way'. (This happens to be the legitimate interpretation that bilingual English-Spanish speakers can assign to Chomsky's famous nonsense phrase, colorless green ideas sleep furiously according to Wikipedia). Now write code to perform the following tasks:
# Split silly into a list of strings, one per word, using Python's split() operation, and save this to a variable called bland.
# Extract the second letter of each word in silly and join them into a string, to get 'eoldrnnnna'.
# Combine the words in bland back into a single string, using join(). Make sure the words in the resulting string are separated with whitespace.
# Print the words of silly in alphabetical order, one per line.

#33 The index() function can be used to look up items in sequences. For example, 'inexpressible'.index('e') tells us the index of the first position of the letter e.
# What happens when you look up a substring, e.g. 'inexpressible'.index('re')?
# Define a variable words containing a list of words. Now use words.index() to look up the position of an individual word.
# Define a variable silly as in the exercise above. Use the index() function in combination with list slicing to build a list phrase consisting of all the words up to (but not including) in in silly.

#34 Write code to convert nationality adjectives like Canadian and Australian to their corresponding nouns Canada and Australia (see http://en.wikipedia.org/wiki/List_of_adjectival_forms_of_place_names).

#35 Read the LanguageLog post on phrases of the form as best as p can and as best p can, where p is a pronoun. Investigate this phenomenon with the help of a corpus and the findall() method for searching tokenized text described in 3.5. http://itre.cis.upenn.edu/~myl/languagelog/archives/002733.html

#36 Study the lolcat version of the book of Genesis, accessible as nltk.corpus.genesis.words('lolcat.txt'), and the rules for converting text into lolspeak at http://www.lolcatbible.com/index.php?title=How_to_speak_lolcat. Define regular expressions to convert English words into corresponding lolspeak words.

#37 Read about the re.sub() function for string substitution using regular expressions, using help(re.sub) and by consulting the further readings for this chapter. Use re.sub in writing code to remove HTML tags from an HTML file, and to normalize whitespace.

#38 An interesting challenge for tokenization is words that have been split across a line-break. E.g. if long-term is split, then we have the string long-\nterm.
# Write a regular expression that identifies words that are hyphenated at a line-break. The expression will need to include the \n character.
# Use re.sub() to remove the \n character from these words.
# How might you identify words that should not remain hyphenated once the newline is removed, e.g. 'encyclo-\npedia'?x

#39 Read the Wikipedia entry on Soundex. Implement this algorithm in Python.

#40 Obtain raw texts from two or more genres and compute their respective reading difficulty scores as in the earlier exercise on reading difficulty. E.g. compare ABC Rural News and ABC Science News (nltk.corpus.abc). Use Punkt to perform sentence segmentation.

#41 Rewrite the following nested loop as a nested list comprehension:
# >>> words = ['attribution', 'confabulation', 'elocution',
# ...          'sequoia', 'tenacious', 'unidirectional']
# >>> vsequences = set()
# >>> for word in words:
# ...     vowels = []
# ...     for char in word:
# ...         if char in 'aeiou':
# ...             vowels.append(char)
# ...     vsequences.add(''.join(vowels))
# >>> sorted(vsequences)
# ['aiuio', 'eaiou', 'eouio', 'euoia', 'oauaio', 'uiieioa']

#42 Use WordNet to create a semantic index for a text collection. Extend the concordance search program in 3.6, indexing each word using the offset of its first synset, e.g. wn.synsets('dog')[0].offset (and optionally the offset of some of its ancestors in the hypernym hierarchy).

#43 With the help of a multilingual corpus such as the Universal Declaration of Human Rights Corpus (nltk.corpus.udhr), and NLTK's frequency distribution and rank correlation functionality (nltk.FreqDist, nltk.spearman_correlation), develop a system that guesses the language of a previously unseen text. For simplicity, work with a single character encoding and just a few languages.

#44 Write a program that processes a text and discovers cases where a word has been used with a novel sense. For each word, compute the WordNet similarity between all synsets of the word and all synsets of the words in its context. (Note that this is a crude approach; doing it well is a difficult, open research problem.)

#45 Read the article on normalization of non-standard words (Sproat et al, 2001), and implement a similar system for text normalization.
