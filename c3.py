import os
from nltk.tokenize import RegexpTokenizer as ret
from nltk.corpus import stopwords as sw
import string
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import words

tokenizer = ret(r'\w+')
stemmer = SnowballStemmer('english')


both = {}
positive = {}
negative = {}
vocab=0 #number of words in our vocabulary
totpos=0
totneg=0 #total occurances of all positive and negative words


for name in os.listdir('pos'):
	fobj=open(os.path.join('pos',name),'r+')
	arr=fobj.read().lower().decode('utf-8')
	arr=tokenizer.tokenize(arr)
	swords = set(sw.words('english'))	#all english stopwords
	refined=[]	#array with refined words
	
	for word in arr:
		if not word in swords:
			if not word in string.punctuation:
				refined.append(stemmer.stem(word))
#above step removes all punctuations and stopwords, as well as stems them(snowball)

	for word in refined:
		totpos+=1
		if word in both:
			both[word]+=1
		else:
			both[word]=1
			vocab+=1
		if word in positive:
			positive[word]+=1
		else:
			positive[word]=1

#store all positive words and their number of occurances
#repeat the same for negative examples

print 'Trained for positive examples!'

for name in os.listdir('neg'):
	fobj=open(os.path.join('neg',name),'r+')
	arr=fobj.read().lower().decode('utf-8')
	arr=tokenizer.tokenize(arr)
	swords = set(sw.words('english'))
	refined=[]
	for word in arr:
		if not word in swords:
			if not word in string.punctuation:
				refined.append(stemmer.stem(word))


	for word in refined:
		totneg+=1
		if word in both:
			both[word]+=1
		else:
			both[word]=1
			vocab+=1
		if word in negative:
			negative[word]+=1
		else:
			negative[word]=1

print 'Trained for negative examples'



fobj2=open('both.txt','w')
fobj2.write('%s' %both)
fobj2.close()
fobj2=open('positive.txt','w')
fobj2.write('%s' %positive)
fobj2.close()
fobj2=open('negative.txt','w')
fobj2.write('%s' %negative)
fobj2.close()
fobj2=open('stats.txt','w')
fobj2.write('vocab = %d\ntotpos = %d\ntotneg = %d' %(vocab,totpos,totneg))


#arrays storing words and their probability of lying in + and - examples
posprob={}
negprob={}

for word in positive:
	posprob[word]=(float)(positive[word]+1)/(totpos+vocab)*1000
for word in negative:
	negprob[word]=(float)(negative[word]+1)/(vocab+totneg)*1000
#multiplication by 1000 just to ensure that probability is not ridiculously small

print 'Calculated posprob and negprob!'

correct=0
asked=0		#for the test cases, correct classification and total test samples


tp=0
tn=0
fp=0
fn=0 	#for precision vs recall



#+ve examples
for name in os.listdir('expneg'):
	fobj=open(os.path.join('expneg',name),'r+')
	arr=fobj.read().lower().decode('utf-8')
	arr=tokenizer.tokenize(arr)
	swords = set(sw.words('english'))
	refined=[]
	for word in arr:
		if not word in swords:
			if not word in string.punctuation:
				refined.append(stemmer.stem(word))
	
	pplus=1
	pminus=1		#these are probabilities of reviews being + or -
	for word in refined:
		if word in posprob:
			pplus*=posprob[word]
		else:
			pplus*=(float)(1)/(vocab + totpos)*1000
		if word in negprob:
			pminus*=negprob[word]
		else:
			pminus*=(float)(1)/(vocab + totneg)*1000
		asked+=1
		if pminus>pplus:
			correct+=1
			tn+=1
		else:
			fp+=1

print 'Tested over positive examples!'

#-ve examples
for name in os.listdir('exppos'):
	fobj=open(os.path.join('exppos',name),'r+')
	arr=fobj.read().lower().decode('utf-8')
	arr=tokenizer.tokenize(arr)
	swords = set(sw.words('english'))
	refined=[]
	for word in arr:
		if not word in swords:
			if not word in string.punctuation:
				refined.append(stemmer.stem(word))
	
	pplus=1
	pminus=1
	for word in refined:
		if word in posprob:
			pplus*=posprob[word]
		else:
			pplus*=(float)(1)/(vocab + totpos)*1000
		if word in negprob:
			pminus*=negprob[word]
		else:
			pminus*=(float)(1)/(vocab + totneg)*1000
		asked+=1
		if pminus<pplus:
			correct+=1
			tp+=1
		else:
			fn+=1

print 'Tested over negative examples!'

accuracy = (float)(correct)/(asked)
fscore = 2*((float)(tp)/(tp+fp))*((float)(tp)/(tp+fn))/(((float)(tp)/(tp+fp))+((float)(tp)/(tp+fn)))
print accuracy
print fscore
