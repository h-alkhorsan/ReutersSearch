import nltk
import re 
from nltk import word_tokenize
from nltk.corpus import reuters, stopwords 
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict

class Corpus:
    def __init__(self, corpus, download=False):
        if download:
            # required 
            nltk.download('reuters')
            nltk.download('stopwords')
            nltk.download('punkt')
            nltk.download('wordnet')
            nltk.download('omw-1.4')

        self.stop_words = stopwords.words('english')
        self.corpus = corpus 
        self.documents = corpus.fileids()
        self.categories = corpus.categories()

    def summary(self):
        n_documents = 'Number of documents: ' + str(len(self.documents)) 
        n_categories = 'Number of categories: ' + str(len(self.categories))
        print(f'Corpus summary:\n{n_documents}\n{n_categories}') 
                
    def _tokenize(self, text, min_length=3):

        words = map(lambda word: word.lower(), word_tokenize(text)) # lower case words
        words = [word for word in words if word not in self.stop_words] # remove stop words

        #tokens = (list(map(lambda token: PorterStemmer().stem(token), words))) # apply stemming (much faster)
        tokens = (list(map(lambda token: WordNetLemmatizer().lemmatize(token), words))) # apply lemmatization (much more accurate)
        regex = re.compile('[a-zA-Z]+') # regex for removing punctuation
        
        # filter the tokens. 1. Remove punctuation 2. Remove words with length less than min_length
        filtered_tokens = list(filter(lambda token: regex.match(token) and len(token) >= min_length, tokens))

        # convert back into list
        return [*filtered_tokens, ]

    def get_raw_document(self, document_id):
        return self.corpus.raw(document_id)
    
    def get_word_count_in_doc(self, document_id):
        return len(self.corpus.words(document_id))

    def get_docs_in_category(self, category):
        return self.corpus.fileids(category)

    def doc_dictionary(self):
        doc_collection = defaultdict(list)
        for category in self.categories:
            for doc in self.get_docs_in_category(category):
                doc_collection[category].append(doc)
        return doc_collection

    def preprocess_docs(self):
        processed_docs = dict()
        for doc in self.documents:
            raw_doc = self.get_raw_document(doc)
            tokenized_doc = self._tokenize(raw_doc)
            processed_docs[doc] = tokenized_doc
        return processed_docs

#corpus = Corpus(reuters, download=False) # use download=True first time running the script
#corpus.summary() # print summary of corpus
#collection = corpus.doc_dictionary() # get dictionary of documents where key=category, value=document_index
#words = corpus.get_words_in_document(collection['dmk'][0]) # get number of words in first document in category 'dmk'
#raw = corpus.get_raw_document(collection['dmk'][0]) # display the raw text of first document in category 'dmk'
#print(corpus.stop_words) # get all the stop words
#print(corpus.collection['dmk'][0]) #index the first doc in category 'dmk'


