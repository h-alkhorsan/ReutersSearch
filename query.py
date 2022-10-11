import nltk
import re
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

class Query:
    def __init__(self, query):
        self.query = query 
        self.stop_words = stopwords.words('english')

    def tokenize(self, min_length=3):
        words = map(lambda word: word.lower(), word_tokenize(self.query))
        words = [word for word in words if word not in self.stop_words]

        #tokens = (list(map(lambda token: PorterStemmer().stem(token), words))) 
        tokens = (list(map(lambda token: WordNetLemmatizer().lemmatize(token), words))) 

        regex = re.compile('[a-zA-Z]+')

        filtered_tokens = list(filter(lambda token: regex.match(token) and len(token) >= min_length, tokens))
        filtered_tokens = set(filtered_tokens)
        
        return [*filtered_tokens, ]
       
