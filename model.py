from locale import normalize
import math
import operator
from query import Query
from corpus import Corpus
from collections import defaultdict

class Document:
    def __init__(self, id=None, score=0, priority=0):
        self.id = id
        self.score = score 
        self.priority = priority

class Model:
    def __init__(self, query, documents):
        self.query = query 
        self.documents = documents
        self.priority_group = defaultdict(list)
  
    def _compute_score(self, query, document):
        term_count = 0

        for term in document:
            if term in query:
                term_count += 1

        weight_decay = float('0.' + '0' * (len(query)-3) + '1')

        doc_score = 0

        k = len(query)
        m = term_count

        for i in range(1, k+1):
            for j in range(1, m+1):
                doc_score += math.pow(1/j, j-1)
            doc_score *= i 
    
        doc_score = round(doc_score * term_count * weight_decay, 2)

        return doc_score 

    def _compute_priority(self, query, document):
        
        doc_priority = len(query)+1
        for term in query:
            if term in document:
                doc_priority -= 1

        return doc_priority

    def priority_assignment(self):
        for doc in self.documents:
            score = self._compute_score(self.query, self.documents[doc])
            priority = self._compute_priority(self.query, self.documents[doc])
            self.priority_group[priority].append(Document(doc, score, priority))


    def rank_scores(self):
        for group in self.priority_group:
            self.priority_group[group].sort(key=operator.attrgetter('score'), reverse=True)

    def normalize_scores(self, docs):
        scores = []

        for doc in docs:
            scores.append(doc.score)

        for score in scores:
            score -= min(scores) / (max(scores) - min(scores)) * 100
        return scores 



    def top_k(self, top_docs = [None] * 10, group_ptr=1):
        if None not in top_docs:
            return top_docs
 
        top_k_ptr = top_docs.index(None)
        get_doc = self.priority_group[group_ptr]

        if get_doc:
            top_docs[top_k_ptr] = get_doc.pop(0)
        else:
            group_ptr += 1

        return self.top_k(top_docs, group_ptr)


