import nltk
import time
from nltk.corpus import reuters, stopwords
from corpus import Corpus
from query import Query 
from model import Model, Document


def main():
    corpus = Corpus(reuters)
    query = input("Search: ")
    
    start = time.time()
    
    docs = corpus.preprocess_docs()

    q = Query(query=query)
    q = q.tokenize()

    model = Model(q, docs)
    model.priority_assignment()
    model.rank_scores()
    top_docs = model.top_k()
    scores = model.normalize_scores(top_docs)

    end = time.time()

    print(f'\nTIME TAKEN: {round(end - start, 2)} seconds')
    print('DOC ID\t\tSCORE\tPRIORITY')
    for i in range(10):
        print(f'{top_docs[i].id}\t{scores[i]}\t{top_docs[i].priority}')

    print('\n')
    print('TOP 10 DOCUMENTS:\n')

    for doc in top_docs:
        print(corpus.get_raw_document(doc.id)[:30] + '...\n')
        
       # fname = doc.id.replace('/', '')
       # with open(f'{fname}.txt', 'w') as f:
       #     f.write(corpus.get_raw_document(doc.id))

if __name__ == '__main__':
    main()


