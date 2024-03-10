from modules.cleaner import Cleaner
from modules.quality_evaluater import QualityEvaluater
from modules.similarity import Similarity
from modules.vectorizer import Vectorizer
from modules.geval import GEval
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

def main():
    k = 10
    train_corpus = list(pd.read_csv('train_corpus.csv')['review/text'])
    test_corpus = list(pd.read_csv('test_corpus.csv')['review/text'])[:k]
    
    model_name = 'mistralai/Mistral-7B-Instruct-v0.2'
    geval_model = GEval(model_name=model_name)

    cleaner = Cleaner()
    vectorizer = Vectorizer()
    similarity = Similarity()
    
    quality_evaluater = QualityEvaluater(cleaner=cleaner, 
                                         vectorizer=vectorizer, 
                                         similarity=similarity, 
                                         corpus=train_corpus)
    
    dct_score = {'cos_sim': [], 'tfidf': [], 'tfidf_standart': [], 'length': [], 'geval': [], 'index': []}

    for i in tqdm(range(len(test_corpus))):
        text = test_corpus[i]
        cos_sim_value = quality_evaluater.get_score_cos_sim(text)
        tfidf_value = quality_evaluater.get_score_tfidf(text)
        tfidf_standart_value = quality_evaluater.get_score_standarted_tfidf(text)
        length_value = quality_evaluater.get_length(text)
        geval_value = geval_model.predict(review=text)

        dct_score['cos_sim'].append(cos_sim_value)
        dct_score['tfidf'].append(tfidf_value)
        dct_score['tfidf_standart'].append(tfidf_standart_value)
        dct_score['length'].append(length_value)
        dct_score['geval'].append(geval_value)
        dct_score['index'].append(i)
    print(dct_score)
    return dct_score
    
if __name__ == "__main__":
    main()