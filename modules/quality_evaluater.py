import numpy as np
from .cleaner import Cleaner
from .vectorizer import Vectorizer
from .similarity import Similarity

class QualityEvaluater:
    """
    Класс QualityEvaluater предназначен для оценки качества текста на основе различных метрик, таких как
    косинусное сходство, TF-IDF сходство и стандартизированное TF-IDF значение. Он использует предварительную
    обработку текста, векторизацию и вычисление сходства для оценки текстов.

    Атрибуты:
        cleaner (Cleaner): Экземпляр класса Cleaner для предварительной обработки текста.
        vectorizer (Vectorizer): Экземпляр класса Vectorizer для преобразования текста в векторное представление.
        similarity (Similarity): Экземпляр класса Similarity для вычисления метрик сходства между векторами.
        corpus (list[str]): Список строк, представляющий корпус текстов для анализа.
        cleaned_corpus (list[str]): Список предварительно обработанных строк корпуса.
        max_length (int): Максимальная длина текста в cleaned_corpus.

    Методы:
        get_cleaned_text(text: str) -> list:
            Возвращает очищенный нежелательных элементов текст.

        get_unnoised_text(text: str) -> list:
            Возвращает текст, очищенный от шаблонных фраз и шума.

        get_embeddings(text: str):
            Возвращает векторные представления для очищенного текста и текста, очищенного от шума.

        get_score_cos_sim(text: str) -> float:
            Возвращает оценку косинусного сходства между векторными представлениями очищенного текста и текста без шума.

        get_score_tfidf(text: str) -> float:
            Возвращает оценку сходства на основе TF-IDF между векторными представлениями очищенного текста и текста без шума.

        get_score_standarted_tfidf(text: str) -> float:
            Возвращает стандартизированное значение TF-IDF для векторного представления очищенного текста.

        get_length(text: str) -> int:
            Возвращает относительную длину очищенного текста по отношению к максимальной длине текста в корпусе.
    
    Пример использования:
        # Предполагается, что классы Cleaner, Vectorizer и Similarity уже реализованы.
        cleaner = Cleaner()
        vectorizer = Vectorizer()
        similarity = Similarity()
        corpus = ["Пример текста.", "Ещё один пример текста."]
        
        evaluater = QualityEvaluater(cleaner, vectorizer, similarity, corpus)
        
        text = "Текст для оценки."
        score_cos_sim = evaluater.get_score_cos_sim(text)
        print(f"Оценка косинусного сходства: {score_cos_sim}")
        
        score_tfidf = evaluater.get_score_tfidf(text)
        print(f"Оценка TF-IDF сходства: {score_tfidf}")
        
        score_standarted_tfidf = evaluater.get_score_standarted_tfidf(text)
        print(f"Стандартизированное значение TF-IDF: {score_standarted_tfidf}")
        
        length_ratio = evaluater.get_length(text)
        print(f"Относительная длина текста: {length_ratio}")
    """

    def __init__(self, 
                 cleaner: Cleaner, 
                 vectorizer: Vectorizer, 
                 similarity: Similarity, 
                 corpus: list[str]):
        """
        Инициализирует экземпляр класса QualityEvaluater.

        Аргументы:
            cleaner (Cleaner): Экземпляр класса для очистки текста.
            vectorizer (Vectorizer): Экземпляр класса для векторизации текста.
            similarity (Similarity): Экземпляр класса для вычисления мер сходства между текстами.
            corpus (list[str]): Список текстов, который будет использоваться как корпус для анализа.

        В процессе инициализации происходит предварительная обработка корпуса текстов, его векторизация
        и определение максимальной длины текста в корпусе.
        """

        self.cleaner = cleaner
        self.vectorizer = vectorizer
        self.similarity = similarity
        self.cleaned_corpus = [self.cleaner.clean_text(item) for item in corpus]
        self.vectorizer.corpus_fit_vectorize(self.cleaned_corpus)
        self.max_length = max([len(text) for text in self.cleaned_corpus])
    
    def get_cleaned_text(self, text: str) -> list:
        """
        Возвращает очищенный от шума и нежелательных элементов текст.

        Аргументы:
            text (str): Исходный текст для очистки.

        Возвращает:
            list: Очищенный текст в виде списка токенов.
        """

        cleaned_text = self.cleaner.clean_text(text)
        return cleaned_text
    
    def get_unnoised_text(self, text: str) -> list:
        """
        Возвращает текст, очищенный от шаблонных фраз и шума.

        Аргументы:
            text (str): Исходный текст для очистки.

        Возвращает:
            list: Текст после удаления шаблонных фраз и шума в виде списка токенов.
        """

        unnoised_text = self.cleaner.remove_noise_boilerplate(vectorizer=self.vectorizer, input_text=text)
        return unnoised_text
    
    def get_embeddings(self, text: str):
        """
        Возвращает векторные представления для очищенного текста и текста, очищенного от шума.

        Аргументы:
            text (str): Исходный текст для получения векторных представлений.

        Возвращает:
            tuple: Кортеж из двух numpy массивов, представляющих векторные представления очищенного текста и текста без шума.
        """

        cleaned_text = self.get_cleaned_text(text)
        unnoised_text = self.get_unnoised_text(text)
        cleand_unnoised_text = self.get_cleaned_text(unnoised_text)
        
        vector_cleaned_text = self.vectorizer.corpus_transform_vectorize([cleaned_text])
        vector_cleand_unnoised_text = self.vectorizer.corpus_transform_vectorize([cleand_unnoised_text])
        
        a = np.array(vector_cleaned_text.todense())[0]
        b = np.array(vector_cleand_unnoised_text.todense())[0]
        
        return a, b
    
    def get_score_cos_sim(self, text: str) -> float:
        """
        Возвращает оценку косинусного сходства между векторными представлениями очищенного текста и текста без шума.

        Аргументы:
            text (str): Исходный текст для оценки.

        Возвращает:
            float: Значение косинусного сходства между двумя векторными представлениями текста.
        """

        a, b = self.get_embeddings(text)
        return self.similarity.calc_cos_sim(a, b)
    
    def get_score_tfidf(self, text: str) -> float:
        """
        Возвращает оценку сходства на основе TF-IDF между векторными представлениями очищенного текста и текста без шума.

        Аргументы:
            text (str): Исходный текст для оценки.

        Возвращает:
            float: Значение сходства на основе TF-IDF между двумя векторными представлениями текста.
        """

        a, b = self.get_embeddings(text)
        return self.similarity.calc_tfidf_sim(a, b)
    
    def get_score_standarted_tfidf(self, text: str) -> float:
        """
        Возвращает стандартизированное значение TF-IDF для векторного представления очищенного текста.

        Аргументы:
            text (str): Исходный текст для оценки.

        Возвращает:
            float: Стандартизированное значение TF-IDF для векторного представления текста.
        """
        cleaned_text = self.get_cleaned_text(text)
        vector_cleaned_text = self.vectorizer.corpus_transform_vectorize([cleaned_text])
        vector_cleaned_text = np.array(vector_cleaned_text.todense())[0]
        return vector_cleaned_text.mean()/vector_cleaned_text.max()
    
    def get_length(self, text: str) -> int:
        """
        Возвращает относительную длину очищенного текста по отношению к максимальной длине текста в корпусе.

        Аргументы:
            text (str): Исходный текст для определения его длины.

        Возвращает:
            int: Относительная длина очищенного текста.
        """

        cleaned_text = self.get_cleaned_text(text)
        return len(cleaned_text)/self.max_length