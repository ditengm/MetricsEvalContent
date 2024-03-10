from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

class Vectorizer:
    """
    Класс для векторизации текстовых данных с использованием метода TF-IDF.
    
    Этот класс предоставляет функциональность для векторизации как отдельных текстов, так и корпуса документов.
    Основан на использовании TfidfVectorizer из библиотеки scikit-learn.
    
    Атрибуты:
        sentence_vectorizer (TfidfVectorizer): Векторизатор для обработки отдельных текстов.
        corpus_vectorizer (TfidfVectorizer): Векторизатор для обработки корпуса документов.
        
    Методы:
        __init__(self): Инициализирует экземпляр класса, создавая два экземпляра TfidfVectorizer.
        
        corpus_fit_vectorize(self, corpus: list[str]): Производит подготовку векторизатора корпуса к работе с переданным корпусом документов.
        
        corpus_transform_vectorize(self, input_text: list[str]): Преобразует предоставленные тексты в разреженную матрицу признаков TF-IDF, используя ранее подготовленный векторизатор корпуса.
        
        text_vectorize(self, input_text: list[str]): Векторизует отдельные тексты или предложения, преобразуя их в матрицу признаков TF-IDF с помощью векторизатора для отдельных текстов. Подходит для разовых задач векторизации текста, где процесс подготовки не отделён от преобразования.
    """
    def __init__(self):
        """
        Инициализация векторизатора.
        Создает два экземпляра TfidfVectorizer: один для работы с отдельными текстами, другой — для работы с корпусом документов.
        """
        self.sentence_vectorizer = TfidfVectorizer()
        self.corpus_vectorizer = TfidfVectorizer()
    
    def corpus_fit_vectorize(self, corpus: list[str]):
        """
        Подготавливает векторизатор корпуса к работе с переданным корпусом документов.
        
        Параметры:
            corpus (list[str]): Список строк, где каждая строка представляет собой документ в корпусе.
            
        Возвращает:
            None
        """
        self.corpus_vectorizer.fit(corpus)
        
    def corpus_transform_vectorize(self, input_text: list[str]):
        """
        Преобразует предоставленные тексты в разреженную матрицу признаков TF-IDF.
        
        Параметры:
            input_text (list[str]): Список строк для векторизации. Это могут быть новые документы, не участвующие в процессе подготовки.
            
        Возвращает:
            scipy.sparse.csr.csr_matrix: Разреженная матрица признаков TF-IDF предоставленных текстов.
        """
        counts_matrix = self.corpus_vectorizer.transform(input_text)
        return counts_matrix
        
    def text_vectorize(self, input_text: list[str]) -> np.array:
        """
        Векторизует отдельные тексты или предложения.
        
        Параметры:
            input_text (list[str]): Список строк, где каждая строка - это текст или предложение для векторизации.
            
        Возвращает:
            np.ndarray: Массив numpy, содержащий плотное представление признаков TF-IDF входных текстов.
        """
        # Использование vectorizer.fit_transform для преобразования текста
        counts_matrix = self.sentence_vectorizer.fit_transform(input_text)
        
        # Преобразование полученной матрицы в плотную матрицу
        dense_matrix = counts_matrix.todense()
        
        # Возврат плотной матрицы в виде массива numpy 
        return np.array(dense_matrix)