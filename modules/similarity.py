import numpy as np

class Similarity:
    """
    Класс для вычисления сходства между векторами.
    
    Этот класс предоставляет методы для вычисления косинусного сходства и сходства на основе TF-IDF между двумя векторами.
    
    Методы:
        calc_cos_sim(self, a: np.array, b: np.array): Вычисляет косинусное сходство между двумя векторами.
        
        calc_tfidf_sim(self, a: np.array, b: np.array): Вычисляет сходство на основе TF-IDF между средними значениями двух векторов.
        
    Примеры использования:
        similarity = Similarity()
        vector_a = np.array([1, 2, 3])
        vector_b = np.array([4, 5, 6])
        
        cos_sim = similarity.calc_cos_sim(vector_a, vector_b)
        print(f"Косинусное сходство: {cos_sim}")
        
        tfidf_sim = similarity.calc_tfidf_sim(vector_a, vector_b)
        print(f"Сходство TF-IDF: {tfidf_sim}")
    """

    def calc_cos_sim(self, a: np.array, b: np.array):
        """
        Вычисляет косинусное сходство между двумя векторами.
        
        Параметры:
            a (np.array): Первый вектор.
            b (np.array): Второй вектор.
            
        Возвращает:
            float: Косинусное сходство между векторами a и b, округленное до двух знаков после запятой.
        """
        cos_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 0.000001)
        return np.round(cos_sim, 2)
    
    def calc_tfidf_sim(self, a: np.array, b: np.array):
        """
        Вычисляет сходство на основе TF-IDF между средними значениями двух векторов.
        
        Параметры:
            a (np.array): Первый вектор.
            b (np.array): Второй вектор.
            
        Возвращает:
            float: Сходство на основе TF-IDF между средними значениями векторов a и b, округленное до двух знаков после запятой.
            
        Примечание:
            Этот метод не является стандартным способом вычисления сходства на основе TF-IDF и представлен для демонстрации.
        """
        return np.round(b.mean() / a.mean(), 2)
