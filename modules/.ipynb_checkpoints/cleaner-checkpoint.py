import re
import unicodedata
import inflect
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import contractions
import emoji
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import sent_tokenize
import numpy as np
from .vectorizer import Vectorizer


class Cleaner:
    """
    Класс Cleaner предназначен для предварительной обработки текста перед его анализом.
    Он включает в себя функции для удаления HTML-тегов, URL, эмоджи, специальных символов,
    преобразования чисел в слова, удаления стоп-слов и знаков препинания, а также лемматизации.
    
    Методы:
        clean_text(self, input_text): Очищает входной текст от различных нежелательных элементов и приводит его к стандартному виду.
        
        emojis_words(self, text): Преобразует эмоджи в слова, используя их текстовое описание.
        
        remove_noise_boilerplate(self, vectorizer, input_text, min_cluster_size=2, num_clusters=3, max_noise_ratio=0.3):
            Функция для удаления шума и шаблонных фраз из текста. Параметры vectorizer, min_cluster_size, num_clusters и max_noise_ratio
            используются для настройки процесса очистки. (Функция не реализована в данном примере)
    
    Пример использования:
        cleaner = Cleaner()
        raw_text = "Some raw text with HTML <html>...</html>, URLs http://example.com, and emojis 😊."
        clean_text = cleaner.clean_text(raw_text)
        print(clean_text)
    """

    def __init__(self):
        """
        Инициализирует экземпляр класса Cleaner, загружая необходимые ресурсы для лемматизации и токенизации.
        """

        self.model_lemmatizer = WordNetLemmatizer()
        self.tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        
    # Функция для очистки текста
    def clean_text(self, input_text):  
        """
        Очищает входной текст от HTML-тегов, URL, эмоджи, специальных символов,
        преобразует числа в слова, удаляет стоп-слова и знаки препинания, и проводит лемматизацию.

        Параметры:
            input_text (str): Входной текст для очистки.

        Возвращает:
            str: Очищенный текст.
        """
        
        # HTML-теги: первый шаг - удалить из входного текста все HTML-теги
        clean_text = re.sub('<[^<]+?>', '', input_text)

        # URL и ссылки: далее - удаляем из текста все URL и ссылки
        clean_text = re.sub(r'http\S+', '', clean_text)

        # Эмоджи и эмотиконы: используем собственную функцию для преобразования эмоджи в текст
        # Важно понимать эмоциональную окраску обрабатываемого текста
        clean_text = self.emojis_words(clean_text)

        # Приводим все входные данные к нижнему регистру
        clean_text = clean_text.lower()

        # Убираем все пробелы
        # Так как все данные теперь представлены словами - удалим пробелы
        clean_text = re.sub('\s+', ' ', clean_text)

        # Преобразование символов с диакритическими знаками к ASCII-символам: используем функцию normalize из модуля unicodedata и преобразуем символы с диакритическими знаками к ASCII-символам
        clean_text = unicodedata.normalize('NFKD', clean_text).encode('ascii', 'ignore').decode('utf-8', 'ignore')

        # Разворачиваем сокращения: текст часто содержит конструкции вроде "don't" или "won't", поэтому развернём подобные сокращения
        clean_text = contractions.fix(clean_text)

        # Убираем специальные символы: избавляемся от всего, что не является "словами"
        clean_text = re.sub('[^a-zA-Z0-9\s]', '', clean_text)

        # Записываем числа прописью: 100 превращается в "сто" (для компьютера)
        temp = inflect.engine()
        words = []
        for word in clean_text.split():
            if word.isdigit():
                words.append(temp.number_to_words(word))
            else:
                words.append(word)
        clean_text = ' '.join(words)

        # Стоп-слова: удаление стоп-слов - это стандартная практика очистки текстов
        stop_words = set(stopwords.words('english'))
        tokens = word_tokenize(clean_text)
        tokens = [token for token in tokens if token not in stop_words]
        clean_text = ' '.join(tokens)

        # Знаки препинания: далее - удаляем из текста все знаки препинания
        clean_text = re.sub(r'[^\w\s]', '', clean_text)
        
        # лемматизируем каждое слово
        clean_text = ' '.join([self.model_lemmatizer.lemmatize(w) for w in clean_text.split(' ')])
        # И наконец - возвращаем очищенный текст
        return clean_text

    # Функция для преобразования эмоджи в слова
    def emojis_words(self, text):
        """
        Преобразует эмоджи в тексте в их словесные описания.

        Параметры:
            text (str): Текст для преобразования эмоджи.

        Возвращает:
            str: Текст с преобразованными эмоджи.
        """
        # Модуль emoji: преобразование эмоджи в их словесные описания
        clean_text = emoji.demojize(text, delimiters=(" ", " "))

        # Редактирование текста путём замены ":" и" _", а так же - путём добавления пробела между отдельными словами
        clean_text = clean_text.replace(":", "").replace("_", " ")

        return clean_text
    
    def remove_noise_boilerplate(self, 
                                 vectorizer: Vectorizer, 
                                 input_text: str, 
                                 min_cluster_size: int = 2, 
                                 num_clusters: int = 3, 
                                 max_noise_ratio: float = 0.3):
        """
        Функция для удаления шума и шаблонных фраз из текста. Данная функция требует дополнительной реализации.

        Параметры:
            vectorizer: Векторизатор для преобразования текста в числовые векторы.
            input_text (str): Входной текст для очистки.
            min_cluster_size (int): Минимальный размер кластера для анализа.
            num_clusters (int): Количество кластеров для анализа.
            max_noise_ratio (float): Максимально допустимое соотношение шума.

        Возвращает:
            Тип возвращаемого значения не указан из-за отсутствия реализации.
        """

        # Разбиение текста на предложения: для идентификации шаблонных фрагментов или "шума" сначала надо выделить из текста предложения, которые мы будем сравнивать друг с другом
        sentences = self.tokenizer.tokenize(input_text)
        
        # для маленьких текстов указываю num_clusters, которая равняется кол-ву предложений
        if len(sentences) <= num_clusters:
            num_clusters = len(sentences)
        # для средних текстов указываю num_clusters, которая равняется трём
        elif len(sentences) < 12:
            num_clusters = 3
        # для средних текстов указываю num_clusters, которая равняется кол-ву предложений делённых на 4
        else:
            num_clusters = int(len(sentences)/4)
            max_noise_ratio += 0.1
        
        # Преобразование предложений в матрицу словесных эмбеддингов
        embeddings_matrix = vectorizer.text_vectorize(sentences)

        # KMean-кластеризация: кластеризация предложений, позволяющая разместить похожие эмбеддинги поблизости друг от друга 
        kmeans_model = KMeans(n_clusters=num_clusters)
        kmeans_model.fit(embeddings_matrix)
        model_labels = kmeans_model.labels_
        model_centroids = kmeans_model.cluster_centers_
        cluster_sizes = np.bincount(model_labels)

        # Идентификация кластеров, содержащих "шум" и шаблонные формулировки
        is_noise = np.zeros(num_clusters, dtype=bool)
        for i, centroid in enumerate(model_centroids):
            if cluster_sizes[i] < min_cluster_size:
                # Игнорируем кластеры, количество предложений в которых меньше, чем пороговое значение - min_cluster_size
                continue
            distances = np.linalg.norm(embeddings_matrix[model_labels == i] - centroid, axis=1)
            median_distance = np.median(distances)
            if np.count_nonzero(distances > median_distance) / cluster_sizes[i] > max_noise_ratio:
                is_noise[i] = True

        # Удаление ненужных данных: предложения, которые идентифицированы как "шум" или шаблонный текст, удаляются
        filtered_sentences = []
        for i, sentence in enumerate(sentences):
            if not is_noise[model_labels[i]]:
                filtered_sentences.append(sentence)

        filtered_text = ' '.join(filtered_sentences)
        return filtered_text