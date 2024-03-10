import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

TEMPLATE_PROMPT = """
You will be given a review of the book.
Your task is to evaluate the content of the book review.
Please make sure that you have read and understood this review carefully.
Please keep this document in mind during the assessment and refer to it if necessary. 
The score can be from 1 to 5.
The content (1-5) is the degree of its informativeness, reasonableness and depth of analysis of the content of the work. A meaningful review contributes to a deeper understanding of the book and helps other readers make an informed decision about reading it.
Rating 1 is not a meaningful review that does not provide any information about the book, a superficial impression without argumentation.
Score 2 is The minimum content. The review contains a limited amount of information about the book, but the reasoning is insufficient or inconclusive.
Score 3 - Satisfactory content. The review provides basic information about the book and contains some arguments or impressions, but may not be deep enough or thorough enough.
Score 4 - High content. The review contains extensive information about the book, well-developed arguments and an analysis of the main aspects of the work.
Score 5 - Excellent content. The review is a complete, informative and in-depth analysis of the book, providing readers with an extensive understanding of its content, structure and significance.
In the answer, write only the number, only 1, 2, 3, 4 or 5 without explanations and unnecessary words.

Steps in evaluating the content of a book review:
1. Read the review carefully to understand the author's perspective and arguments.
2. Analyze the amount and quality of information the review provides about the book.
3. Assess the reasoning behind the review and its depth of analysis.
4. Determine the overall contribution of the review to the reader's understanding of the book.
5. Based on the analysis, assign a score between 1 and 5.


Review: {review}
Score (write only 1, 2, 3, 4, or 5):
"""

class GEval:
    """
    Класс GEval предназначен для оценки сгенерированного текста с использованием предобученной языковой модели.
    
    Атрибуты:
        model (AutoModelForCausalLM): Экземпляр предобученной языковой модели для генерации текста.
        tokenizer (AutoTokenizer): Токенизатор, соответствующий модели, для преобразования текста в токены и обратно.
        score_tokens (list[int]): Список идентификаторов токенов, используемых для оценки сгенерированного текста.
        weight_tokens (torch.tensor): Веса, соответствующие токенам оценок, для вычисления итогового балла.
        
    Методы:
        init(self, model_name: str):
            Инициализирует экземпляр класса, загружая модель и токенизатор по указанному имени модели.
            
        find_tokens(self, generated_tokens: list, score_tokens: list) -> list:
            Возвращает индексы токенов из списка generated_tokens, которые присутствуют в списке score_tokens.
            
        predict(self, review: str) -> float:
            Генерирует продолжение текста на основе входного примера review и вычисляет итоговый балл сгенерированного текста.
    
    Пример использования:
        # Инициализация оценщика с предобученной моделью
        g_eval = GEval('gpt2')
        
        # Оценка сгенерированного текста
        score = g_eval.predict("Пример текста для генерации.")
        
        print(f"Итоговый балл: {score}")
    """

    def __init__(self, model_name: str):
        """
        Инициализирует экземпляр класса GEval.
        
        Аргументы:
            model_name (str): Имя предобученной модели, которая будет использоваться для генерации текста.
        """

        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.score_tokens = self.tokenizer.convert_tokens_to_ids(['1', '2', '3', '4', '5'])
        self.weight_tokens = torch.tensor([1., 2., 3., 4., 5.])
        
    def find_tokens(self, generated_tokens: list, score_tokens: list):
        """
        Находит индексы токенов из списка score_tokens в списке generated_tokens.
        
        Аргументы:
            generated_tokens (list): Список идентификаторов сгенерированных токенов.
            score_tokens (list): Список идентификаторов токенов оценок.
        
        Возвращает:
            list: Список индексов токенов оценок в сгенерированном тексте.
        """

        indexies = []
        for i in range(len(generated_tokens)):
            id_token = generated_tokens[i]
            for id_score_token in score_tokens:
                if id_token == id_score_token:
                    indexies.append(i)
        return indexies
        
    def predict(self, review: str) -> float:
        """
        Генерирует продолжение текста на основе входного примера и вычисляет итоговый балл сгенерированного текста.
        
        Аргументы:
            review (str): Текст, на основе которого будет сгенерировано продолжение.
        
        Возвращает:
            float: Итоговый балл сгенерированного текста.
        """

        prompt = TEMPLATE_PROMPT.format(review=review)
        
        messages = [
            {"role": "user", "content": prompt}
        ]
        
        encodeds = self.tokenizer.apply_chat_template(messages, return_tensors="pt")
        model_inputs = encodeds.to(self.model.device)
        
        generation = self.model.generate(model_inputs, 
                               max_new_tokens=25, 
                               pad_token_id=self.tokenizer.eos_token_id,
                               output_scores=True,
                               return_dict_in_generate=True,
                               do_sample=False)
        
        generated_ids = generation['sequences']
        all_logits = generation['scores']
        generated_tokens = generated_ids[:, len(model_inputs[0]):][0].tolist()
        decoded = self.tokenizer.batch_decode(generated_ids)
        ids = self.find_tokens(generated_tokens, self.score_tokens)
        try:
            logits_token = all_logits[ids[0] - 1][0]
        except:
            logits_token = all_logits[-1]
            
        log_probs = torch.nn.functional.softmax(logits_token[self.score_tokens], dim=0).detach().cpu()
        score = torch.dot(log_probs, self.weight_tokens).item()
        return score