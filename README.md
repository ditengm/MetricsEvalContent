# MetricsEvalContent
Creating metrics to evaluate the content of book reviews.

[Description of Solution](https://docs.google.com/document/d/11E63EaqB57ujiGSTePvnWjp8-KprapkCGiFZnl0nJpw/edit?usp=sharing)

# Methods

## **TF-IDF Cosine Similarity**

TF-IDF Cosine Score is a metric that shows the proximity of a noise-free recall and a noise-free recall. It is assumed that by removing noise from the review, the most important information will remain in the text and, by calculating the cosine proximity, you can see how much a review with noise is similar to a review without noise.
- If TF-IDF Cosine Score is ~ 1, then the quality of the review is most likely good. This may mean that the noise search algorithm has not found the noise.
- If TF-IDF Cosine Score is ~ 0.5, then the quality of the review is most likely neutral. This may mean that the noise search algorithm found the noise and partially deleted it, which means that the review is meaningful, but there was noise in it. 
- If TF-IDF Cosine Score is ~ 0, then the quality of the review is most likely poor. This may mean that the noise search algorithm has found the noise and removed it completely. 

### Algorithm

1. Clearing train_corpus
2. I teach TF-IDF on train_corpus
3. Iterating on test_corpus, let the iteration be called test_corpus_i
4. Removing noise from test_corpus_i (removing unnecessary sentences using K-Means), let test_corpus_i be called unnoised_test_corpus_i without noise
5. Clearing unnoised_test_corpus_i
6. I count tf-idf from unnoised_test_corpus_i = tf-idf(unnoised_test_corpus_i)
7. I count tf-idf from test_corpus_i = tf-idf(test_corpus_i)
8. I consider the cosine proximity from tf-idf(unnoised_test_corpus_i) and tf-idf(test_corpus_i)


## **TF-IDF Score**
TF-IDF Score is a metric that shows the ratio of a noise-free recall and a noise-free recall. It is assumed that by removing noise from the review, the most important information will remain in the review and, calculating the ratio, you can see how much a review with noise is similar to a review without noise.

- If TF-IDF Score is ~ 1, then the quality of the review is most likely good. This may mean that the noise search algorithm has not found the noise.
- If TF-IDF Score is ~ 0.5, then the quality of the review is most likely neutral. This may mean that the noise search algorithm found the noise and partially deleted it, which means that the review is meaningful, but there was noise in it. 
- If TF-IDF Score is ~ 0, then the quality of the review is most likely poor. This may mean that the noise search algorithm has found the noise and removed it completely. 

### Algorithm

1. Clearing train_corpus
2. I teach TF-IDF on train_corpus
3. Iterating on test_corpus, let the iteration be called test_corpus_i
4. Removing noise from test_corpus_i (removing unnecessary sentences using K-Means), let test_corpus_i be called unnoised_test_corpus_i without noise
5. Clearing unnoised_test_corpus_i
6. I count tf-idf from unnoised_test_corpus_i = tf-idf(unnoised_test_corpus_i)
7. I count tf-idf from test_corpus_i = tf-idf(test_corpus_i)
8. I divide the average tf-idf value(test_corpus_i) by the average tf-idf value(unnoised_test_corpus_i)


## **Standard TF-IDF Score**
The Standard TF-IDF Score is a metric that shows the “content” of a review. It is implied that if you divide the average tf-idf value by the maximum tf-idf value, then in this way you can evaluate the “content”.
The metric is not normalized, so it is worth looking at its absolute values in comparison with other values.

### Algorithm

1. Clearing train_corpus
2. I teach TF-IDF on train_corpus
3. Iterating on test_corpus, let the iteration be called test_corpus_i
4. Clearing test_corpus_i
5. I count tf-idf from test_corpus_i = tf-idf(test_corpus_i)
6. I divide the average value of tf-idf(test_corpus_i) by the maximum value of tf-idf(test_corpus_i) 

## **Length Score**
Length Score is a metric that shows the length of a review in relation to other reviews. It is implied that if you divide the average tf-idf value by the maximum tf-idf value, then in this way you can evaluate the “content”.

### Algorithm

1. Clearing train_corpus
2. I teach TF-IDF on train_corpus
3. Iterating on test_corpus, let the iteration be called test_corpus_i
4. Clearing test_corpus_i
5. I count the length of test_corpus_i = len(test_corpus_i) 
6. I divide len(test_corpus_i) by the maximum length of the recall in the case

## **G-Eval**
G-eval is a metric that in our case shows the quality of the review. The metric is based on an understanding of LLM as a quality criterion, that is, the quality of the review. The metric is from one to five. If one, then a bad review, if five, then an excellent review.

### Algorithm

1. Uploading LLM
2. I ask a prompt with explanations of the data and 3.quality criteria
3. I ask LLM to generate the steps for evaluating the review
4. I insert into the prompt the steps that the model generated in the prompt
5. I run the reviews through this prompt, in which I give
- Explanation of the data
- Definition of the quality criterion
- Explanation of the quality criterion
- The assessment steps, which was written by LLM herself
Feedback
- Please write a rating from 1 to 5
6. I take the weighted average among the tokens and the grid of ratings - this will be the quality

### Visualistaion of Methods
![PipeLine](https://github.com/ditengm/MetricsEvalContent/blob/main/imgs/Screenshot%202024-03-10%20at%2020.51.54.png?raw=true)

![PipeLine](https://github.com/ditengm/MetricsEvalContent/blob/main/imgs/Screenshot%202024-03-10%20at%2020.51.46.png?raw=true)

![PipeLine](https://github.com/ditengm/MetricsEvalContent/blob/main/imgs/Screenshot%202024-03-10%20at%2020.51.32.png?raw=true)

![PipeLine](https://github.com/ditengm/MetricsEvalContent/blob/main/imgs/Screenshot%202024-03-10%20at%2020.51.25.png?raw=true)

![PipeLine](https://github.com/ditengm/MetricsEvalContent/blob/main/imgs/Screenshot%202024-03-10%20at%2020.51.16.png?raw=true)

## Conclusion

- GEval shows itself to be the best, so I would take it as the main metric. Using quantization and weight distribution on multiple video cards, it can be run on a local computer. 
- Standard TF-IDF Score - also took it into account, especially if there are few resources.
- TF-IDF Cosine Similarity and TF-IDF Score are good metrics for detecting average text quality.
- Length Score - shows the length of the text, which should also be included in the review analysis


