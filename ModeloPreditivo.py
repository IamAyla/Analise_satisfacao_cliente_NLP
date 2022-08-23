#Importando as bibliotecas a serem utilizadas
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

#Ferramentas NLTK para processamento de texto 
import re, nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from wordcloud import WordCloud, STOPWORDS

#Pacotes de modelagens
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'all'

#Análise exploratória
#Lendo as avaliações do dataset
review_df = pd.read_csv('olist_order_reviews_dataset.csv')
review_df.head()
review_df.shape

#Verificando se há valores nulos
review_df.isnull().sum()

#Pré-processamento
#Separando em dois campos de estudo: Vamos estudar separadamente o título e a descrição da avaliação 
review_data_title = review_df['review_comment_title']
review_data = review_df.drop(['review_comment_title'], axis = 1)

#Eliminando os valores NaN
review_data = review_data.dropna()
review_data_title = review_data_title.dropna()

#Reorganizando os índices das revisões e a visualizando os dados
review_data = review_data.reset_index(drop=True)
review_data.head(3)
review_data.shape

#Reorganizando os índices das revisões de títulos e a visualizando os dados
review_data_title = review_data_title.reset_index(drop=True)
review_data.head(10)
review_data_title.shape

import nltk
nltk.download('stopwords')
nltk.download('punkt')

comments = []
stop_words = set(stopwords.words('portuguese'))

for words in review_data['review_comment_message']:
  only_letters = re.sub('^a-zA-A', ' ', words) #somente letras
  tokens = nltk.word_tokenize(only_letters) #tokenizando as sentenças
  lower_case = [l.lower() for l in tokens] #convertendo em minúsculo
  filtered_result = list(filter(lambda l: l not in stop_words, lower_case)) #removendo as stopwords dos comentários
  comments.append(' '.join(filtered_result))
  
  #Visualizando as dados de revisões já limpos
#Comentários

#Usando wordcloud para visualizar os comentários
unique_string = (' ').join(comments)
wordcloud = WordCloud(width = 2000, height = 1000, background_color = 'white').generate(unique_string)
plt.figure(figsize = (20,12))
plt.imshow(wordcloud, interpolation = 'bilinear')
plt.axis('off')
plt.show()

#Usando CountVectorizer para chegar nos unigramas mais importantes
co = CountVectorizer(ngram_range=(1,1))
counts = co.fit_transform(comments)
important_unigrams = pd.DataFrame(counts.sum(axis=0),columns=co.get_feature_names()).T.sort_values(0,ascending=False)
important_unigrams

#Usando CountVectorizer para chegar nos trigramas mais importantes
co = CountVectorizer(ngram_range=(3,3))
counts = co.fit_transform(comments)
important_trigrams = pd.DataFrame(counts.sum(axis=0),columns=co.get_feature_names()).T.sort_values(0,ascending=False)
important_trigrams

#Antes de remover os valores NaN
plt.figure(figsize = (14,6))
sns.countplot(review_df['review_score'], color = 'blue')
#Temos uma faixa de 60.000 notas 5 e 10.000 notas 1

#Depois de remover os valores NaN
plt.figure(figsize = (14,6))
sns.countplot(review_data['review_score'], color = 'green')

#Processando os títulos dos dados
comments_titles = []
stop_words = set(stopwords.words('portuguese'))

for words in review_data_title:
  only_letters = re.sub('^a-zA-A', ' ', words) #somente letras
  tokens = nltk.word_tokenize(only_letters) #tokenizando as sentenças
  lower_case = [l.lower() for l in tokens] #convertendo em minúsculo
  filtered_result = list(filter(lambda l: l not in stop_words, lower_case)) #removendo as stopwords dos comentários
  comments_titles.append(' '.join(filtered_result))
  comments_titles
  
#Usando wordcloud para visualizar os títulos dos comentários
unique_string = (' ').join(comments_titles)
wordcloud = WordCloud(width = 2000, height = 1000, background_color = 'white').generate(unique_string)
plt.figure(figsize = (20,12))
plt.imshow(wordcloud, interpolation = 'bilinear')
plt.axis('off')
plt.show()

#Usando CountVectorizer para chegar nos unigramas mais importantes
co = CountVectorizer(ngram_range=(1,1))
counts = co.fit_transform(comments_titles)
important_unigrams_title = pd.DataFrame(counts.sum(axis=0),columns=co.get_feature_names()).T.sort_values(0,ascending=False)
important_unigrams_title

#Usando CountVectorizer para chegar nos bigramas mais importantes
co = CountVectorizer(ngram_range=(2,2))
counts = co.fit_transform(comments_titles)
important_bigrams_tile = pd.DataFrame(counts.sum(axis=0),columns=co.get_feature_names()).T.sort_values(0,ascending=False)
important_bigrams_tile

#Usando CountVectorizer para chegar nos trigramas mais importantes
co = CountVectorizer(ngram_range=(3,3))
counts = co.fit_transform(comments_titles)
important_trigrams_title = pd.DataFrame(counts.sum(axis=0),columns=co.get_feature_names()).T.sort_values(0,ascending=False)
important_trigrams_title

#Chequando as porcentagens
percent_scores = 100 * review_data['review_score'].value_counts()/len(review_data)
percent_scores

#Máquina de Análise de sentimentos
#Mapeando as notas
review_data['Sentiment_rating'] = np.where(review_data.review_score > 3,1,0)

#Removendo neutros
review_data = review_data[review_data.review_score != 3]

#Verificando as quantidades das classes
review_data['Sentiment_rating'].value_counts()

#Olhando 
review_data.head()
review_data['Sentiment_rating'].shape

comments = []
stop_words = set(stopwords.words('portuguese'))

for words in review_data['review_comment_message']:
  only_letters = re.sub('^a-zA-A', ' ', words) #somente letras
  tokens = nltk.word_tokenize(only_letters) #tokenizando as sentenças
  lower_case = [l.lower() for l in tokens] #convertendo em minúsculo
  filtered_result = list(filter(lambda l: l not in stop_words, lower_case)) #removendo as stopwords dos comentários
  comments.append(' '.join(filtered_result))
  
co_counts = CountVectorizer(stop_words = set(stopwords.words('portuguese')), 
                            ngram_range = (1,4)) #Unigramas para trigramas
co_data = co_counts.fit_transform(comments)
co_data

#Dividindo as pontuações de sentimento em dados de treino e teste
X_train_co, X_test_co, y_train_co, y_test_co = train_test_split(co_data, 
                                                               review_data['Sentiment_rating'], test_size = 0.2, random_state = 7)

y_test_co.value_counts()/y_test_co.shape[0]
#Temos  71% de emoções positivas e 29% de emoções negativas

#Definindo e treinando modelos
lr_model = LogisticRegression(max_iter = 200)
lr_model.fit(X_train_co, y_train_co)

#Avaliando a máquina
#Resultados da predição
test_pred = lr_model.predict(X_test_co)

print('Acurácia: ', accuracy_score(y_test_co, test_pred))
print('F1 score: ', f1_score(y_test_co, test_pred))
print('Confusion Matrix: ', confusion_matrix(y_test_co, test_pred))

lr_weights = pd.DataFrame(list(zip(co_counts.get_feature_names(), #pegando todos os n-gram feature names
                                   lr_model.coef_[0])), #pegando os coeficientes da regressão logística
                          columns = ['words', 'weights']) #definindo nomes das colunas 

Positive_sentiments = pd.DataFrame(lr_weights.sort_values(['weights'], ascending = False) [:15]) #15 mais importantes
Positive_sentiments

Negative_sentiments = pd.DataFrame(lr_weights.sort_values(['weights'], ascending = False) [-15:]) #-15 mais importantes
Negative_sentiments

