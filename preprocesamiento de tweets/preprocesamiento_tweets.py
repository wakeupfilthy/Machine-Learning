import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.probability import FreqDist
import string
from nltk.corpus import stopwords
import spacy

# Cargar el modelo de SpaCy para español
nlp = spacy.load("es_core_news_sm")
tweets_normalizados = []
with open('tweets_asco.txt', 'r', encoding='utf-8') as file:
    text = file.read()
#Expresiones regulares para eliminar los usuarios, fechas, enlaces y hashtags
modified_text = re.sub(r"'Asco' \d{18} ", '', text)
modified_text_1 = re.sub(r'@\w+ ', '', modified_text)
modified_text_2 = re.sub(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}', '\n', modified_text_1)
modified_text_3 = re.sub(r'https?://\S+', '', modified_text_2)
modified_text_4 = re.sub(r'#', '', modified_text_3)
modified_text_5 = re.sub(r"\.|\\\\|!|¡|\?|¿", '', modified_text_4)
#print(modified_text_4)
digitos = set('0123456789')

# Eliminar los números del texto
texto_sin_numeros = ''.join(c for c in modified_text_5 if c not in digitos)
#print(texto_sin_numeros)
tweets = []

# Dividir el texto en tweets separados por saltos de línea
tweets_text = texto_sin_numeros.split('\n')

# Eliminar tweets vacíos si los hay
tweets_text = [tweet.strip() for tweet in tweets_text if tweet.strip()]

# Agregar cada tweet a un arreglo dentro del arreglo que contiene todos los tweets
for tweet_text in tweets_text:
    tweet_tokens = tweet_text.split()  # Convertir el texto del tweet en tokens
    tweets.append(tweet_tokens)  # Agregar los tokens del tweet al arreglo de tweets

for tweet in tweets:
    texto = ' '.join(tweet)
    tokens = word_tokenize(texto)
    tokens_lowercase = [token.lower() for token in tokens]
    punctuation_signs = set(string.punctuation)
    punctuation_signs.add("‘")
    punctuation_signs.add("’")
    punctuation_signs.add("…")
    punctuation_signs.add("'")
    punctuation_signs.add("''")
    punctuation_signs.add("..")
    punctuation_signs.add("...")
    punctuation_signs.add("....")
    punctuation_signs.add("-_-")
    punctuation_signs.add("_")
    punctuation_signs.add("-")
    punctuation_signs.add("``")
    punctuation_signs.add(".....")
    #print(punctuation_signs)
    #elimina los signos de puntuación sobre el texto en minusculas y guardamos el resultado en un arreglo tk
    tk=[w for w in tokens_lowercase if not w in punctuation_signs]
    #print(tk)
    #eliminamos las stop words haciendo uso de un set de stopwords en español de la librería nltk
    stop_words=set(stopwords.words("spanish"))
    #Se muestra el set de stopwords
    #print(stop_words)
    #Se eliminan las stopwords del texto anterior y se guarda en un nuevo arreglo filtrado
    filtered_text=[w for w in tk if not w in stop_words]
    #print(filtered_text)
    #Se une los tokens filtrados en un solo texto
    texto_filtrado = ' '.join(filtered_text)
    #Procesar el texto filtrado con SpaCy para realizar la lematización
    doc = nlp(texto_filtrado)
    # Se guardan los lemas de cada token en el nuevo arreglo
    lemmatized_text = [token.lemma_ for token in doc]
    #print(lemmatized_text)
    tweets_normalizados.append(lemmatized_text)

# Guardar los tweets normalizados en un archivo de texto
with open('tweets_preprocesados_asco.txt', 'w', encoding='utf-8') as file:
    for tweet in tweets_normalizados:
        file.write(' '.join(tweet) + '\n')