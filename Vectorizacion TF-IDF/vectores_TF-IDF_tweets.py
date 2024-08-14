from sklearn.feature_extraction.text import TfidfVectorizer 

def vectorizar_tweets(n, corpus, archivo):
    # Obtener los vectores TF-IDF de bigramas
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(n,n))
    tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)
    # Obtener el vocabulario
    feature_names = tfidf_vectorizer.get_feature_names_out()
    # Imprimir el vocabulario
    print(f"Vectores TF-IDF de {n}-gramas:")
    print("Vocabulario (Características):")
    print(feature_names)
    # Imprimir la matriz TF-IDF
    print("\nMatriz TF-IDF:")
    print(tfidf_matrix.toarray())

    #Guardar los vectores TF-IDF en un archivo de texto
    with open(archivo, 'w', encoding='utf-8') as file:
        for vector in tfidf_matrix.toarray():
            file.write(str(vector) + '\n')

with open('tweets_emonegativas_prepro.txt', 'r', encoding='utf-8') as file:
    text = file.read()
corpus = text.split('\n')
#Vectorizar los tweets con unigramas
vectorizar_tweets(1, corpus, 'vectores_TF-IDF_unigramas.txt')
#Vectorizar los tweets con bigramas
vectorizar_tweets(2, corpus, 'vectores_TF-IDF_bigramas.txt')
#Vectorizar los tweets con trigramas
vectorizar_tweets(3, corpus, 'vectores_TF-IDF_trigramas.txt')
#Vectorizar los tweets con cuatrigramas
vectorizar_tweets(4, corpus, 'vectores_TF-IDF_cuatrigramas.txt')