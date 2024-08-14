from sklearn.feature_extraction.text import TfidfVectorizer 

# Datos de ejemplo (lista de documentos) 
corpus = [     'Este es un ejemplo de texto',
               'Otro ejemplo de texto',     
               'Algunos ejemplos son buenos' ]

# Crear un objeto TfidfVectorizer
tfidf_vectorizer   =   TfidfVectorizer(ngram_range=(2,4)) #   Considera   unigramas   y bigramas. 
# Ajustar el vectorizador y transformar los datos de texto 
tfidf_matrix = tfidf_vectorizer.fit_transform(corpus) 
# Obtener el vocabulario (características) 
feature_names = tfidf_vectorizer.get_feature_names_out()
# Imprimir el vocabulario 
print("Vocabulario (Características):")
print(feature_names) 
# Imprimir la matriz TF-IDF 
print("\nMatriz TF-IDF:") 
print(tfidf_matrix.toarray())
top_nouns = sorted(tfidf_vectorizer.vocabulary_, key=lambda x: tfidf_matrix[0, tfidf_vectorizer.vocabulary_[x]], reverse=True)[:10] 
print("\nTop 3 palabras más importantes del primer documento:")
print(top_nouns)


