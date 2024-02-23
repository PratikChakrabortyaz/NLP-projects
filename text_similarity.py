from collections import Counter
import math

# Function to read text from a file
def read_text_from_file(file_path):
    with open(file_path, 'r') as file:
        return file.read()

# Function to tokenize a text into words
def tokenize(text):
    # Split the text into words
    words = text.lower().split()
    # Remove punctuation
    words = [word.strip(".,?!") for word in words]
    return words

# Function to calculate term frequency (TF)
def calculate_tf(text):
    # Tokenize the text
    tokens = tokenize(text)
    # Count the frequency of each word
    word_counts = Counter(tokens)
    # Calculate the term frequency for each word
    total_words = len(tokens)
    tf = {word: count / total_words for word, count in word_counts.items()}
    return tf

# Function to calculate inverse document frequency (IDF)
def calculate_idf(documents):
    # Count the number of documents containing each word
    document_freq = Counter()
    for document in documents:
        tokens = set(tokenize(document))
        document_freq.update(tokens)
    # Calculate IDF for each word
    num_documents = len(documents)
    idf = {word: math.log(num_documents / (1 + freq)) for word, freq in document_freq.items()}
    return idf

# Function to calculate TF-IDF vector for a text
def calculate_tfidf(text, idf):
    # Calculate TF for the text
    tf = calculate_tf(text)
    # Calculate TF-IDF vector using TF and IDF
    tfidf = {word: tf[word] * idf[word] for word in tf}
    return tfidf

# Function to calculate cosine similarity between two TF-IDF vectors
def cosine_similarity(tfidf1, tfidf2):
    # Compute dot product of two TF-IDF vectors
    dot_product = sum(tfidf1[word] * tfidf2[word] for word in set(tfidf1) & set(tfidf2))
    # Compute magnitude of each TF-IDF vector
    magnitude1 = math.sqrt(sum(value ** 2 for value in tfidf1.values()))
    magnitude2 = math.sqrt(sum(value ** 2 for value in tfidf2.values()))
    # Compute cosine similarity
    similarity = dot_product / (magnitude1 * magnitude2)
    return similarity

# Read text from files
file1_path = "t1.txt"
file2_path = "t2.txt"
text1 = read_text_from_file(file1_path)
text2 = read_text_from_file(file2_path)

# Calculate TF-IDF vectors for each text
documents = [text1, text2]
idf = calculate_idf(documents)
tfidf1 = calculate_tfidf(text1, idf)
tfidf2 = calculate_tfidf(text2, idf)

# Calculate cosine similarity between the two TF-IDF vectors
similarity = cosine_similarity(tfidf1, tfidf2)
print("Cosine similarity between the two paragraphs:", similarity)
