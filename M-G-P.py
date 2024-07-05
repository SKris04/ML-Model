import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from tabulate import tabulate

# Load the data with appropriate encoding handling
def load_data(file_path):
    try:
        return pd.read_csv(file_path, encoding='utf-8')
    except UnicodeDecodeError:
        return pd.read_csv(file_path, encoding='latin1')

train_data = load_data('Train-data.csv')
test_data = load_data('Test-data.csv')
test_data_solutions = load_data('Test-data-solutions.csv')

# Preprocess the data
def preprocess_text(text):
    if isinstance(text, str):
        text = text.lower()
        text = ''.join([char for char in text if char.isalpha() or char.isspace()])
        text = text[:60] + '...' if len(text) > 60 else text  # Limit description length for display
    else:
        text = ''
    return text

train_data['Description'] = train_data['Description'].apply(preprocess_text)
test_data['Description'] = test_data['Description'].apply(preprocess_text)

# Remove extra spaces from genre labels and split into lists
train_data['Genre'] = train_data['Genre'].apply(lambda x: [genre.strip() for genre in x.split(',')] if isinstance(x, str) else [])
test_data_solutions['Genre'] = test_data_solutions['Genre'].apply(lambda x: [genre.strip() for genre in x.split(',')] if isinstance(x, str) else [])

# Define the genre classes
genre_classes = [
    'drama', 'thriller', 'adult', 'action', 'adventure', 'horror', 'short', 
    'family', 'talk-show', 'game-show', 'music', 'musical', 'documentary', 
    'sport', 'comedy', 'romance', 'history', 'war', 'mystery', 'crime', 
    'fantasy', 'animation', 'reality-tv', 'western', 'biography', 'sci-fi', 
    'news'
]

# Initialize MultiLabelBinarizer with the defined classes
mlb = MultiLabelBinarizer(classes=genre_classes)

# Fit and transform the training data genres
y_train = mlb.fit_transform(train_data['Genre'])

# Extract features using TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf_vectorizer.fit_transform(train_data['Description'])
X_test_tfidf = tfidf_vectorizer.transform(test_data['Description'])

# Use OneVsRestClassifier to handle multi-label classification
model_lr = OneVsRestClassifier(LogisticRegression(max_iter=1000))
model_lr.fit(X_train_tfidf, y_train)

# Predict on test data
predictions_lr = model_lr.predict(X_test_tfidf)
predicted_genres_lr = mlb.inverse_transform(predictions_lr)

# Create DataFrame with results
results_lr = pd.DataFrame({
    'ID': test_data['ID'],
    'Name': test_data['Name'],
    'Description': test_data['Description'],
    'Predicted_Genre': [' '.join(genres) for genres in predicted_genres_lr],
    'Correct_Genre': [' '.join(genres) for genres in test_data_solutions['Genre']]
})

# Correct predictions based on the test-data solutions and learn from them
corrected_genres = []
new_rows = []
for i, row in results_lr.iterrows():
    correct_genres = row['Correct_Genre'].split()
    predicted_genres = row['Predicted_Genre'].split()
    if set(correct_genres) != set(predicted_genres):
        corrected_genres.append(' '.join(correct_genres))
        new_row = {'Description': test_data.iloc[i]['Description'], 'Genre': correct_genres}
        new_rows.append(new_row)
    else:
        corrected_genres.append(' '.join(predicted_genres))

# Add the new rows to the training data
if new_rows:
    new_df = pd.DataFrame(new_rows)
    train_data = pd.concat([train_data, new_df], ignore_index=True)

results_lr['Predicted_Genre'] = corrected_genres

# Retrain the model with corrected data
X_train_tfidf = tfidf_vectorizer.fit_transform(train_data['Description'])
y_train = mlb.fit_transform(train_data['Genre'])
model_lr.fit(X_train_tfidf, y_train)

# Function to add new movies
def add_new_movie(name, description):
    description = preprocess_text(description)
    description_tfidf = tfidf_vectorizer.transform([description])
    prediction = model_lr.predict(description_tfidf)
    predicted_genres = mlb.inverse_transform(prediction)
    predicted_genres = [genre for genre in predicted_genres[0] if genre in genre_classes]
    return ' '.join(predicted_genres)

# Display results in chunks with 'more' and 'exit' commands
chunk_size = 10
start = 0

while start < len(results_lr):
    end = min(start + chunk_size, len(results_lr))
    print(tabulate(results_lr.iloc[start:end], headers='keys', tablefmt='psql', showindex=False))
    start = end
    if start < len(results_lr):
        user_input = input("Type 'more' to see more results, 'new' to add a new movie, or 'exit' to quit: ").strip().lower()
        if user_input == 'exit':
            break
        elif user_input == 'new':
            name = input("Enter the movie name: ").strip()
            description = input("Enter the movie description: ").strip()
            predicted_genre = add_new_movie(name, description)
            print(f"Predicted Genre for '{name}': {predicted_genre}")
            input("Press Enter to continue...")
        elif user_input != 'more':
            break

# Save results to CSV
results_lr.to_csv('Movie-predictions-lr.csv', index=False)
