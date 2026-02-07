import pandas as pd
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

data = {
    'Movie_Name': [
        'Inception', 'Interstellar', 'The Dark Knight', 'Batman Begins', 'Toy Story',
        'Finding Nemo', 'Shrek', 'The Lion King', 'The Godfather', 'Pulp Fiction',
        'Goodfellas', 'The Conjuring', 'Annabelle', 'It', 'The Hangover',
        'Superbad', 'Step Brothers', 'Gladiator', 'Braveheart', 'Troy',
        'Avatar', 'Star Wars', 'The Matrix', 'The Silence of the Lambs', 'Se7en',
        'Zodiac', 'Coco', 'Up', 'Cars', 'Joker',
        'Parasite', 'The Departed', 'Spiderman No Way Home', 'Avengers Endgame', 'Iron Man',
        'The Shining', 'Psycho', 'A Quiet Place', 'The Wolf of Wall Street', 'The Social Network',
        'Deadpool', '21 Jump Street', 'The Mask', 'Saving Private Ryan', '1917',
        'Dunkirk', 'Blade Runner 2049', 'Arrival', 'Gravity', 'The Martian'
    ],
    'Genre': [
        'Sci-Fi Action', 'Sci-Fi Drama', 'Action Crime', 'Action Adventure', 'Animation Kids',
        'Animation Adventure', 'Animation Comedy', 'Animation Drama', 'Crime Drama', 'Crime Thriller',
        'Crime Drama', 'Horror Supernatural', 'Horror Supernatural', 'Horror Thriller', 'Comedy Adult',
        'Comedy Teen', 'Comedy Adult', 'Action History Drama', 'Action History Drama', 'Action History Adventure',
        'Sci-Fi Adventure', 'Sci-Fi Adventure', 'Sci-Fi Action', 'Thriller Crime Drama', 'Thriller Crime Crime',
        'Thriller Crime Mystery', 'Animation Kids Musical', 'Animation Kids Drama', 'Animation Kids Comedy', 'Crime Drama Thriller',
        'Thriller Drama Mystery', 'Crime Drama Thriller', 'Action Hero Sci-Fi', 'Action Hero Sci-Fi', 'Action Hero Sci-Fi',
        'Horror Psychological', 'Horror Psychological', 'Horror Sci-Fi Thriller', 'Biography Drama Comedy', 'Biography Drama Tech',
        'Comedy Action Hero', 'Comedy Crime Action', 'Comedy Fantasy', 'War Drama History', 'War Drama Action',
        'War Drama History', 'Sci-Fi Mystery Drama', 'Sci-Fi Mystery Drama', 'Sci-Fi Thriller Drama', 'Sci-Fi Adventure Drama'
    ],
    'Rating': [
        8.8, 8.7, 9.0, 8.2, 8.3, 8.2, 7.9, 8.5, 9.2, 8.9,
        8.7, 7.5, 5.8, 7.3, 7.7, 7.6, 6.9, 8.5, 8.3, 7.3,
        7.9, 8.6, 8.7, 8.6, 8.6, 7.7, 8.4, 8.3, 7.2, 8.4,
        8.5, 8.5, 8.2, 8.4, 7.9, 8.4, 8.5, 7.5, 8.2, 7.8,
        8.0, 7.2, 6.9, 8.6, 8.2, 7.8, 8.0, 7.9, 7.7, 8.0
    ]
}

df = pd.DataFrame(data)
cv = CountVectorizer()
matrix = cv.fit_transform(df['Genre'])
sim_matrix = cosine_similarity(matrix)

while True:
    print("\n" + "="*40)
    usr_input = input(
        "Enter Movie Name or random or 'exit' to end: ").strip()

    if usr_input.lower() == 'exit':
        break

    if usr_input.lower() == 'random':
        res = random.choice(df['Movie_Name'].values)
        print(f"Random Movie is: {res}")
        continue

    if usr_input in df['Movie_Name'].values:
        idx = df[df['Movie_Name'] == usr_input].index[0]
        scores = list(enumerate(sim_matrix[idx]))
        sorted_list = sorted(scores, key=lambda x: x[1], reverse=True)

        print(f"\nRecommended for {usr_input}:")
        for i in range(1, 4):
            m_idx = sorted_list[i][0]
            print(f"{i}. {df.iloc[m_idx]['Movie_Name']}")
    else:
        print("Sorry, Not found.")
