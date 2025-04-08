import csv

unique_genres = set()

# Adjust the file path as necessary.
csv_path = '/Users/kumarsatyam/python/basicsofai/project/bigdataset/archive/filtered_movies.csv'

with open(csv_path, 'r', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile, delimiter='|')
    for row in reader:
        genres = row['Genre'].split(',')
        for genre in genres:
            unique_genres.add(genre.strip())

print(f'Number of unique genres: {len(unique_genres)}')
print('Unique genres:', unique_genres)