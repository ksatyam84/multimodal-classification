import pandas as pd
import ast
from datetime import datetime

def process_movie_data(input_file, output_file):
    # Load the dataset
    df = pd.read_csv(input_file, low_memory=False)
    
    # Convert JSON strings to Python objects for genres
    df['genres'] = df['genres'].apply(lambda x: [g['name'] for g in ast.literal_eval(x)])
    
    # Helper function to safely parse dates
    def safe_date_parse(date_str):
        try:
            return pd.to_datetime(date_str, errors='coerce').strftime('%Y-%m-%d')
        except:
            return None
    
    # Process and format the data
    processed_df = pd.DataFrame({
        'Release_Date': df['release_date'].apply(safe_date_parse),
        'Title': df['title'],
        'Overview': df['overview'],
        'Popularity': df['popularity'],
        'Vote_Count': df['vote_count'],
        'Vote_Average': df['vote_average'],
        'Original_Language': df['original_language'],
        'Genre': df['genres'].apply(lambda x: ', '.join(x)),
        'Poster_Url': 'https://image.tmdb.org/t/p/original' + df['poster_path']
    })
    
    # Drop rows with missing values (including invalid dates)
    processed_df = processed_df.dropna()
    
    # Save to CSV with pipe delimiter
    processed_df.to_csv(output_file, sep='|', index=False, encoding='utf-8')
    
    print(f"Successfully saved {len(processed_df)} movies to {output_file}")
    print(f"Dropped {len(df) - len(processed_df)} rows with invalid/missing data")
    return processed_df

# Example usage
input_csv = "movies_metadata.csv"
output_csv = "formatted_movies.csv"
movie_data = process_movie_data(input_csv, output_csv)