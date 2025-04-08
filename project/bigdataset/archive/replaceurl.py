# python
import pandas as pd
from imdb import Cinemagoer
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm  # for progress bar

# Initialize IMDb (we'll create instances per thread)
def get_imdb_instance():
    return Cinemagoer()

def get_imdb_poster(args):
    title, year = args
    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            ia = get_imdb_instance()
            movies = ia.search_movie(title)
            if not movies:
                return None, None
            
            # Find best match (filter by year if available)
            movie = None
            for m in movies:
                if year and 'year' in m and m['year'] == year:
                    movie = ia.get_movie(m.movieID)
                    break
            if not movie:
                movie = ia.get_movie(movies[0].movieID)
            
            return title, movie.get('full-size cover url')
        except Exception as e:
            if "timed out" in str(e).lower():
                if attempt < max_attempts - 1:
                    continue  # Retry on read timeout errors
            return title, None

def update_posters_parallel(df, max_workers=4):
    # Filter rows where Poster_Url contains "image.tmdb.org"
    df_to_update = df[df['Poster_Url'].str.contains("image.tmdb.org", na=False)]
    
    # Prepare arguments for parallel processing
    args_list = [
        (row['Title'], int(row['Release_Date'][:4]) if pd.notna(row['Release_Date']) else None)
        for _, row in df_to_update.iterrows()
    ]
    
    # Process in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(executor.map(get_imdb_poster, args_list),
                            total=len(df_to_update),
                            desc="Updating posters"))
    
    # Create a mapping from title to new poster URL
    updates = {title: url for title, url in results if url}
    
    # Update only the rows that need to be replaced
    df.loc[df['Title'].isin(updates.keys()), 'Poster_Url'] = df['Title'].map(updates)
    
    updated_count = len(updates)
    print(f"\nSuccessfully updated {updated_count}/{len(df_to_update)} poster URLs ({updated_count/len(df_to_update):.1%})")
    return df

if __name__ == "__main__":
    # Load your formatted CSV
    df = pd.read_csv('formatted_movies_updated.csv', sep='|')
    
    # Update poster URLs in parallel for URLs containing "image.tmdb.org"
    df_updated = update_posters_parallel(df)
    
    # Save back to CSV
    df_updated.to_csv('formatted_movies_updated1.csv', sep='|', index=False)