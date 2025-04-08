import pandas as pd

# Load CSV file with pipe delimiter
df = pd.read_csv('formatted_movies_updated1.csv', sep='|')

# Filter rows: keep only those, where Poster_Url does not contain "image.tmdb.org"
filtered_df = df[~df['Poster_Url'].str.contains("image.tmdb.org", na=False)]

# Save filtered DataFrame to a new CSV file
filtered_df.to_csv('filtered_movies.csv', sep='|', index=False)

print(f"Filtered CSV saved with {len(filtered_df)} rows.")


##### Cinemagoer