import pandas as pd
import psycopg2
from psycopg2.extras import execute_values

# ---- CONFIG ----
CSV_PATH = "steam.csv"  # replace with your CSV path
DB_NAME = "videoGames"
DB_USER = "postgres"
DB_PASS = "root"
DB_HOST = "localhost"
DB_PORT = "5432"
TABLE_NAME = "steam_games"  # DB table name

# ---- LOAD CSV ----
df = pd.read_csv(CSV_PATH, sep=',', encoding='utf-8', engine='python')
df.columns = df.columns.str.strip()

# ---- CLEAN DATA ----
# Drop rows with missing appid
df = df.dropna(subset=['appid'])

# Ensure appid is integer
df['appid'] = pd.to_numeric(df['appid'], errors='coerce')
df = df.dropna(subset=['appid'])
df['appid'] = df['appid'].astype(int)

# Convert date format (DD-MM-YYYY → YYYY-MM-DD)
df['release_date'] = pd.to_datetime(df['release_date'], format='%d-%m-%Y', errors='coerce')

# Split semicolon-separated strings into lists for Postgres arrays
def to_pg_array(x):
    if pd.isna(x):
        return []
    return [i.strip() for i in str(x).split(';')]

for col in ['platforms', 'categories', 'genres', 'steamspy_tags']:
    if col in df.columns:
        df[col] = df[col].apply(to_pg_array)

# Ensure numeric columns
for col in ['positive_ratings', 'negative_ratings', 'price']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

# ---- CALCULATE RATING ----
def calc_rating(pos, neg):
    total = pos + neg
    if total == 0:
        return 0
    return round((pos / total) * 100, 2)

df['rating_percent'] = df.apply(
    lambda row: calc_rating(row['positive_ratings'], row['negative_ratings']),
    axis=1
)

# ---- CONNECT TO POSTGRES ----
conn = psycopg2.connect(
    dbname=DB_NAME,
    user=DB_USER,
    password=DB_PASS,
    host=DB_HOST,
    port=DB_PORT
)
cur = conn.cursor()

# ---- INSERT DATA ----
insert_query = f"""
INSERT INTO games (
    appid, name, release_date, developer, platforms, categories, genres,
    steamspy_tags, positive_ratings, negative_ratings, rating_percent, price, short_description
) VALUES %s
ON CONFLICT (appid) DO UPDATE
SET name = EXCLUDED.name,
    release_date = EXCLUDED.release_date,
    developer = EXCLUDED.developer,
    platforms = EXCLUDED.platforms,
    categories = EXCLUDED.categories,
    genres = EXCLUDED.genres,
    steamspy_tags = EXCLUDED.steamspy_tags,
    positive_ratings = EXCLUDED.positive_ratings,
    negative_ratings = EXCLUDED.negative_ratings,
    rating_percent = EXCLUDED.rating_percent,
    price = EXCLUDED.price,
    short_description = EXCLUDED.short_description;
"""

# Build records safely
records = []
for _, row in df.iterrows():
    records.append((
        row['appid'],
        row['name'],
        row['release_date'].to_pydatetime() if pd.notna(row['release_date']) else None,
        row['developer'],
        row['platforms'],
        row['categories'],
        row['genres'],
        row['steamspy_tags'],
        int(row['positive_ratings']),
        int(row['negative_ratings']),
        row['rating_percent'],
        float(row['price']),
        row['short_description']
    ))

execute_values(cur, insert_query, records)
conn.commit()
cur.close()
conn.close()

print(f"✅ Data inserted successfully! Rows processed: {len(records)}")
