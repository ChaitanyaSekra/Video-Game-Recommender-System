from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict
import pandas as pd
import psycopg2
from psycopg2.extras import DictCursor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


# --- DATABASE CLASS ---
class GameDatabase:
    def __init__(self, dbname: str, user: str, password: str, host: str = 'localhost', port: int = 5432):
        self.conn = psycopg2.connect(
            dbname=dbname,
            user=user,
            password=password,
            host=host,
            port=port
        )
        self.conn.autocommit = True

    def execute_query(self, query: str, params: tuple = None) -> List[Dict]:
        with self.conn.cursor(cursor_factory=DictCursor) as cursor:
            cursor.execute(query, params or ())
            if cursor.description:
                return [dict(row) for row in cursor.fetchall()]
            return []


db = GameDatabase('videoGames', 'postgres', 'root')

# --- FEATURES FOR RECOMMENDATION ---
feature_options = {
    "categories": ['categories'],
    "genres": ['genres'],
    "Both":['genres','categories']
}

# --- LOAD TOP 500 GAMES BY RATING ---
def get_games_df() -> pd.DataFrame:
    query = """
    SELECT 
        appid, 
        name, 
        short_description, 
        categories, 
        genres, 
        developer,
        platforms, 
        price, 
        release_date, 
        rating_percent 
    FROM games
    ORDER BY rating_percent DESC
    LIMIT 500;
    """
    result = db.execute_query(query)
    df = pd.DataFrame(result)

    if df.empty:
        raise ValueError("No data returned from the database!")

    df.columns = df.columns.astype(str).str.strip().str.lower()
    for col in ['name', 'short_description', 'categories', 'genres', 'developer', 'platforms']:
        if col not in df.columns:
            raise KeyError(f"Missing expected column: '{col}'")
        df[col] = df[col].astype(str).fillna('')
    return df


# --- BUILD SIMILARITY MATRIX ---
def compute_similarity_matrix(df, features=['short_description', 'categories', 'genres']):
    combined_data = df[features].apply(lambda row: ' '.join([str(i) for i in row.values]), axis=1)
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(combined_data)
    return cosine_similarity(tfidf_matrix, tfidf_matrix)


# --- GET RECOMMENDATIONS ---
def get_recommendations(game_name: str, df: pd.DataFrame, cosine_sim, threshold: float = 0.6):
    game_name = game_name.strip().lower()
    # Exact match after stripping spaces & lowercasing
    matched_rows = df[df['name'].str.lower().str.strip() == game_name]
    if matched_rows.empty:
        return []

    idx = matched_rows.index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:]

    recommendations = []
    for i, score in sim_scores:
        if score >= threshold:
            row = df.iloc[i]
            recommendations.append({
                'name': row['name'],
                'short_description': row['short_description'],
                'categories': row['categories'],
                'genres': row['genres'],
                'developer': row['developer'],
                'platforms': row['platforms'],
                'price': row['price'],
                'release_date': row['release_date'],
                'rating_percent': row['rating_percent'],
                'similarity': round(score, 2)
            })
    return recommendations


# --- ROUTES ---
@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/recommend")
@app.post("/recommend")
async def recommend(request: Request):
    df = get_games_df()
    games = sorted(df['name'].dropna().unique().tolist())

    form = await request.form() if request.method == "POST" else None
    selected_game = form.get("game") if form else ""
    selected_feature = form.get("feature") if form else "all"
    recommendations = []

    if form and selected_game:
        cosine_sim = compute_similarity_matrix(df, feature_options.get(selected_feature, ['short_description','categories','genres']))
        recommendations = get_recommendations(selected_game, df, cosine_sim)

    return templates.TemplateResponse("recommend.html", {
        "request": request,
        "games": games,
        "recommendations": recommendations,
        "selected_game": selected_game,
        "selected_feature": selected_feature,
        "feature_options": feature_options
    })
