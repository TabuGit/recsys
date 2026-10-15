# RecSys Project: Recommender System for a Movie Service

Educational project: implementation of a recommender system for an online cinema
based on the MovieLens dataset (ml-latest-small).

## Dataset

**MovieLens ml-latest-small** (GroupLens Research):
- `movies.csv` — 9742 movies: `movieId`, `title`, `genres`
- `ratings.csv` — 100836 ratings: `userId`, `movieId`, `rating`, `timestamp`

Genre format: `Action|Adventure|Comedy` (separator — `|`).
Rating scale: 0.5 to 5.0 with step 0.5.

## Part I. Implemented Approaches

### 1. Popular (heuristic, baseline)
**File:** `popular.py`

Top-10 movies by mean rating among movies with at least 50 ratings.
Used as a fallback for new users (cold start).

**Pros:** always works, no user data required, simple.
**Cons:** no personalization — all users see the same list.

### 2. Content-Based (similar movies)
**File:** `content_based.py`

Genres are vectorized with TF-IDF, then cosine similarity is computed
between all movie pairs. For a given title, the top-N most similar
movies are returned.

**Pros:** works without user data, solves cold start for new items.
**Cons:** ignores individual user taste, limited to metadata.

### 3. Collaborative Filtering (SVD)
**File:** `collab_based.py`

The `user × movie` matrix is decomposed via truncated SVD (k=20).
Predicted ratings for all unseen movies are ranked; the top-N are
recommended to the user.

**Pros:** personalized, discovers non-obvious relations.
**Cons:** cold start for new users and items, requires behavioral data.

## Part II. Problems and Solutions

### a) Model-level problems

#### 1. Cold Start (user)
**Problem:** new users have no rating history, so nothing can be recommended.

**Solution:** short onboarding survey ("pick 3 favorite genres"),
show Popular (baseline) until enough data is collected.

**Drawback:** users churn if forced to fill out a survey.

**Best practice:** hybrid models (e.g., LightFM) that combine user/item
features with interactions; gradual data collection via non-intrusive
prompts; Popular as a temporary fallback.

#### 2. Ethical Issues
**Problem:** recommendations may amplify bias (e.g., promote only
blockbusters and bury indie films), forming a "filter bubble".

**Solution:** add diversity to outputs, monitor catalog coverage,
introduce explicit filters.

**Drawback:** diversity lowers short-term CTR, business may resist.

**Best practice:** diversity and serendipity metrics alongside accuracy;
A/B tests; manual moderation of controversial recommendations.

#### 3. Cold Start (item)
**Problem:** a new movie without ratings is not included in
collaborative recommendations.

**Solution:** use content-based features (genres, cast, description)
for initial promotion.

**Drawback:** requires maintaining up-to-date metadata.

### b) Production-level problems

#### 1. Slow Response
**Problem:** the collaborative model is slow on a large catalog.

**Solution:** precompute recommendations in batch mode and store
them in Redis / DB.

**Drawback:** recommendations do not reflect the latest user actions (freshness).

**Best practice:** two-stage pipeline — fast retrieval (Faiss, ANN)
plus heavy ranking; online updates for active users.

#### 2. Scalability
**Problem:** SVD does not scale to millions of users.

**Solution:** use ALS / implicit (parallelized), shard by user.

#### 3. Model Drift
**Problem:** the model "ages" as user tastes change.

**Solution:** scheduled retraining (every N days), real-time online metrics.

## Part III. Metrics

### ML Metrics
- **Precision@K** — share of relevant items among top-K recommendations
- **Recall@K** — share of found relevant items out of all relevant items
- **NDCG@K** — ranking quality with position weighting
- **MAP** — mean average precision
- **Coverage** — % of the catalog recommended by the system

### Business Metrics
- **CTR** — click-through rate on recommendations
- **Conversion Rate** — share of users who performed the target action
- **Average Order Value**
- **Retention** — user retention
- **Session Time**

### Infrastructure Metrics
- **Latency** — API response time
- **Throughput** — requests per second
- **GPU/CPU utilization**
- **Cache hit rate**

### Success Criteria
1. **Offline:** Precision@10 > 0.3, NDCG@10 > 0.5
2. **Online:** CTR +5–15% vs baseline (Popular)
3. **Business:** retention up, churn down
4. **Infrastructure:** latency < 200 ms, catalog coverage > 60%

## How to Run

```bash
pip install -r requirements.txt
python popular.py
python content_based.py
python collab_based.py