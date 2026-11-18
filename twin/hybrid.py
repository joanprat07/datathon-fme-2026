"""Hybrid content/collaborative features and a directly trained purchase model.
Run: python -m twin.hybrid (requires artifacts/train).
The frozen training state and transforms are saved together; never refit embeddings at serving.
Only load joblib artifacts produced by your own team.
"""
from pathlib import Path
import json
import joblib
import numpy as np
import lightgbm as lgb
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.metrics import log_loss, roc_auc_score
from twin.state import State, FEATURES, CATEGORICAL, to_ms
from twin.rt import ARTIFACTS

EXTRA = ['content_session', 'content_user', 'collab_session', 'collab_user']

class HybridState(State):
    def prepare(self):
        texts = [' '.join(sorted(self.tokens[i])) + ' ' + self.sub_names[self.sub[i]] for i in range(self.P)]
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=12000)
        x = self.vectorizer.fit_transform(texts)
        self.text_svd = TruncatedSVD(n_components=min(48, min(x.shape)-1), random_state=7)
        self.content = normalize(self.text_svd.fit_transform(x)).astype('float32')
        # The snapshot contains only events before the training cutoff.
        counts = np.diff(self.up_ptr)
        users = np.repeat(np.arange(len(counts)), counts)
        weights = np.log1p(self.up_v + 3*self.up_c)
        matrix = csr_matrix((weights, (users, self.up_idx)), shape=(len(counts), self.P))
        self.cf_svd = TruncatedSVD(n_components=min(48, min(matrix.shape)-1), random_state=7)
        self.cf_svd.fit(matrix)
        self.collab = normalize(self.cf_svd.components_.T).astype('float32')

    def featurize(self, ts, page, dev, reg, ui, rv, cand, slot):
        base = super().featurize(ts, page, dev, reg, ui, rv, cand, slot)
        cols = []
        for embedding in (self.content, self.collab):
            session = embedding[rv[-5:]].mean(0) if rv else np.zeros(embedding.shape[1])
            ix, v, c = self.user_products(ui)
            profile = np.average(embedding[ix], axis=0, weights=v+3*c+1e-8) if len(ix) else np.zeros(embedding.shape[1])
            session /= max(np.linalg.norm(session), 1e-8)
            profile /= max(np.linalg.norm(profile), 1e-8)
            cols.extend([embedding[cand] @ session, embedding[cand] @ profile])
        return np.column_stack([base, *cols]).astype('float32')

class HybridModel:
    def __init__(self, folder):
        folder = Path(folder)
        self.state = joblib.load(folder/'state.joblib')
        self.model = lgb.Booster(model_file=str(folder/'purchase.txt'))

    def recommend(self, user_id, recent_views, ts, page=-1, device=-1, region=-1):
        st = self.state
        ui = st.uidx.get(user_id, -1)
        rv = [st.idx[p] for p in recent_views if p in st.idx]
        # Full catalogue scoring avoids a candidate-recall bottleneck; benchmark before live use.
        cand = np.flatnonzero(st.ok[:st.P])
        X = st.featurize(ts, page, device, region, ui, rv, cand, 1)
        scores = self.model.predict(X, num_threads=4)
        return [st.pid[i] for i in cand[np.argsort(-scores)[:10]]]

def main():
    from twin import train
    from twin.build import VALID_START
    if not (ARTIFACTS/'train'/'sessions.parquet').exists():
        raise SystemExit('Missing artifacts/train. Run python -m twin.build train first.')
    st = HybridState('train')
    st.prepare()
    train.FEATURES = FEATURES + EXTRA
    train.CACHE = ARTIFACTS/'train'/'matrix_hybrid.npz'
    m = train.build_matrix(st)  # rebuilt to keep transforms and features consistent
    X, y, w = m['X'], m['buy'], m['w']
    late = m['ts'] >= to_ms(VALID_START)
    tr, va = ~late, late & ~m['anon']
    if len(np.unique(y[tr])) < 2 or len(np.unique(y[va])) < 2:
        raise SystemExit('Training and validation must each contain purchases and negatives.')
    ds = lgb.Dataset(X[tr], label=y[tr], weight=w[tr], feature_name=train.FEATURES, categorical_feature=CATEGORICAL)
    dv = lgb.Dataset(X[va], label=y[va], weight=w[va], reference=ds)
    model = lgb.train(dict(objective='binary', learning_rate=.03, num_leaves=31,
                          min_data_in_leaf=150, lambda_l2=15, num_threads=4, verbosity=-1,
                          seed=7), ds, num_boost_round=1500, valid_sets=[dv],
                      callbacks=[lgb.early_stopping(80)])
    p = model.predict(X[va])
    metrics = dict(purchase_auc=roc_auc_score(y[va], p, sample_weight=w[va]),
                   purchase_logloss=log_loss(y[va], p, sample_weight=w[va]),
                   validation_rows=int(va.sum()), validation_purchases=int(y[va].sum()),
                   best_iteration=model.best_iteration, features=train.FEATURES,
                   status='offline candidate; not live validated')
    out = ARTIFACTS/'model_hybrid'
    out.mkdir(parents=True, exist_ok=True)
    model.save_model(str(out/'purchase.txt'))
    joblib.dump(st, out/'state.joblib', compress=3)
    (out/'metrics.json').write_text(json.dumps(metrics, indent=2))
    print(metrics)

if __name__ == '__main__':
    from twin.hybrid import main as run
    run()
