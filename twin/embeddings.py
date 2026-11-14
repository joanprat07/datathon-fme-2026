"""Local content embeddings: hashing + truncated SVD, fitted on catalogue text.

This is latent lexical similarity, not a pretrained language model. Retrieval only;
the existing ranker decides whether the additional candidates should be shown.
"""
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize


def semantic_candidates(st, rv, ui, k=60):
    if not hasattr(st, '_semantic'):
        vectorizer = HashingVectorizer(n_features=8192, alternate_sign=False,
                                      ngram_range=(1, 2), norm='l2')
        texts = [' '.join(sorted(st.tokens[i])) + ' ' + st.sub_names[st.sub[i]]
                 for i in range(st.P)]
        x = vectorizer.transform(texts)
        svd = TruncatedSVD(n_components=min(64, st.P - 1), random_state=7)
        z = normalize(svd.fit_transform(x)).astype(np.float32)
        st._semantic = vectorizer, svd, z
    vectorizer, svd, z = st._semantic
    if len(z) < st.P:
        texts = [' '.join(sorted(st.tokens[i])) + ' ' + st.sub_names[st.sub[i]]
                 for i in range(len(z), st.P)]
        z = np.vstack([z, normalize(svd.transform(vectorizer.transform(texts)))])
        st._semantic = vectorizer, svd, z
    anchors = rv[-5:]
    if not anchors and ui >= 0:
        ix, views, carts = st.user_products(ui)
        anchors = ix[np.argsort(-(views + 3 * carts))[:10]].tolist()
    if not anchors:
        return np.empty(0, dtype=int)
    query = z[anchors].mean(axis=0)
    score = z @ query
    score[~st.ok[:st.P]] = -np.inf
    return np.argsort(-score)[:k]
