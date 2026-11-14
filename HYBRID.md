# Model híbrid experimental

Nou mòdul: twin/hybrid.py. Contingut TF-IDF + SVD, embeddings col·laboratius
SVD de la matriu dispersa usuari-producte (log(1+visites+3*carrets)), quatre
similituds sessió/usuari i un LightGBM de compra directa amb 60 variables.
Els embeddings es construeixen només amb el snapshot anterior al tall de train.
Els pesos compensen el mostreig de sessions negatives. La validació és temporal.
El model conserva les recompres en l'entrenament i deixa que les dades decideixin.

## Entrenament
Dependències: requests pandas numpy scipy scikit-learn lightgbm pyarrow joblib.
Amb .env configurat i des de l'arrel:

    python -m twin.build train
    python -m twin.hybrid

Desa artifacts/model_hybrid/purchase.txt, state.joblib i metrics.json.
El joblib només s'ha de carregar si és un fitxer propi i de confiança.
L'estat conserva vocabularis i transformadors exactes d'entrenament.

## Inferència offline
Importar HybridModel de twin.hybrid, carregar artifacts/model_hybrid i cridar
recommend(user_id, recent_views, ts). ts és un enter en mil·lisegons UTC.
page/device/region són codis del mateix estat desat, no codis d'un altre snapshot.
Puntua tot el catàleg actiu: cal mesurar-ne la latència en la màquina final.

## Què està implementat i què falta
Implementats embeddings de contingut i col·laboratius, variables de similitud,
entrenament directe de compra, validació temporal i inferència offline.
No hi ha encoder semàntic preentrenat, ensemble ni pesos entrenats amb dades reals.
El snapshot train no és un catàleg viu: NO connectar aquesta inferència directament
al repte. Falta integrar actualitzacions de catàleg i vectors per productes nous,
perfils nous, registre de decisions amb 60 variables i servei amb deadlines.
El twin.live inclòs continua executant el Twin anterior, no aquest model híbrid.
Comparar amb twin-v1 sobre les mateixes files i fer A/B després de la integració.
Les mètriques offline no demostren increment de conversió ni primer lloc.

## Proves fetes
Construcció de les dues representacions sobre snapshot sintètic, matriu de 60
variables i similituds finites amb usuari anònim. Entrenament real pendent perquè
artifacts/train no és disponible en aquesta màquina.
