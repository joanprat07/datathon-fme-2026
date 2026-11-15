# Recomanador v5tcs

Recomanador en temps real per a la botiga de RawTree (Datathon FME 2026, equip 28). Llegeix les
peticions de `shop.requests`, tria 10 productes per a cada client i escriu la llista a
`team_NN.submissions` en menys de 3 segons. Aprèn del feedback i dels esdeveniments de la botiga
mentre està en marxa.

El model que hi ha és la versió final, `v5tcs`:

```
v5tcs = v5-hybrid + m2-intent + m3-live + trend2 + cold8 + seedtrend
```

| Peça | Què fa |
|---|---|
| `v5-hybrid` | Ranker LightGBM de 60 variables: p(clic) × p(compra \| clic). Inclou 4 similituds de text i de co-visites amb la sessió i amb el perfil del client. |
| `m2-intent` | Omple les 10 posicions segons la intenció del client, en lloc de posar com a màxim 5 productes per subcategoria. |
| `m3-live` | Bandit que s'actualitza amb el feedback i oblida amb una semivida de 2 h. |
| `trend2` | Dona més pes als productes que pugen els últims ~20 minuts (multiplicador entre 0,64 i 1,96). |
| `cold8` | Els productes sense historial es puntuen amb un segon model (`model_v7s`), entrenat per a productes nous. |
| `seedtrend` | Un producte nou comença amb la taxa heretada de productes semblants, perquè no surti com a "en alça" només per ser nou. |

No es recomana mai un producte que el client ja ha comprat. Si falla qualsevol cosa en puntuar,
la resposta són els més venuts que hi ha en estoc.

## Contingut

```
twin/            el codi del recomanador
  rt.py            client de RawTree (llegeix .env)
  build.py         baixa els agregats de RawTree a artifacts/train o artifacts/live
  state.py         estat en memòria i càlcul de variables (el mateix per entrenar i per servir)
  train.py         ranker base -> artifacts/model
  hybrid.py        matriu d'entrenament amb les 4 variables híbrides
  hybrid_live.py   embeddings + ranker de 60 variables -> artifacts/model_v5
  train_v7.py      ranker per a productes nous -> artifacts/model_v7s
  embeddings.py    vectors de text per a productes que apareixen en directe
  recommend.py     candidats, rànquing, bandit i filtres
  variants.py      definició de les peces de v5tcs i lectura de config/twin_arms.txt
  live.py          bucle en directe
  supervisor.py    arrenca live.py i el torna a arrencar si s'atura
  monitor.py       conversió i fiabilitat a partir del nostre feedback
  declog.py        guarda les variables de cada resposta a artifacts/declog/
  train_feedback.py  reentrena amb aquest registre i el feedback
config/twin_arms.txt   quin model serveix el bucle (es rellegeix cada 15 s)
artifacts/model/       ranker base ja entrenat (els altres models s'han de generar)
tests/                 proves amb pytest
start_live.ps1         arrenca el supervisor en una finestra pròpia (Windows)
```

`artifacts/` no es puja a Git excepte `artifacts/model/`. Les dades baixades, `model_v5` i
`model_v7s` s'han de generar amb els passos de sota.

## Instal·lació

Python 3.13 o més nou.

```bash
python -m venv venv
source venv/bin/activate          # Windows: .\venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env              # i posa-hi TEAM_ID i RAWTREE_API_KEY
```

El fitxer `.env` no es puja mai a Git.

## Execució

Totes les ordres es fan des de l'arrel del repositori.

| Pas | Ordre | Temps aprox. |
|---|---|---|
| 1. Dades d'entrenament | `python -m twin.build train` | 4 min |
| 2. Ranker base | `python -m twin.train --refit` | 6 min |
| 3. Matriu híbrida | `python -m twin.hybrid` | |
| 4. Model v5 | `python -m twin.hybrid_live` | |
| 5. Model per a productes nous | `python -m twin.train_v7 --out model_v7s` | uns minuts |
| 6. Dades en directe | `python -m twin.build live` | 3 min |
| 7. Prova sense enviar res | `python -m twin.live --dry --seconds 60` | 1 min |
| 8. En directe | `python -m twin.supervisor` | indefinit |
| 9. Resultats | `python -m twin.monitor 60` (últims 60 min) | segons |

Els passos 1 a 5 només cal tornar-los a fer si canvien les variables o els models. El pas 6 s'ha
de repetir just abans d'arrencar el bucle: el bucle repassa els esdeveniments des del moment de
la instantània, i si és antiga tarda a posar-se al dia.

Notes:
- Només hi pot haver **un** bucle en directe per equip. Abans d'arrencar-ne un, comprova que no
  n'hi hagi cap altre en marxa (`start_live.ps1` ja ho comprova).
- Cada reinici fa perdre aproximadament un minut de peticions. Per canviar de model no cal
  reiniciar: n'hi ha prou d'editar `config/twin_arms.txt`.
- Al pas 5 no sabem amb quines opcions exactes es va entrenar `model_v7s` per a la final. Les
  opcions disponibles són a la capçalera de `twin/train_v7.py`.

## Configuració del model

`config/twin_arms.txt`:

```
# comentari (la primera línia ha de ser un comentari)
v5tcs=v5-hybrid+m2-intent+m3-live+trend2+cold8+seedtrend: 100
holdout: 0
```

Cada línia és `nom=recepta: pes`. Amb més d'una línia, el trànsit es reparteix entre les
receptes, i cada resposta porta el nom com a `model_version`, de manera que `twin.monitor` les
pot comparar. `holdout` és el percentatge de peticions que es responen amb els més venuts, per
mesurar la millora.

Si el fitxer no existeix, es fan servir les variables d'entorn `TWIN_ARMS` (la mateixa sintaxi en
una sola línia, separada per comes) i `TWIN_HOLDOUT` (fracció, p. ex. `0.1`). Si el fitxer
existeix, té prioritat. `TWIN_ARMS_FILE` permet fer servir un altre fitxer.

## Proves

```bash
python -m pytest -q tests
```

`tests/test_v9.py` necessita `artifacts/model_v5` i `artifacts/model_v7s`. Sense aquests models,
executa la resta amb `--ignore=tests/test_v9.py`.

Més detalls del disseny (separació temporal de les dades, biaix de posició, cold start) a
[twin/README.md](twin/README.md).
