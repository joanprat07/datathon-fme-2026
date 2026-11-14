# Passos per entrenar el model m1-rebuy

Per a qui té les dades d'entrenament del Shopper Twin (`artifacts\train\`). Temps total: 10–20 minuts si les dades ja hi són; més si s'han de tornar a baixar.

**Què és m1-rebuy.** El mateix twin, però sense la regla que prohibeix recomanar productes que l'usuari ja ha comprat, i entrenat amb aquestes files en lloc de llençar-les. Així el model aprèn de l'historial si una compra anterior ajuda o perjudica. Ho explica `docs/ANALISI_TWIN.md`, apartat 4.

Totes les ordres van a la terminal de VS Code (PowerShell), a la carpeta del repo, amb `(.venv)` al principi de la línia. Si no hi surt, activa l'entorn amb `.venv\Scripts\Activate.ps1`.

---

## 0. Abans de començar

**Si tens `twin.live` corrent en aquest ordinador, digues-ho a l'equip abans d'aturar-lo.** Només una màquina de l'equip pot estar enviant respostes, i cal saber quina és.

Desa o fes commit de qualsevol canvi teu pendent: el pas 1 canvia de branca.

---

## 1. Baixa la branca amb les variants

**Amb GitHub Desktop:** **Fetch origin** → **Current branch** → tria `variants-m1-m3`.

**Amb la terminal:**

```powershell
git fetch
git switch variants-m1-m3
```

Comprova que el canvi de branca ha funcionat (ha d'escriure `True`):

```powershell
Test-Path twin\variants.py
```

**No executis `python -m twin.train` per «veure l'ajuda»:** no té ajuda, es posa a entrenar directament i, sense `--out`, sobreescriu el model actual.

**Revisa el diff** de `twin\live.py`, `twin\recommend.py` i `twin\train.py`: són teus i els he modificat. Sense opcions noves, fan el mateix que abans (ho he comprovat amb 60 respostes, idèntiques abans i després).

---

## 2. Instal·la i comprova

```powershell
pip install -r starter\requirements.txt
python -m tests.test_twin_variants
```

Han de sortir **6 línies `OK`**. Les xifres que surten pel mig són de dades inventades per a la prova: no les miris.

---

## 3. Entrena el model

**Mira primer si tens les dades:**

```powershell
dir artifacts\train
```

**Cas A · Hi ha `matrix.npz`** (les features ja calculades). És el més ràpid, uns minuts:

```powershell
python -m twin.train --keep-bought --out model_rebuy --refit
```

**Cas B · Hi ha `sessions.parquet` però no `matrix.npz`.** La mateixa ordre: primer recalcula les features (pot trigar bastant) i després entrena.

**Cas C · No hi ha la carpeta `artifacts\train`.** Primer baixa les dades (uns 4 min) i després entrena:

```powershell
python -m twin.build train
python -m twin.train --keep-bought --out model_rebuy --refit
```

**No facis servir `--out model`** ni deixis `--out` sense posar: sobreescriuries el model actual, que és el que està en viu.

Ha d'acabar amb `saved to ...\artifacts\model_rebuy`.

---

## 4. Llegeix el resultat

Busca aquestes línies a la sortida i apunta-les:

| Línia | Què vol dir |
|---|---|
| `repurchase_share_of_purchases` | Quina part de les compres de validació són de productes que l'usuari ja havia comprat |
| `top1_bought_model` | Quantes vegades el primer producte del model és el que es va comprar, **sense** la regla |
| `top1_bought_model_if_blocked` | El mateix, **amb** la regla (bloquejant els ja comprats) |
| `purchase_auc` | Qualitat general. Compara-la amb el teu model original: 0,954 |

**Com decidir:**

- `repurchase_share_of_purchases` < 5 %: les recompres són rares, m1 gairebé no canviarà res. Avisa l'equip i no cal continuar.
- `top1_bought_model` clarament més alt que `top1_bought_model_if_blocked`: a l'historial, deixar recomanar el ja comprat ajuda. Val la pena provar-ho en viu. Continua.
- Si són gairebé iguals, la regla no importa gaire. Es pot provar igualment, però no espereu gaire diferència.

Passa aquestes xifres a l'equip: el model en viu pot dir una altra cosa, perquè l'enunciat avisa que qui ja ha comprat un producte «tendeix a no voler-lo de nou».

---

## 5. Comprova que el model es carrega

```powershell
python -c "from twin.recommend import Ranker; from twin.rt import ARTIFACTS; Ranker(ARTIFACTS / 'model_rebuy'); print('model_rebuy OK')"
```

Ha d'escriure `model_rebuy OK`. Si surt «Model format error», digues-ho: és un problema de salts de línia.

Prova en sec (no envia res):

```powershell
$env:TWIN_ARMS = "m1-rebuy:100"
python -m twin.live --dry --seconds 60
```

Has de veure `arms: m1-rebuy 100%` i `errors 0`. No ha de dir `arm 'm1-rebuy' skipped`.

---

## 6. Puja el model

**Amb GitHub Desktop:**

1. Comprova que estàs a la branca `variants-m1-m3`.
2. A la llista de canvis han de sortir **tres fitxers**: `artifacts/model_rebuy/click.txt`, `buy.txt` i `meta.json`.
3. Comprova que **no surten** `artifacts/model/...` (el model actual), ni el `.env`, ni res de `artifacts/train` o `artifacts/live`.
4. Missatge, per exemple «Model m1-rebuy», → **Commit to variants-m1-m3** → **Push origin**.

**Amb la terminal:**

```powershell
git status
git add artifacts/model_rebuy
git commit -m "Model m1-rebuy"
git push
```

Si `git status` no mostra `artifacts/model_rebuy/`, fes primer `git pull`: la branca ja porta el `.gitignore` que permet pujar aquesta carpeta.

---

## 7. Avisa l'equip

Envia:

- que el model ja és a la branca `variants-m1-m3`;
- les xifres del pas 4.

Qui té el servei en viu ha de fer **Pull**, aturar amb `Ctrl+C` i tornar-lo a engegar:

```powershell
python -m twin.build live
$env:TWIN_ARMS = "m1-rebuy:34,m2-intent:33,m3-live:33"
$env:TWIN_HOLDOUT = "0.1"
python -m twin.supervisor
```

---

## Si alguna cosa falla

| Missatge | Causa | Solució |
|---|---|---|
| `No module named twin.variants` | No estàs a la branca bona | Pas 1 |
| `No such file ... matrix.npz` o `sessions.parquet` | Falten les dades d'entrenament | Cas C del pas 3 |
| `Model format error` | Salts de línia canviats per git | Torna a entrenar i comprova que `.gitattributes` diu `artifacts/model*/*.txt -text` |
| `arm 'm1-rebuy' skipped` | No troba `artifacts\model_rebuy\meta.json` | Comprova que el pas 3 ha acabat amb `saved to ...model_rebuy` |
| `the saved model was trained with a different feature list` | Has entrenat amb una versió de `state.py` diferent | Fes `git pull` a la branca i torna a entrenar |
