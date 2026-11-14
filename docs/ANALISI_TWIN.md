# Anàlisi del Shopper Twin i tres variants per provar en viu

Dissabte 14/11, 18:45. Escrit a partir del codi de `twin/` que hi ha al repo després del pull, del model desat a `artifacts/model/` i dels documents de pla (MASTER_CONTEXT, AGENT_INSTRUCTIONS, P1–P4).

**Què no he pogut mirar.** No tinc accés a RawTree ni al leaderboard, i les dades d'entrenament (`artifacts/train/`) no són al repo. Per tant no he vist els resultats en viu del twin ni he pogut reentrenar res. Tot el que dic sobre resultats esperats és una hipòtesi que ha de confirmar l'A/B; el que dic sobre errors surt de llegir i executar el codi.

## 1. Resum

1. **El model no es carrega en un clon de Windows.** `artifacts/model/click.txt` i `buy.txt` arriben amb salts de línia CRLF (git els converteix) i LightGBM els rebutja amb «Model format error». Ho he reproduït aquí amb LightGBM 4.7 a Linux; convertits a LF, es carreguen. Ja ho he arreglat al teu disc i a `.gitattributes`.
2. **El bucle en viu es pot aturar per una sola fila rara**, i no hi ha res que el torni a engegar. He protegit aquest punt i he afegit un supervisor.
3. **La decisió amb més impacte potencial és una regla, no el model:** el twin no recomana mai un producte que l'usuari ja ha comprat, tot i que el seu propi README diu que al log aquests productes converteixen entre 14 i 80 vegades més. És la variant `m1-rebuy`.
4. **Del pla falta la part d'aprenentatge:** no hi ha reentrenament amb el feedback propi, ni experiments aleatoritzats, ni política per intenció. Les variants `m2-intent` i `m3-live` en cobreixen una part sense reentrenar.
5. **La regla d'A/B del pla (3.000 peticions per braç) només detecta millores del 50 %.** Cal triar pocs experiments i llargs (secció 5).

## 2. El pla i el que hi ha fet

| Pla (AGENT_INSTRUCTIONS) | Al codi de `twin/` | Comentari |
|---|---|---|
| Candidats + ranker LightGBM | Sí: ~300–500 candidats, `p(clic) × p(compra \| clic)` | Dos models en lloc d'un de compra directa. Raonable: les compres són escasses |
| `slot` com a feature, inferència a slot 1 | Sí | |
| Sense features d'identitat | **No del tot:** `i_sub` i `i_cat` són categòriques | `i_sub` és la 4a feature del model de clic (9 % del guany) i la 3a del de compra (8 %) |
| Profile dropout per a usuaris nous | Sí (25 % de còpies anònimes) | Top-1 de compra: 48 % amb perfil, 43 % sense, 35 % popularitat |
| Slate per intenció | No: màxim 5 per subcategoria | Variant `m2-intent` |
| Bandit observat/esperat | Sí, sobre clics, sense oblit | Variant `m3-live` hi afegeix oblit i compres |
| Tendència per producte, rebaixes de preu | Només tendència per subcategoria, com a feature | Variant `m3-live` |
| Reentrenament horari amb feedback | **No** | El log a `scratch` desa ítems i clics esperats, però no les features |
| Braços d'experiment (`xperm`, `xwide`) | No: només `baseline-holdout` | |
| A/B entre versions | No (una sola `TWIN_VERSION`) | Ara sí: `TWIN_ARMS` |
| Supervisor, standby, recàrrega en calent | No | Ara hi ha `twin.supervisor`. Canviar de model demana reiniciar |

## 3. Errors i riscos, per gravetat

### Greus (poden deixar-nos sense respondre)

**E1 · Model il·legible en Windows.** Vegeu el resum. Afecta qualsevol màquina que cloni el repo amb Git per a Windows, inclosa la «definitiva» si és Windows. A la màquina on es va entrenar funciona perquè allà el fitxer és l'original. *Arreglat:* fitxers reescrits amb LF al teu disc i `artifacts/model/*.txt -text` a `.gitattributes`. Per comprovar-ho: `python -c "import lightgbm as lgb; lgb.Booster(model_file='artifacts/model/click.txt')"`.

**E2 · Una fila inesperada mata el procés.** A `live.py`, el pas 1 del bucle (aplicar events, canvis de catàleg i feedback) no tenia cap `try`. Un preu que no és un número, un `created_at` amb un format nou o el producte 1.201 afegit durant l'event (`RuntimeError: product capacity exhausted` a `State._register`) aturaven el servei. Ho he comprovat injectant un preu invàlid: abans petava, ara la fila se salta i es compta (`skipped rows` al log). *Arreglat.*

**E3 · Ningú reinicia el procés.** Si cau de nit, totes les peticions van al baseline fins que algú se n'adoni. *Afegit:* `python -m twin.supervisor`. Un reinici costa el temps de carregar l'estat més posar-se al dia amb els events.

**E4 · Capacitat de productes nous.** `State(spare_products=1200)`. A les 18:10 ja n'hi havia 200 de nous (5.200 al catàleg), unes 25 per hora, i la final promet «productes nous». Amb E2 arreglat, passar del límit ja no atura res, però els productes que no hi càpiguen no es podran recomanar ni comptaran com a vistos. *Recomanació:* pujar-ho a 2.500 si la memòria ho permet (les tres matrius de co-visites passen d'uns 460 MB a uns 680 MB).

### Mitjans (fan perdre respostes o qualitat)

**E5 · Una crida encallada bloqueja fins a 8 s.** El poll té un límit de 4 s (`bounded(4, ...)`) i l'enviament fa dos intents de 4 s. Durant aquest temps totes les peticions caduquen. Amb el p50 real de 250–320 ms que vam mesurar, un límit d'1,5–2 s perdria molt menys. *No ho he tocat:* és un llindar del teu company.

**E6 · Si l'enviament falla dues vegades, les respostes es perden en silenci** i `answered` les compta igualment. No es reintenten a la volta següent.

**E7 · La regla de recompra descarta el senyal més fort del log.** Detall a la variant `m1-rebuy`.

**E8 · Vocabulari de regions sense desar.** `f_region` és categòrica i el seu codi surt de `sorted(sp.region.unique())` de cada snapshot. Si el snapshot `live` conté una regió que el de `train` no tenia, tots els codis posteriors es desplacen i el model llegeix regions equivocades sense cap error. Passarà si la final porta regions noves. *Recomanació:* desar `region_names` a `meta.json` en entrenar i reutilitzar-lo en servir.

**E9 · El bandit no oblida.** Acumula des de l'inici (i ho restaura tot en reiniciar). A les 10:00, 24 hores d'evidència de la pràctica pesaran més que la primera hora de la final. Variant `m3-live`.

**E10 · `i_sub` com a categòrica.** El model ha après quines subcategories venien a l'historial. Si la final canvia les tendències, aquest coneixement queda desfasat. El bandit per subcategoria ho compensa en part. Per saber quant en depèn el model n'hi ha prou amb reentrenar sense aquesta columna i comparar la validació.

### Lleus

- **E11** · `views_7d` i `purch_7d` s'incrementen amb cada event i no caduquen: diumenge al migdia seran «8 dies», i `i_trend7` haurà pujat un 15 % respecte del que el model va veure en entrenar.
- **E12** · El fallback d'error és el baseline oficial (més venuts de sempre): vàlid, però val zero punts.
- **E13** · Els feeds d'events i feedback avancen el cursor sense solapament: una fila que es fa visible tard es perd.
- **E14** · L'exploració dels slots 8–10 continua activa durant la final. El pla deia explorar-hi només productes nous.
- **E15** · `BANDIT_RESTORE` llegeix tot `scratch` amb un límit de 25 s; quan la taula creixi, el bandit arrencarà buit sense avisar gaire.

## 4. Les tres variants

Cada una canvia una sola cosa. Totes corren dins del mateix procés, sobre el mateix estat, i el nom del braç s'escriu com a `model_version`, de manera que `python -m twin.monitor` les compara. Sense `TWIN_ARMS`, el bucle fa exactament el mateix que abans: ho he comprovat comparant 60 respostes abans i després dels canvis, idèntiques.

### M1 · `m1-rebuy`: deixar de bloquejar el que ja s'ha comprat

- **Què canvia.** Els productes que l'usuari ja ha comprat tornen a ser candidats, i el ranker es reentrena **amb** aquestes files, que ara es llencen (199.922 files). Així les features `u_p_buys` i `u_p_buy_days`, que avui tenen importància zero perquè mai veuen un valor diferent de zero, aprenen del log què vol dir una compra anterior i quants dies han de passar.
- **Per què.** El README del twin diu que al log aquests productes converteixen un 0,7–4 % contra un 0,05 % la resta. L'enunciat diu que qui ja ha comprat «tendeix a no voler-lo de nou». Les dues coses poden ser certes alhora (afinitat molt alta, amb un descompte després de comprar), i el log són 2 milions de respostes del mateix simulador. Ara ho decidim amb una regla a mà; la variant deixa que ho decideixin les dades.
- **Primer pas, 6 minuts i sense risc:** a la màquina que té `artifacts/train/matrix.npz`,
  `python -m twin.train --keep-bought --out model_rebuy --refit`. Imprimeix `repurchase_share_of_purchases` (quina part de les compres de validació són recompres) i el top-1 amb la regla i sense. Si la part és petita, no cal seguir.
- **Risc.** Si el simulador en viu penalitza la recompra més que el log, la conversió baixarà. Per això entra com a braç i no com a canvi directe. No ajuda els usuaris nous.
- **Expectativa.** És l'única de les tres que pot moure molt el resultat, en qualsevol dels dos sentits.

### M2 · `m2-intent`: omplir la llista per intenció

- **Què canvia.** Desapareix el límit fix de 5 productes per subcategoria. Cada producte triat redueix el valor dels següents de la mateixa categoria en proporció a la probabilitat que ja hagi satisfet aquella intenció (la fórmula del pla de P4). Si una categoria concentra el valor, la llista s'hi concentra; si està repartit, es diversifica.
- **Sense reentrenar.** La intenció s'estima amb el repartiment del valor dels candidats per categoria, que és el que el pla preveu quan no hi ha `intent.lgb`.
- **Expectativa, sent honest.** Amb probabilitats de compra de l'ordre de l'1–3 %, la fórmula penalitza poc, així que a la pràctica aquesta variant mesura sobretot si **el límit de 5 per subcategoria ajuda o fa nosa**. Efecte probablement petit. Paràmetres a `Variant`: `lam` (força) i `intent_level` (`cat` o `sub`).

### M3 · `m3-live`: una capa viva que oblida

- **Què canvia.** Substitueix els multiplicadors del bandit base per:
  1. el mateix observat/esperat de clics per producte i subcategoria, però amb **semivida de 2 hores**;
  2. observat/esperat de **compres per categoria** (per producte o subcategoria n'hi ha massa poques), limitat a ×0,8–1,3;
  3. **tendència per producte** amb els events orgànics: activitat dels últims 20 minuts contra les últimes 6 hores, limitada a ×0,8–1,4;
  4. **rebaixa de preu** de més del 10 %: ×1,15 durant una hora;
  5. a les 10:00 de diumenge, l'evidència acumulada es redueix a un quart, un sol cop.
- **Per què.** Són els 10 punts d'«aprenentatge del stream» del jurat i la resposta a E9: que el sistema s'adapti quan a les 10:00 canviïn les tendències.
- **Límit.** L'estat d'aquesta capa no es desa: després d'un reinici comença neutra i triga una o dues hores a tornar a aprendre. La capa s'alimenta de totes les respostes de tots els braços, de manera que ja està calenta quan se li dona trànsit.

Es poden combinar amb `+`: `m1-rebuy+m3-live` aplica els dos canvis.

## 5. Com provar-les

```powershell
python -m tests.test_twin_variants                         # proves amb dades sintètiques i el model real (10 s)
python -m twin.train --keep-bought --out model_rebuy --refit   # només per a m1 (cal artifacts/train)
python -m twin.build live                                  # snapshot fresc abans d'engegar
$env:TWIN_HOLDOUT = "0.1"
$env:TWIN_ARMS = "twin-v1:50,m1-rebuy:50"
python -m twin.supervisor                                  # o: python -m twin.live --dry --seconds 60 per assajar
python -m twin.monitor 120                                 # comparació per model_version
```

A Linux o Mac: `TWIN_ARMS="twin-v1:50,m1-rebuy:50" python -m twin.supervisor`. Un braç el model del qual no existeix es descarta sol i el seu trànsit va als altres.

**Quant de trànsit cal.** Amb una conversió de l'ordre de l'1 % i unes 10.800 peticions per hora, per veure una diferència de 2 errors estàndard:

| Millora relativa real | Peticions per braç | Hores amb dos braços al 50 % |
|---|---|---|
| 50 % | 3.200 | 0,6 |
| 30 % | 8.800 | 1,6 |
| 20 % | 19.800 | 3,7 |
| 10 % | 79.200 | 14,7 |

El llindar de 3.000 peticions del pla només distingeix millores del 50 %. Conseqüències: pocs braços alhora, mirar també el `click_pct` (més freqüent, dona senyal abans) i no promocionar res per una diferència petita d'una hora.

**Ordre que proposo.**

1. Ara: reentrenar `model_rebuy` i llegir els números offline.
2. Fins a mitjanit: `twin-v1:50,m1-rebuy:50`. És l'únic amb un efecte prou gran per veure'l en poques hores.
3. De mitjanit a les 08:00: el guanyador contra el guanyador `+m3-live`, al 50 %.
4. `m2-intent` només si sobra temps: el seu efecte esperat queda per sota del que es pot mesurar en una nit.
5. 08:30: triar. 09:30: un sol braç, `TWIN_HOLDOUT=0`, snapshot fresc i reinici.

## 6. Què he canviat

| Fitxer | Canvi |
|---|---|
| `twin/variants.py` | Nou. `Variant`, les tres variants, repartiment estable per `request_id`, `intent_slate`, `LiveLayer` |
| `twin/recommend.py` | `recommend()` accepta una variant. Amb la de defecte fa el mateix que abans. Retorna també la compra esperada per slot. Els productes ja comprats entren com a candidats quan la variant no els bloqueja |
| `twin/live.py` | `TWIN_ARMS`; un ranker per carpeta de model; el pas 1 protegit fila a fila (E2); la capa viva s'alimenta d'events, catàleg i feedback; recompte per braç al log |
| `twin/train.py` | `--keep-bought` i `--out`. Sempre informa de quina part de les compres de validació són recompres. Sense opcions, entrena igual que abans |
| `twin/supervisor.py` | Nou. Reinicia el bucle si s'atura |
| `twin/README.md` | `TWIN_ARMS` i el supervisor |
| `tests/twin_synth.py`, `tests/test_twin_variants.py` | Nous. Snapshot sintètic amb els esquemes de `twin.build` i sis proves |
| `artifacts/model/*.txt`, `.gitattributes` | Salts de línia LF perquè LightGBM els pugui llegir (E1) |

**Verificat aquí:** les sis proves; 60 respostes del model base idèntiques abans i després; el bucle sencer amb quatre braços i holdout contra taules simulades (90 respostes, cap d'invàlida, feedback aplicat, una fila de catàleg corrupta saltada); 5–8 ms per petició amb 5.000 productes en totes les variants.

**No verificat:** res contra RawTree ni amb les dades reals; `--keep-bought` només amb dades sintètiques; que cap variant millori la conversió. `twin/live.py`, `recommend.py` i `train.py` són del teu company: abans de desplegar, que revisi el diff.
