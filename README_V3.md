# Twin V3: candidat experimental

## Estat real
Implementació preparada a partir de les millores locals de Twin v2. Els pesos
LightGBM inclosos continuen sent els de twin-v1. No s'ha entrenat un nou ranker
amb dades reals ni s'ha demostrat una millora de conversió. No desplegat.

## Mancances principals
1. Mostreig doble: build conserva 1/4 dels negatius, train en conserva 0,3.
   El pes correcte és 4/0,3 per negatiu de train i 4 per negatiu de validació.
   Abans només es corregia el segon mostreig. El nou codi ho corregeix i usa
   pesos en AUC, log-loss i calibratge. Cal reconstruir la matriu i reentrenar.
   Aquests pesos corregeixen el mostreig, no el biaix de la política antiga.
2. Selecció limitada als veïns observats: un producte semblant sense co-visites
   pot quedar fora dels candidats. S'afegeix recuperació per contingut.
3. Les mètriques de reordenació sobre deu ítems no mesuren la cobertura de
   candidats. Cal mesurar recall de compra abans del ranker, per usuaris nous,
   productes nous i usuaris coneguts.
4. El model de compra aprèn d'un carrusel antic. El reentrenament amb feedback
   propi és implementat, però necessita decision logs reals i validació temporal.
5. La fiabilitat depèn de xarxa, rellotge, cues i estoc; els tests sintètics no
   acrediten el termini de tres segons contra RawTree.

## Arquitectura
Fonts SQL → estat en memòria → candidats clàssics + embeddings → dos LightGBM
→ correcció amb feedback → deu productes actius diferents → enviament.
El log de decisions permet entrenar amb les features que realment es van usar.

## Embeddings implementats
Hashing de paraules i bigrames de descripció/subcategoria a 8.192 dimensions,
seguit de SVD a 64 dimensions i normalització. Recupera 60 candidats segons
les últimes cinc visites; sense visites usa fins a deu productes de l'historial.
Els nous productes es projecten amb el mateix transformador. La construcció
inicial es fa abans del bucle. És una representació lèxica latent local, no
un encoder semàntic preentrenat. No necessita downloads ni GPU.
Activació: TWIN_EMBEDDINGS=1. El ranker de 56 variables es manté compatible.
Risc: candidats addicionals poden tenir probabilitats sobreestimades fora de
la distribució antiga. Cal comparar amb embeddings desactivats.

## Millores grans següents
- Embeddings col·laboratius: factoritzar una matriu dispersa usuari-producte
  (visites, carretons i compres ponderats), amb tall temporal abans del train.
  Recuperar candidats per afinitat latent, i combinar amb contingut per cold start.
- Encoder de text preentrenat: comparar embeddings de descripció amb el SVD
  actual. Desar vectors i model/versionat; no fer crides externes per petició.
- Ranker ampliat: afegir similitud de sessió, usuari i producte com a features;
  exigeix un nou esquema i nou entrenament, no carregar pesos v1 amb més columnes.
- Compra directa com a tercer model i combinació validada temporalment amb
  P(clic)*P(compra|clic). Comparar calibratge i rendiment per segment.
- Experiments independents de recompra, exploració i preu; no combinar abans
  de saber què millora. Comparar finestres simultànies i incertesa per usuari.

## Correccions de servei incloses
Es conserva protecció per fila, supervisor, braços i decision log de v2.
Es descarten peticions de més de 2,2 s, es redueix el límit d'espera de lectura
a 1,2 s i d'enviament a 0,7 s. No es reenvia automàticament una petició amb
resultat ambigu: el servidor podria haver-la acceptat. El comptador answered
només compta enviaments HTTP confirmats (en dry compta simulacions).
Un HTTP confirmat no prova que l'organitzador l'hagi considerat puntual.

## Riscos encara pendents
Els futurs amb timeout no cancel·len la crida subjacent; cal aïllar transport
per procés per garantir límits durs. Els feeds encara poden perdre files amb
el mateix timestamp o arribada tardana. Falta persistir vocabularis de categories
i regions, i mesurar latència després de l'ACK. El límit de 80 ms d'aplicació
no s'aplica dins un lot gran. No declarar aquests problemes resolts.
El mostreig històric de validació també exigeix revisar el baseline position-only.

## Llançament i validació
Copiar artifacts/live de la màquina de servei i configurar .env amb les claus
de l'equip. Instal·lar requests pandas numpy lightgbm pyarrow scikit-learn.
Primer executar: python -m tests.test_twin_variants.
Després: python -m twin.live --dry --seconds 60.
Per servir: python -m twin.supervisor (una sola instància per equip).
Per embeddings, definir TWIN_EMBEDDINGS=1 abans d'arrencar.
Reentrenament corregit: python -m twin.build train; python -m twin.train
--out model_corrected --refit. Registrar explícitament aquest model com a braç
abans d'usar-lo; no substituir el model base sense comparació.
La final necessita el guanyador verificat, holdout=0 i servei estable.
