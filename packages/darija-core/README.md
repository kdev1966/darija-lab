# darija-core

**Socle de traitement du texte tunisien (الدارجة التونسية).**
Normalisation, Arabizi, alternance codique, classification de dialecte.

Zéro dépendance d'exécution. C'est délibéré : ce paquet est la base d'autres
projets, il ne doit rien leur imposer.

```bash
python3.12 -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest -q     # 98 tests
```

---

## Pourquoi ce paquet existe

Les chaînes de traitement arabes normalisent vers l'arabe standard. Pour du
darija, c'est destructeur : elles corrigent l'orthographe dialectale, remplacent
`برشا` par `كثيرا`, et suppriment `ڨ` parce qu'il n'appartient pas au bloc arabe
de base — transformant `ڨلب` en `لب`.

Ce paquet fait l'inverse. Il ne canonise que ce qui ne distingue **jamais** deux
mots (élongation typographique, diacritiques, porteurs de hamza), et laisse le
dialecte intact. C'est vérifié par un test : chacun de `برشا` `علاش` `قداش`
`كيفاش` `اللي` `باهي` traverse les trois niveaux de normalisation sans changer.

---

## Les cinq briques

### `darija.normalize` — normalisation préservant le dialecte

Trois niveaux : `LIGHT` (élongation + diacritiques), `STANDARD` (+ unification
orthographique), `AGGRESSIVE` (+ réduction des répétitions, filtrage). Les
lettres maghrébines `ڨ ڥ پ چ گ`, qui notent des phonèmes absents de la fusha,
survivent à tous les niveaux.

```python
from darija.normalize import Level, normalize
normalize("مـــاذا")                      # "ماذا"
normalize("برشاااااا", Level.AGGRESSIVE)  # "برشاا"
```

### `darija.arabizi` — Arabizi ↔ arabe

L'Arabizi (`3arabi`) est la forme écrite la plus répandue du tunisien en ligne,
et presque aucun outil arabe ne la traite. Les chiffres portent le signal parce
qu'ils sont *iconiques* : `3`→`ع`, `7`→`ح`, `9`→`ق`, `5`→`خ`.

```python
from darija.arabizi import to_arabic, is_arabizi
to_arabic("9alb")                 # "قالب"
to_arabic("gal")                  # "ڨال"   (ou "قال" avec g_as_qaf=True)
is_arabizi("chnowa a7welek")      # True
is_arabizi("i bought an iphone 13")  # False — un nombre n'est pas un chiffre-lettre
```

La translittération est **intrinsèquement ambiguë** : `salem` peut être `سالم`
ou `سلام`. Elle est déterministe et utile pour indexer, chercher et normaliser
une entrée — ce n'est pas une orthographe à publier.

### `darija.codeswitch` — alternance arabe / français / Arabizi

Le tunisien écrit alterne les codes en permanence, y compris **sans changer
d'alphabet**. C'est le cas difficile, et celui que ce module traite :

```python
from darija.codeswitch import segment
segment("ken 3andek le temps ajoutili chwaya de sucre")
# [arabizi] ken 3andek
# [fr     ] le temps ajoutili
# [arabizi] chwaya
# [fr     ] de sucre
```

L'étiquetage se fait mot à mot puis se lisse par le contexte : le
code-switching se fait par syntagmes, pas mot par mot. Les mots hybrides
(`ajoutili` = radical français + clitique arabe) sont rattachés à leur
voisinage, faute de mieux — c'est une limite assumée, pas un bug.

### `darija.markers` — marqueurs morphologiques, pour **inspecter**

```python
from darija.markers import explain
print(explain("شنوة أحوالك برشا اللي ماناكلش باش نمشي"))
# score=1.000  (6 occurrences, 6 marqueurs distincts)
#   negation_ma_sh   x1 [morphologie] circumfixe de négation ما...ش
#   n_prefix_1sg     x1 [morphologie] préfixe n- de 1re personne
#   ...
```

L'option `modern=True` restreint aux marqueurs du tunisien **contemporain**
(`برشا`, `شنوة`, `باهي`…), ce qui sépare le registre parlé actuel du registre
littéraire ancien — où ces mots sont quasi absents.

⚠️ Un score fondé sur les marqueurs plafonne autour d'**AUC 0.77**. Utilisez ce
module pour expliquer et inspecter ; utilisez `dialect` pour décider.

### `darija.dialect` — classifieur contrastif, pour **décider**

Rapport de vraisemblance logarithmique sur 4-grammes de caractères. Mesuré sur
corpus tunisien réel :

| trait | AUC |
|---|---:|
| **4-grammes de caractères (LLR)** | **0.960** |
| unigrammes de mots (LLR) | 0.947 |
| taux de marqueurs | 0.771 |
| longueur moyenne des mots | 0.659 |
| ratio type/token | 0.223 |

```python
from darija.dialect import train, evaluate
model = train(textes_tunisiens, textes_msa, labels=("tunisien", "msa"))
model.predict(texte)      # ("tunisien", 0.94)  ou None si trop court
model.save("modele.json.gz")
```

**Aucun modèle pré-entraîné n'est fourni** — il faut vos données.

Deux propriétés contraignent l'usage, et elles sont dans le code :

1. **C'est un classifieur binaire contrastif, pas un détecteur absolu.** Il
   apprend « positif plutôt que ce négatif-là ». Entraîné contre du MSA, il ne
   dit rien d'utile face à de l'algérien. Réentraînez quand la classe négative
   change.

2. **Il exige du texte.** Sous `MIN_WORDS` (25), le score sature et ne veut plus
   rien dire. `predict()` renvoie `None` plutôt qu'un chiffre faussement
   confiant. Ce garde-fou vient d'une erreur constatée sur un modèle réel, où un
   extrait de 10 mots d'arabe standard obtenait le **même score maximal** qu'un
   texte authentiquement dialectal.

---

## Ligne de commande

```bash
echo "chnowa a7welek" | darija translit
echo "ken 3andek le temps" | darija segment
darija markers --file post.txt
darija detect --file post.txt        # rapport complet, en JSON
```

---

## Ce que ce paquet ne fait pas

- **Pas de segmentation morphologique.** Découper les clitiques (`ع`, `ب`, `ل`,
  pronoms suffixés) est un problème distinct, et un tokeniseur qui devine fait
  plus de mal que de bien sur du texte spontané. `tokenize()` coupe sur l'espace,
  point.
- **Pas de traduction, pas de correction orthographique.**
- **Pas de modèle livré.** Voir `dialect` ci-dessus.

---

## Provenance

La méthode contrastive sur n-grammes de caractères et l'approche de
normalisation préservant le dialecte sont reprises de
`tuni-folk-gemini` (Chouaieb Nemri, Apache-2.0), un
pipeline de fine-tuning pour la poésie populaire tunisienne. **Seuls le code et
la méthode ont été repris** — aucune donnée, aucun modèle entraîné.

Le garde-fou `MIN_WORDS` et le découpage intra-latin de `codeswitch` sont des
corrections apportées ici, à partir de défauts constatés à l'usage sur
l'implémentation d'origine.

---

## `darija.data` — récupérer les corpus et entraîner

```bash
pip install -e ".[data]"     # huggingface-hub + pyarrow, pour les jeux HF
darija data budget           # coût et licences, AVANT de télécharger
darija data fetch            # vers data/raw/
darija data train --contrast vs_maghrebi
```

Dix sources déclarées dans `data/sources.py`, chacune avec un **rôle** :

| Source | Rôle | Licence |
|---|---|---|
| LinTO (4,5 M lignes) | 🇹🇳 positif | CC BY 4.0 |
| TSAC (Facebook, ~17k) | 🇹🇳 positif | LGPL-3.0 |
| arbml (Twitter, ~50k) | 🇹🇳 positif | ⚠️ non déclarée |
| TUNIZI (YouTube, Arabizi) | 🇹🇳 positif | ⚠️ non déclarée |
| **OMCD** (YouTube marocain, ~8k) | ❌ négatif | ⚠️ non déclarée |
| **MAC** (Twitter marocain, 18k) | ❌ négatif | ⚠️ non déclarée |
| **oea_algd** (Twitter algérien, ~6k) | ❌ négatif | ⚠️ non déclarée |
| Wikipédia `ary` marocain | ❌ négatif | CC BY-SA 4.0 |
| Wikipédia `arz` égyptien | ❌ négatif | CC BY-SA 4.0 |
| Wikipédia `ar` standard | ❌ négatif | CC BY-SA 4.0 |

Le marocain et l'égyptien ne sont **jamais** des données tunisiennes : ce sont
les contre-exemples. Un classifieur contrastif n'apprend pas « voici du
tunisien », il apprend « ceci plutôt que cela ». Sans négatifs proches, un
modèle entraîné sur « tunisien contre MSA » étiquettera du marocain comme
tunisien.

Trois choix de conception :

- **Lecture en flux.** Les dumps sont décompressés à la volée et coupés à un
  plafond d'octets. Le XML n'est jamais écrit : pas de pic à 9 Go, et on ne
  télécharge que ce qu'on lit.
- **Équilibrage à la construction**, pas au téléchargement — le cache reste
  complet et rééchantillonnable.
- **Blocs de 60 mots**, au-dessus de `dialect.MIN_WORDS`, pour que chaque
  échantillon soit décidable.

### Six biais, chacun invisible avant correction du précédent

C'est l'apport principal de ce module — la méthode, bien plus que le modèle :

| # | biais | symptôme mesuré | correctif |
|---|---|---|---|
| 1 | **genre** | AUC 1.000 contre Wikipédia | réseaux sociaux des deux côtés |
| 2 | **alphabet** | AUC 1.000 malgré le genre | `arabic_only` |
| 3 | **plateforme** | traits `😂😂😂😂`, `#الج` | retrait emoji, hashtags, ponctuation |
| 4 | **entités** | traits `مغرب`, `تونسي`, `لطفي` | `strip_entities` |
| 5 | **provenance** | 0,998 en interne, **70 %** ailleurs | plusieurs corpus tunisiens |
| 6 | **registre** | 25,6 % du marocain **formel** pris pour du tunisien | registres équilibrés des deux côtés |

Les deux derniers sont les plus coûteux :

```
provenance   TSAC seul          89,6 %  ->  + ARBML            99,8 %
registre     social seul  ary   25,6 %  ->  + LinTO / + ary     0,0 %
```

### Le seuil de décision est appris, pas fixé à 0,5

Bug trouvé en corrigeant le n° 6. Les ancres `lo`/`hi` préservent l'**ordre**
des scores mais pas la position de la frontière : `lo` s'accroche au minimum de
la classe négative, donc à une seule valeur extrême. Un négatif atypique
suffisait à décaler toute l'échelle et à faire classer « tunisien »
**l'intégralité** des contre-exemples — AUC intacte à 0,99, `predict()`
inutilisable.

Les seuils appris valent 0,599 et 0,856 selon la configuration : **0,5 était
faux dans les deux cas**. `train()` le fixe désormais par l'indice de Youden, et
`predict()` l'utilise.

### Validation source par source

```
darija data validate --model models/vs_maghreb.json.gz
```

```
source     role           n   mediane  >=seuil
linto      positive    4000     0.955    99.2%
tsac       positive     885     0.918    99.6%
arbml_tn   positive    3758     0.903    96.1%
ary        negative    4000     0.522     0.0%   <- prose formelle
omcd       negative    1506     0.757     0.0%
mac        negative    2100     0.793     0.8%
dz         negative     354     0.805     5.4%
```

La commande sort en erreur si la pire provenance positive passe sous 90 %.

### Ce que le modèle sait et ne sait pas

**Il sait** : distinguer le tunisien du marocain et de l'algérien, sur les
réseaux sociaux comme sur la prose formelle, et il généralise à des provenances
jamais vues.

**Il ne sait pas** : le registre **littéraire ancien**. La poésie populaire n'est
reconnue tunisienne que dans 43 % des cas — ce registre n'est dans aucune des
deux classes, le modèle ne le couvre pas et ne le prétend pas. Contrepartie
assumée de l'équilibrage : avant celui-ci, la poésie passait à 55,6 %, mais
`ary` était mal classé à 25,6 %.

### Un piège à connaître sur LinTO

LinTO concatène ses sources **par blocs**, et ses 40 premiers pourcents sont de
l'arabe standard :

```
lignes      0-32 000   اللي 0.0-1.0    برشا 0.0     <- MSA
lignes 48 000-80 000   اللي 2.2-86.3   برشا 13.5    <- dialecte
```

Un `--max-lines` qui tronque en tête n'aurait ramené aucun dialecte, sans le
signaler. `fetch` échantillonne donc **uniformément** (réservoir), jamais en tête.

### Sources écartées, et pourquoi

- **QADI, NADI** — QADI ne distribue que des identifiants de tweets, à hydrater
  via l'API X payante ; NADI exige un accord de données.
- **Libyen** — les corpus de la littérature (ASALDA, 9 350 commentaires) ne sont
  pas publiés. Le seul dépôt trouvé est un `.xlsx` d'avis de restaurants : genre
  différent, ce qui réintroduirait le biais n° 1.
- **TUNIZI** ne contribue à aucun contraste `arabic_only` : il est intégralement
  en Arabizi et le filtre d'alphabet l'élimine.

## Prochaines étapes

1. Étendre au registre formel : le modèle échoue sur la prose encyclopédique
   (`ary` à 48 %). Ajouter des positifs tunisiens non issus des réseaux sociaux.
2. Réduire la fuite thématique — `عيد`, `منتخب`, `وسلم` subsistent : les corpus
   ne parlent pas des mêmes sujets.
3. Un modèle Arabizi distinct, puisque le filtre d'alphabet écarte TUNIZI.
4. Le libyen, si un corpus publiable apparaît.
3. Étendre `ARABIZI_STOPWORDS` et `FRENCH_STOPWORDS` à partir de données réelles
   plutôt que d'intuition.
