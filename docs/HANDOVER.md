# Historique et mesures

Journal détaillé du travail qui a produit ce dépôt. `CLAUDE.md` en donne le
résumé opérationnel ; ce document conserve le **chemin** et les **mesures**, car
plusieurs conclusions sont contre-intuitives et ne se redécouvrent pas seules.

---

## 1. Point de départ : l'audit de `tuni-folk-gemini`

Dépôt tiers (Chouaieb Nemri, Apache-2.0) : pipeline SFT → RL entraînant Gemini à
composer de la poésie populaire tunisienne. **Ce dépôt n'est pas le nôtre et
n'est plus utilisé** ; seules la méthode et une partie du code ont été repris.

### Défauts trouvés

| gravité | défaut |
|---|---|
| 🔴 | `Dockerfile` installe `pip install .` sans l'extra `server` → le conteneur n'a ni flask ni gunicorn, **ne démarre pas**. Vérifié en reproduisant l'installation. |
| 🔴 | `make venv` et la CI installent `.[dev]` seulement → `make test` échoue (1/105), **CI rouge sur main**. |
| 🔴 | Faille de reward hacking : avec les références exactes d'un exemple de production, le mot `بيت` répété 6 fois obtient **reward = +1.000**, le maximum. `self_repetition` retourne 0.0 sous 8 tokens, donc le garde-fou anti-boucle est aveugle. |
| 🟡 | `make evaluate` cassé (passe `--corpus`, non accepté ; `--tuned-endpoint` requis manquant). |
| 🟡 | `docs/09-results.md` annonce des runs SUCCEEDED que les `job.json` committés contredisent (PENDING, RUNNING). |
| 🟡 | `evaluate.py` n'impose pas `thinkingLevel`, alors que la doc affirme le contraire. |

**Points forts réels :** README entièrement traçable (tous les chiffres
vérifiés), un seul scorer partagé entre tests / gate / service, distinction
méthodologique rare entre discriminateur, contrainte et garde.

### Pourquoi on ne l'a pas repris

18 des 22 cibles du Makefile exigent le corpus (non versionné) ou GCP. Les runs
RL ont duré 1 j 19 h et 2 j 02 h de Vertex — plusieurs milliers de dollars. Un
seul commit, pas d'amont où proposer une correction.

---

## 2. Le modèle `tunisianity` livré : inutilisable pour nos besoins

Mesuré sur textes de longueur réelle :

| texte | score |
|---|---:|
| darija **moderne** (prose, 105 mots) | **1.000** |
| MSA classique (63 mots) | 0.983 |
| poésie du corpus (160 mots) | **0.824** |

Le classement est **inversé**. Sa classe négative était *de la poésie générée par
un LLM* — tout ce qui sort de ce contraste atterrit haut. Ce n'est pas un
détecteur de dialecte.

Deuxième défaut : sous ~30 mots le score sature à 1.000 quel que soit le
contenu. D'où `MIN_WORDS = 25` et le refus de `predict()` dans `darija-core`.

**Ce qui a été repris :** la méthode (LLR contrastif sur 4-grammes de
caractères) et l'approche de normalisation. Aucune donnée, aucun modèle.

---

## 3. Extraction du corpus de poésie

Les métadonnées n'étaient **pas** stockées comme champs : le dataset SFT ne
contient que `systemInstruction` et `contents`, tout le reste étant encodé dans
la chaîne arabe du prompt.

```
انظم الملزومة في غرض «الأخضر/الغزل».   ->  genre, gharad
وليكن على الميزان الفرعي «بورجيلة».     ->  wazn_sub
وليكن بنَفَس أهل «قابس (بني زيد)».       ->  region
```

Les `uid` viennent d'une seconde jointure : le jeu RL stocke un `source_uid` à
côté d'une empreinte de novelty ; la recalculer réidentifie la source (833
récupérés).

3 942 exemples se réduisent à **2 028 textes uniques** (chaque poème produisait
un `compose` et un `continue` partageant la même cible). Tout décompte lexical
fait directement sur le dataset SFT est donc faussé d'un facteur ~2.

**Le poète est irrécupérable** : `sft/dataset.py` ne l'a jamais mis dans le
prompt. `extract.py --from-diwan <chemin>` le remplirait, si le Diwan d'origine
était obtenu.

Nature des données : dialecte authentique en **registre littéraire ancien**.
`اللي` 47.7/10k mots, `ما...ش` 8.6/10k — mais `برشا` et `شنوة`, parmi les mots
les plus caractéristiques du tunisien contemporain, y sont **quasi absents**
(2 occurrences chacun).

---

## 4. Les six biais, dans l'ordre de découverte

C'est l'apport principal du projet. Chacun n'est devenu visible qu'après
correction du précédent.

### 1 — Genre
Positifs = commentaires de réseaux sociaux, négatifs = Wikipédia. **AUC 1.000**
qui ne mesure que « commentaire contre article ».
→ Même support des deux côtés (`genre_controlled`).

### 2 — Alphabet
Toujours 1.000. TUNIZI est à **99,9 %** en caractères latins, OMCD à 0 % : il
suffisait de détecter l'alphabet.
→ `arabic_only`.

### 3 — Plateforme
Traits dominants du modèle algérien : `😂😂😂😂`, `#الج`. Il séparait **Twitter
de Facebook**, avec une AUC de 0,9995.
→ Retrait emoji, hashtags, mentions, ponctuation répétée.

### 4 — Entités
Traits `مغرب`, `تونسي`, `لطفي`. Ce dernier apparaît **126 fois dans TSAC et zéro
fois** dans les quatre autres corpus — une personnalité tunisienne discutée sur
les pages TV dont TSAC est issu.
→ `strip_entities` (pays, villes, médias, noms).

**Faux positifs corrigés en route :** `نهار` (jour), `شمس` (soleil), `حوار`
(dialogue), `بيضا` (blanc), `وليدي` (mon fils) avaient été listés comme noms
propres. Ce sont d'abord des mots courants → `AMBIGUOUS_EXCLUDED`, soustrait par
construction.

**Ordre des filtres :** la ponctuation part **avant** les noms propres, sinon
`لطفي!!` n'est pas reconnu (la normalisation STANDARD ne retire pas la
ponctuation).

### 5 — Provenance ← le plus coûteux

AUC interne de 0,998, et **70,5 %** seulement sur ARBML (tunisien Twitter, jamais
vu). Le modèle avait appris TSAC, pas le tunisien.

```
positifs                médiane LinTO   bien classés
TSAC seul               0.576           89,6 %
TSAC + ARBML            0.695           99,8 %
```

→ Les contrastes contrôlés tirent leurs positifs de **toutes** les provenances.

### 6 — Registre

Le marocain **encyclopédique** (`ary`) classé tunisien dans **25,6 %** des cas,
contre 0,1 % pour le marocain de réseaux sociaux. Cause : le registre formel
n'avait jamais été vu, d'aucun côté.

```
entraînement            ary classé TN   LinTO
social seul             25,6 %          94,5 %
registres équilibrés     0,0 %          99,6 %
```

→ LinTO (prose formelle) chez les positifs **et** `ary` chez les négatifs.

**Contrepartie mesurée :** la poésie tombe de 55,6 % à 43,2 %. Ce registre
n'appartient à aucune classe.

---

## 5. Le bug du seuil de décision

Trouvé en corrigeant le n° 6 : la configuration équilibrée classait **100 % des
négatifs comme tunisiens**.

Les ancres `lo`/`hi` préservent l'**ordre** des scores mais pas la position de la
frontière. `lo = min(neg) - 0.1 × plage` s'accroche à une seule valeur extrême :
un négatif atypique décale toute l'échelle au-dessus de 0,5. AUC intacte à 0,99,
`predict()` inutilisable.

Seuils réellement appris : **0,599** et **0,856** selon la configuration. `0.5`
codé en dur était faux dans les deux cas, probablement depuis le début et
silencieusement.

→ `train()` fixe le seuil par l'**indice de Youden**, `predict()` l'utilise, deux
tests de régression le verrouillent (dont un qui injecte un négatif aberrant).

---

## 6. Ce que le modèle a appris

Après tous les filtres, les traits dominants sont du vocabulaire dialectal :

```
tunisien   : برشا  محلا  ماسط  ياسر
maghrébin  : ديال  بحال  بغيت  زوين  كلشي  دشي
```

`ماسط` est de l'argot tunisien (~670 occurrences avec ses variantes
`مااااسط` `ممسطو` `الماسط`), quasi absent des corpus voisins qui n'ont que
`وسط` et `فلسطين`. Je l'avais d'abord pris pour un artefact — vérification faite,
c'est du vrai lexique.

---

## 7. Sources : ce qui a été retenu et écarté

### Retenues

| clé | rôle | volume | licence |
|---|---|---|---|
| `linto` | 🇹🇳 positif | 4,5 M lignes | CC BY 4.0 |
| `tsac` | 🇹🇳 positif | ~17k Facebook | LGPL-3.0 |
| `arbml_tn` | 🇹🇳 positif | ~50k Twitter | ⚠️ aucune |
| `tunizi` | 🇹🇳 positif | Arabizi YouTube | ⚠️ aucune |
| `omcd` | ❌ négatif | ~8k YouTube MA | ⚠️ aucune |
| `mac` | ❌ négatif | 18k Twitter MA | ⚠️ aucune |
| `dz` | ❌ négatif | ~6k Twitter DZ | ⚠️ aucune |
| `ary` `arz` `ar` | ❌ négatifs | Wikipédia | CC BY-SA 4.0 |

### Écartées, et pourquoi

- **QADI** — ne distribue que des identifiants de tweets, à hydrater via l'API X
  payante. **NADI** — accord de données requis.
- **Libyen** — les corpus de la littérature (ASALDA 9 350 commentaires, 6 000
  tweets) ne sont pas publiés. Le seul dépôt trouvé est un `.xlsx` d'avis de
  restaurants : genre différent, réintroduirait le biais n° 1.
- **ElecMorocco2016** — `.xlsx` uniquement, dépendance supplémentaire pour peu.

### Erreurs commises sur les sources

- URLs GitHub **devinées** au premier essai (mauvaise branche `main` au lieu de
  `master`, chemin TSAC inventé) → 404. Toujours vérifier via l'API GitHub.
- TSAC annoncé « sans licence » : le dépôt contient un fichier LICENSE, **LGPL-3.0**.
- LinTO annoncé « quasi non dialectal » : c'était un artefact d'échantillon de
  tête. Sur l'ensemble il est **plus dialectal que TSAC**.

---

## 8. Chiffres de référence

```
darija-core                98 tests · ruff propre · 13 modules · 0 dépendance runtime
tunisian-poetry-corpus     2 028 textes · 396 681 mots · 93 476 formes
modèle vs_maghreb          AUC 0.9988 · précision 98,9 % · seuil 0,838
cache data/raw/            23 Mo, 7 corpus
```

---

## 9. Questions ouvertes

1. **Licence du corpus de poésie** — bloque toute publication.
2. **Aucune application** ne consomme la bibliothèque. `apps/` est vide.
3. **Arabizi non couvert** par le classifieur, alors que c'est la forme écrite
   majoritaire du tunisien.
4. **Fuite thématique résiduelle** — `عيد`, `منتخب`, `وسلم` subsistent parmi les
   traits : les corpus ne parlent pas des mêmes sujets.
5. **Noms de personnes partiellement traités** — beaucoup de prénoms arabes sont
   des mots courants (`كريم`, `امين`, `نور`), la liste reste volontairement courte.
