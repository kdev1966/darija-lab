# darija-lab — contexte projet

Mono-repo de traitement du **tunisien (الدارجة التونسية)**. Bibliothèques
réutilisables, corpus, et applications qui les consomment.

**Langue de travail : le français.** Le code, les docstrings et les commentaires
sont en français ; les identifiants restent en anglais.

```
packages/darija-core/        bibliothèque socle (98 tests, ruff propre)
apps/darija-bench/           banc d'évaluation de LLM (22 tests, ruff propre)
data/tunisian-poetry-corpus/ 2 028 textes de poésie populaire, 396 681 mots
docs/HANDOVER.md             historique détaillé et mesures
docs/ROADMAP.md              le choix de la première application, et pourquoi
```

---

## Ce qu'il faut savoir avant de toucher au code

### 1. `darija-core` est un socle sans dépendance d'exécution

C'est délibéré. Il sert de base à tout le reste, il ne doit rien imposer.
Les extras : `[dev]` (pytest, ruff), `[data]` (huggingface-hub, pyarrow — pour
`darija.data` uniquement).

```bash
cd packages/darija-core
python3.12 -m venv .venv && .venv/bin/pip install -e ".[dev,data]"
.venv/bin/python -m pytest -q     # 98 tests
.venv/bin/ruff check src tests
```

### 2. La normalisation ne corrige JAMAIS le dialecte vers la fusha

C'est la raison d'être du paquet. `برشا` `علاش` `قداش` `اللي` traversent les
trois niveaux sans changer, et les lettres maghrébines `ڨ ڥ پ چ` survivent au
niveau `AGGRESSIVE` — sans quoi `ڨلب` deviendrait `لب`. Un test le verrouille.

### 3. Le classifieur de dialecte est **contrastif binaire**

Il n'apprend pas « voici du tunisien », il apprend « ceci plutôt que cela ».
**Le choix des négatifs détermine ce que le modèle sait faire.** Entraîné contre
du MSA seul, il étiquette du marocain comme tunisien.

### 4. Sept biais ont été trouvés et corrigés — ne pas les réintroduire

Chacun n'est devenu visible qu'après correction du précédent. Tous sont
documentés dans le code et verrouillés par des tests.

| # | biais | symptôme mesuré | correctif |
|---|---|---|---|
| 1 | genre | AUC 1.000 contre Wikipédia | même support des deux côtés |
| 2 | alphabet | AUC 1.000 malgré le genre | `arabic_only` |
| 3 | plateforme | traits `😂😂😂😂`, `#الج` | emoji, hashtags, ponctuation |
| 4 | entités | traits `مغرب`, `تونسي`, `لطفي` | `strip_entities` |
| 5 | provenance | 0,998 interne mais **70 %** ailleurs | plusieurs corpus tunisiens |
| 6 | registre | 25,6 % du marocain formel mal classé | registres équilibrés |
| 7 | registre d'assistant | 0,4 % de faux positifs sur la fusha encyclopédique, mais **33 %** sur la fusha conversationnelle | conjonction classifieur + marqueurs |

**Le septième s'est révélé en construisant `darija-bench`**, et il suit
exactement le même schéma que le sixième. `darija data validate` mesure 0,4 %
de faux positifs sur `ar` — mais `ar` est de la prose encyclopédique. Une
réponse d'assistant en fusha sur un sujet du quotidien est un troisième
registre : sur six passages de ce type, le classifieur en a classé **deux**
comme tunisiens (0,842 et 0,867 pour un seuil de 0,838).

Le correctif exploite une complémentarité mesurée : les marqueurs ne séparent
pas le tunisien du marocain (qui partage `علاش` `كيفاش` `وين` `اللي`), mais la
fusha n'en utilise aucun — 0 ou 1 marqueur distinct contre 2 à 5. Chaque signal
couvre l'angle mort de l'autre ; en conjonction, 0 faux positif sur 6.

**Corrigé depuis par la vérité terrain.** La règle exigeait 2 marqueurs, seuil
fixé sur six textes écrits à la main. Mesuré ensuite sur `HkayetErwi` — 432
blocs de récit tunisien humain, CC BY-SA 4.0 — ce réglage rejetait **37 % du
tunisien authentique** pour éviter un seul faux positif. Ramené à 1 marqueur :
87 % conservés au lieu de 63 %.

Leçon, et c'est la même que celle des six autres biais : **calibrer sur des
textes qu'on a écrits soi-même revient à mesurer sa propre main.** Il faut un
positif de référence externe avant de fixer un seuil.

Le classifieur, lui, reconnaît le récit authentique à 94 % avec 6,9 %
d'indécision : il n'est pas aveugle au registre narratif. Ce sont les sorties
de LLM qui sont réellement moins tunisiennes que du tunisien humain.

**Une AUC élevée ne prouve rien ici.** Toujours valider source par source :

```bash
darija data validate --model models/vs_maghreb.json.gz
```

La commande sort en erreur si la pire provenance tunisienne passe sous 90 %.

### 5. Le seuil de décision est appris, jamais 0,5

Les ancres `lo`/`hi` préservent l'ordre des scores mais pas la position de la
frontière : `lo` s'accroche au minimum de la classe négative, donc à une seule
valeur extrême. Un négatif atypique suffisait à faire classer « tunisien »
l'intégralité des contre-exemples, AUC intacte. Les seuils réels observés :
**0,599 et 0,856** selon la configuration.

---

## État mesuré du modèle de référence (`vs_maghreb`)

```
source     role           n   mediane  >=seuil
linto      positive    4000     0.955    99.2%
tsac       positive     885     0.918    99.6%
arbml_tn   positive    3758     0.903    96.1%
ary        negative    4000     0.522     0.0%   (prose formelle)
omcd       negative    1506     0.757     0.0%
mac        negative    2100     0.793     0.8%
dz         negative     354     0.805     5.4%
```

**Sait faire :** distinguer le tunisien du marocain et de l'algérien, sur
réseaux sociaux comme sur prose formelle, avec généralisation à des provenances
jamais vues.

**Ne sait pas faire :**
- **L'Arabizi.** TUNIZI est à 99,9 % en alphabet latin et le filtre `arabic_only`
  l'élimine → **0 bloc**. Le modèle ignore la forme écrite majoritaire du
  tunisien. C'est le trou le plus net.
- **Le registre littéraire ancien.** La poésie populaire n'est reconnue
  tunisienne que dans 43 % des cas. Ce registre n'est dans aucune classe.

---

## ⚠️ Licences — à vérifier avant toute publication

`darija data budget` liste l'état déclaré de chaque source.

- **Corpus de poésie** : issu de la مدوّنة الشعر الشعبي التونسي, corpus national
  publié en 10 volumes. Le dépôt d'origine est Apache-2.0 **pour le code
  seulement** ; le texte n'est pas couvert. **Usage interne / recherche tant que
  ce point n'est pas tranché.**
- **Sources d'entraînement** : LinTO (CC BY 4.0) et TSAC (LGPL-3.0) déclarent une
  licence. **arbml, TUNIZI, OMCD, MAC, oea_algd n'en déclarent aucune** — absence
  de licence ≠ domaine public.
- Le corpus algérien contenait des données personnelles (nom, pseudo, âge) :
  seule la colonne de texte est extraite, mentions `@` retirées.

---

## Pièges rencontrés, à ne pas refaire

- **LinTO concatène ses sources par blocs** : les 40 premiers pourcents sont de
  l'arabe standard. Une troncature en tête ne ramène aucun dialecte, sans le
  signaler. `fetch` échantillonne par réservoir.
- **QADI et NADI sont inutilisables** : QADI ne distribue que des identifiants de
  tweets, à hydrater via l'API X payante ; NADI exige un accord de données.
- **Aucun corpus libyen publiable** n'existe à ce jour.
- **Il n'existe pas de Wikipédia tunisienne** — seuls `ar`, `arz` (égyptien) et
  `ary` (marocain), qui servent donc de négatifs.
- **Collision module / fonction** : `darija.normalize` est à la fois un module et
  une fonction ré-exportée. Dans les tests, importer depuis le sous-module
  (`from darija.normalize import normalize`). Le module `assemble` a été renommé
  pour cette raison (c'était `build`).

---

## Conventions

- Python ≥ 3.11, `ruff` avec docstrings obligatoires (`D`), ligne à 100.
- Tests : un test doit dire **pourquoi** la règle existe, pas seulement la
  vérifier. Les tests de régression citent la mesure qui les a motivés.
- `data/raw/` et `models/` sont gitignorés (23 Mo de corpus téléchargés).
- Ne jamais annoncer une performance sur la seule AUC interne.

---

## Où en est le projet

**Le choix est tranché : `apps/darija-bench`.** Le socle a désormais un
consommateur, qui utilise ses cinq modules.

La réserve du roadmap — « un benchmark ne vaut que si quelqu'un construit des
modèles tunisiens » — a été levée par un recadrage : les systèmes à évaluer
existent déjà (Claude, GPT, Gemini, Qwen, Jais…), tous revendiquent l'arabe, et
aucun ne dit ce qu'il fait du tunisien. Il n'y a plus de pari sur des modèles à
venir.

**Ne pas relancer l'analyse « produire ou mesurer »** si la question revient —
elle est close, et `docs/ROADMAP.md` en garde le raisonnement.

### État du banc

Instrument complet et testé, **mais aucune campagne réelle n'a encore tourné** :
il n'y avait pas de clé d'API dans l'environnement au moment de l'écriture. Les
chiffres qui existent viennent de textes écrits à la main, pas de sorties de
modèles.

La première campagne fera donc deux choses à la fois : produire un premier
classement, et **valider la règle de décision elle-même** (voir biais nº 7).

### Suite naturelle

- **Faire tourner une campagne** — c'est ce qui manque, et rien d'autre ne
  peut le remplacer.
- **Un petit générateur par LoRA** comme premier système maison à passer au
  banc. L'ordre reste celui du roadmap : l'instrument d'abord, il le guide.
- **Arabizi** — le banc l'évalue via translittération approximative. Injecter
  du TUNIZI translittéré à l'entraînement du classifieur reste à tester ;
  c'est le trou le plus net du socle.
- **Registre littéraire** — la poésie fournirait une troisième classe.
- **Licence du corpus de poésie** — bloque toujours toute publication.
