# darija-lab — contexte projet

Mono-repo de traitement du **tunisien (الدارجة التونسية)**. Bibliothèques
réutilisables, corpus, et applications qui les consomment.

**Langue de travail : le français.** Le code, les docstrings et les commentaires
sont en français ; les identifiants restent en anglais.

```
packages/darija-core/        bibliothèque socle (115 tests, ruff propre)
apps/darija-bench/           banc d'évaluation et tri de corpus (63 tests)
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
.venv/bin/python -m pytest -q     # 115 tests
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

### 4. Huit biais ont été trouvés et corrigés — ne pas les réintroduire

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
| 8 | marqueurs non discriminants | la règle « ≥ 1 marqueur » déclenchait sur 86,6 % du tunisien et **86,0 % du marocain** | `markers.DISCRIMINANT` |

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

**Le huitième est le prolongement du septième.** Compter les dix-neuf
marqueurs rendait la règle inopérante : elle déclenchait sur 86,6 % du
tunisien et 86,0 % du marocain — un écart de +0,6 point. Trois d'entre eux
sont aussi fréquents ailleurs qu'en tunisien, voire plus : le préfixe `ن-`
note le *je* en tunisien et le *nous* en arabe classique (66,6 % de la fusha),
`اللي` est **cinq fois plus fréquent en marocain**, `علاش` aussi. La décision
ne compte donc que `markers.DISCRIMINANT` — écart porté à +29,8 points.

Corollaire mesuré : ce filtre **ne sert pas partout**. Sur du texte humain il
coûte 10 à 37 points et ne gagne rien sur les contre-exemples, que le
classifieur seul rejette déjà à 99,3-100 %. Il est donc appliqué dans le banc
(sorties de LLM) et optionnel dans le tri (`--strict`).

**Le septième biais est désormais attaqué à la racine, pas seulement rattrapé
par une règle.** Idée empruntée à `tuni-folk-gemini` : si le classifieur trébuche
sur la fusha des LLM, il faut la lui montrer comme négatif. Un T4 gratuit sur
Colab produit ce corpus sans clé d'API — 544 réponses de Qwen2.5-7B sur les
sujets du banc, 855 blocs. Le contraste `vs_maghreb_llm` en résulte.

Mesuré **sur les 8 prompts jamais vus à l'entraînement** :

| | référence | `vs_maghreb_llm` |
|---|---|---|
| fusha de LLM classée tunisienne | 4,9 % | **0,0 %** |
| récit tunisien humain reconnu | 94,0 % | 94,2 % |
| pire provenance tunisienne | 96,1 % | 96,7 % |
| algérien mal classé | 5,4 % | **7,6 %** |
| marocain (`mac`) mal classé | 0,8 % | **1,8 %** |

Le gain sur le registre visé est net et le tunisien ne perd rien ; le coût est
sur le voisinage maghrébin, et il est réel. `vs_maghreb` reste donc la
référence tant que ce compromis n'est pas arbitré.

**Deux pièges à ne pas refaire de ce corpus** — tous deux documentés dans
`darija_bench.adversarial` :
1. Le modèle **ré-émet parfois la consigne**, qui est écrite *en tunisien*,
   après un jeton `user`. Sans troncature, du positif authentique entre dans la
   classe négative.
2. Le partage entraînement / validation se fait **par prompt**. Seize réponses
   d'une même consigne sont des quasi-doublons : les répartir au hasard fait
   mesurer la mémorisation. C'est exactement l'erreur du premier essai, qui a
   fait annoncer « 8,7 % → 4,2 % » quand la mesure hors des blocs vus donnait
   « 66,7 % → 60,0 % ».

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
  tunisien. C'était le trou le plus net.

  **Atténué, pas comblé.** `arabizi.to_arabic` ne notait aucune voyelle brève
  et produisait des formes inexistantes — `barcha` sortait `بارشا` au lieu de
  `برشا`, que ni le classifieur ni les marqueurs ne reconnaissent. Deux
  correctifs : les voyelles brèves internes sont supprimées, et les
  chiffres-lettres comptent comme des consonnes (`9alb` → `قلب`, non `قالب`).

  Mesuré sur TUNIZI : les blocs reconnus tunisiens passent de **47 % à 77 %**,
  la position médiane de 8 % à 31 %, et le tri du corpus de 0 % à **76,6 %**.

  **Puis l'alphabet réduit.** Ce que le latin ne peut pas noter — `س` contre
  `ص`, `ت` contre `ط`, `ا` contre `ة` — est confondu **des deux côtés** :
  corpus d'entraînement et texte translittéré. Le contraste
  `vs_maghreb_arabizi` est identique au contraste de référence, à ce repli
  près.

  ```
  source                 vs_maghreb   vs_maghreb_arabizi
  linto (positif)            96,3 %             96,3 %
  tsac (positif)             86,7 %             87,6 %
  arbml_tn (positif)         84,6 %             84,9 %
  négatifs                 0-0,7 %            0-1,3 %
  TUNIZI translittéré          77 %               87 %
  ```

  Les positifs sont intacts, les négatifs cèdent moins d'un point. Passer le
  modèle par `--arabizi-model` porte le tri de TUNIZI à **86,7 %**, contre 0 %
  il y a deux jours.

  Ce qui reste : le classifieur n'a toujours jamais vu d'Arabizi natif, et
  l'ambiguïté brève/longue est irréductible sans dictionnaire (`gal` → `ڨل` au
  lieu de `ڨال`). La voie directe — entraîner **sur** de l'Arabizi — reste
  bloquée faute de négatif maghrébin propre en écriture latine ; voir le
  contraste `vs_moroccan_latin` et son avertissement.
- **Le registre littéraire ancien.** La poésie populaire n'est reconnue
  tunisienne que dans 42,3 % des cas. Ce registre n'est dans aucune classe.

  **Ce chiffre mêle deux populations**, et `darija-bench triage` les sépare :

  | genre | n | reconnu tunisien | position médiane |
  |---|---|---|---|
  | malzuma | 1006 | 59,7 % | 25 % |
  | qasim | 385 | 41,9 % | 20 % |
  | song | 359 | 32,3 % | 32 % |
  | **prose** | **114** | **0 %** | **−18 %** |

  Les entrées `prose` ont un profil de métadonnées **disjoint** des poèmes :
  100 % ont un titre et 0 % une prosodie (`gharad`, `wazn_sub`, `modes`,
  `is_usul`), là où les poèmes ont l'inverse. Ce n'est pas de la poésie
  dialectale, et 91 % passent sous le seuil. Les filtrer relève la mesure.
  Leur fonction éditoriale exacte n'a pas été établie — les indices lexicaux
  d'un appareil critique (`تحقيق`, `الطبعة`) n'y sont pas plus fréquents que
  dans les poèmes.

  Deux causes distinctes expliquent le reste, que le taux global confondait :
  le **classifieur** rejette 20 à 27 % des vraies formes poétiques (registre
  absent de son entraînement), tandis que les **marqueurs** recalent les
  formes courtes — `song` a 65 % de textes sans marqueur discriminant pour
  89 mots de médiane, alors que sa position médiane est la meilleure du
  corpus. Le classifieur les reconnaît, la règle les rejette.

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
