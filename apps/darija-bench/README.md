# darija-bench

Un LLM interrogé en tunisien répond-il **en tunisien** — ou glisse-t-il vers la
fusha ou vers un autre dialecte maghrébin ?

Personne ne mesure ça aujourd'hui. Les modèles à évaluer, eux, existent déjà :
c'est ce qui distingue ce banc d'un pari sur des modèles tunisiens à venir.

```
42 prompts (34 en écriture arabe, 8 en Arabizi) × 2 conditions × N modèles
```

---

## Installation

```bash
cd apps/darija-bench
python3.12 -m venv .venv
.venv/bin/pip install -e ../../packages/darija-core -e ".[dev,anthropic]"
.venv/bin/python -m pytest -q      # 22 tests
.venv/bin/ruff check src tests
```

Les extras `[anthropic]`, `[openai]`, `[google]` sont indépendants : installer
celui dont on a besoin suffit. Aucun n'est requis pour **mesurer** des réponses
déjà collectées.

## Usage

```bash
# 1. Inspecter le protocole
darija-bench prompts

# 2. Collecter (appels facturés, confirmation demandée)
darija-bench run --model anthropic:claude-opus-5 --out replies.jsonl

# 3. Mesurer (gratuit, rejouable autant qu'on veut)
darija-bench report --replies replies.jsonl \
  --dialect-model ../../packages/darija-core/models/vs_maghreb.json.gz
```

Collecte et mesure sont **séparées** : les appels coûtent de l'argent, et
améliorer le scorer ne doit jamais obliger à les repayer. Une campagne
interrompue reprend là où elle s'est arrêtée.

L'identifiant de modèle est obligatoire pour OpenAI et Google : leurs
catalogues ne sont pas vérifiables depuis ce dépôt, et une chaîne devinée
produirait un 404 opaque au milieu d'une campagne.

---

## Les deux conditions

| condition | consigne | ce qu'elle mesure |
|---|---|---|
| `implicite` | aucune | le réflexe — le modèle suit-il la langue qu'on lui adresse ? |
| `explicite` | « réponds en tunisien » | la capacité — sait-il le faire quand on le demande ? |

L'écart entre les deux est l'observation intéressante. Un modèle qui échoue en
implicite et réussit en explicite ne manque pas de compétence, il manque de
calibration — et les deux défauts appellent des réponses différentes.

---

## Comment se décide « tunisien »

**Conjonction de deux signaux.** Ce n'était pas le design prévu ; il vient
d'une mesure qui a réfuté le premier.

`vs_maghreb` rejette l'arabe standard sans l'avoir jamais vu à l'entraînement —
0,4 % de faux positifs sur 4 000 blocs de Wikipédia arabe. Un axe unique
semblait donc suffire. Mais `ar` est encyclopédique, et une réponse d'assistant
en fusha sur un sujet du quotidien est un **troisième registre** : sur six
passages de ce type, le classifieur seul en classe **deux comme tunisiens**
(0,842 et 0,867 pour un seuil de 0,838). Soit 33 % de faux positifs là où
l'encyclopédique en donnait 0,4 %.

Les marqueurs échouent, eux, à séparer le tunisien du marocain — qui partage
`علاش` `كيفاش` `وين` `اللي`. Mais la fusha n'en utilise **aucun** : 0 ou 1
marqueur distinct côté fusha, 2 à 5 côté tunisien.

Chaque signal couvre l'angle mort de l'autre.

### La vérité terrain a corrigé cette règle

`HkayetErwi` — 432 blocs de **récit tunisien humain**, sous CC BY-SA 4.0 — a
servi de premier positif de référence. Il a démenti le diagnostic :

| minimum de marqueurs exigé | tunisien authentique conservé | faux positifs fusha |
|---|---|---|
| classifieur seul | 94,0 % | 2 / 6 |
| **≥ 1 (règle actuelle)** | **87,0 %** | 1 / 6 |
| ≥ 2 (règle initiale) | 63,2 % | 0 / 6 |

Exiger deux marqueurs rejetait **37 % du tunisien réel** pour éviter un unique
faux positif sur six textes écrits par l'auteur du banc. C'était calibrer
l'instrument sur sa propre main plutôt que sur la langue.

Le classifieur, lui, reconnaît le récit authentique à 94 %, avec seulement
6,9 % dans la bande d'indécision. **Il n'est donc pas aveugle au registre
narratif** — ce sont les sorties de modèles qui sont réellement moins
tunisiennes que du tunisien humain.

### Réserves, à lire avant de citer un chiffre

- **Aucun taux de « réponses tunisiennes » n'est publié**, et c'est délibéré :
  le seuil coupe au milieu de la distribution des sorties de modèles, et le
  registre déplace le niveau de base plus que le bruit de mesure. Un tel taux
  dépendrait de la composition du jeu de prompts, pas du modèle évalué.
- **Le corpus de validation n'est pas versé dans le dépôt.** CC BY-SA
  imposerait le partage à l'identique aux œuvres dérivées. Les tests se sautent
  quand il n'est pas en cache local ; la CI ne télécharge rien.
- **L'Arabizi passe par une translittération approximative.** `to_arabic` rend
  `barcha` en `بارشا`, que le motif des marqueurs ne reconnaît même pas. Le
  rapport sépare toujours ces lignes ; les mélanger donnerait un chiffre unique
  flatteur et faux.
- **Sous 25 mots, rien n'est décidable.** `min_words` du classifieur. Ces
  réponses sont comptées à part, jamais comme des échecs : un modèle laconique
  n'est pas un modèle qui parle fusha.
- **Les adaptateurs OpenAI et Google n'ont pas été exercés** contre l'API
  réelle, faute de clé. Seul Anthropic l'a été.

---

## Ce que consomme le banc

Les cinq modules de `darija-core`, chacun pour ce à quoi il a été conçu :

| module | rôle ici |
|---|---|
| `dialect` | décide : tunisien vs autre maghrébin |
| `markers` | écarte la fusha, et explique chaque verdict |
| `arabizi` | translittère les réponses en écriture latine |
| `normalize` | compte les mots, mesure la part d'écriture |
| `codeswitch` | profil d'alternance avec le français |

Aucune mesure n'est réinventée : tout ce qui décide vient du classifieur
validé, tout ce qui explique vient des marqueurs.

---

## Licence

Voir la note du dépôt racine. Le banc ne publie que des **prompts et des
scorers**, jamais les corpus — c'est ce qui rend son exposition juridique bien
plus légère que celle d'un générateur.
