# Quelle application construire — décision prise

**État : tranché. C'est le benchmark → `apps/darija-bench`.** Le choix
appartenait à Othman ; ce document conserve le raisonnement qui y a mené, pour
que la question ne soit pas rouverte de zéro.

## Ce qui a levé la réserve

L'analyse ci-dessous se terminait sur un pari : « un benchmark ne vaut que si
quelqu'un construit des modèles tunisiens, ce qui reste marginal ».

Cette réserve reposait sur une erreur de cadrage. Elle supposait que les
systèmes à évaluer étaient **des modèles tunisiens à venir**. Or ils existent
déjà : Claude, GPT, Gemini, Qwen, Jais, AceGPT revendiquent tous l'arabe, et
aucun ne dit ce qu'il fait du tunisien. La question — *quand on lui parle en
tunisien, répond-il en tunisien ?* — est mesurable aujourd'hui, avec
l'instrument déjà construit. Il n'y a plus de pari.

Un second critère a départagé, appliqué au code plutôt qu'aux intentions :
lequel des candidats **consomme réellement le socle** ? Le benchmark utilise
les cinq modules, chacun pour ce à quoi il a été conçu. Le générateur LoRA en
utilise deux, et aux marges. L'explorateur de patrimoine deux aussi, et reste
bloqué par la licence.

Ce que la construction a appris, et qui n'était pas prévu : le classifieur seul
ne suffisait pas — voir le biais nº 7 dans `CLAUDE.md`.

---

## L'analyse d'origine

Conservée telle quelle. Elle reste valable pour l'ordre des travaux : mesurer
d'abord, produire ensuite.

---

## Les quatre candidats identifiés

| # | projet | état |
|---|---|---|
| 1 | Socle de traitement du Darija | ✅ **fait** — c'est `packages/darija-core` |
| 2 | Générateur de poésie tunisienne par LoRA | candidat |
| 3 | Benchmark d'évaluation Darija pour LLM | candidat |
| 4 | Moteur d'exploration du patrimoine | en attente (licence) |

### 2 — Générateur par LoRA sur modèle ouvert

Refaire ce que `tuni-folk-gemini` a fait, sans Vertex : LoRA sur Gemma 3 4B ou
Qwen 3, avec les 3 631 paires **déjà formatées** du dépôt d'origine.

Faisable : médiane 233 tokens par exemple, 99,7 % sous 2 000 tokens. Un GPU grand
public ou un Colab gratuit suffit — quelques heures, pas les milliers de dollars
qu'a coûté le pipeline Vertex.

### 3 — Benchmark d'évaluation

Un jeu de prompts en tunisien, les réponses de plusieurs modèles, un score
objectif par axe. `darija-core` fournit déjà tous les instruments :

- `dialect` — le modèle répond-il en tunisien, ou glisse-t-il vers la fusha ou le
  marocain ? **Personne ne mesure ça aujourd'hui.**
- `markers` — expliquer *pourquoi* une réponse est jugée peu tunisienne.
- `codeswitch` — le modèle alterne-t-il avec le français comme un locuteur réel ?
- `arabizi` — évaluer sur la forme écrite majoritaire, que tous les benchmarks
  arabes ignorent.
- **La méthodologie des six biais** — l'atout décisif : un benchmark qui mesure le
  genre ou la plateforme plutôt que la langue est pire qu'inutile.

### 4 — Exploration du patrimoine

Recherche par rime, forme, thème, poète, région sur les 2 028 textes. Vrais
utilisateurs (chercheurs, étudiants, musiciens). **Bloqué pour tout déploiement
public** par la licence du corpus.

---

## Produire du tunisien, ou mesurer ceux qui prétendent en produire

La question posée en fin de session. La réponse n'est pas symétrique.

### La dépendance ne va que dans un sens

On ne peut pas construire un bon générateur **sans** mesure. Commencer par
produire, c'est évaluer à l'œil — et retomber exactement dans le piège de cette
session : une AUC de 1.000 qui ne mesurait rien, pendant six itérations.

L'inverse est faux : un benchmark se construit et se valide sans avoir rien
produit, il suffit de systèmes existants à comparer.

**La mesure est en amont.** En produisant d'abord, on finit par la construire
quand même — mais dans l'urgence, mal, et pliée à son propre modèle.

### La nature du travail diffère

Produire fabrique quelque chose dont la qualité se **ressent** : retour immédiat,
satisfaisant, mais non fiable. Mesurer fabrique un **instrument** : sa qualité se
démontre — classe-t-il un bon système au-dessus d'un mauvais ? Retour lent,
abstrait, mais vérifiable.

C'est la différence entre un travail où l'on peut se tromper longtemps sans le
savoir, et un travail où l'erreur finit par se voir.

### La position concurrentielle est radicalement différente

Produire, c'est entrer dans une course où Google, OpenAI et les laboratoires
arabes financés à neuf chiffres sont déjà. Un LoRA sur 4 milliards de paramètres
ne battra pas Gemini sur la génération.

Mesurer, c'est être **pratiquement seul** — et ces mêmes laboratoires ont besoin
de l'instrument, car ils ne savent pas évaluer le tunisien. Position de levier,
pas de compétition.

### Le risque juridique n'est pas le même

Concret ici. Un générateur **mémorise** des données dont la licence n'est pas
tranchée (corpus de poésie, cinq sources sans licence déclarée) ; le publier les
redistribue sous une autre forme.

Un benchmark publie des **prompts et des scorers**, pas les corpus. La poésie peut
servir de référence interne sans jamais être diffusée. Exposition bien plus
légère.

### « Fini » ne veut pas dire la même chose

Un générateur n'est jamais fini : toujours un meilleur modèle, plus de données,
un rendement décroissant. Un benchmark **converge** : une fois validé, c'est un
artefact stable, et y ajouter un système à évaluer ne coûte presque rien.

---

## Recommandation

**Commencer par la mesure** — non par prudence, mais parce qu'elle est en amont :
elle sera nécessaire de toute façon, et la construire d'abord évite de la
construire mal.

**Réserve maintenue :** un benchmark ne vaut que si quelqu'un construit des
modèles tunisiens. C'est marginal aujourd'hui. C'est un pari sur la trajectoire
des LLM en langues sous-dotées — raisonnable, mais un pari.

> *Levée depuis — voir « Ce qui a levé la réserve » en tête de document. Les
> systèmes à évaluer sont les modèles généralistes existants, pas des modèles
> tunisiens à venir.*

**Compromis proposé :** le benchmark d'abord, puis **un** petit générateur — non
pour concurrencer, mais comme premier système à évaluer. Il valide l'instrument,
et l'instrument le guide. C'est le seul ordre où les deux se renforcent au lieu
de se disperser.

---

## Si le choix se porte sur le benchmark

Le trou le plus net à combler d'abord : **l'Arabizi**. Le classifieur en contient
0 bloc (`arabic_only` élimine intégralement TUNIZI), alors que c'est la forme
écrite majoritaire du tunisien. Un benchmark qui l'ignorerait raterait le cas le
plus fréquent.

Données : TUNIZI côté tunisien ; des jeux marocains en écriture latine existent
(voir « Sentiment Analysis Dataset in Moroccan Dialect: Bridging the Gap Between
Arabic and Latin Scripted dialect »).
