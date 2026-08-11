# darija-lab

**Mono-repo de traitement du tunisien (الدارجة التونسية).**
Bibliothèques réutilisables, corpus, et applications qui les consomment.

```
packages/darija-core/          socle : normalisation, Arabizi, alternance
                               codique, marqueurs, classification de dialecte
data/tunisian-poetry-corpus/   2 028 textes de poésie populaire, 396 681 mots
apps/                          applications (vide)
docs/HANDOVER.md               historique détaillé et mesures
CLAUDE.md                      contexte projet
```

---

## Démarrer

```bash
cd packages/darija-core
python3.12 -m venv .venv && .venv/bin/pip install -e ".[dev,data]"
.venv/bin/python -m pytest -q          # 98 tests
.venv/bin/ruff check src tests
```

Puis, en ligne de commande :

```bash
echo "chnowa a7welek" | .venv/bin/darija translit
echo "ken 3andek le temps" | .venv/bin/darija segment
.venv/bin/darija data budget           # coût et licences, avant de télécharger
```

---

## Ce que fait `darija-core`

Cinq briques indépendantes, **zéro dépendance d'exécution** :

| module | rôle |
|---|---|
| `normalize` | normalisation orthographique **qui préserve le dialecte** |
| `arabizi` | translittération Arabizi ↔ arabe, et détection |
| `codeswitch` | segmentation arabe / français / Arabizi |
| `markers` | 20 marqueurs morphologiques, pour **inspecter** |
| `dialect` | classifieur contrastif entraînable, pour **décider** |
| `data` | récupération des corpus, assemblage, entraînement, validation |

La différence avec les chaînes arabes usuelles tient en une phrase : elles
normalisent vers l'arabe standard, ce qui détruit le dialecte. Ici `برشا`
`علاش` `قداش` `اللي` traversent intacts, et les lettres maghrébines `ڨ ڥ پ چ`
survivent — sans quoi `ڨلب` deviendrait `لب`.

---

## L'essentiel à savoir

**Le classifieur de dialecte est contrastif.** Il n'apprend pas « voici du
tunisien », il apprend « ceci plutôt que cela ». Le choix des contre-exemples
détermine ce qu'il sait faire.

**Une AUC élevée n'y prouve rien.** Six biais successifs ont produit des AUC
proches de 1.000 sans qu'aucune ne mesure le dialecte : genre, alphabet,
plateforme, entités nommées, provenance, registre. Ils sont documentés dans
`CLAUDE.md`, corrigés dans le code, et verrouillés par des tests.

La validation utile se fait **source par source** :

```bash
darija data validate --model models/vs_maghreb.json.gz
```

---

## ⚠️ Licences

Plusieurs corpus d'entraînement **ne déclarent aucune licence**, et le corpus de
poésie provient d'une publication nationale dont les droits ne sont pas
tranchés. Usage interne / recherche tant que ce point n'est pas réglé.
`darija data budget` liste l'état déclaré de chaque source.

Le code de ce dépôt est sous Apache-2.0.
