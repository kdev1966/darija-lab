# tunisian-poetry-corpus

Corpus structuré de poésie populaire tunisienne (الشعر الشعبي التونسي / الملحون),
reconstruit à partir des artefacts de `tuni-folk-gemini`.

```
2 028 textes · 396 681 mots · 93 476 formes distinctes
```

## ⚠️ Licence — à lire avant tout usage

**Le statut juridique de ce texte n'est pas établi.** Le dépôt d'origine
(`tuni-folk-gemini`) est sous Apache-2.0, mais la licence y est formulée avec
soin : *« Apache-2.0 pour **le code** de ce dépôt »*. Elle ne couvre pas le texte.

La source est la **مدوّنة الشعر الشعبي التونسي**, corpus national publié en 10
volumes, vraisemblablement sous droits. Son auteur a délibérément choisi de ne
pas verser le corpus au dépôt — le texte n'y subsiste que parce qu'un exemple
d'apprentissage supervisé contient nécessairement le poème, puisque c'est sa
cible.

Traitez ces données comme **usage interne / recherche uniquement** tant que ce
point n'est pas tranché. Ne publiez ni le corpus, ni un modèle entraîné dessus,
sans avoir clarifié les droits.

**Le texte n'est donc pas dans ce dépôt.** Il l'a été : `corpus.csv` et
`corpus.jsonl` étaient versionnés quand le dépôt était privé, et sont devenus
publics en même temps que lui. Ils ont été retirés de tout l'historique
(`git filter-repo`), et `.gitignore` les empêche de revenir. Seul
`report.json` subsiste — ce ne sont que des décomptes.

`extract.py` les régénère en local à partir des artefacts de `tuni-folk-gemini`.

## Schéma

`data/corpus.jsonl` — une ligne JSON par texte. `data/corpus.csv` — mêmes
données, pour inspection au tableur. `data/report.json` — décomptes et couverture.

| Champ | Type | Couverture | Description |
|---|---|---:|---|
| `text` | str | 100 % | le texte, tel qu'imprimé (diacritiques comprises) |
| `genre` | str | 100 % | `qasim` `malzuma` `mawqif` `musaddas` `song` `riddle` `quatrain` `prose` |
| `genre_ar` | str | 100 % | le même, en arabe |
| `is_usul` | bool | 100 % | vrai pour les 4 formes à topologie de rimes vérifiable |
| `gharad` | str | 92.9 % | thème / propos (25 valeurs) |
| `wazn_sub` | str | 39.3 % | sous-mètre (بورجيلة, محدود القسيم, العروبي…) |
| `region` | str | 33.6 % | attache régionale |
| `uid` | str | 41.1 % | identifiant d'origine (`vol01-0088`) |
| `title` | str | 5.6 % | titre — entrées en prose uniquement |
| `poet` | str | **0 %** | ⚠️ voir ci-dessous |
| `n_lines` `n_words` | int | 100 % | décomptes dérivés |
| `modes` | list | 100 % | modes de prompt d'origine (`compose`, `continue`) |
| `split` | str | 100 % | `train` / `validation` du découpage SFT |

## Le poète est absent, et pourquoi

`sft/dataset.py` construit le prompt à partir du genre, du thème, du sous-mètre
et de la région — **jamais du poète**. Le champ n'a donc laissé aucune trace
dans les artefacts, et aucune reconstruction ne peut l'inventer.

Il existe dans la source d'origine (front matter `الشاعر` des fichiers markdown
du Diwan). Si vous obtenez ce répertoire un jour :

```bash
python extract.py --from-diwan "/chemin/vers/ديوان الشعر الشعبي التونسي"
```

Ce mode passe par le lecteur `tunifolk.data.diwan` et remplit `poet`, `uid` et
`title` pour l'ensemble du corpus.

## Comment les métadonnées ont été récupérées

Elles n'étaient **pas** stockées comme champs. Le dataset SFT ne contient que
`systemInstruction` et `contents` ; tout le reste était encodé dans la chaîne
arabe du prompt et a été réextrait par motif :

```
انظم الملزومة في غرض «الأخضر/الغزل».   ->  genre=malzuma, gharad=الأخضر/الغزل
وليكن على الميزان الفرعي «بورجيلة».     ->  wazn_sub=بورجيلة
وليكن بنَفَس أهل «قابس (بني زيد)».       ->  region=قابس (بني زيد)
```

Les `uid` proviennent d'une jointure distincte : le jeu de prompts RL stocke un
`source_uid` à côté d'une empreinte de novelty du poème source. Recalculer cette
empreinte sur notre texte réidentifie la source — d'où les 833 `uid` récupérés,
soit les poèmes qui servaient aussi à l'étape RL.

Chaque poème source produisait jusqu'à deux exemples SFT (`compose` et
`continue`) partageant la même cible : **3 942 exemples se réduisent à 2 028
textes uniques.** C'est aussi pourquoi tout décompte lexical fait directement sur
le dataset SFT est faussé par un facteur ~2.

## Reproduire

```bash
python extract.py                                  # depuis les artefacts (défaut)
python extract.py --from-artifacts ../../../tuni-folk-gemini
python extract.py --from-diwan "/chemin/vers/Diwan"   # complet, avec le poète
```

Aucune dépendance hors bibliothèque standard, sauf pour la jointure des `uid` et
le mode `--from-diwan`, qui importent `tunifolk` depuis le dépôt voisin. Si ce
dépôt est absent, la jointure est signalée et sautée — l'extraction aboutit
quand même.

## Nature des données

C'est du **dialecte authentique, dans un registre littéraire ancien**. La
grammaire tunisienne y est solide (`اللي` 47.7/10k mots, négation `ما...ش`
8.6/10k, `قداش`, `علاش`, `كيفاش`, `توا`), mais deux mots parmi les plus
caractéristiques du tunisien contemporain — `برشا` et `شنوة` — y sont quasi
absents (2 occurrences chacun).

Utile pour : morphologie dialectale, lexique, rimes, modélisation du registre
littéraire. **Inadapté** au tunisien parlé actuel, à l'Arabizi, à l'alternance
codique avec le français, au registre des réseaux sociaux.
