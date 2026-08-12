"""Mesure de la tunisianité d'une réponse de modèle.

La décision est une **conjonction de deux signaux**, parce qu'aucun des deux ne
suffit seul. Ce n'était pas le design prévu : il vient d'une mesure qui a
réfuté le premier.

**Ce qu'on croyait.** Le classifieur contrastif ``vs_maghreb`` rejette l'arabe
standard sans l'avoir jamais vu à l'entraînement — sur 4 000 blocs de Wikipédia
arabe, médiane 0,786 pour un seuil appris à 0,838, soit 0,4 % au-dessus. Un
seul axe semblait donc couvrir les deux dérives, vers un autre maghrébin comme
vers la fusha.

**Ce qu'on a mesuré.** ``ar`` est de la prose encyclopédique. Une réponse
d'assistant en fusha sur un sujet du quotidien est un troisième registre, et
c'est celui que ce banc rencontre. Sur six passages de ce type écrits à la
main, le classifieur en classe **deux comme tunisiens** (0,842 et 0,867 pour un
seuil de 0,838) — 33 % de faux positifs là où l'encyclopédique en donnait
0,4 %. C'est le biais nº 6 du dépôt qui se répète : un agrégat rassurant qui ne
généralise pas au registre qui compte.

**Ce qui marche.** La fusha n'utilise pas les marqueurs du tunisien. Le
classifieur écarte très bien le marocain mais trébuche sur la fusha
conversationnelle ; les marqueurs font l'inverse. En conjonction, chacun
couvre l'angle mort de l'autre.

**Mais tous les marqueurs ne se valent pas**, et l'ignorer rendait la règle
inopérante. Mesuré sur les corpus du dépôt : la règle « au moins un marqueur »
déclenchait sur 86,6 % du tunisien et **86,0 % du marocain** — un tirage à pile
ou face. Trois coupables : le préfixe ``ن-`` note le *je* en tunisien et le
*nous* en arabe classique (66,6 % de la fusha), ``اللي`` est cinq fois plus
fréquent en marocain qu'en tunisien, ``علاش`` aussi.

La décision ne compte donc que :data:`darija.markers.DISCRIMINANT`. Effet
mesuré : le rappel sur 432 blocs de récit tunisien authentique passe de 88,0 %
à 76,9 %, et le déclenchement sur la fusha de 67,1 % à **2,0 %**. Onze points
de rappel contre soixante-cinq de précision, sur le registre où le classifieur
échoue.

Les marqueurs écartés restent **affichés** : ils expliquent, ils ne décident
pas.

Réserve indépendante : **la translittération de l'Arabizi est approximative**.
``arabizi.to_arabic`` rend ``barcha`` en ``بارشا`` et non ``برشا`` — que le
motif des marqueurs ne reconnaît même pas. Les scores sur l'Arabizi sont donc
indicatifs, et le rapport les sépare toujours du reste.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from darija import arabizi, codeswitch, markers
from darija.dialect import DialectModel
from darija.normalize import Level, normalize, script_ratio

#: Part de caractères latins au-delà de laquelle on tente la translittération.
#: Le seuil est haut à dessein : une réponse arabe contenant deux mots français
#: ne doit pas être translittérée, ça la détruirait.
LATIN_DOMINANT: float = 0.60

#: Nombre de marqueurs **distincts** exigés en plus du classifieur.
#:
#: Cette valeur a été fixée à 2 sur six textes de fusha écrits à la main, puis
#: **corrigée par la vérité terrain**. Mesuré sur ``HkayetErwi`` — 432 blocs de
#: récit tunisien authentique, sous licence CC BY-SA 4.0 :
#:
#: ===================  ==========================  ====================
#: minimum exigé        tunisien authentique gardé  faux positifs fusha
#: ===================  ==========================  ====================
#: classifieur seul     94,0 %                      2/6
#: ``>= 1``             87,0 %                      1/6
#: ``>= 2`` (ancien)    63,2 %                      0/6
#: ===================  ==========================  ====================
#:
#: Exiger deux marqueurs rejetait **37 % du tunisien authentique** pour éviter
#: un unique faux positif sur six textes que j'avais écrits moi-même. C'était
#: calibrer l'instrument sur son auteur plutôt que sur la langue.
#:
#: On compte les marqueurs distincts et non les occurrences : la diversité ne
#: dépend pas de la longueur, dix ``اللي`` ne prouvent rien de plus qu'un seul.
MIN_DISTINCT_MARKERS: int = 1


@dataclass(frozen=True)
class Verdict:
    """Le résultat de mesure d'une réponse.

    ``scorable`` est faux quand la réponse est trop courte pour le classifieur.
    Ce n'est pas un échec du modèle évalué : c'est une réponse dont on ne sait
    rien. Le rapport les compte séparément plutôt que de les traiter comme des
    échecs, ce qui gonflerait artificiellement le taux d'erreur.

    Les deux signaux de la conjonction — ``above_classifier`` et ``n_markers``
    — restent exposés séparément à côté du verdict ``is_tunisian``. C'est
    délibéré : la règle de conjonction est provisoire, et on doit pouvoir la
    réviser sur des données déjà collectées sans repayer les appels.
    """

    prompt_id: str
    model: str
    condition: str
    script: str
    n_words: int
    scorable: bool
    skipped: str | None = None
    score: float | None = None
    label: str | None = None
    above_classifier: bool | None = None
    n_markers: int | None = None
    is_tunisian: bool | None = None
    transliterated: bool = False
    script_ratio: dict[str, float] = field(default_factory=dict)
    codeswitch: dict[str, float] = field(default_factory=dict)
    explanation: str = ""

    def as_dict(self) -> dict[str, object]:
        """Vue sérialisable, pour le fichier de résultats."""
        return asdict(self)


def prepare(reply: str) -> tuple[str, bool]:
    """Ramène une réponse à une forme que le classifieur peut lire.

    Le filtre ``arabic_only`` de l'entraînement a éliminé l'intégralité de
    TUNIZI, donc le classifieur n'a **jamais vu d'Arabizi**. Lui en soumettre
    directement ne mesurerait rien. On translittère donc — en sachant que la
    conversion est lossy (voir le module).

    Returns:
      Le texte à scorer, et un drapeau disant s'il a été translittéré.

    """
    if not reply.strip():
        return "", False
    ratio = script_ratio(reply)
    if ratio.get("latin", 0.0) >= LATIN_DOMINANT and arabizi.is_arabizi(reply):
        return arabizi.to_arabic(reply), True
    return reply, False


def count_words(text: str) -> int:
    """Compte les mots comme le fait le classifieur, après normalisation."""
    return len(normalize(text, Level.STANDARD).split())


def evaluate(
    reply: str,
    model: DialectModel,
    *,
    prompt_id: str,
    model_name: str,
    condition: str,
    script: str,
) -> Verdict:
    """Mesure une réponse.

    Args:
      reply: le texte brut renvoyé par le modèle évalué.
      model: le classifieur de dialecte, chargé une fois par exécution.
      prompt_id: identifiant du prompt, reporté tel quel.
      model_name: nom du modèle évalué, reporté tel quel.
      condition: ``implicite`` ou ``explicite``.
      script: écriture du prompt d'origine, pour séparer les agrégats.

    """
    text, translit = prepare(reply)
    n_words = count_words(text)

    common = {
        "prompt_id": prompt_id,
        "model": model_name,
        "condition": condition,
        "script": script,
        "n_words": n_words,
        "transliterated": translit,
    }

    if not text:
        return Verdict(**common, scorable=False, skipped="réponse vide")
    if n_words < model.min_words:
        return Verdict(
            **common,
            scorable=False,
            skipped=f"trop court ({n_words} mots, minimum {model.min_words})",
            script_ratio=dict(script_ratio(text)),
        )

    predicted = model.predict(text)
    score = model.score(text)
    above = score >= model.threshold
    # Seuls les marqueurs discriminants entrent dans la décision. Compter
    # les dix-neuf rendait la règle inopérante : 86,6 % du tunisien mais
    # 86,0 % du marocain. Voir markers.DISCRIMINANT.
    distinct = len({m.marker for m in markers.find(text)} & markers.DISCRIMINANT)
    return Verdict(
        **common,
        scorable=True,
        score=round(score, 4),
        label=predicted[0] if predicted else None,
        above_classifier=above,
        n_markers=distinct,
        is_tunisian=above and distinct >= MIN_DISTINCT_MARKERS,
        script_ratio=dict(script_ratio(text)),
        codeswitch=dict(codeswitch.profile(text)),
        explanation=markers.explain(text),
    )
