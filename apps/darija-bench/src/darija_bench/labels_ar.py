"""Les libellés arabes de l'interface, séparés du code qui les calcule.

L'outil mesure du tunisien ; le lire en français était une bizarrerie. La page
est donc en arabe et en RTL, mais **rien de mesuré n'est traduit ici** : les
seuils, les bornes et les décisions restent dans `anchors` et `scoring`. Ce
module ne fait que nommer.

Deux libertés prises, et assumées :

- Les gloses des marqueurs sont **en arabe standard**, pas en tunisien. La
  glose décrit un fait de langue à quelqu'un qui lit ; l'écrire dans la langue
  décrite serait joli et illisible.
- La barre reste **en LTR** au milieu d'une page RTL. Elle porte une échelle
  numérique croissante de gauche à droite, et une position sur une droite
  graduée n'est pas du texte. L'inverser rendrait « 86 % » plus à gauche que
  « 57 % ».
"""

from __future__ import annotations

from typing import Final

#: Zones de dispersion — les clés viennent de :func:`anchors.zone`.
ZONES: Final[dict[str, str]] = {
    "bas": "تحت أدنى عُشر من الحكي البشري",
    "sous_typique": "تحت المجال المعتاد، لكن ضمن مدى الحكي البشري",
    "centre": "ضمن النصف الأوسط من الحكي البشري",
    "haut": "فوق ثلاثة أرباع الحكي البشري",
}

#: Catégories de marqueurs, telles que ``markers.MARKERS`` les nomme.
CATEGORIES: Final[dict[str, str]] = {
    "morphologie": "صرف",
    "mot-outil": "أداة",
    "lexique": "معجم",
}

#: Gloses des marqueurs, par clé de ``markers.MARKERS``. Une clé absente
#: retombe sur la glose française plutôt que de disparaître de la page.
MARQUEURS: Final[dict[str, str]] = {
    "n_prefix_1sg": "سابقة ن- للمتكلّم المفرد (نمشي)",
    "negation_ma_sh": "أداة النفي ما...ش (ماناكلش)",
    "future_bash": "أداة الاستقبال باش",
    "progressive_qaed": "صيغة الاستمرار قاعد",
    "relativizer_elli": "الموصول اللي",
    "interrog_chnowa": "شنوة (ماذا)",
    "interrog_3lach": "علاش (لماذا)",
    "interrog_9adech": "قدّاش (كم)",
    "interrog_kifach": "كيفاش (كيف)",
    "interrog_win": "وين (أين)",
    "interrog_waqtech": "وقتاش (متى)",
    "existential_famma": "فمّا (يوجد)",
    "now_tawa": "توّا (الآن)",
    "quant_barsha": "برشا (كثير)",
    "num_zouz": "زوز (اثنان)",
    "adj_behi": "باهي (حسن)",
    "enough_yezzi": "يزّي (كفى)",
    "little_chwaya": "شويّة (قليل)",
    "ok_yakhi": "ياخي (أليس كذلك)",
}
