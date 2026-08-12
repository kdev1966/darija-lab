"""Ce que la preparation du negatif adversarial doit garantir, et pourquoi."""

from __future__ import annotations

from darija_bench import adversarial


def test_la_fuite_de_gabarit_est_coupee():
    # Qwen2.5-7B re-emet la consigne apres un jeton `user` sur 9 reponses de
    # 544. Les consignes de ce banc sont ECRITES EN TUNISIEN : la fuite injecte
    # donc du positif authentique dans la classe negative. Un modele entraine
    # dessus apprendrait que le tunisien est de la fusha.
    brut = "النص العربي الفصيح هنا.\nuser\nخويا الصغير ما يحبش يقرا. شنوة نعمل معاه؟"
    propre = adversarial.clean(brut)
    assert "شنوة" not in propre and "ما يحبش" not in propre
    assert propre.startswith("النص")


def test_l_alphabet_etranger_disparait():
    # Liste blanche et non liste noire : une liste noire enumerant les blocs
    # CJK avait laisse passer la ponctuation pleine chasse, ce qui a produit
    # 15 blocs de « ， ， ： » identiques des deux cotes du partage.
    assert "，" not in adversarial.clean("مرحبا ， ： 正品 بالعالم")
    assert "正品" not in adversarial.clean("مرحبا 正品 بالعالم")


def test_le_partage_ne_met_aucun_prompt_des_deux_cotes():
    # 16 reponses tirees d'une meme consigne sont des quasi-doublons. Les
    # repartir au hasard ferait mesurer la memorisation : c'est ainsi qu'un
    # gain de « 8,7 % -> 4,2 % » avait ete annonce, quand la mesure hors des
    # blocs vus donnait 66,7 % -> 60,0 %.
    mot = "الكلمة"
    rows = [
        {"prompt_id": f"tn-{i:03d}", "texte": f"{mot} {i} " * 40}
        for i in range(1, 21)
        for _ in range(4)
    ]
    train, val = adversarial.split_by_prompt(rows)
    assert train and val
    assert not (set(train) & set(val))


def test_le_partage_est_reproductible():
    # Sans graine fixe, chaque reconstruction du corpus changerait le jeu de
    # validation — et toute comparaison avec une mesure anterieure serait
    # muette sur ce qui a bouge.
    rows = [
        {"prompt_id": f"tn-{i:03d}", "texte": "كلمة " * 60} for i in range(1, 13)
    ]
    assert adversarial.split_by_prompt(rows) == adversarial.split_by_prompt(rows)


def test_les_fragments_sont_ecartes():
    # En dessous de 20 mots la reponse est une troncature ou un refus ; elle ne
    # represente pas le registre qu'on cherche a capturer.
    rows = [
        {"prompt_id": "tn-001", "texte": "لا"},
        {"prompt_id": "tn-002", "texte": "كلمة " * 60},
    ]
    train, val = adversarial.split_by_prompt(rows)
    assert "لا" not in train + val
