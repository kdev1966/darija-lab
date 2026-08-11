"""Interface locale : coller un texte, voir ce que l'instrument en dit.

Le banc mesurait jusqu'ici des campagnes entières en ligne de commande. C'est
utile pour comparer des modèles, inutile pour la question qu'on se pose le plus
souvent : *« ce texte-là, il est tunisien ou pas ? »*

Cette page répond à celle-là. On y colle n'importe quoi — une réponse de
modèle, un message, un bout de conte — et elle rend la mesure complète avec ce
qui l'a produite : la position entre les deux ancres humaines, les marqueurs
reconnus un par un, l'écriture, l'alternance codique.

**Aucune dépendance.** ``http.server`` de la bibliothèque standard suffit pour
un outil local, et ``darija-core`` n'impose rien à personne — le dépôt tient à
cette propriété, ce n'est pas ici qu'on va la casser.

**Local seulement.** Le serveur écoute sur ``127.0.0.1`` et rien d'autre. Ce
n'est pas un service : pas d'authentification, pas de limite de débit, aucune
protection. L'exposer sur un réseau serait une erreur.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer

from darija import arabizi, codeswitch, markers
from darija.dialect import DialectModel
from darija.normalize import Level, normalize, script_ratio

from . import anchors
from .scoring import MIN_DISTINCT_MARKERS, prepare

#: Taille maximale acceptée, en octets. Un outil local n'a pas besoin de plus,
#: et ça évite qu'un copier-coller malheureux fasse ramer la page.
MAX_BODY = 200_000


@dataclass
class Measure:
    """Tout ce que l'instrument sait dire d'un texte."""

    text: str
    model: DialectModel

    def as_dict(self) -> dict[str, object]:
        """Vue sérialisable, telle que la page la consomme."""
        scored, translit = prepare(self.text)
        n_words = len(normalize(scored, Level.STANDARD).split())
        out: dict[str, object] = {
            "n_words": n_words,
            "min_words": self.model.min_words,
            "transliterated": translit,
            "transliteration": scored if translit else None,
            "script": {k: round(v, 3) for k, v in script_ratio(self.text).items()},
            "arabizi_score": round(arabizi.arabizi_score(self.text), 3),
            "threshold": round(self.model.threshold, 4),
            "anchor_low": anchors.BAS,
            "anchor_high": anchors.HAUT,
        }
        if not scored.strip():
            out["status"] = "vide"
            return out
        if n_words < self.model.min_words:
            # Sous le minimum, `predict` rend None. Le dire plutôt que de
            # rendre un score inventé : indécidable n'est pas faux.
            out["status"] = "trop_court"
            return out

        score = self.model.score(scored)
        found = markers.find(scored)
        distinct = sorted({m.marker for m in found})
        out.update(
            status="mesure",
            score=round(score, 4),
            position=round(anchors.position(score), 4),
            above_classifier=score >= self.model.threshold,
            n_markers=len(distinct),
            min_markers=MIN_DISTINCT_MARKERS,
            is_tunisian=score >= self.model.threshold and len(distinct) >= MIN_DISTINCT_MARKERS,
            markers=[
                {
                    "name": name,
                    "category": markers.MARKERS[name][1],
                    "gloss": markers.MARKERS[name][2],
                    "count": sum(1 for m in found if m.marker == name),
                }
                for name in distinct
            ],
            codeswitch={k: round(v, 3) for k, v in codeswitch.profile(scored).items()},
        )
        return out


PAGE = """<!doctype html>
<html lang="fr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>darija-bench — mesurer un texte</title>
<style>
:root {
  --bg:#faf9f7; --fg:#1c1a17; --muted:#6b6560; --line:#ddd8d2;
  --card:#fff; --accent:#8a5a2b; --ok:#2f6b3f; --no:#8c3a3a;
}
@media (prefers-color-scheme:dark){:root{
  --bg:#16150f; --fg:#ece7df; --muted:#9b938a; --line:#33302a;
  --card:#1f1d17; --accent:#d19a5e; --ok:#7fb98a; --no:#d98a8a;}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
  font:16px/1.55 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;}
.wrap{max-width:860px;margin:0 auto;padding:2rem 1.25rem 4rem}
h1{font-size:1.35rem;margin:0 0 .25rem;font-weight:650}
.sub{color:var(--muted);margin:0 0 1.5rem;font-size:.92rem}
textarea{width:100%;min-height:170px;padding:.9rem;border:1px solid var(--line);
  border-radius:10px;background:var(--card);color:var(--fg);
  font:inherit;font-size:1.05rem;resize:vertical}
textarea:focus{outline:2px solid var(--accent);outline-offset:1px}
.row{display:flex;gap:.75rem;align-items:center;margin:.85rem 0 0;flex-wrap:wrap}
button{background:var(--accent);color:#fff;border:0;border-radius:8px;
  padding:.6rem 1.15rem;font:inherit;font-weight:600;cursor:pointer}
button:disabled{opacity:.5;cursor:default}
.hint{color:var(--muted);font-size:.85rem}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;
  padding:1.1rem 1.2rem;margin-top:1.5rem}
.big{font-size:2.4rem;font-weight:700;line-height:1;margin:0}
.scale{position:relative;height:12px;background:linear-gradient(90deg,#c9b8a4,#7fb98a);
  border-radius:99px;margin:1.1rem 0 .35rem}
.pin{position:absolute;top:-5px;width:4px;height:22px;background:var(--fg);border-radius:2px}
.ends{display:flex;justify-content:space-between;color:var(--muted);font-size:.8rem}
table{width:100%;border-collapse:collapse;margin-top:.6rem;font-size:.92rem}
td,th{text-align:left;padding:.35rem .5rem;border-bottom:1px solid var(--line)}
th{color:var(--muted);font-weight:600}
.ar{font-size:1.15rem}
.tag{display:inline-block;padding:.12rem .5rem;border-radius:99px;
  font-size:.78rem;border:1px solid var(--line);color:var(--muted)}
.ok{color:var(--ok)} .no{color:var(--no)}
.warn{border-left:3px solid var(--accent);padding-left:.9rem;color:var(--muted)}
details{margin-top:.9rem} summary{cursor:pointer;color:var(--muted);font-size:.9rem}
</style></head><body><div class="wrap">
<h1>Ce texte est-il en tunisien&nbsp;?</h1>
<p class="sub">Collez n'importe quoi — une réponse de modèle, un message, un conte.
L'écriture latine est translittérée automatiquement.</p>

<textarea id="t" dir="auto" placeholder="برشا علاش قداش... ou chnowa a7welek..."></textarea>
<div class="row">
  <button id="go">Mesurer</button>
  <span class="hint" id="hint"></span>
</div>
<div id="out"></div>

<script>
const $=s=>document.querySelector(s);
const pct=x=>(x*100).toFixed(0)+"%";

function render(d){
  if(d.status==="vide") return `<div class="card warn">Rien à mesurer.</div>`;
  if(d.status==="trop_court") return `<div class="card warn">
    <strong>Indécidable</strong> — ${d.n_words} mots, il en faut ${d.min_words}.<br>
    Le classifieur ne rend pas de verdict en dessous. Ce n'est pas un échec du
    texte&nbsp;: c'est un texte dont on ne sait rien.</div>`;

  const p=Math.max(-0.35,Math.min(1.35,d.position));
  const left=((p+0.35)/1.7*100).toFixed(1);
  const verdict=d.is_tunisian
    ? `<span class="ok">tunisien</span>`
    : `<span class="no">pas assez tunisien</span>`;

  const mk=d.markers.length
    ? d.markers.map(m=>`<tr><td>${m.gloss}</td>
        <td><span class="tag">${m.category}</span></td>
        <td>${m.count}</td></tr>`).join("")
    : `<tr><td colspan="3" class="hint">aucun marqueur tunisien détecté</td></tr>`;

  const cs=Object.entries(d.codeswitch).filter(([,v])=>v>0)
    .map(([k,v])=>`${k} ${pct(v)}`).join(" · ")||"—";

  return `<div class="card">
    <p class="big">${pct(d.position)}</p>
    <p class="sub" style="margin:.3rem 0 0">position entre deux textes humains — ${verdict}</p>
    <div class="scale"><div class="pin" style="left:${left}%"></div></div>
    <div class="ends"><span>récit en fusha</span><span>récit tunisien humain</span></div>
    <table>
      <tr><th>score brut</th><td>${d.score}
        <span class="hint">(seuil ${d.threshold})</span></td></tr>
      <tr><th>marqueurs distincts</th><td>${d.n_markers}
        <span class="hint">(minimum ${d.min_markers})</span></td></tr>
      <tr><th>mots</th><td>${d.n_words}</td></tr>
      <tr><th>alternance codique</th><td>${cs}</td></tr>
    </table>
    ${d.transliterated?`<div class="warn" style="margin-top:.9rem">
      Écriture latine détectée et translittérée avant mesure. La conversion est
      approximative&nbsp;: lisez ce score comme un indice.
      <div class="ar" dir="rtl" style="margin-top:.4rem">${d.transliteration}</div></div>`:""}
    <details open><summary>marqueurs reconnus</summary>
      <table><tr><th>marqueur</th><th>type</th><th>n</th></tr>${mk}</table>
    </details>
  </div>`;
}

async function go(){
  const t=$("#t").value;
  $("#go").disabled=true; $("#hint").textContent="mesure…";
  try{
    const r=await fetch("/api/measure",{method:"POST",
      headers:{"content-type":"application/json"},body:JSON.stringify({text:t})});
    const d=await r.json();
    $("#out").innerHTML=r.ok?render(d):`<div class="card warn">${d.error||"erreur"}</div>`;
    $("#hint").textContent="";
  }catch(e){ $("#hint").textContent="erreur : "+e.message; }
  $("#go").disabled=false;
}
$("#go").addEventListener("click",go);
$("#t").addEventListener("keydown",e=>{if((e.metaKey||e.ctrlKey)&&e.key==="Enter")go();});
</script>
</div></body></html>
"""


def make_handler(model: DialectModel) -> type[BaseHTTPRequestHandler]:
    """Construit le gestionnaire, le modèle étant chargé une fois pour toutes."""

    class Handler(BaseHTTPRequestHandler):
        """Sert une page et un point de mesure."""

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802 - imposé par la classe de base
            """Sert la page."""
            if self.path not in ("/", "/index.html"):
                self._send(404, b"introuvable", "text/plain; charset=utf-8")
                return
            self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")

        def do_POST(self) -> None:  # noqa: N802 - imposé par la classe de base
            """Mesure le texte reçu."""
            if self.path != "/api/measure":
                self._send(404, b'{"error":"introuvable"}', "application/json")
                return
            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_BODY:
                self._send(413, b'{"error":"texte trop long"}', "application/json")
                return
            try:
                payload = json.loads(self.rfile.read(length) or b"{}")
                result = Measure(str(payload.get("text", "")), model).as_dict()
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                self._send(
                    400,
                    json.dumps({"error": f"requête illisible : {exc}"}).encode("utf-8"),
                    "application/json; charset=utf-8",
                )
                return
            self._send(
                200,
                json.dumps(result, ensure_ascii=False).encode("utf-8"),
                "application/json; charset=utf-8",
            )

        def log_message(self, *args: object) -> None:
            """Silence : le serveur tourne au premier plan, les logs gênent."""

    return Handler


def serve(model: DialectModel, *, port: int = 8000) -> None:
    """Démarre le serveur local. Bloquant, arrêt par Ctrl-C."""
    httpd = HTTPServer(("127.0.0.1", port), make_handler(model))
    print(f"darija-bench — http://127.0.0.1:{port}  (Ctrl-C pour arrêter)")
    print(f"  échelle : {anchors.BAS} = récit fusha, {anchors.HAUT} = récit tunisien humain")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\narrêté.")
    finally:
        httpd.server_close()
