"""Look up DOI candidates for every cited reference in Crossref (manual verification follows)."""
import json
import re
import urllib.parse
import urllib.request

bib = open("refs.bib", encoding="utf8").read()
tex = open("main.tex", encoding="utf8").read()
cited = set()
for m in re.finditer(r"\\cite\{([^}]*)\}", tex):
    cited |= {k.strip() for k in m.group(1).split(",")}
entries = re.findall(r"@(\w+)\{(\w+),(.*?)\n\}", bib, re.S)
out = {}
for typ, key, body in entries:
    if key not in cited:
        continue
    t = re.search(r"title\s*=\s*\{(.*?)\},", body, re.S)
    a = re.search(r"author\s*=\s*\{(.*?)\}", body, re.S)
    title = re.sub(r"[{}\\$]", "", t.group(1)) if t else ""
    auth = re.sub(r"[{}\\\"']", "", a.group(1)).split(",")[0] if a else ""
    q = urllib.parse.quote(f"{title} {auth}")
    try:
        r = json.load(urllib.request.urlopen(
            f"https://api.crossref.org/works?query.bibliographic={q}&rows=1", timeout=30))
        it = r["message"]["items"][0]
        out[key] = dict(doi=it.get("DOI"), title=(it.get("title") or [""])[0], score=it.get("score", 0),
                        year=(it.get("issued", {}).get("date-parts") or [[None]])[0][0])
    except Exception as e:  # noqa: BLE001
        out[key] = dict(doi=None, title=str(e)[:60], score=0, year=None)
    print(f"{key:16s} | {title[:50]:50s} || {out[key]['doi']} | {out[key]['title'][:60]} | "
          f"{out[key]['year']} | {out[key]['score']:.0f}", flush=True)
json.dump(out, open("doi_candidates.json", "w"), indent=1)
print(len(cited), "cited keys")
