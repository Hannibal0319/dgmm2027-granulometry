"""Verify candidate DOIs one by one against Crossref (title, year, container), politely rate-limited."""
import json
import re
import sys
import time
import urllib.request

sys.stdout.reconfigure(encoding="utf-8")
CAND = {
    "maragos1989": "10.1109/34.192465",
    "jones1996": "10.1016/0167-8655(96)00066-9",
    "adams1993": "10.1006/cgip.1993.1024",
    "soille2001": "10.1109/34.969120",
    "rosenfeld1968": "10.1016/0031-3203(68)90013-7",
    "vanderwalt2014": "10.7717/peerj.453",
    "borgefors1986": "10.1016/S0734-189X(86)80047-0",
    "normand2003": "10.1007/978-3-540-39966-7_14",
    "schneider2014": "10.1017/CBO9781139003858",
    "soille2003": "10.1007/978-3-662-05088-0",
    "vincent1994": "10.1007/978-94-011-1040-2_34",
    "hildebrand1997": "10.1046/j.1365-2818.1997.1340694.x",
    "murota2003": "10.1137/1.9780898718508",
    "haase2008": "10.37236/886",
    "jarnik1926": "10.1007/BF01216795",
}
tex = open("main.tex", encoding="utf8").read()
cited = set()
for m in re.finditer(r"\\cite\{([^}]*)\}", tex):
    cited |= {k.strip() for k in m.group(1).split(",")}
print("cited:", sorted(cited))
ok = {}
for key, doi in CAND.items():
    if key not in cited:
        continue
    url = "https://api.crossref.org/works/" + urllib.request.quote(doi, safe="") + "?mailto=13dzsoldosp@gmail.com"
    try:
        it = json.load(urllib.request.urlopen(url, timeout=30))["message"]
        title = (it.get("title") or [""])[0]
        cont = (it.get("container-title") or [""])[0]
        year = (it.get("issued", {}).get("date-parts") or [[None]])[0][0]
        print(f"{key:15s} {doi:40s} | {title[:70]} | {cont[:40]} | {year}", flush=True)
        ok[key] = dict(doi=doi, title=title, container=cont, year=year)
    except Exception as e:  # noqa: BLE001
        print(f"{key:15s} {doi:40s} | NOT FOUND ({e})", flush=True)
    time.sleep(2.0)
json.dump(ok, open("doi_verified.json", "w"), indent=1, ensure_ascii=False)
