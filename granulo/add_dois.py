"""Insert the verified DOIs (see verify_dois.py) into refs.bib."""
import re

DOIS = {
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
}
s = open("refs.bib", encoding="utf8").read()
for key, doi in DOIS.items():
    m = re.search(r"(@\w+\{" + key + r",.*?)(\n\})", s, re.S)
    assert m, key
    body = m.group(1)
    if "doi =" in body:
        continue
    s = s[:m.start()] + body.rstrip().rstrip(",") + ",\n  doi = {" + doi + "}" + m.group(2) + s[m.end():]
s = s.replace("Radial Decomposition of Discs and Spheres", "Radial Decomposition of Disks and Spheres")
open("refs.bib", "w", encoding="utf8").write(s)
print(s.count("doi ="), "DOIs in refs.bib")
