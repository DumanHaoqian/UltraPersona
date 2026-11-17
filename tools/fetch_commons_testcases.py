"""Download openly licensed people photos from Wikimedia Commons as test cases.

Only files whose license is CC0 / public domain / CC BY / CC BY-SA are kept.
Every file gets a row in <out>/sources.csv (title, url, author, license).
Faces are counted later by tools/bucket_testcases.py.

Usage: python tools/fetch_commons_testcases.py <out_dir> [max_per_query]
"""
import csv
import html
import os
import re
import sys
import time
import urllib.parse
import urllib.request
import json

API = "https://commons.wikimedia.org/w/api.php"
UA = "UltraPersona-testcase-fetcher/0.1 (research test set; contact via repo)"
QUERIES_V1 = [
    "portrait photograph smiling woman",
    "portrait photograph man headshot",
    "couple portrait photograph",
    "two friends photograph",
    "three people portrait photograph",
    "family portrait photograph",
    "group photo four people",
    "group photo five people",
    "team group photo",
    "students group photo",
    "graduation group photo",
    "volunteers group photo",
]
QUERIES = [
    "Wikimedia meetup group photo",
    "Wikipedia editors group photo",
    "Wikimania group photo",
    "hackathon participants group photo",
    "conference speakers posing group photo",
    "colleagues posing group photo office",
    "students posing photo campus",
    "friends posing photo smiling",
    "couple posing photo outdoors",
    "two women smiling photo",
    "two men posing photo",
    "three friends posing photo",
    "five people posing group photo",
    "workshop participants group photo 2019",
    "volunteers posing group photo 2022",
]
OK_LICENSE = re.compile(r"^(cc0|public domain|pd|cc by(-sa)? [0-9.]+)", re.I)


def api(params):
    params = {**params, "format": "json"}
    req = urllib.request.Request(API + "?" + urllib.parse.urlencode(params), headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def strip_tags(s):
    return html.unescape(re.sub(r"<[^>]+>", "", s or "")).strip()


def main():
    out = sys.argv[1]
    per_query = int(sys.argv[2]) if len(sys.argv) > 2 else 15
    raw = os.path.join(out, "raw")
    os.makedirs(raw, exist_ok=True)
    csv_path = os.path.join(out, "sources.csv")
    seen = set()
    if os.path.exists(csv_path):
        with open(csv_path, encoding="utf-8") as f:
            seen = {row["file"] for row in csv.DictReader(f)}
    new_file = not os.path.exists(csv_path)
    fout = open(csv_path, "a", newline="", encoding="utf-8")
    w = csv.DictWriter(fout, fieldnames=["file", "query", "title", "page_url", "author", "license", "license_url", "width", "height"])
    if new_file:
        w.writeheader()

    for q in QUERIES:
        data = api({
            "action": "query", "generator": "search", "gsrsearch": f"{q} filetype:bitmap",
            "gsrnamespace": 6, "gsrlimit": 50, "prop": "imageinfo",
            "iiprop": "url|size|mime|extmetadata", "iiurlwidth": 1280,
        })
        pages = sorted(data.get("query", {}).get("pages", {}).values(), key=lambda p: p.get("index", 0))
        kept = 0
        for p in pages:
            if kept >= per_query:
                break
            ii = (p.get("imageinfo") or [{}])[0]
            meta = ii.get("extmetadata", {})
            lic = strip_tags(meta.get("LicenseShortName", {}).get("value", ""))
            if not OK_LICENSE.match(lic) or ii.get("mime") != "image/jpeg":
                continue
            if ii.get("width", 0) < 1000 or ii.get("height", 0) < 700:
                continue
            fname = re.sub(r"[^A-Za-z0-9._-]+", "_", p["title"].replace("File:", ""))[:120]
            if fname in seen:
                continue
            url = ii.get("thumburl") or ii["url"]
            try:
                req = urllib.request.Request(url, headers={"User-Agent": UA})
                with urllib.request.urlopen(req, timeout=120) as r, open(os.path.join(raw, fname), "wb") as f:
                    f.write(r.read())
            except Exception as e:  # noqa: BLE001
                print("skip", fname, e)
                continue
            seen.add(fname)
            kept += 1
            w.writerow({
                "file": fname, "query": q, "title": p["title"], "page_url": ii.get("descriptionurl", ""),
                "author": strip_tags(meta.get("Artist", {}).get("value", ""))[:200], "license": lic,
                "license_url": meta.get("LicenseUrl", {}).get("value", ""),
                "width": ii.get("width"), "height": ii.get("height"),
            })
            fout.flush()
            time.sleep(0.5)
        print(f"{q}: kept {kept}")
    fout.close()


if __name__ == "__main__":
    main()
