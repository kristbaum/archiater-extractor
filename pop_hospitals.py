#!/usr/bin/env python3
"""Extract historic hospitals from the POP (Plateforme Ouverte du Patrimoine) API.

Queries the Mérimée base ("Patrimoine architectural") for a free-text term,
resolves a WGS84 coordinate for every notice, and writes CSV + GeoJSON.

Coordinates are resolved in three tiers, recorded in the `coord_source` column:
  pop      - POP_COORDONNEES, when present and non-zero
  ban      - geocoded from the notice address via the Base Adresse Nationale
  commune  - centroid of the commune (INSEE code), when no address resolves

POP stores addresses inverted ("Hôpital (rue de l') 3"), so they are rewritten
to "3 rue de l'Hôpital" before geocoding.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any, Iterator

import requests

POP_SEARCH = "https://api.pop.culture.gouv.fr/search/simple"
POP_NOTICE = "https://pop.culture.gouv.fr/notice/merimee/{ref}"
BAN_BULK = "https://api-adresse.data.gouv.fr/search/csv/"
GEO_COMMUNES = "https://geo.api.gouv.fr/communes"

PAGE_SIZE = 500
USER_AGENT = "pop-hospitals/1.0 (+heritage data extraction)"

# BAN returns the best match *within* the requested commune even when the street
# does not exist there, so a weak match must be rejected rather than trusted.
BAN_MIN_SCORE = 0.5

# A coordinate further than this from its commune centroid is reported as suspect.
MAX_KM_FROM_COMMUNE = 50.0

# Fields kept in the tabular output, in order.
COLUMNS = [
    "ref",
    "title",
    "denomination",
    "address",
    "commune",
    "insee",
    "departement",
    "region",
    "century",
    "protection",
    "latitude",
    "longitude",
    "coord_source",
    "coord_score",
    "coord_label",
    "coord_check",
    "notice_url",
]


def session() -> requests.Session:
    s = requests.Session()
    s.headers["User-Agent"] = USER_AGENT
    return s


def get_json(s: requests.Session, url: str, **kw) -> Any:
    """GET with retry on transient failures."""
    for attempt in range(5):
        try:
            r = s.get(url, timeout=60, **kw)
            r.raise_for_status()
            return r.json()
        except requests.HTTPError as exc:
            # 404/400 are answers, not failures; only server errors are worth retrying.
            if exc.response is None or exc.response.status_code < 500 or attempt == 4:
                raise
        except (requests.RequestException, ValueError):
            if attempt == 4:
                raise
        wait = 2**attempt
        print(f"  retry in {wait}s", file=sys.stderr)
        time.sleep(wait)
    raise AssertionError("unreachable")


# --------------------------------------------------------------------------
# POP
# --------------------------------------------------------------------------


def fetch_notices(s: requests.Session, text: str, base: str) -> list[dict]:
    """Page through /search/simple with a stable sort so pages cannot overlap."""
    hits: list[dict] = []
    offset = 0
    total = None
    while True:
        params = [
            ("bases[]", base),
            ("text", text),
            ("size", str(PAGE_SIZE)),
            ("from", str(offset)),
            ("sort[0][REF.keyword]", "asc"),
        ]
        data = get_json(s, POP_SEARCH, params=params)
        total = data["total"]
        page = data.get("hits", [])
        if not page:
            break
        hits.extend(h["_source"] for h in page)
        offset += len(page)
        print(f"  fetched {offset}/{total}", file=sys.stderr)
        if offset >= total:
            break
        time.sleep(0.2)

    # `from` paging is only as stable as the sort; drop any duplicate REF anyway.
    seen: set[str] = set()
    unique = []
    for h in hits:
        ref = h.get("REF")
        if ref and ref not in seen:
            seen.add(ref)
            unique.append(h)
    if total is not None and len(unique) != total:
        print(
            f"  warning: {len(unique)} unique notices for a reported total of {total}",
            file=sys.stderr,
        )
    return unique


def first(value: Any) -> str:
    """POP mixes scalars and arrays in the same field."""
    if isinstance(value, list):
        return str(value[0]).strip() if value else ""
    return str(value or "").strip()


def joined(value: Any) -> str:
    if isinstance(value, list):
        return "; ".join(str(v).strip() for v in value if str(v).strip())
    return str(value or "").strip()


def pop_coords(notice: dict) -> tuple[float, float] | None:
    """POP encodes 'no coordinate' as the literal point 0,0."""
    c = notice.get("POP_COORDONNEES")
    if not isinstance(c, dict):
        return None
    lat, lon = c.get("lat"), c.get("lon")
    if not lat or not lon:
        return None
    return float(lat), float(lon)


# --------------------------------------------------------------------------
# Addresses
# --------------------------------------------------------------------------

ADDRESS_RE = re.compile(r"^\s*(.*?)\s*\(([^)]*)\)\s*(.*?)\s*$")


def normalize_address(adrs: str) -> str:
    """'Hôpital (rue de l') 3' -> "3 rue de l'Hôpital"."""
    adrs = (adrs or "").strip()
    if not adrs:
        return ""
    m = ADDRESS_RE.match(adrs)
    if not m:
        return adrs
    name, kind, number = m.groups()
    kind = kind.strip()
    if not kind:
        return adrs
    # "rue de l'" glues onto the name; "rue" needs a space.
    sep = "" if kind.endswith("'") else " "
    street = f"{kind}{sep}{name}".strip()
    return f"{number} {street}".strip() if number else street


def geocode_bulk(s: requests.Session, rows: list[tuple[str, str, str]]) -> dict[str, dict]:
    """Geocode (ref, address, insee) triples in one BAN bulk-CSV request."""
    if not rows:
        return {}
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["ref", "adresse", "insee"])
    w.writerows(rows)
    payload = buf.getvalue().encode("utf-8")

    r = s.post(
        BAN_BULK,
        files={"data": ("query.csv", payload, "text/csv")},
        data={"columns": "adresse", "citycode": "insee"},
        timeout=300,
    )
    r.raise_for_status()

    out: dict[str, dict] = {}
    for row in csv.DictReader(io.StringIO(r.content.decode("utf-8"))):
        lat, lon = row.get("latitude"), row.get("longitude")
        if not lat or not lon:
            continue
        score = float(row.get("result_score") or 0)
        if score < BAN_MIN_SCORE:
            continue
        out[row["ref"]] = {
            "lat": float(lat),
            "lon": float(lon),
            "score": round(score, 3),
            "label": row.get("result_label", ""),
        }
    return out


def commune_centroids(s: requests.Session) -> dict[str, tuple[float, float]]:
    data = get_json(s, GEO_COMMUNES, params={"fields": "code,centre", "format": "json"})
    out = {}
    for c in data:
        centre = c.get("centre")
        if centre:
            lon, lat = centre["coordinates"]
            out[c["code"]] = (lat, lon)
    return out


def resolve_stray_communes(
    s: requests.Session,
    wanted: list[tuple[str, str]],
) -> dict[str, tuple[float, float]]:
    """Locate communes absent from the bulk list: (insee, name) pairs.

    The bulk endpoint omits the arrondissements of Paris/Lyon/Marseille, and POP
    still carries INSEE codes of communes that have since been merged away.
    """
    found: dict[str, tuple[float, float]] = {}
    for insee, name in wanted:
        if insee:
            try:
                c = get_json(s, f"{GEO_COMMUNES}/{insee}", params={"fields": "centre"})
                lon, lat = c["centre"]["coordinates"]
                found[insee] = (lat, lon)
                continue
            except requests.HTTPError:
                pass  # merged or invalid code; fall through to a name lookup

        if not name:
            continue
        try:
            d = get_json(
                s,
                "https://api-adresse.data.gouv.fr/search/",
                params={"q": name, "type": "municipality", "limit": 1},
            )
        except requests.RequestException:
            continue
        if not d["features"]:
            continue
        props = d["features"][0]["properties"]
        # Commune names repeat across France; only trust a same-department hit.
        if insee and not str(props.get("citycode", "")).startswith(insee[:2]):
            continue
        lon, lat = d["features"][0]["geometry"]["coordinates"]
        if insee:
            found[insee] = (lat, lon)
    return found


def haversine_km(a: tuple[float, float], b: tuple[float, float]) -> float:
    r = 6371.0
    lat1, lon1, lat2, lon2 = map(math.radians, (*a, *b))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * r * math.asin(math.sqrt(h))


# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------


def build_rows(
    notices: list[dict],
    ban: dict[str, dict],
    centroids: dict[str, tuple[float, float]],
) -> Iterator[dict]:
    for n in notices:
        ref = n["REF"]
        insee = first(n.get("INSEE"))

        lat = lon = None
        source = score = label = ""

        coords = pop_coords(n)
        if coords:
            lat, lon = coords
            source = "pop"
        elif ref in ban:
            hit = ban[ref]
            lat, lon, source = hit["lat"], hit["lon"], "ban"
            score, label = hit["score"], hit["label"]
        elif insee in centroids:
            lat, lon = centroids[insee]
            source, label = "commune", "centroïde de la commune"

        check = ""
        if lat is not None and source != "commune":
            ref_point = centroids.get(insee)
            if ref_point is None:
                check = "no_commune_reference"
            else:
                km = haversine_km((lat, lon), ref_point)
                check = "ok" if km <= MAX_KM_FROM_COMMUNE else f"far_from_commune_{km:.0f}km"
        elif source == "commune":
            check = "ok"

        yield {
            "ref": ref,
            "title": first(n.get("TICO")) or first(n.get("TITR")),
            "denomination": joined(n.get("DENO")),
            "address": normalize_address(first(n.get("ADRS"))),
            "commune": first(n.get("COM")),
            "insee": insee,
            "departement": first(n.get("DPT")),
            "region": first(n.get("REG")),
            "century": joined(n.get("SCLE")),
            "protection": joined(n.get("PROT")),
            "latitude": lat if lat is not None else "",
            "longitude": lon if lon is not None else "",
            "coord_source": source or "none",
            "coord_score": score,
            "coord_label": label,
            "coord_check": check,
            "notice_url": POP_NOTICE.format(ref=ref),
        }


def write_csv(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)


def write_geojson(rows: list[dict], path: Path) -> None:
    features = [
        {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [r["longitude"], r["latitude"]],
            },
            "properties": {k: v for k, v in r.items() if k not in ("latitude", "longitude")},
        }
        for r in rows
        if r["latitude"] != ""
    ]
    path.write_text(
        json.dumps({"type": "FeatureCollection", "features": features}, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--text", default="hôpital", help="free-text query (default: %(default)s)")
    p.add_argument("--base", default="merimee", help="POP base (default: %(default)s)")
    p.add_argument("--outdir", type=Path, default=Path("."), help="output directory")
    p.add_argument("--no-geocode", action="store_true", help="keep only POP coordinates")
    args = p.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    s = session()

    print(f"Querying POP base '{args.base}' for {args.text!r}...", file=sys.stderr)
    notices = fetch_notices(s, args.text, args.base)
    print(f"{len(notices)} notices", file=sys.stderr)

    ban: dict[str, dict] = {}
    centroids: dict[str, tuple[float, float]] = {}
    if not args.no_geocode:
        missing = [n for n in notices if not pop_coords(n)]
        print(f"{len(notices) - len(missing)} with POP coordinates, {len(missing)} to geocode", file=sys.stderr)

        todo = [
            (n["REF"], normalize_address(first(n.get("ADRS"))), first(n.get("INSEE")))
            for n in missing
            if normalize_address(first(n.get("ADRS"))) and first(n.get("INSEE"))
        ]
        if todo:
            print(f"Geocoding {len(todo)} addresses via BAN...", file=sys.stderr)
            ban = geocode_bulk(s, todo)
            print(f"  {len(ban)} resolved above score {BAN_MIN_SCORE}", file=sys.stderr)

        print("Fetching commune centroids...", file=sys.stderr)
        centroids = commune_centroids(s)

        stray = {
            (first(n.get("INSEE")), first(n.get("COM")))
            for n in missing
            if n["REF"] not in ban and first(n.get("INSEE")) not in centroids
        }
        if stray:
            print(f"Resolving {len(stray)} communes missing from the bulk list...", file=sys.stderr)
            extra = resolve_stray_communes(s, sorted(stray))
            print(f"  {len(extra)} resolved", file=sys.stderr)
            centroids.update(extra)

    rows = list(build_rows(notices, ban, centroids))

    csv_path = args.outdir / "hopitaux_merimee.csv"
    geo_path = args.outdir / "hopitaux_merimee.geojson"
    write_csv(rows, csv_path)
    write_geojson(rows, geo_path)

    located = sum(1 for r in rows if r["latitude"] != "")
    by_source: dict[str, int] = {}
    for r in rows:
        by_source[r["coord_source"]] = by_source.get(r["coord_source"], 0) + 1
    suspect = sum(1 for r in rows if r["coord_check"].startswith("far_from_commune"))

    print(f"\n{len(rows)} notices, {located} located ({located / len(rows):.0%})", file=sys.stderr)
    for src in ("pop", "ban", "commune", "none"):
        if src in by_source:
            print(f"  {src:<8} {by_source[src]}", file=sys.stderr)
    if suspect:
        print(f"  {suspect} coordinates far from their commune (see coord_check)", file=sys.stderr)
    print(f"\nWrote {csv_path} and {geo_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
