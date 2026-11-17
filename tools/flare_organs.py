#!/usr/bin/env python3
"""Поджелудочная железа и надпочечники из FLARE-MedFM/PancancerCTSeg.

Набор данных: https://huggingface.co/datasets/FLARE-MedFM/PancancerCTSeg
(MICCAI 2026 Challenge: Pan-cancer segmentation in CT scans, https://www.codabench.org/competitions/7149/).

Программа умеет три вещи:

  list      показать подмножества, ссылки, число случаев, источники и лицензии;
  inspect   прочитать дерево файлов подмножества через API Hugging Face
            (без скачивания) и посчитать файлы, объём и парность images/labels;
  download  скачать подмножество через huggingface_hub.

Доступ к набору закрыт условиями (gated = auto). Перед скачиванием примите условия
на странице набора и экспортируйте токен:  export HF_TOKEN=...

Примеры:
  python tools/flare_organs.py list
  python tools/flare_organs.py inspect --organ all
  python tools/flare_organs.py download --organ pancreas --out data
  python tools/flare_organs.py list --json

Зависимости: только стандартная библиотека (list/inspect).
Для download нужен пакет huggingface_hub:  pip install -U huggingface_hub
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

DATASET_ID = "FLARE-MedFM/PancancerCTSeg"
DATASET_URL = f"https://huggingface.co/datasets/{DATASET_ID}"
API_TREE_URL = f"https://huggingface.co/api/datasets/{DATASET_ID}/tree/main/"
USER_AGENT = (
    "radiomics-atlas-flare-organs/1.0 "
    "(+https://github.com/adrenolitik/Radiomics-Atlas)"
)

ORGANS = {
    "pancreas": {
        "name_en": "Pancreas",
        "name_ru": "Поджелудочная железа",
        "subset": "train_per_cancer_type/Dataset006_PancreaticCancer",
        "cases": 1797,
        "labels": "опухоль поджелудочной железы (labelsTr) + маска органа (labelsTr_organ)",
        "organ_masks": True,
        "sources": [
            {
                "name": "PanTS",
                "cases": 876,
                "license": "CC BY-NC-ND 4.0",
                "url": "https://github.com/MrGiovanni/PanTS",
            },
            {
                "name": "PANORAMA",
                "cases": 479,
                "license": "CC BY 4.0",
                "url": "https://panorama.grand-challenge.org/",
            },
            {
                "name": "MSD-Pancreas",
                "cases": 281,
                "license": "CC BY-SA 4.0",
                "url": "http://medicaldecathlon.com/",
            },
            {
                "name": "PanTrack",
                "cases": 161,
                "license": "CC BY-NC-SA 4.0",
                "url": "https://huggingface.co/datasets/mrokuss/PanTrack",
            },
        ],
        "notes": [
            "labelsTr_organ/: 0 = фон, 1 = весь орган, 2 = опухоль; маска органа получена TotalSegmentator и вручную не проверялась.",
            "Объединённый train_label этого набора содержит только опухоль, без масок органов.",
            "В четырёх случаях маска органа пустая: Pancreas_PanTS_00002916, Pancreas_PanTS_00006447, Pancreas_PanTS_00007898, Kidney_KiTS23_00516.",
        ],
    },
    "adrenal": {
        "name_en": "Adrenal glands",
        "name_ru": "Надпочечники",
        "subset": "train_per_cancer_type/Dataset005_AdrenalCancer",
        "cases": 52,
        "labels": "опухоль надпочечника, адренокортикальный рак (labelsTr)",
        "organ_masks": False,
        "sources": [
            {
                "name": "Adrenal-ACC-Ki67-Seg",
                "cases": 52,
                "license": "CC BY 3.0",
                "url": "https://doi.org/10.7937/1FPG-VM46",
            },
        ],
        "notes": [
            "Отдельной маски нормального надпочечника в наборе нет: labelsTr содержит только опухоль.",
            "Комбинированные маски органа (labelsTr_organ) опубликованы только для лёгких, печени, поджелудочной железы и почек.",
        ],
    },
}

ORGAN_CHOICES = ("pancreas", "adrenal", "all")


# --------------------------------------------------------------------------- #
# Служебные функции
# --------------------------------------------------------------------------- #

def organ_url(key: str) -> str:
    """Ссылка на папку подмножества на Hugging Face."""
    return f"https://huggingface.co/datasets/{DATASET_ID}/tree/main/{ORGANS[key]['subset']}"


def _headers() -> dict:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    headers = {"User-Agent": USER_AGENT}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _next_link(link_header: str):
    """URL следующей страницы из заголовка Link (пагинация API Hugging Face)."""
    if not link_header:
        return None
    for part in link_header.split(","):
        segments = part.split(";")
        if len(segments) < 2:
            continue
        rel = ";".join(segments[1:]).replace(" ", "")
        if 'rel="next"' in rel or "rel=next" in rel:
            return segments[0].strip().strip("<>")
    return None


def fetch_tree(path: str) -> list:
    """Все файлы и папки под path (рекурсивно), с обходом пагинации."""
    url = f"{API_TREE_URL}{quote(path)}?recursive=true"
    entries: list = []
    while url:
        request = Request(url, headers=_headers())
        try:
            with urlopen(request, timeout=60) as response:
                entries.extend(json.load(response))
                url = _next_link(response.headers.get("Link"))
        except HTTPError as exc:
            if exc.code in (401, 403):
                raise SystemExit(
                    f"Нет доступа к {path} (HTTP {exc.code}).\n"
                    f"Примите условия на {DATASET_URL} и задайте HF_TOKEN."
                )
            raise SystemExit(f"Ошибка API Hugging Face для {path}: HTTP {exc.code}")
        except URLError as exc:
            raise SystemExit(f"Нет связи с huggingface.co: {exc.reason}")
    return entries


def human_size(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.0f} {unit}" if unit in ("B", "KB") else f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TB"


def summarize(key: str, entries: list) -> dict:
    """Сводка по подмножеству: файлы и объём по папкам, парность images/labels."""
    subset = ORGANS[key]["subset"]
    files = [entry for entry in entries if entry.get("type") == "file"]
    grouped: dict = {}
    for entry in files:
        relative = entry["path"][len(subset) + 1:]
        folder = relative.rsplit("/", 1)[0] if "/" in relative else "(root)"
        grouped.setdefault(folder, []).append(entry)

    folders = {
        name: {
            "files": len(items),
            "bytes": sum(int(item.get("size") or 0) for item in items),
        }
        for name, items in sorted(grouped.items())
    }

    images = {
        entry["path"].rsplit("/", 1)[-1].replace("_0000.nii.gz", "")
        for entry in grouped.get("imagesTr", [])
    }
    labels = {
        entry["path"].rsplit("/", 1)[-1].replace(".nii.gz", "")
        for entry in grouped.get("labelsTr", [])
    }

    return {
        "organ": key,
        "name_ru": ORGANS[key]["name_ru"],
        "subset": subset,
        "url": organ_url(key),
        "total_files": len(files),
        "total_bytes": sum(int(entry.get("size") or 0) for entry in files),
        "folders": folders,
        "images": len(images),
        "labels": len(labels),
        "labels_without_image": len(labels - images),
        "images_without_label": len(images - labels),
        "organ_masks_published": ORGANS[key]["organ_masks"],
    }


# --------------------------------------------------------------------------- #
# Команды
# --------------------------------------------------------------------------- #

def cmd_list(as_json: bool) -> int:
    if as_json:
        payload = {
            "dataset": DATASET_ID,
            "url": DATASET_URL,
            "organs": {
                key: {
                    **{k: v for k, v in value.items() if k != "sources"},
                    "sources": value["sources"],
                    "url": organ_url(key),
                }
                for key, value in ORGANS.items()
            },
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    print(f"Набор: {DATASET_ID}")
    print(f"Ссылка: {DATASET_URL}")
    print("Доступ: gated (нужно принять условия на странице набора и задать HF_TOKEN)")
    print("Лицензия набора: CC BY-NC 4.0; у исходных наборов лицензии различаются")
    print()

    for key, organ in ORGANS.items():
        print(f"== {organ['name_ru']} / {organ['name_en']} ==")
        print(f"  папка:    {organ['subset']}")
        print(f"  ссылка:   {organ_url(key)}")
        print(f"  случаев:  {organ['cases']}")
        print(f"  разметка: {organ['labels']}")
        print(f"  маска органа: {'есть (labelsTr_organ)' if organ['organ_masks'] else 'нет, только опухоль'}")
        print("  источники:")
        for source in organ["sources"]:
            print(
                f"    - {source['name']}: {source['cases']} случаев, "
                f"{source['license']} — {source['url']}"
            )
        for note in organ["notes"]:
            print(f"  ! {note}")
        print()

    print("Скачивание: python tools/flare_organs.py download --organ all --out data")
    return 0


def cmd_inspect(key: str, as_json: bool) -> int:
    keys = list(ORGANS) if key == "all" else [key]
    summaries = []
    for organ_key in keys:
        print(f"Читаю дерево файлов: {ORGANS[organ_key]['subset']} ...", file=sys.stderr)
        summaries.append(summarize(organ_key, fetch_tree(ORGANS[organ_key]["subset"])))

    if as_json:
        print(json.dumps(summaries, ensure_ascii=False, indent=2))
        return 0

    for summary in summaries:
        print(f"== {summary['name_ru']} ==")
        print(f"  папка:  {summary['subset']}")
        print(f"  ссылка: {summary['url']}")
        for folder, info in summary["folders"].items():
            print(f"  {folder:<16} {info['files']:>5} файл(ов)  {human_size(info['bytes']):>9}")
        print(
            f"  всего:  {summary['total_files']} файл(ов), "
            f"{human_size(summary['total_bytes'])}"
        )
        print(
            f"  пары images/labels: {summary['images']} / {summary['labels']}"
            + (
                "  -> ок"
                if not summary["labels_without_image"] and not summary["images_without_label"]
                else f"  -> расхождение: без изображения {summary['labels_without_image']}, "
                f"без разметки {summary['images_without_label']}"
            )
        )
        print(
            "  маска органа: "
            + ("есть (labelsTr_organ)" if summary["organ_masks_published"] else "нет, только опухоль")
        )
        print()
    return 0


def _write_sources_file(out_dir: str, keys: list) -> str:
    lines = [
        "# Источники данных: FLARE PancancerCTSeg",
        "",
        f"Набор: {DATASET_URL}",
        "Доступ закрыт условиями (gated); лицензия набора — CC BY-NC 4.0,",
        "у исходных наборов лицензии различаются (см. таблицу ниже).",
        "",
        "## Скачанные подмножества",
        "",
    ]
    for key in keys:
        organ = ORGANS[key]
        lines.append(f"### {organ['name_ru']} / {organ['name_en']}")
        lines.append("")
        lines.append(f"- Папка: `{organ['subset']}`")
        lines.append(f"- Ссылка: {organ_url(key)}")
        lines.append(f"- Случаев: {organ['cases']}")
        lines.append(f"- Разметка: {organ['labels']}")
        lines.append("")
        lines.append("| Источник | Случаев | Лицензия | Ссылка |")
        lines.append("|----------|---------|----------|--------|")
        for source in organ["sources"]:
            lines.append(
                f"| {source['name']} | {source['cases']} | {source['license']} | {source['url']} |"
            )
        lines.append("")
        for note in organ["notes"]:
            lines.append(f"- Примечание: {note}")
        lines.append("")

    lines.extend(
        [
            "## Цитирование",
            "",
            "MICCAI 2026 Challenge: Pan-cancer segmentation in CT scans —",
            "https://www.codabench.org/competitions/7149/",
            "",
            "При использовании данных цитируйте также исходные наборы из таблиц выше.",
            "",
        ]
    )
    path = os.path.join(out_dir, "FLARE_SOURCES.md")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    return path


def cmd_download(key: str, out_dir: str, as_json: bool) -> int:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise SystemExit(
            "Для скачивания нужен huggingface_hub:\n"
            "    pip install -U huggingface_hub"
        )

    keys = list(ORGANS) if key == "all" else [key]
    os.makedirs(out_dir, exist_ok=True)
    results = []

    for organ_key in keys:
        organ = ORGANS[organ_key]
        print(
            f"Скачиваю {organ['name_ru']} ({organ['subset']}) в {out_dir} ...",
            file=sys.stderr,
        )
        try:
            path = snapshot_download(
                repo_id=DATASET_ID,
                repo_type="dataset",
                local_dir=out_dir,
                allow_patterns=[f"{organ['subset']}/**"],
            )
        except Exception as exc:  # noqa: BLE001 - сообщение зависит от версии клиента
            message = str(exc)
            lowered = message.lower()
            if "gated" in lowered or "401" in message or "403" in message:
                raise SystemExit(
                    "Доступ к набору не подтверждён.\n"
                    f"1) примите условия: {DATASET_URL}\n"
                    "2) создайте токен с правом read: https://huggingface.co/settings/tokens\n"
                    "3) export HF_TOKEN=<токен> и повторите запуск (токен не коммитьте).\n"
                    f"Ответ клиента: {message}"
                )
            raise SystemExit(f"Не удалось скачать {organ['subset']}: {message}")
        results.append({"organ": organ_key, "subset": organ["subset"], "path": path})

    sources_file = _write_sources_file(out_dir, keys)
    if as_json:
        print(json.dumps({"downloads": results, "sources_file": sources_file}, ensure_ascii=False, indent=2))
        return 0

    for result in results:
        print(f"Готово: {ORGANS[result['organ']]['name_ru']} -> "
              f"{os.path.join(result['path'], ORGANS[result['organ']]['subset'])}")
    print(f"Описание источников и лицензий: {sources_file}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--json", action="store_true", help="вывести результат в JSON"
    )

    parser = argparse.ArgumentParser(
        prog="flare_organs.py",
        description=(
            "Поджелудочная железа и надпочечники из набора "
            f"{DATASET_ID}."
        ),
        epilog=f"Набор данных: {DATASET_URL}",
        parents=[common],
    )
    parser.add_argument(
        "--version", action="version", version="flare_organs.py 1.0"
    )

    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser(
        "list",
        parents=[common],
        help="показать подмножества, ссылки, случаи, источники и лицензии",
    )

    inspect_parser = subparsers.add_parser(
        "inspect",
        parents=[common],
        help="посчитать файлы и объём подмножества через API (без скачивания)",
    )
    inspect_parser.add_argument(
        "--organ", choices=ORGAN_CHOICES, default="all", help="что смотреть (по умолчанию all)"
    )

    download_parser = subparsers.add_parser(
        "download",
        parents=[common],
        help="скачать подмножество через huggingface_hub",
    )
    download_parser.add_argument(
        "--organ", choices=ORGAN_CHOICES, default="all", help="что скачивать (по умолчанию all)"
    )
    download_parser.add_argument(
        "--out", default="data", help="каталог для данных (по умолчанию data)"
    )

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command in (None, "list"):
        return cmd_list(args.json)
    if args.command == "inspect":
        return cmd_inspect(args.organ, args.json)
    if args.command == "download":
        return cmd_download(args.organ, args.out, args.json)

    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
