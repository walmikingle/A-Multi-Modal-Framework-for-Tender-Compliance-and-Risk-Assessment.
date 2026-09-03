import json
import pickle
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

CACHE_DIR = PROJECT_ROOT / "data" / "cache"
OUTPUT_FILE = PROJECT_ROOT / "evaluation" / "evidence_inventory.json"


KEYWORDS = {
    "identity": [
        "tender no",
        "tender number",
        "tender id",
        "bid no",
        "bid number",
        "nit no",
        "e-nit",
        "notice no",
        "reference no"
    ],
    "authority": [
        "issued by",
        "department",
        "authority",
        "municipal corporation",
        "institute",
        "ministry",
        "executive engineer",
        "managing director"
    ],
    "work": [
        "name of work",
        "scope of work",
        "description of work",
        "work description",
        "services",
        "supply",
        "hiring",
        "procurement"
    ],
    "money": [
        "estimated cost",
        "estimated amount",
        "tender amount",
        "tender value",
        "contract value",
        "emd",
        "earnest money",
        "bid security",
        "fee"
    ],
    "dates": [
        "last date",
        "submission",
        "opening",
        "date of opening",
        "deadline",
        "due date",
        "closing date",
        "issue date",
        "validity"
    ],
    "eligibility": [
        "eligibility",
        "eligible bidder",
        "qualification",
        "experience",
        "turnover",
        "registration",
        "similar work",
        "technical qualification"
    ],
    "conditions": [
        "penalty",
        "forfeit",
        "security deposit",
        "performance guarantee",
        "contract period",
        "completion period",
        "liquidated",
        "disqualification",
        "non-responsive"
    ],
    "tables": [
        "table",
        "item",
        "quantity",
        "unit",
        "rate",
        "amount"
    ]
}


def normalize(text):
    return re.sub(
        r"\s+",
        " ",
        str(text)
    ).strip()


def categorize(text):
    lower = text.lower()

    categories = []

    for category, keywords in KEYWORDS.items():

        for keyword in keywords:

            if keyword in lower:

                categories.append(
                    category
                )

                break

    return categories


def load_cache(cache_path):

    metadata_path = (
        cache_path / "metadata.json"
    )

    items_path = (
        cache_path / "items.pkl"
    )

    if not (
        metadata_path.exists()
        and items_path.exists()
    ):
        return None

    with open(
        metadata_path,
        "r",
        encoding="utf-8"
    ) as file:

        metadata = json.load(file)

    with open(
        items_path,
        "rb"
    ) as file:

        items = pickle.load(file)

    return metadata, items


def main():

    inventory = {}

    cache_paths = sorted(
        [
            path
            for path in CACHE_DIR.iterdir()
            if path.is_dir()
        ]
    )

    for cache_path in cache_paths:

        loaded = load_cache(
            cache_path
        )

        if loaded is None:
            continue

        metadata, items = loaded

        pdf_name = metadata.get(
            "pdf_name"
        )

        if not pdf_name:
            continue

        print(
            f"\nProcessing {pdf_name} "
            f"({len(items)} items)"
        )

        categorized = {
            category: []
            for category in KEYWORDS
        }

        for index, item in enumerate(items):

            item_type = item.get(
                "type",
                ""
            )

            text = normalize(
                item.get(
                    "text",
                    ""
                )
            )

            if not text:
                continue

            categories = categorize(
                text
            )

            record = {
                "item_index": index,
                "type": item_type,
                "page": item.get(
                    "page"
                ),
                "text": text
            }

            for category in categories:

                categorized[
                    category
                ].append(record)

        inventory[
            pdf_name
        ] = {
            "cache_directory": str(
                cache_path
            ),
            "item_count": len(items),
            "categories": categorized
        }

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            inventory,
            file,
            indent=4,
            ensure_ascii=False
        )

    print(
        "\n" + "=" * 70
    )

    print(
        "EVIDENCE INVENTORY CREATED"
    )

    print(
        f"Output:\n{OUTPUT_FILE}"
    )

    print(
        f"Tenders found: "
        f"{len(inventory)}"
    )


if __name__ == "__main__":
    main()