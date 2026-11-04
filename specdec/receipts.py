"""Synthetic receipts from named template families, with their true field values.

Every merchant, item and number here is made up. Nothing comes from a real receipt.

A template family is a whole layout and business type. Splits assign entire
families, so an evaluation family's layout and vocabulary never appear in training.
"""

import json
import random
from collections.abc import Callable
from dataclasses import dataclass

INSTRUCTION = (
    "Extract the fields from this receipt. Reply with one JSON object with the keys "
    '"merchant", "date", "currency", "subtotal", "tax", "total" and "item_count". '
    "Use numbers for amounts and an ISO 8601 date.\n\nReceipt:\n"
)

CURRENCIES = {
    "USD": ("$", 2),
    "CAD": ("C$", 2),
    "EUR": ("EUR ", 2),
    "GBP": ("GBP ", 2),
    "CHF": ("CHF ", 2),
    "JPY": ("JPY ", 0),
}


@dataclass(frozen=True)
class Item:
    name: str
    quantity: int
    unit_price: float
    is_refund: bool = False

    @property
    def amount(self) -> float:
        sign = -1 if self.is_refund else 1
        return sign * self.quantity * self.unit_price


@dataclass(frozen=True)
class ReceiptFacts:
    merchant: str
    date: str  # ISO 8601.
    currency: str
    items: tuple[Item, ...]
    subtotal: float
    taxes: tuple[tuple[str, float], ...]
    tip: float
    total: float

    def truth(self) -> dict:
        return {
            "merchant": self.merchant,
            "date": self.date,
            "currency": self.currency,
            "subtotal": self.subtotal,
            "tax": round(sum(amount for _, amount in self.taxes), 2),
            "total": self.total,
            "item_count": len(self.items),
        }


def money(amount: float, currency: str) -> str:
    symbol, decimals = CURRENCIES[currency]
    sign = "-" if amount < 0 else ""
    return f"{sign}{symbol}{abs(amount):,.{decimals}f}"


def plain(amount: float, currency: str) -> str:
    return f"{amount:,.{CURRENCIES[currency][1]}f}"


def make_facts(
    rng: random.Random,
    merchants: list[str],
    catalog: list[tuple[str, float]],
    tax_names: list[str],
    tax_rates: list[float],
    allow_refund: bool = False,
    tip_rate: float = 0.0,
    max_items: int = 6,
) -> ReceiptFacts:
    currency = rng.choice(list(CURRENCIES))
    decimals = CURRENCIES[currency][1]
    scale = 110.0 if currency == "JPY" else 1.0
    names = rng.sample(catalog, k=min(rng.randint(1, max_items), len(catalog)))
    items = [
        Item(name, rng.randint(1, 3), round(price * rng.uniform(0.8, 1.2) * scale, decimals))
        for name, price in names
    ]
    if allow_refund and rng.random() < 0.5:
        name, price = rng.choice(catalog)
        items.append(Item(f"REFUND {name}", 1, round(price * scale, decimals), is_refund=True))
    subtotal = round(sum(item.amount for item in items), decimals)
    taxes = tuple(
        (name, round(subtotal * rate, decimals))
        for name, rate in zip(tax_names, tax_rates, strict=True)
    )
    tip = round(subtotal * tip_rate * rng.uniform(0.8, 1.3), decimals)
    total = round(subtotal + sum(amount for _, amount in taxes) + tip, decimals)
    date = f"20{rng.randint(23, 26)}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"
    return ReceiptFacts(
        rng.choice(merchants), date, currency, tuple(items), subtotal, taxes, tip, total
    )


def month_name_date(iso_date: str) -> str:
    year, month, day = iso_date.split("-")
    months = "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split()
    return f"{int(day)} {months[int(month) - 1]} {year}"


def slash_date(iso_date: str) -> str:
    year, month, day = iso_date.split("-")
    return f"{month}/{day}/{year}"


def grocery(rng):
    facts = make_facts(
        rng,
        ["Maple Lane Market", "Harbour Fresh Foods", "Sunrise Grocer", "Oak Hill Provisions"],
        [
            ("Whole Milk 2L", 4.5),
            ("Sourdough Loaf", 5.25),
            ("Free Range Eggs", 6.1),
            ("Cheddar Block", 8.75),
            ("Bananas", 1.8),
            ("Olive Oil", 11.5),
            ("Ground Coffee", 12.25),
            ("Tomato Sauce", 2.95),
        ],
        ["Sales Tax"],
        [rng.choice([0.05, 0.07, 0.0825])],
    )
    lines = [facts.merchant.upper(), f"Date: {facts.date}", "-" * 32]
    lines += [
        f"{item.quantity} x {item.name:<20} {plain(item.amount, facts.currency):>8}"
        for item in facts.items
    ]
    lines += ["-" * 32, f"SUBTOTAL {plain(facts.subtotal, facts.currency):>23}"]
    lines += [f"{name.upper()} {plain(amount, facts.currency):>23}" for name, amount in facts.taxes]
    lines += [f"TOTAL ({facts.currency}) {money(facts.total, facts.currency):>18}"]
    return facts, "\n".join(lines)


def restaurant(rng):
    facts = make_facts(
        rng,
        ["The Copper Table", "Luna Trattoria", "Saffron House", "Blue Door Bistro"],
        [
            ("Margherita Pizza", 16.0),
            ("Caesar Salad", 12.5),
            ("Grilled Salmon", 26.0),
            ("House Wine", 9.5),
            ("Tiramisu", 8.0),
            ("Sparkling Water", 4.0),
        ],
        ["GST", "PST"],
        [0.05, 0.07],
        tip_rate=0.18,
    )
    lines = [facts.merchant, f"Served on {month_name_date(facts.date)}", ""]
    lines += [
        f"{item.name} ({item.quantity}): {plain(item.amount, facts.currency)}"
        for item in facts.items
    ]
    lines += ["", f"Food and drink: {plain(facts.subtotal, facts.currency)}"]
    lines += [f"{name}: {plain(amount, facts.currency)}" for name, amount in facts.taxes]
    lines += [
        f"Gratuity: {plain(facts.tip, facts.currency)}",
        f"Amount paid: {money(facts.total, facts.currency)} {facts.currency}",
    ]
    return facts, "\n".join(lines)


def fuel(rng):
    facts = make_facts(
        rng,
        ["Northway Fuel", "Prairie Gas Stop", "Coastline Petroleum"],
        [
            ("Regular Unleaded", 42.0),
            ("Premium Unleaded", 55.0),
            ("Diesel", 61.0),
            ("Car Wash", 12.0),
        ],
        ["Fuel Tax"],
        [0.1],
        max_items=2,
    )
    lines = [f"*** {facts.merchant} ***", f"DATE {slash_date(facts.date)}", ""]
    lines += [
        f"PUMP ITEM {i + 1}  {item.name}  {plain(item.amount, facts.currency)}"
        for i, item in enumerate(facts.items)
    ]
    lines += [
        f"NET AMOUNT      {plain(facts.subtotal, facts.currency)}",
        f"FUEL TAX        {plain(facts.taxes[0][1], facts.currency)}",
        f"SALE TOTAL {facts.currency} {plain(facts.total, facts.currency)}",
    ]
    return facts, "\n".join(lines)


def pharmacy(rng):
    facts = make_facts(
        rng,
        ["Wellspring Pharmacy", "Cornerstone Drugs", "Lakeside Apothecary"],
        [
            ("Pain Relief 24ct", 9.99),
            ("Allergy Tablets", 14.5),
            ("Vitamin D", 11.25),
            ("Hand Cream", 7.75),
            ("Bandages", 5.5),
        ],
        ["VAT"],
        [0.2],
        allow_refund=True,
    )
    lines = [facts.merchant, f"Receipt date | {facts.date}", "Item | Qty | Amount"]
    lines += [
        f"{item.name} | {item.quantity} | {plain(item.amount, facts.currency)}"
        for item in facts.items
    ]
    lines += [
        f"Net | | {plain(facts.subtotal, facts.currency)}",
        f"VAT | | {plain(facts.taxes[0][1], facts.currency)}",
        f"Total due | {facts.currency} | {plain(facts.total, facts.currency)}",
    ]
    return facts, "\n".join(lines)


def electronics(rng):
    facts = make_facts(
        rng,
        ["Circuit Point", "Volt and Wire", "Gridline Electronics"],
        [
            ("USB-C Cable", 19.0),
            ("Wireless Mouse", 39.0),
            ("Laptop Stand", 59.0),
            ("Noise Cancelling Headset", 189.0),
            ("Portable SSD", 129.0),
        ],
        ["HST"],
        [0.13],
        allow_refund=True,
        max_items=4,
    )
    lines = [f"Merchant: {facts.merchant}", f"Sold on: {slash_date(facts.date)}", "Lines:"]
    lines += [
        f"  - {item.name}, qty {item.quantity}, {money(item.amount, facts.currency)}"
        for item in facts.items
    ]
    lines += [
        f"Before tax: {money(facts.subtotal, facts.currency)}",
        f"HST 13%: {money(facts.taxes[0][1], facts.currency)}",
        f"Charged to card: {money(facts.total, facts.currency)} ({facts.currency})",
    ]
    return facts, "\n".join(lines)


def hotel(rng):
    facts = make_facts(
        rng,
        ["Harbourview Inn", "The Alder Lodge", "Stonebridge Suites"],
        [("Room Night", 149.0), ("Breakfast", 18.0), ("Parking", 25.0), ("Late Checkout", 40.0)],
        ["Lodging Tax", "Service Charge"],
        [0.08, 0.03],
        max_items=4,
    )
    lines = [f"GUEST FOLIO - {facts.merchant}", f"Checkout {month_name_date(facts.date)}", ""]
    lines += [
        f"{item.quantity:>2} {item.name:<16}{plain(item.amount, facts.currency):>12}"
        for item in facts.items
    ]
    lines += [f"Room and extras  {plain(facts.subtotal, facts.currency):>12}"]
    lines += [f"{name:<17}{plain(amount, facts.currency):>12}" for name, amount in facts.taxes]
    lines += [f"Balance settled  {money(facts.total, facts.currency):>12} {facts.currency}"]
    return facts, "\n".join(lines)


def coffee(rng):
    facts = make_facts(
        rng,
        ["Bright Bean Cafe", "Second Cup Roasters", "Morning Ritual Coffee"],
        [("Flat White", 4.75), ("Cappuccino", 4.5), ("Almond Croissant", 3.95), ("Cold Brew", 5.0)],
        ["Tax"],
        [0.05],
        tip_rate=0.1,
        max_items=3,
    )
    lines = [f"{facts.merchant} / {facts.date}"]
    lines += [
        f"{item.name} x{item.quantity} {plain(item.amount, facts.currency)}" for item in facts.items
    ]
    lines += [
        f"sub {plain(facts.subtotal, facts.currency)} tax {plain(facts.taxes[0][1], facts.currency)} "
        f"tip {plain(facts.tip, facts.currency)}",
        f"total {facts.currency} {plain(facts.total, facts.currency)}",
    ]
    return facts, "\n".join(lines)


def hardware(rng):
    facts = make_facts(
        rng,
        ["Ironwood Hardware", "Timber and Nail", "Fieldstone Supply"],
        [
            ("Deck Screws 1lb", 12.5),
            ("Paint Roller", 8.25),
            ("Wood Stain", 24.0),
            ("Utility Knife", 9.5),
            ("Work Gloves", 15.0),
        ],
        ["State Tax", "County Tax"],
        [0.04, 0.0225],
        allow_refund=True,
    )
    lines = [
        facts.merchant,
        f"Invoice date: {month_name_date(facts.date)}",
        "QTY  DESCRIPTION  AMOUNT",
    ]
    lines += [
        f"{item.quantity}  {item.name}  {plain(item.amount, facts.currency)}"
        for item in facts.items
    ]
    lines += [f"Merchandise {plain(facts.subtotal, facts.currency)}"]
    lines += [f"{name} {plain(amount, facts.currency)}" for name, amount in facts.taxes]
    lines += [f"Invoice total {money(facts.total, facts.currency)} {facts.currency}"]
    return facts, "\n".join(lines)


def online_order(rng):
    facts = make_facts(
        rng,
        ["Parcel and Pine", "Everyday Goods Online", "Nimbus Home"],
        [
            ("Linen Sheet Set", 89.0),
            ("Ceramic Mug", 14.0),
            ("Scented Candle", 22.0),
            ("Throw Blanket", 49.0),
            ("Shipping", 8.0),
        ],
        ["Tax"],
        [0.0875],
        allow_refund=True,
    )
    record = {
        "order_from": facts.merchant,
        "ordered_at": f"{facts.date}T10:15:00",
        "lines": [
            {"sku": item.name, "qty": item.quantity, "amount": item.amount} for item in facts.items
        ],
        "net": facts.subtotal,
        "tax": facts.taxes[0][1],
        "grand_total": facts.total,
        "ccy": facts.currency,
    }
    return facts, f"Order confirmation\n{json.dumps(record)}"


def ride(rng):
    facts = make_facts(
        rng,
        ["Swift Ride", "Metro Hop", "Lantern Cabs"],
        [("Base Fare", 6.0), ("Distance Fee", 14.0), ("Time Fee", 5.0), ("Booking Fee", 2.5)],
        ["Transit Levy"],
        [0.05],
        tip_rate=0.15,
        max_items=4,
    )
    lines = [f"Thanks for riding with {facts.merchant}", f"Trip on {month_name_date(facts.date)}"]
    lines += [f"{item.name}: {money(item.amount, facts.currency)}" for item in facts.items]
    lines += [
        f"Fare subtotal: {money(facts.subtotal, facts.currency)}",
        f"Transit Levy: {money(facts.taxes[0][1], facts.currency)}",
        f"Tip: {money(facts.tip, facts.currency)}",
        f"You paid {money(facts.total, facts.currency)} {facts.currency}",
    ]
    return facts, "\n".join(lines)


def bookstore(rng):
    facts = make_facts(
        rng,
        ["Inkwell Books", "The Reading Room", "Dogear Booksellers"],
        [("Hardcover Novel", 28.0), ("Paperback", 16.0), ("Bookmark Set", 6.0), ("Atlas", 45.0)],
        ["GST"],
        [0.05],
        allow_refund=True,
        max_items=4,
    )
    lines = [f"{facts.merchant} * sales slip * {slash_date(facts.date)}"]
    lines += [
        f"{item.quantity} @ {plain(item.unit_price, facts.currency)} {item.name} = "
        f"{plain(item.amount, facts.currency)}"
        for item in facts.items
    ]
    lines += [
        f"Goods {plain(facts.subtotal, facts.currency)}; GST {plain(facts.taxes[0][1], facts.currency)}",
        f"Pay {facts.currency} {plain(facts.total, facts.currency)}",
    ]
    return facts, "\n".join(lines)


def parking(rng):
    facts = make_facts(
        rng,
        ["CityPark Garage", "Riverside Parking", "Central Lot 14"],
        [("Hourly Parking", 9.0), ("Daily Maximum", 28.0), ("EV Charging", 11.0)],
        ["Parking Tax"],
        [0.2],
        max_items=2,
    )
    lines = [facts.merchant.upper(), f"EXIT {facts.date} 17:42", "TICKET 448120"]
    lines += [f"{item.name} {plain(item.amount, facts.currency)}" for item in facts.items]
    lines += [
        f"NET {plain(facts.subtotal, facts.currency)}",
        f"PARKING TAX {plain(facts.taxes[0][1], facts.currency)}",
        f"PAID {money(facts.total, facts.currency)} {facts.currency}",
    ]
    return facts, "\n".join(lines)


FAMILIES: dict[str, Callable] = {
    "grocery": grocery,
    "restaurant": restaurant,
    "fuel": fuel,
    "pharmacy": pharmacy,
    "electronics": electronics,
    "hotel": hotel,
    "coffee": coffee,
    "hardware": hardware,
    "online_order": online_order,
    "ride": ride,
    "bookstore": bookstore,
    "parking": parking,
}

# Whole families per split. Validation and test families never reach training.
SPLITS = {
    "train": ["grocery", "restaurant", "pharmacy", "electronics", "coffee", "online_order"],
    "validation": ["fuel", "hardware", "ride"],
    "test": ["hotel", "bookstore", "parking"],
}


def build_examples(split: str, per_family: int, seed: int) -> list[dict]:
    """Return examples with prompt text and true fields. The same seed gives the same list."""
    examples = []
    for family in SPLITS[split]:
        rng = random.Random(f"{seed}:{family}")
        for index in range(per_family):
            facts, text = FAMILIES[family](rng)
            examples.append(
                {
                    "id": f"{split}-{family}-{index:04d}",
                    "split": split,
                    "family": family,
                    "prompt": INSTRUCTION + text,
                    "truth": facts.truth(),
                }
            )
    return examples
