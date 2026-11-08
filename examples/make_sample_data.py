"""Generate the SYNTHETIC sample data in this folder.

Every poll, candidate and number here is made up for demonstration.
Replace these files with real polls before drawing any conclusions.
"""

import csv
import random
from datetime import date, timedelta
from pathlib import Path

random.seed(7)
HERE = Path(__file__).parent
ELECTION = date(2026, 11, 3)

# True (hidden) support over time, plus each fictional pollster's lean.
RACES = {
    "Example Senate": {"Avery Hart": 46.0, "Blake Moreno": 45.0, "Casey Lin": 4.0},
    "Example Governor": {"Dana Price": 44.0, "Elliot Shaw": 47.0},
}
POLLSTERS = {
    "Northfield Research": {},
    "Keystone Polling": {"Avery Hart": 2.5, "Dana Price": 2.0},  # leans toward these candidates
    "Civic Pulse": {"Blake Moreno": 1.5, "Elliot Shaw": 1.5},
    "Lakeshore Univ.": {},
    "Granite Insights": {},
}
SCANDAL = date(2026, 8, 5)  # fictional event hurting Avery Hart

rows = []
pid = 0
for race, base in RACES.items():
    for k in range(40):
        end = ELECTION - timedelta(days=random.randint(2, 200))
        pollster = random.choice(list(POLLSTERS))
        n = random.choice([400, 600, 800, 1000, 1500])
        pop = random.choice(["lv", "lv", "rv", "a"])
        pid += 1
        shares = {}
        for cand, true in base.items():
            val = true + POLLSTERS[pollster].get(cand, 0.0)
            if cand == "Avery Hart" and end >= SCANDAL:
                val -= 4.0 * 0.5 ** ((end - SCANDAL).days / 120)
            shares[cand] = max(val + random.gauss(0, 100 * (0.25 / n) ** 0.5), 0.0)
        scale = min(1.0, 98.0 / sum(shares.values()))  # leave room for undecideds
        for cand, val in shares.items():
            rows.append([f"P{pid:03d}", end.isoformat(), pollster, race, pop, n, cand, round(val * scale, 1)])

rows.sort(key=lambda r: (r[1], r[0]))
with open(HERE / "polls.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["poll_id", "end_date", "pollster", "race", "population", "sample_size", "candidate", "pct"])
    w.writerows(rows)

with open(HERE / "fundamentals.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["date", "metric", "value"])
    d = date(2025, 11, 1)
    while d <= date(2026, 10, 1):
        m = (d.year - 2025) * 12 + d.month
        w.writerow([d.isoformat(), "incumbent_net_approval", round(-12 + 0.4 * m + random.gauss(0, 1.5), 1)])
        w.writerow([d.isoformat(), "real_income_growth", round(0.8 + random.gauss(0, 0.4), 2)])
        w.writerow([d.isoformat(), "unemployment_change", round(0.2 - 0.02 * m + random.gauss(0, 0.05), 2)])
        w.writerow([d.isoformat(), "inflation", round(3.4 - 0.05 * m + random.gauss(0, 0.2), 2)])
        d = date(d.year + (d.month == 12), d.month % 12 + 1, 1)
print(f"wrote {len(rows)} poll rows")
