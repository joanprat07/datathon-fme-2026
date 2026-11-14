"""Show what is in our team database: submissions, feedback and scratch.

    python team.py          # 5 latest rows of each table
    python team.py 20       # 20 latest rows
"""
import sys

import pandas as pd

from twin.rt import TEAM, q

pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", 60)


def show(title: str, df: pd.DataFrame) -> None:
    print(f"\n=== {title} ===")
    print(df.to_string(index=False))


N = int(sys.argv[1]) if len(sys.argv) > 1 else 5

if __name__ == "__main__":
    for table in ["submissions", "feedback", "scratch"]:
        try:
            count = q(f"SELECT count() AS n FROM {TEAM}.{table}")["n"][0]
        except Exception as e:
            print(f"\n=== {TEAM}.{table}: could not read ({e}) ===")
            continue
        print(f"\n=== {TEAM}.{table}: {count} rows ===")
        if count == 0:
            print("(empty)")
            continue
        order = "ORDER BY ts DESC" if table == "feedback" else ""
        print(q(f"SELECT * FROM {TEAM}.{table} {order} LIMIT {N}").to_string(index=False))

    show("Feedback: only rows where something was bought", q(f"""
        SELECT ts, request_id, served_by, clicked, purchased, revenue_eur
        FROM {TEAM}.feedback
        WHERE request_id != '__init__' AND purchased::String != ''
        ORDER BY ts DESC LIMIT {N}"""))
