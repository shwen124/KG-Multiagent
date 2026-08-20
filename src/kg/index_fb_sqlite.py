"""SQLite adjacency index for FB+CVT-REV (avoid loading 134M edges into RAM dicts)."""

from __future__ import annotations

import sqlite3
from pathlib import Path


def build_sqlite_index(edges_tsv: Path, db_path: Path, batch_size: int = 500_000) -> dict:
    """Create outgoing (h,r)->t and incoming (t,r)->h indexes."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=OFF;")
    conn.execute("PRAGMA synchronous=OFF;")
    conn.execute("PRAGMA temp_store=MEMORY;")
    conn.execute(
        """
        CREATE TABLE edges (
            head INTEGER NOT NULL,
            relation INTEGER NOT NULL,
            tail INTEGER NOT NULL
        );
        """
    )

    buf = []
    n = 0
    with edges_tsv.open("r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            _eid, h, r, t = line.rstrip("\n").split("\t")
            buf.append((int(h), int(r), int(t)))
            if len(buf) >= batch_size:
                conn.executemany("INSERT INTO edges VALUES (?,?,?)", buf)
                n += len(buf)
                buf.clear()
                if n % (batch_size * 4) == 0:
                    print(f"  indexed {n:,} edges...", flush=True)
    if buf:
        conn.executemany("INSERT INTO edges VALUES (?,?,?)", buf)
        n += len(buf)

    print("  creating indexes...", flush=True)
    conn.execute("CREATE INDEX idx_out ON edges(head, relation);")
    conn.execute("CREATE INDEX idx_in ON edges(tail, relation);")
    conn.commit()
    conn.close()
    return {"num_edges": n, "db_path": str(db_path)}


class FreebaseSQLiteStore:
    """Query helpers on the SQLite edge store (vids are 1-based)."""

    def __init__(self, db_path: Path):
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.conn.execute("PRAGMA query_only=ON;")

    def close(self) -> None:
        self.conn.close()

    def tails(self, head: int, relation: int) -> list[int]:
        cur = self.conn.execute(
            "SELECT tail FROM edges WHERE head=? AND relation=?",
            (head, relation),
        )
        return [row[0] for row in cur.fetchall()]

    def heads(self, tail: int, relation: int) -> list[int]:
        cur = self.conn.execute(
            "SELECT head FROM edges WHERE tail=? AND relation=?",
            (tail, relation),
        )
        return [row[0] for row in cur.fetchall()]

    def has_edge(self, head: int, relation: int, tail: int) -> bool:
        cur = self.conn.execute(
            "SELECT 1 FROM edges WHERE head=? AND relation=? AND tail=? LIMIT 1",
            (head, relation, tail),
        )
        return cur.fetchone() is not None
