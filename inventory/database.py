"""Inventory SQLite repository sharing StoreSense's connection and lock.

SQL stays here; the service uses only repository methods so a later backend can
replace this implementation without changing inventory decisions.
"""
import json
import time


FIELDS = ("id", "sku", "name", "shelf_quantity", "shelf_capacity",
          "shelf_low_threshold", "storeroom_quantity", "storeroom_low_threshold",
          "updated_at", "alert_type")


class InventoryRepository:
    def __init__(self, store):
        self.store = store
        with store.lock:
            store.db.executescript("""
                CREATE TABLE IF NOT EXISTS inventory_stock (
                    id INTEGER PRIMARY KEY,
                    sku TEXT NOT NULL UNIQUE REFERENCES products(sku),
                    shelf_quantity INTEGER NOT NULL DEFAULT 0 CHECK(shelf_quantity >= 0),
                    shelf_capacity INTEGER CHECK(shelf_capacity IS NULL OR shelf_capacity >= 0),
                    shelf_low_threshold INTEGER NOT NULL DEFAULT 5 CHECK(shelf_low_threshold >= 0),
                    storeroom_quantity INTEGER NOT NULL DEFAULT 0 CHECK(storeroom_quantity >= 0),
                    storeroom_low_threshold INTEGER NOT NULL DEFAULT 10 CHECK(storeroom_low_threshold >= 0),
                    updated_at REAL NOT NULL,
                    alert_type TEXT
                );
                CREATE TABLE IF NOT EXISTS inventory_events (
                    id INTEGER PRIMARY KEY,
                    product_id INTEGER NOT NULL REFERENCES inventory_stock(id),
                    event_type TEXT NOT NULL,
                    quantity INTEGER,
                    source TEXT NOT NULL,
                    ts REAL NOT NULL,
                    metadata TEXT
                );
                CREATE INDEX IF NOT EXISTS ix_inventory_events_product_ts
                    ON inventory_events(product_id, ts);
            """)

    def _row(self, row):
        return dict(zip(FIELDS, row))

    def _select(self):
        return ("SELECT i.id,i.sku,p.name,i.shelf_quantity,i.shelf_capacity,"
                "i.shelf_low_threshold,i.storeroom_quantity,i.storeroom_low_threshold,"
                "i.updated_at,i.alert_type FROM inventory_stock i "
                "JOIN products p ON p.sku=i.sku")

    def list(self):
        return [self._row(r) for r in self.store.q(self._select() + " ORDER BY p.name")]

    def get(self, product_id):
        rows = self.store.q(self._select() + " WHERE i.id=?", product_id)
        return self._row(rows[0]) if rows else None

    def by_sku(self, sku):
        rows = self.store.q(self._select() + " WHERE i.sku=?", sku)
        return self._row(rows[0]) if rows else None

    def insert(self, sku, values, source="api"):
        now = time.time()
        with self.store.lock:
            cur = self.store.db.execute(
                "INSERT INTO inventory_stock(sku,shelf_quantity,shelf_capacity,shelf_low_threshold,"
                "storeroom_quantity,storeroom_low_threshold,updated_at) VALUES(?,?,?,?,?,?,?)",
                (sku, values["shelf_quantity"], values["shelf_capacity"],
                 values["shelf_low_threshold"], values["storeroom_quantity"],
                 values["storeroom_low_threshold"], now))
            self._event(cur.lastrowid, "RESTOCK", None, source,
                        {"created": True, "shelf_quantity": values["shelf_quantity"],
                         "storeroom_quantity": values["storeroom_quantity"]})
            self.store.db.commit()
            return cur.lastrowid

    def change(self, product_id, values, event_type, quantity=None, source="api", metadata=None):
        """Apply stock and event in one transaction."""
        with self.store.lock:
            cols = list(values)
            assignments = ",".join(f"{col}=?" for col in cols)
            self.store.db.execute(
                f"UPDATE inventory_stock SET {assignments},updated_at=? WHERE id=?",
                (*[values[c] for c in cols], time.time(), product_id))
            self._event(product_id, event_type, quantity, source, metadata)
            self.store.db.commit()

    def transfer(self, product_id, quantity, source):
        with self.store.lock:
            row = self.store.db.execute(
                "SELECT shelf_quantity,storeroom_quantity,shelf_capacity FROM inventory_stock WHERE id=?",
                (product_id,)).fetchone()
            if row is None:
                raise KeyError("product not found")
            shelf, room, capacity = row
            if room < quantity:
                raise RuntimeError("insufficient storeroom inventory")
            if capacity is not None and shelf + quantity > capacity:
                raise ValueError("transfer exceeds shelf capacity")
            self.store.db.execute(
                "UPDATE inventory_stock SET shelf_quantity=?,storeroom_quantity=?,updated_at=? WHERE id=?",
                (shelf + quantity, room - quantity, time.time(), product_id))
            self._event(product_id, "TRANSFER_TO_SHELF", quantity, source,
                        {"shelf_before": shelf, "storeroom_before": room})
            self.store.db.commit()

    def set_alert(self, product_id, alert_type, message=None):
        with self.store.lock:
            old = self.store.db.execute(
                "SELECT alert_type FROM inventory_stock WHERE id=?", (product_id,)).fetchone()
            if old is None or old[0] == alert_type:
                return False
            self.store.db.execute("UPDATE inventory_stock SET alert_type=? WHERE id=?", (alert_type, product_id))
            self._event(product_id, "ALERT" if alert_type else "ALERT_RESOLVED", None,
                        "system", {"type": alert_type or old[0], "message": message})
            self.store.db.commit()
            return True

    def _event(self, product_id, event_type, quantity, source, metadata=None):
        self.store.db.execute(
            "INSERT INTO inventory_events(product_id,event_type,quantity,source,ts,metadata) "
            "VALUES(?,?,?,?,?,?)",
            (product_id, event_type, quantity, source, time.time(), json.dumps(metadata or {})))

    def events(self, product_id=None, limit=100):
        query = "SELECT id,product_id,event_type,quantity,source,ts,metadata FROM inventory_events"
        args = ()
        if product_id is not None:
            query += " WHERE product_id=?"
            args = (product_id,)
        query += " ORDER BY id DESC LIMIT ?"
        return [dict(id=r[0], product_id=r[1], event_type=r[2], quantity=r[3],
                     source=r[4], timestamp=r[5], metadata=json.loads(r[6] or "{}"))
                for r in self.store.q(query, *args, limit)]
