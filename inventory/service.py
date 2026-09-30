"""Inventory operations and decisions, independent of FastAPI and camera code."""
import sqlite3
import threading
from functools import wraps

from .database import InventoryRepository


class DuplicateSKU(ValueError):
    pass


class InsufficientStock(ValueError):
    pass


def evaluate_inventory(product):
    shelf_low = product["shelf_quantity"] <= product["shelf_low_threshold"]
    room_low = product["storeroom_quantity"] <= product["storeroom_low_threshold"]
    if shelf_low and room_low:
        kind, severity = "REORDER_REQUIRED", "critical"
        message = f"{product['name']} is low on the shelf and storeroom stock is also low. Reorder inventory."
    elif shelf_low:
        kind, severity = "SHELF_REFILL", "warning"
        message = f"{product['name']} shelf stock is low. Refill from storeroom."
    elif room_low:
        kind, severity = "STOREROOM_LOW", "warning"
        message = f"{product['name']} storeroom stock is low. Reorder inventory."
    else:
        return None
    return {"type": kind, "severity": severity, "product_id": product["id"], "message": message}


def _nonnegative(name, value):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def synchronized(fn):
    @wraps(fn)
    def run(self, *args, **kwargs):
        with self._lock:
            return fn(self, *args, **kwargs)
    return run


class InventoryService:
    def __init__(self, store, engine=None):
        self.store = store
        self.repo = getattr(store, "inventory_repo", None) or InventoryRepository(store)
        self.engine = engine
        self._lock = threading.RLock()
        if engine:
            for product in self.repo.list():
                decision = evaluate_inventory(product)
                self.repo.set_alert(product["id"], decision["type"] if decision else None,
                                    decision["message"] if decision else None)
                self._publish(product, decision)

    def list_products(self):
        return self.repo.list()

    def get_product(self, product_id):
        product = self.repo.get(product_id)
        if product is None:
            raise KeyError("product not found")
        return product

    @synchronized
    def create_product(self, sku, name, shelf_quantity=0, shelf_capacity=None,
                       shelf_low_threshold=5, storeroom_quantity=0,
                       storeroom_low_threshold=10):
        sku, name = sku.strip(), name.strip()
        if not sku or not name:
            raise ValueError("sku and name are required")
        values = dict(shelf_quantity=shelf_quantity, shelf_capacity=shelf_capacity,
                      shelf_low_threshold=shelf_low_threshold,
                      storeroom_quantity=storeroom_quantity,
                      storeroom_low_threshold=storeroom_low_threshold)
        for key, value in values.items():
            if value is not None:
                _nonnegative(key, value)
        if shelf_capacity is not None and shelf_quantity > shelf_capacity:
            raise ValueError("shelf quantity exceeds capacity")
        if self.repo.by_sku(sku):
            raise DuplicateSKU("SKU already has inventory")
        catalog = self.store.product_get(sku)
        if catalog is None:
            self.store.product_upsert({"sku": sku, "name": name})
        elif catalog["name"] != name:
            raise ValueError("name differs from the existing catalog product")
        try:
            product_id = self.repo.insert(sku, values)
        except sqlite3.IntegrityError as exc:
            raise DuplicateSKU("SKU already has inventory") from exc
        return self._evaluate(self.get_product(product_id))

    @synchronized
    def update_product(self, product_id, **changes):
        product = self.get_product(product_id)
        allowed = {"name", "shelf_capacity", "shelf_low_threshold", "storeroom_low_threshold"}
        if not changes or set(changes) - allowed:
            raise ValueError("unsupported product fields")
        if "name" in changes:
            if not isinstance(changes["name"], str):
                raise ValueError("name is required")
            name = changes["name"].strip()
            if not name:
                raise ValueError("name is required")
        for key, value in changes.items():
            if key != "name" and (value is not None or key != "shelf_capacity"):
                _nonnegative(key, value)
        capacity = changes.get("shelf_capacity", product["shelf_capacity"])
        if capacity is not None and product["shelf_quantity"] > capacity:
            raise ValueError("shelf quantity exceeds capacity")
        if "name" in changes:
            self.store.product_upsert({"sku": product["sku"], "name": changes.pop("name")})
        if changes:
            self.repo.change(product_id, changes, "CONFIG_UPDATE", source="api")
        return self._evaluate(self.get_product(product_id))

    @synchronized
    def update_shelf_stock(self, product_id, quantity, source="api"):
        _nonnegative("quantity", quantity)
        product = self.get_product(product_id)
        if product["shelf_capacity"] is not None and quantity > product["shelf_capacity"]:
            raise ValueError("shelf quantity exceeds capacity")
        if quantity != product["shelf_quantity"]:
            self.repo.change(product_id, {"shelf_quantity": quantity}, "SHELF_UPDATE", quantity, source)
        return self._evaluate(self.get_product(product_id))

    @synchronized
    def update_storeroom_stock(self, product_id, quantity, source="api"):
        _nonnegative("quantity", quantity)
        product = self.get_product(product_id)
        if quantity != product["storeroom_quantity"]:
            self.repo.change(product_id, {"storeroom_quantity": quantity}, "STOREROOM_UPDATE", quantity, source)
        return self._evaluate(self.get_product(product_id))

    @synchronized
    def transfer_to_shelf(self, product_id, quantity, source="employee"):
        _nonnegative("quantity", quantity)
        if quantity == 0:
            raise ValueError("transfer quantity must be positive")
        self.get_product(product_id)
        try:
            self.repo.transfer(product_id, quantity, source)
        except RuntimeError as exc:
            raise InsufficientStock(str(exc)) from exc
        return self._evaluate(self.get_product(product_id))

    @synchronized
    def update_shelf_fill(self, product_id, fill_percentage, source="camera"):
        product = self.get_product(product_id)
        if product["shelf_capacity"] is None:
            raise ValueError("shelf capacity is unknown")
        if isinstance(fill_percentage, bool) or not isinstance(fill_percentage, (int, float)) or not 0 <= fill_percentage <= 100:
            raise ValueError("fill percentage must be between 0 and 100")
        return self.update_shelf_stock(product_id, round(product["shelf_capacity"] * fill_percentage / 100), source)

    @synchronized
    def observe_cells(self, cells):
        """Adapter for valid marked shelf observations; anonymous and hidden cells are ignored."""
        by_sku = {}
        invalid = set()
        for cell in cells or []:
            sku = cell.get("sku")
            if not sku:
                continue
            if (cell.get("occluded") or cell.get("misplaced")
                    or cell.get("status") not in ("OK", "LOW", "EMPTY")
                    or isinstance(cell.get("est_units"), bool)
                    or not isinstance(cell.get("est_units"), int)
                    or cell["est_units"] < 0):
                invalid.add(sku)
            else:
                by_sku[sku] = by_sku.get(sku, 0) + cell["est_units"]
        for sku, quantity in by_sku.items():
            if sku in invalid:
                continue
            product = self.repo.by_sku(sku)
            if product and (product["shelf_capacity"] is None or quantity <= product["shelf_capacity"]):
                self.update_shelf_stock(product["id"], quantity, "camera")

    def _evaluate(self, product):
        decision = evaluate_inventory(product)
        if self.repo.set_alert(product["id"], decision["type"] if decision else None,
                               decision["message"] if decision else None):
            self._publish(product, decision)
        return self.get_product(product["id"])

    def _publish(self, product, decision):
        if self.engine is None:
            return
        key = f"inventory:{product['id']}"
        self.engine.resolve(key)
        if decision:
            severity = 3 if decision["severity"] == "critical" else 2
            self.engine.fire(key, "inventory", severity, decision["message"], decision["type"])

    def status(self):
        products = self.repo.list()
        alerts = [evaluate_inventory(p) for p in products]
        alerts = [a for a in alerts if a]
        return {"low_shelf_count": sum(p["shelf_quantity"] <= p["shelf_low_threshold"] for p in products),
                "low_storeroom_count": sum(p["storeroom_quantity"] <= p["storeroom_low_threshold"] for p in products),
                "critical_reorder_count": sum(a["type"] == "REORDER_REQUIRED" for a in alerts),
                "alerts": alerts}
