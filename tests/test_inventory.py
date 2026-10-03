"""Inventory decisions, storage, camera adapter, and HTTP integration without a camera."""
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

import storesense
from inventory.service import DuplicateSKU, InsufficientStock


class InventoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = storesense.Store(str(Path(self.temp.name) / "store.db"))
        self.engine = storesense.Engine(self.store)
        self.service = self.engine.inventory
        self.client = TestClient(storesense.make_app(self.engine, {}, self.store))

    def tearDown(self):
        self.client.close()
        self.store.db.close()
        self.temp.cleanup()

    def create(self):
        return self.service.create_product("COKE-500", "Coca Cola 500ml", 30, 40, 10, 100, 20)

    def active(self):
        return [a for a in self.engine.snapshot()["alerts"] if a["kind"] == "inventory"]

    def test_creation_validation_and_catalog_sharing(self):
        p = self.create()
        self.assertEqual(self.store.product_get("COKE-500")["name"], p["name"])
        self.assertIsNone(p["alert_type"])
        with self.assertRaises(DuplicateSKU):
            self.create()
        with self.assertRaises(ValueError):
            self.service.update_shelf_stock(p["id"], -1)
        with self.assertRaises(ValueError):
            self.service.create_product("BAD", "Bad", shelf_quantity=-1)
        self.assertEqual(len(self.service.repo.events(p["id"])), 1)

    def test_existing_catalog_and_unknown_capacity(self):
        self.store.product_upsert({"sku": "EXISTING", "name": "Existing item", "mrp": 12})
        product = self.service.create_product("EXISTING", "Existing item", 3, None, 1, 20, 5)
        self.assertEqual(self.store.product_get("EXISTING")["mrp"], 12)
        with self.assertRaises(ValueError):
            self.service.update_shelf_fill(product["id"], 50)
        self.service.update_product(product["id"], shelf_capacity=10)
        self.assertEqual(self.service.update_shelf_fill(product["id"], 50)["shelf_quantity"], 5)

    def test_schema_is_safe_on_reopen(self):
        pid = self.create()["id"]
        again = storesense.Store(str(Path(self.temp.name) / "store.db"))
        try:
            self.assertEqual(storesense.Engine(again).inventory.get_product(pid)["sku"], "COKE-500")
            self.assertEqual(len(again.inventory_repo.events(pid)), 1)
        finally:
            again.db.close()

    def test_decisions_dedup_resolution_and_history(self):
        p = self.create()
        pid = p["id"]
        self.service.update_shelf_stock(pid, 7, "camera")
        self.assertEqual(self.active()[0]["action"], "SHELF_REFILL")
        n = len(self.service.repo.events(pid))
        self.service.update_shelf_stock(pid, 7, "camera")
        self.assertEqual(len(self.service.repo.events(pid)), n)
        self.assertEqual(len(self.active()), 1)

        self.service.update_storeroom_stock(pid, 10, "employee")
        self.assertEqual(self.active()[0]["action"], "REORDER_REQUIRED")
        self.assertEqual(len(self.active()), 1)
        self.service.update_shelf_stock(pid, 30)
        self.assertEqual(self.active()[0]["action"], "STOREROOM_LOW")
        self.service.update_storeroom_stock(pid, 100)
        self.assertEqual(self.active(), [])
        self.assertIsNone(self.service.get_product(pid)["alert_type"])
        self.assertTrue(any(e["event_type"] == "ALERT_RESOLVED"
                            for e in self.service.repo.events(pid)))

    def test_atomic_transfer(self):
        pid = self.create()["id"]
        self.service.update_shelf_stock(pid, 5)
        result = self.service.transfer_to_shelf(pid, 20)
        self.assertEqual((result["shelf_quantity"], result["storeroom_quantity"]), (25, 80))
        self.assertEqual(self.active(), [])
        before = len(self.service.repo.events(pid))
        with self.assertRaises(InsufficientStock):
            self.service.transfer_to_shelf(pid, 81)
        with self.assertRaises(ValueError):
            self.service.transfer_to_shelf(pid, 16)
        self.assertEqual(len(self.service.repo.events(pid)), before)
        self.assertEqual(self.service.get_product(pid)["storeroom_quantity"], 80)

    def test_camera_adapter_ignores_occlusion_and_duplicates(self):
        pid = self.create()["id"]
        cell = {"sku": "COKE-500", "status": "LOW", "est_units": 7,
                "occluded": False, "misplaced": False}
        self.service.observe_cells([cell])
        count = len(self.service.repo.events(pid))
        self.service.observe_cells([cell])
        self.assertEqual(len(self.service.repo.events(pid)), count)
        self.service.observe_cells([{**cell, "occluded": True, "est_units": 0}])
        self.assertEqual(self.service.get_product(pid)["shelf_quantity"], 7)
        self.service.observe_cells([cell, {**cell, "occluded": True}])
        self.assertEqual(self.service.get_product(pid)["shelf_quantity"], 7)

    def test_engine_shelf_path_uses_inventory_decision(self):
        pid = self.create()["id"]
        cell = {"slot": "slot-1", "sku": "COKE-500", "name": "Coca Cola 500ml",
                "loc": "row 1", "status": "LOW", "est_units": 7, "full_units": 40,
                "occluded": False, "misplaced": False, "eta_min": None, "method": "front"}
        self.engine.on_shelf("shelfA", [cell])
        self.assertEqual(self.service.get_product(pid)["shelf_quantity"], 7)
        self.assertEqual([a["kind"] for a in self.engine.snapshot()["alerts"]], ["inventory"])
        before = len(self.service.repo.events(pid))
        self.engine.on_shelf("shelfA", [cell])
        self.assertEqual(len(self.service.repo.events(pid)), before)
        self.engine.on_shelf("shelfA", [{**cell, "occluded": True, "est_units": 0}])
        self.assertEqual(self.service.get_product(pid)["shelf_quantity"], 7)

    def test_http_and_live_state(self):
        response = self.client.post("/api/inventory/products", json={
            "sku": "COKE-500", "name": "Coca Cola 500ml", "shelf_quantity": 30,
            "shelf_capacity": 40, "shelf_low_threshold": 10,
            "storeroom_quantity": 100, "storeroom_low_threshold": 20})
        self.assertEqual(response.status_code, 201, response.text)
        pid = response.json()["id"]
        self.assertEqual(self.client.post("/api/inventory/products", json={
            "sku": "COKE-500", "name": "Coca Cola 500ml"}).status_code, 409)
        self.assertEqual(self.client.patch(f"/api/inventory/products/{pid}/shelf",
            json={"quantity": -1}).status_code, 422)
        self.assertEqual(self.client.get("/api/inventory/products/999").status_code, 404)
        self.assertEqual(self.client.patch(f"/api/inventory/products/{pid}/shelf",
            json={"quantity": 7, "source": "camera"}).status_code, 200)
        self.assertEqual(self.client.patch(f"/api/inventory/products/{pid}/storeroom",
            json={"quantity": 10, "source": "employee"}).status_code, 200)
        self.assertEqual(self.client.get("/api/inventory/status").json()["critical_reorder_count"], 1)
        self.assertEqual(self.client.get("/api/state").json()["inventory"]["alerts"][0]["type"],
                         "REORDER_REQUIRED")
        self.assertEqual(self.client.post(f"/api/inventory/products/{pid}/transfer-to-shelf",
            json={"quantity": 11}).status_code, 409)
        self.assertEqual(self.client.post(f"/api/inventory/products/{pid}/transfer-to-shelf",
            json={"quantity": 5}).status_code, 200)
        self.assertTrue(self.client.get(f"/api/inventory/products/{pid}/events").json()["events"])
        self.assertTrue(self.client.get("/api/inventory/events").json()["events"])
        self.assertEqual(len(self.client.get("/api/inventory/products").json()["products"]), 1)
        self.assertEqual(self.client.patch(f"/api/inventory/products/{pid}",
            json={"storeroom_low_threshold": 15}).status_code, 200)
        with self.client.websocket_connect("/ws") as ws:
            self.assertIn("inventory", ws.receive_json())

    def test_alert_restored_after_restart(self):
        pid = self.create()["id"]
        self.service.update_shelf_stock(pid, 5)
        restarted = storesense.Engine(self.store)
        active = [a for a in restarted.snapshot()["alerts"] if a["kind"] == "inventory"]
        self.assertEqual([a["action"] for a in active], ["SHELF_REFILL"])


if __name__ == "__main__":
    unittest.main()
