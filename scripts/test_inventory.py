"""Run five inventory scenarios locally, using an isolated temporary SQLite file."""
import tempfile
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import storesense


with tempfile.TemporaryDirectory() as temp:
    store = storesense.Store(str(Path(temp) / "demo.db"))
    engine = storesense.Engine(store)
    inv = engine.inventory
    pid = inv.create_product("DEMO-1", "Demo product", 30, 40, 10, 50, 20)["id"]
    scenarios = [
        ("A shelf low", lambda: inv.update_shelf_stock(pid, 5, "demo")),
        ("B storeroom low", lambda: inv.update_storeroom_stock(pid, 10, "demo")),
        ("C shelf healthy", lambda: inv.update_shelf_stock(pid, 30, "demo")),
        ("D transfer", lambda: inv.transfer_to_shelf(pid, 5, "demo")),
        ("E restore", lambda: inv.update_storeroom_stock(pid, 50, "demo")),
    ]
    for label, action in scenarios:
        product = action()
        alerts = inv.status()["alerts"]
        print(f"{label}: shelf={product['shelf_quantity']} storeroom={product['storeroom_quantity']} "
              f"alert={alerts[0]['type'] if alerts else 'none'}")
    print(f"Recorded {len(inv.repo.events(pid))} inventory events")
    store.db.close()
