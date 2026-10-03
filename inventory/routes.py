"""FastAPI inventory router; all writes go through InventoryService."""
from fastapi import APIRouter, HTTPException

from .schemas import ProductCreate, ProductPatch, QuantityUpdate, TransferRequest, validate_source
from .service import DuplicateSKU, InsufficientStock


def make_router(service):
    router = APIRouter(prefix="/api/inventory", tags=["inventory"])

    def call(fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        except (DuplicateSKU, InsufficientStock) as exc:
            raise HTTPException(409, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    def data(body, **options):
        return body.model_dump(**options) if hasattr(body, "model_dump") else body.dict(**options)

    def sourced(fn, product_id, body):
        def run():
            validate_source(body.source)
            return fn(product_id, body.quantity, body.source)
        return call(run)

    @router.get("/products")
    def products():
        return {"products": service.list_products()}

    @router.post("/products", status_code=201)
    def create(body: ProductCreate):
        return call(service.create_product, **data(body))

    @router.get("/products/{product_id}")
    def product(product_id: int):
        return call(service.get_product, product_id)

    @router.patch("/products/{product_id}")
    def patch(product_id: int, body: ProductPatch):
        return call(service.update_product, product_id, **data(body, exclude_unset=True))

    @router.patch("/products/{product_id}/shelf")
    def shelf(product_id: int, body: QuantityUpdate):
        return sourced(service.update_shelf_stock, product_id, body)

    @router.patch("/products/{product_id}/storeroom")
    def storeroom(product_id: int, body: QuantityUpdate):
        return sourced(service.update_storeroom_stock, product_id, body)

    @router.post("/products/{product_id}/transfer-to-shelf")
    def transfer(product_id: int, body: TransferRequest):
        return sourced(service.transfer_to_shelf, product_id, body)

    @router.get("/events")
    def events(limit: int = 100):
        return {"events": service.repo.events(limit=max(1, min(limit, 500)))}

    @router.get("/products/{product_id}/events")
    def product_events(product_id: int, limit: int = 100):
        call(service.get_product, product_id)
        return {"events": service.repo.events(product_id, max(1, min(limit, 500)))}

    @router.get("/status")
    def status():
        return service.status()

    return router
