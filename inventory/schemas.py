"""Request validation for inventory endpoints."""
from typing import Optional

from pydantic import BaseModel, Field

from .models import SOURCES


class ProductCreate(BaseModel):
    sku: str = Field(min_length=1)
    name: str = Field(min_length=1)
    shelf_quantity: int = Field(default=0, ge=0)
    shelf_capacity: Optional[int] = Field(default=None, ge=0)
    shelf_low_threshold: int = Field(default=5, ge=0)
    storeroom_quantity: int = Field(default=0, ge=0)
    storeroom_low_threshold: int = Field(default=10, ge=0)


class ProductPatch(BaseModel):
    name: Optional[str] = None
    shelf_capacity: Optional[int] = Field(default=None, ge=0)
    shelf_low_threshold: Optional[int] = Field(default=None, ge=0)
    storeroom_low_threshold: Optional[int] = Field(default=None, ge=0)


class QuantityUpdate(BaseModel):
    quantity: int = Field(ge=0)
    source: str = "api"


class TransferRequest(BaseModel):
    quantity: int = Field(gt=0)
    source: str = "employee"


def validate_source(source):
    if source not in SOURCES:
        raise ValueError(f"source must be one of {', '.join(SOURCES)}")
