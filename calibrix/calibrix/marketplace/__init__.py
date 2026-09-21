# SPDX-License-Identifier: MIT
"""Calibrix kernel marketplace: sell calibration kernels as portable artifacts.

The ComfyUI node (`comfyui_nodes/calibrix_node.py`) is the client: paste a
license key, the node verifies it offline (HMAC) and unlocks the licensed
kernel spec. This package provides the seller side:

- licenses  : HMAC-signed, offline-verifiable license keys
- store     : JSON-backed listings/orders (dev + small-scale production)
- stripe_provider : Stripe Checkout (optional `stripe` extra) + offline mock
- server    : dependency-free HTTP storefront + webhook endpoint
"""

from .licenses import (
    License,
    checksum_spec,
    get_signing_secret,
    issue_license,
    verify_license,
)
from .store import JsonStore, MarketplaceStore, Order, OrderStatus, Listing
from .stripe_provider import MockStripeProvider, StripeProvider, make_provider

__all__ = [
    "License", "checksum_spec", "get_signing_secret", "issue_license",
    "verify_license",
    "JsonStore", "MarketplaceStore", "Order", "OrderStatus", "Listing",
    "MockStripeProvider", "StripeProvider", "make_provider",
]
