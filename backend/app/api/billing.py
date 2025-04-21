from fastapi import APIRouter, HTTPException, Depends, Header, Request
from typing import Dict, Optional
from pydantic import BaseModel, EmailStr
import os
from app.utils.billing import billing_manager

router = APIRouter()

class SubscriptionRequest(BaseModel):
    email: EmailStr
    tier: str = "pro"

class APIKeyResponse(BaseModel):
    api_key: str
    tier: str
    features: list

@router.post("/create-checkout", response_model=Dict)
async def create_checkout(request: SubscriptionRequest):
    """Create a Stripe checkout session for subscription"""
    try:
        result = billing_manager.create_checkout_session(request.email, request.tier)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/webhook")
async def stripe_webhook(request: Request, stripe_signature: str = Header(None)):
    """Handle Stripe webhook events"""
    if not stripe_signature:
        raise HTTPException(status_code=400, detail="Missing Stripe signature")
    
    payload = await request.body()
    result = billing_manager.handle_webhook(payload, stripe_signature)
    
    if result["status"] == "error":
        raise HTTPException(status_code=400, detail=result["message"])
    
    return result

@router.get("/usage/{api_key}")
async def check_usage(api_key: str):
    """Check API usage and limits"""
    result = billing_manager.check_usage_limit(api_key)
    if not result["allowed"] and "reason" in result:
        raise HTTPException(status_code=403, detail=result["reason"])
    return result

@router.get("/invoice/{email}")
async def get_invoice(email: EmailStr, period: str = "current"):
    """Get invoice for a user"""
    result = billing_manager.generate_invoice(email, period)
    if result["status"] == "error":
        raise HTTPException(status_code=400, detail=result["message"])
    return result

@router.get("/tiers")
async def get_subscription_tiers():
    """Get available subscription tiers"""
    return billing_manager.subscription_tiers 