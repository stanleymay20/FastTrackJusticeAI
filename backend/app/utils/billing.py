import os
import stripe
from typing import Dict, List, Optional, Union
from datetime import datetime
import json
from pathlib import Path
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Initialize Stripe with API key from environment variable
stripe.api_key = os.getenv("STRIPE_SECRET_KEY")

# Define subscription tiers
SUBSCRIPTION_TIERS = {
    "free": {
        "name": "Free",
        "price_id": os.getenv("STRIPE_FREE_PRICE_ID"),
        "judgments_per_day": 5,
        "features": ["Basic judgment generation", "Standard scroll phases"]
    },
    "pro": {
        "name": "Professional",
        "price_id": os.getenv("STRIPE_PRO_PRICE_ID"),
        "judgments_per_day": 50,
        "features": ["Advanced judgment generation", "All scroll phases", "PDF exports", "Priority support"]
    },
    "enterprise": {
        "name": "Enterprise",
        "price_id": os.getenv("STRIPE_ENTERPRISE_PRICE_ID"),
        "judgments_per_day": 500,
        "features": ["Unlimited judgments", "Custom scroll phases", "API access", "Dedicated support", "White-labeling"]
    }
}

class BillingManager:
    """Manages billing operations and subscription tracking"""
    
    def __init__(self):
        self.subscription_tiers = SUBSCRIPTION_TIERS
        self.usage_log_path = Path("logs/usage")
        self.usage_log_path.mkdir(parents=True, exist_ok=True)
    
    def create_checkout_session(self, user_email: str, tier: str = "pro") -> Dict:
        """Create a Stripe checkout session for subscription"""
        try:
            if tier not in self.subscription_tiers:
                raise ValueError(f"Invalid subscription tier: {tier}")
            
            price_id = self.subscription_tiers[tier]["price_id"]
            
            session = stripe.checkout.Session.create(
                success_url=f"{os.getenv('APP_URL')}/success?session_id={{CHECKOUT_SESSION_ID}}",
                cancel_url=f"{os.getenv('APP_URL')}/cancel",
                line_items=[{
                    "price": price_id,
                    "quantity": 1
                }],
                mode="subscription",
                customer_email=user_email,
                metadata={
                    "tier": tier,
                    "user_email": user_email
                }
            )
            
            logger.info(f"Created checkout session for {user_email} with tier {tier}")
            return {"session_id": session.id, "url": session.url}
        
        except Exception as e:
            logger.error(f"Error creating checkout session: {e}")
            raise
    
    def create_api_key(self, user_email: str, tier: str) -> str:
        """Generate an API key for the user"""
        import secrets
        
        # Generate a secure API key
        api_key = f"ftj_{secrets.token_urlsafe(32)}"
        
        # Store the API key with user info
        key_data = {
            "api_key": api_key,
            "user_email": user_email,
            "tier": tier,
            "created_at": datetime.now().isoformat(),
            "is_active": True
        }
        
        # Save to file
        key_file = self.usage_log_path / "api_keys.json"
        keys = []
        if key_file.exists():
            with open(key_file, "r") as f:
                keys = json.load(f)
        
        keys.append(key_data)
        
        with open(key_file, "w") as f:
            json.dump(keys, f, indent=2)
        
        logger.info(f"Created API key for {user_email} with tier {tier}")
        return api_key
    
    def verify_api_key(self, api_key: str) -> Optional[Dict]:
        """Verify an API key and return user info"""
        key_file = self.usage_log_path / "api_keys.json"
        if not key_file.exists():
            return None
        
        with open(key_file, "r") as f:
            keys = json.load(f)
        
        for key_data in keys:
            if key_data["api_key"] == api_key and key_data["is_active"]:
                return key_data
        
        return None
    
    def log_usage(self, api_key: str, judgment_count: int = 1) -> bool:
        """Log API usage for billing purposes"""
        key_data = self.verify_api_key(api_key)
        if not key_data:
            return False
        
        # Get today's date for the log file
        today = datetime.now().strftime("%Y-%m-%d")
        usage_file = self.usage_log_path / f"usage_{today}.json"
        
        # Load existing usage data
        usage_data = {}
        if usage_file.exists():
            with open(usage_file, "r") as f:
                usage_data = json.load(f)
        
        # Update usage for this API key
        if api_key not in usage_data:
            usage_data[api_key] = {
                "user_email": key_data["user_email"],
                "tier": key_data["tier"],
                "judgment_count": 0,
                "last_used": datetime.now().isoformat()
            }
        
        usage_data[api_key]["judgment_count"] += judgment_count
        usage_data[api_key]["last_used"] = datetime.now().isoformat()
        
        # Save updated usage data
        with open(usage_file, "w") as f:
            json.dump(usage_data, f, indent=2)
        
        logger.info(f"Logged {judgment_count} judgments for API key {api_key[:8]}...")
        return True
    
    def check_usage_limit(self, api_key: str) -> Dict:
        """Check if user has exceeded their daily usage limit"""
        key_data = self.verify_api_key(api_key)
        if not key_data:
            return {"allowed": False, "reason": "Invalid API key"}
        
        tier = key_data["tier"]
        daily_limit = self.subscription_tiers[tier]["judgments_per_day"]
        
        # Get today's usage
        today = datetime.now().strftime("%Y-%m-%d")
        usage_file = self.usage_log_path / f"usage_{today}.json"
        
        if not usage_file.exists():
            return {"allowed": True, "remaining": daily_limit}
        
        with open(usage_file, "r") as f:
            usage_data = json.load(f)
        
        if api_key not in usage_data:
            return {"allowed": True, "remaining": daily_limit}
        
        current_usage = usage_data[api_key]["judgment_count"]
        remaining = max(0, daily_limit - current_usage)
        
        return {
            "allowed": remaining > 0,
            "remaining": remaining,
            "limit": daily_limit,
            "used": current_usage
        }
    
    def generate_invoice(self, user_email: str, period: str = "current") -> Dict:
        """Generate an invoice for a user"""
        try:
            # Get customer ID from email
            customers = stripe.Customer.list(email=user_email)
            if not customers.data:
                raise ValueError(f"No customer found for email: {user_email}")
            
            customer_id = customers.data[0].id
            
            # Get invoices for the customer
            invoices = stripe.Invoice.list(
                customer=customer_id,
                limit=10
            )
            
            if not invoices.data:
                return {"status": "no_invoices", "message": "No invoices found"}
            
            # Return the most recent invoice
            latest_invoice = invoices.data[0]
            
            return {
                "status": "success",
                "invoice_id": latest_invoice.id,
                "amount": latest_invoice.amount_paid / 100,  # Convert from cents
                "currency": latest_invoice.currency,
                "date": datetime.fromtimestamp(latest_invoice.created).isoformat(),
                "pdf_url": latest_invoice.invoice_pdf
            }
        
        except Exception as e:
            logger.error(f"Error generating invoice: {e}")
            return {"status": "error", "message": str(e)}
    
    def handle_webhook(self, payload: Dict, sig_header: str) -> Dict:
        """Handle Stripe webhook events"""
        try:
            event = stripe.Webhook.construct_event(
                payload, sig_header, os.getenv("STRIPE_WEBHOOK_SECRET")
            )
            
            # Handle different event types
            if event.type == "checkout.session.completed":
                session = event.data.object
                user_email = session.metadata.get("user_email")
                tier = session.metadata.get("tier", "pro")
                
                # Create API key for the user
                api_key = self.create_api_key(user_email, tier)
                
                logger.info(f"Subscription completed for {user_email}, created API key")
                return {"status": "success", "api_key": api_key}
            
            elif event.type == "customer.subscription.updated":
                subscription = event.data.object
                # Handle subscription updates (e.g., tier changes)
                logger.info(f"Subscription updated: {subscription.id}")
                return {"status": "success", "message": "Subscription updated"}
            
            elif event.type == "customer.subscription.deleted":
                subscription = event.data.object
                # Handle subscription cancellations
                logger.info(f"Subscription cancelled: {subscription.id}")
                return {"status": "success", "message": "Subscription cancelled"}
            
            return {"status": "success", "message": f"Processed {event.type}"}
        
        except Exception as e:
            logger.error(f"Error handling webhook: {e}")
            return {"status": "error", "message": str(e)}

# Create a singleton instance
billing_manager = BillingManager() 