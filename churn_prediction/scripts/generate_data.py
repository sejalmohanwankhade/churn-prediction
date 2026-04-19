"""
generate_data.py — Generate synthetic customer churn dataset
Usage: python scripts/generate_data.py
Output: data/customer_churn_dataset.csv
"""

import random
import csv
import os

random.seed(42)

NEGATIVE_REVIEWS = [
    "Terrible service, thinking about cancelling immediately.",
    "Very disappointed with the product quality and support.",
    "Worst experience ever. Going to switch to a competitor.",
    "Support team is useless. Never resolves my issues.",
    "Overpriced for what you get. Not worth the money at all.",
    "Keeps breaking down. I'm frustrated and want to quit.",
    "Billing errors every month. Completely unacceptable service.",
    "Service downtime is ridiculous. I've had enough problems.",
]

NEUTRAL_REVIEWS = [
    "Service is okay. Nothing exceptional about the experience.",
    "Average product, does the job but nothing impressive.",
    "Had some issues but support eventually fixed them.",
    "It's fine. Sometimes slow but generally works alright.",
    "Decent service for the price. Could be improved though.",
]

POSITIVE_REVIEWS = [
    "Excellent service! Very happy with everything so far.",
    "Amazing support team and reliable product. Highly recommend!",
    "Love the service. Fast, efficient, and great value for money.",
    "Outstanding experience. Would definitely recommend to friends.",
    "Perfect! No issues at all. Very satisfied customer.",
    "Great product and fantastic customer support. Keep it up!",
    "Wonderful service. Exceeded all my expectations completely.",
]

CONTRACTS = ["Month-to-month", "One year", "Two year"]
PAYMENTS = ["Electronic check", "Mailed check", "Bank transfer", "Credit card"]

def generate_customer(i: int) -> dict:
    # Assign risk profile
    risk = random.choices(["high", "medium", "low"], weights=[0.25, 0.35, 0.40])[0]

    if risk == "high":
        review = random.choice(NEGATIVE_REVIEWS)
        tenure = random.randint(1, 12)
        charges = round(random.uniform(75, 120), 2)
        complaints = random.randint(3, 9)
        calls = random.randint(4, 12)
        contract = "Month-to-month"
        payment = random.choice(["Electronic check", "Mailed check"])
        churn = 1 if random.random() < 0.82 else 0
    elif risk == "medium":
        review = random.choice(NEUTRAL_REVIEWS)
        tenure = random.randint(6, 30)
        charges = round(random.uniform(50, 80), 2)
        complaints = random.randint(1, 3)
        calls = random.randint(1, 4)
        contract = random.choice(["Month-to-month", "One year"])
        payment = random.choice(PAYMENTS)
        churn = 1 if random.random() < 0.35 else 0
    else:
        review = random.choice(POSITIVE_REVIEWS)
        tenure = random.randint(18, 72)
        charges = round(random.uniform(30, 60), 2)
        complaints = random.randint(0, 1)
        calls = random.randint(0, 2)
        contract = random.choice(["One year", "Two year"])
        payment = random.choice(["Bank transfer", "Credit card"])
        churn = 1 if random.random() < 0.08 else 0

    return {
        "customer_id": f"CUST_{i:05d}",
        "review_text": review,
        "tenure_months": tenure,
        "monthly_charges": charges,
        "num_complaints": complaints,
        "num_support_calls": calls,
        "contract_type": contract,
        "payment_method": payment,
        "churn": churn,
        "risk_profile": risk
    }


def main():
    os.makedirs("data", exist_ok=True)
    customers = [generate_customer(i) for i in range(1, 5001)]

    filepath = "data/customer_churn_dataset.csv"
    fieldnames = list(customers[0].keys())

    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(customers)

    total = len(customers)
    churned = sum(c["churn"] for c in customers)
    print(f"✅ Dataset generated: {filepath}")
    print(f"   Total customers : {total:,}")
    print(f"   Churned         : {churned:,} ({churned/total:.1%})")
    print(f"   Retained        : {total - churned:,} ({(total-churned)/total:.1%})")


if __name__ == "__main__":
    main()
