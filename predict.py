import sys
import joblib
import pandas as pd

def main():
    if len(sys.argv) < 2:
        print("Usage: python predict.py <tenure_months> <monthly_charges> <num_support_tickets> <uses_mobile_app(0/1)> <contract_type> <payment_method>")
        print("Example: python predict.py 5 80 4 0 month-to-month credit_card")
        return

    tenure_months = int(sys.argv[1])
    monthly_charges = float(sys.argv[2])
    num_support_tickets = int(sys.argv[3])
    uses_mobile_app = int(sys.argv[4])
    contract_type = sys.argv[5]
    payment_method = sys.argv[6]

    model = joblib.load("churn_model.joblib")

    row = pd.DataFrame([{
        "tenure_months": tenure_months,
        "monthly_charges": monthly_charges,
        "num_support_tickets": num_support_tickets,
        "uses_mobile_app": uses_mobile_app,
        "contract_type": contract_type,
        "payment_method": payment_method,
    }])

    proba = model.predict_proba(row)[0, 1]
    pred = model.predict(row)[0]

    print(f"Churn probability: {proba:.3f}")
    print(f"Predicted churn label: {int(pred)} (1 = churn, 0 = stay)")

if __name__ == "__main__":
    main()
