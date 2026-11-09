"""Data audit, cleaning, SQLite loading, and validation pipeline."""
import os
import sqlite3
import pandas as pd
import numpy as np

RAW = "data/raw"
DB_PATH = "data/instamart.db"

def audit():
    print("Auditing raw CSV files...")
    for fn in ["customers.csv", "orders.csv", "products.csv", "tickets.csv"]:
        path = os.path.join(RAW, fn)
        if os.path.exists(path):
            df = pd.read_csv(path, keep_default_na=False)
            print(f" - {fn}: shape={df.shape}")

def clean_data():
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    
    # Load raw data
    orders = pd.read_csv(os.path.join(RAW, "orders.csv"), keep_default_na=False)
    customers = pd.read_csv(os.path.join(RAW, "customers.csv"), keep_default_na=False)
    products = pd.read_csv(os.path.join(RAW, "products.csv"), keep_default_na=False)
    tickets = pd.read_csv(os.path.join(RAW, "tickets.csv"), keep_default_na=False)
    
    # Cleaning steps
    # 1. Standardize city / gender
    customers["city"] = customers["city"].astype(str).str.strip().str.title()
    customers["gender"] = customers["gender"].astype(str).str.strip().str.capitalize()
    customers.loc[~customers["gender"].isin(["Male", "Female"]), "gender"] = np.nan
    
    # 2. Treat invalid ages as missing
    customers["age"] = pd.to_numeric(customers["age"], errors="coerce")
    customers.loc[(customers["age"] < 18) | (customers["age"] > 100), "age"] = np.nan
    
    # 3. Resolve duplicate orders by keeping latest timestamp
    orders["order_ts"] = pd.to_datetime(orders["order_ts"], errors="coerce")
    orders = orders.sort_values("order_ts").drop_duplicates(subset=["order_id"], keep="last")
    
    # 4. Impute missing amounts from catalogue prices
    prod_prices = dict(zip(products["product_id"], products["price_inr"]))
    def fix_amount(row):
        try:
            amt = float(row["order_amount"])
            if amt <= 0:
                raise ValueError()
            return amt
        except:
            p_price = prod_prices.get(row["product_id"], 100)
            qty = float(row["quantity"]) if str(row["quantity"]).isdigit() else 1
            return round(p_price * qty * (1 - float(row.get("discount_pct", 0) or 0) / 100), 2)
            
    orders["order_amount"] = orders.apply(fix_amount, axis=1)
    
    # 5. Delivery minutes cleaning
    orders["delivery_minutes"] = pd.to_numeric(orders["delivery_minutes"], errors="coerce")
    orders.loc[orders["delivery_minutes"] < 0, "delivery_minutes"] = np.nan
    
    # Write to SQLite
    customers.to_sql("customers", conn, if_exists="replace", index=False)
    products.to_sql("products", conn, if_exists="replace", index=False)
    orders.to_sql("orders", conn, if_exists="replace", index=False)
    tickets.to_sql("tickets", conn, if_exists="replace", index=False)
    conn.commit()
    conn.close()
    print("Data cleaned and loaded into SQLite successfully.")
    return {
        "customers": customers,
        "products": products,
        "orders": orders,
        "tickets": tickets,
    }

def validate():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM customers")
    cust_count = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM orders")
    ord_count = c.fetchone()[0]
    conn.close()
    assert cust_count > 0, "Customers table is empty!"
    assert ord_count > 0, "Orders table is empty!"
    print(f"Validation passed: customers={cust_count}, orders={ord_count}")

if __name__ == "__main__":
    audit()
    clean_data()
    validate()
