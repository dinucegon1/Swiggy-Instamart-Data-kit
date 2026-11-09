"""Refund agent tool loop and database integration."""
import sqlite3
import json

DB_PATH = "data/instamart.db"

def get_order(order_id):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("SELECT * FROM orders WHERE order_id = ?", (order_id,))
    row = cur.fetchone()
    conn.close()
    if not row:
        return json.dumps({"error": "Order not found"})
    return json.dumps(dict(row))
