-- Seven SQL Learning Patterns

-- 1. Monthly Order Volume and Revenue
SELECT strftime('%Y-%m', order_ts) AS month, COUNT(*) AS total_orders, SUM(order_amount) AS total_revenue
FROM orders WHERE status = 'completed' GROUP BY month ORDER BY month;

-- 2. Top Selling Product Categories
SELECT p.category, SUM(o.quantity) AS units_sold, SUM(o.order_amount) AS revenue
FROM orders o JOIN products p ON o.product_id = p.product_id
WHERE o.status = 'completed' GROUP BY p.category ORDER BY revenue DESC;

-- 3. Customer Lifetime Value (Top 10 Customers)
SELECT customer_id, COUNT(*) AS orders_placed, SUM(order_amount) AS lifetime_spend
FROM orders WHERE status = 'completed' GROUP BY customer_id ORDER BY lifetime_spend DESC LIMIT 10;

-- 4. Late Deliveries Analysis (> 45 minutes)
SELECT COUNT(*) AS late_orders, AVG(delivery_minutes) AS avg_late_time
FROM orders WHERE status = 'completed' AND delivery_minutes > 45;

-- 5. City-wise Average Order Value
SELECT c.city, AVG(o.order_amount) AS avg_order_value, COUNT(o.order_id) AS total_orders
FROM orders o JOIN customers c ON o.customer_id = c.customer_id
WHERE o.status = 'completed' GROUP BY c.city ORDER BY avg_order_value DESC;

-- 6. Ticket Category Breakdown
SELECT true_category, COUNT(*) AS ticket_count FROM tickets GROUP BY true_category;

-- 7. Churn / Inactive Customers (No orders in last 90 days of dataset)
SELECT customer_id FROM customers WHERE customer_id NOT IN (
    SELECT DISTINCT customer_id FROM orders WHERE order_ts >= datetime('2025-09-01')
);