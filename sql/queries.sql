-- Q1: Revenue by plan. Only successful recharges. INNER JOIN recharges -> subscribers -> plans.
SELECT
    p.plan_name,
    SUM(r.amount_inr) AS total_revenue,
    AVG(r.amount_inr) AS avg_revenue,
    COUNT(DISTINCT r.subscriber_id) AS distinct_subscribers
FROM recharges r
INNER JOIN subscribers s ON s.subscriber_id = r.subscriber_id
INNER JOIN plans p ON p.plan_id = s.plan_id
WHERE r.status = 'success'
GROUP BY p.plan_name
ORDER BY total_revenue DESC;

-- Q2: Subscribers who never had a successful recharge. LEFT JOIN, keep unmatched (recharge_id IS NULL).
SELECT
    s.subscriber_id,
    s.tenure_months
FROM subscribers s
LEFT JOIN recharges r
    ON r.subscriber_id = s.subscriber_id AND r.status = 'success'
WHERE r.recharge_id IS NULL;

-- Q3: Ranking of subscribers by total successful recharge amount within each city.
WITH totals AS (
    SELECT
        s.subscriber_id,
        s.city,
        SUM(r.amount_inr) AS total_amount
    FROM subscribers s
    INNER JOIN recharges r
        ON r.subscriber_id = s.subscriber_id AND r.status = 'success'
    GROUP BY s.subscriber_id, s.city
)
SELECT
    subscriber_id,
    city,
    total_amount,
    RANK()       OVER (PARTITION BY city ORDER BY total_amount DESC) AS rank_,
    DENSE_RANK() OVER (PARTITION BY city ORDER BY total_amount DESC) AS dense_rank_,
    ROW_NUMBER() OVER (PARTITION BY city ORDER BY total_amount DESC, subscriber_id) AS row_number_
FROM totals
ORDER BY city, total_amount DESC, subscriber_id;

-- Q4: Running total of successful recharges for subscriber TP00010.
SELECT
    recharge_id,
    recharge_date,
    amount_inr,
    SUM(amount_inr) OVER (ORDER BY recharge_date, recharge_id) AS running_total
FROM recharges
WHERE subscriber_id = 'TP00010' AND status = 'success'
ORDER BY recharge_date, recharge_id;
