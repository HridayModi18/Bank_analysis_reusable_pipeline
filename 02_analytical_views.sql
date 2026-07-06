
USE creditcard;
--vw_Analytics_Features is added in database

CREATE OR REPLACE VIEW vw_Analytics_Features AS
--cte
WITH CalculatedDOB AS (
    SELECT *,
        CASE 
            WHEN data_nascimento LIKE '%/%/%' THEN STR_TO_DATE(data_nascimento, '%d/%m/%Y') 
            WHEN data_nascimento LIKE '%-%-%' THEN STR_TO_DATE(data_nascimento, '%Y-%m-%d') 
            ELSE NULL 
        END AS parsed_dob
    FROM bankchurn_clean
),
CalculatedAge AS (
    SELECT *,
        TIMESTAMPDIFF(YEAR, parsed_dob, CURDATE()) AS calculated_age
    FROM CalculatedDOB
)
--cte ended
SELECT 
    customer_id, 
    nome, 
    email, 
    numero_conta, 
    data_nascimento,
    LOWER(TRIM(cidade)) AS cidade,
    LOWER(TRIM(country)) AS country,
    gender,
    calculated_age AS age,
    CASE WHEN calculated_age < 30 THEN '18-30'
        WHEN calculated_age BETWEEN 30 AND 45 THEN '31-45'
        WHEN calculated_age BETWEEN 46 AND 60 THEN '46-60'
        ELSE '60+'
    END AS age_band,
    credit_score, 
    tenure, 
    balance, 
    limite_credito,
    balance / NULLIF(limite_credito, 0) AS credit_utilization,
    products_number, 
    credit_card, 
    active_member, 
    estimated_salary,
    CASE WHEN estimated_salary IS NULL THEN 1 ELSE 0 END AS salary_missing_flag,
    churn
FROM CalculatedAge;

--vw_Data_Quality_Report is added in database , and now working fine
CREATE OR REPLACE VIEW vw_Data_Quality_Report AS
    (SELECT COUNT(*) FROM reject_records) AS total_rejected,
    (SELECT COUNT(*) FROM duplicate_records) AS total_duplicates,
    (SELECT COUNT(*) FROM bankchurn_clean) AS total_clean,
    (SELECT COUNT(*) FROM bankChurn WHERE estimated_salary IS NULL) * 100.0 / (SELECT COUNT(*) FROM bankChurn) AS salary_missing_pct;
