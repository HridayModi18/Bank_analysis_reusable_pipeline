
USE creditcard;  -- name need to be change , its of previous project
DROP TABLE IF EXISTS bankchurn_clean;
DROP TABLE IF EXISTS reject_records;
DROP TABLE IF EXISTS duplicate_records;
CREATE TABLE bankchurn_clean (
    customer_id BIGINT PRIMARY KEY,
    nome VARCHAR(255),
    email VARCHAR(255),
    numero_conta VARCHAR(50),
    data_nascimento VARCHAR(50),
    cidade VARCHAR(100),
    country VARCHAR(100),
    gender VARCHAR(20),
    age INT,
    credit_score INT,
    tenure INT,
    balance DOUBLE,
    limite_credito DOUBLE,
    products_number INT,
    credit_card INT,
    active_member INT,
    estimated_salary DOUBLE,
    churn INT
);
CREATE TABLE reject_records (
    customer_id DOUBLE,
    nome TEXT,
    email TEXT,
    numero_conta TEXT,
    data_nascimento TEXT,
    cidade TEXT,
    country TEXT,
    gender TEXT,
    age BIGINT,
    credit_score BIGINT,
    tenure BIGINT,
    balance DOUBLE,
    limite_credito DOUBLE,
    products_number BIGINT,
    credit_card BIGINT,
    active_member BIGINT,
    estimated_salary DOUBLE,
    churn BIGINT,
    rejection_reason VARCHAR(255)
);
CREATE TABLE duplicate_records (
    customer_id DOUBLE,
    nome TEXT,
    email TEXT,
    numero_conta TEXT,
    data_nascimento TEXT,
    cidade TEXT,
    country TEXT,
    gender TEXT,
    age BIGINT,
    credit_score BIGINT,
    tenure BIGINT,
    balance DOUBLE,
    limite_credito DOUBLE,
    products_number BIGINT,
    credit_card BIGINT,
    active_member BIGINT,
    estimated_salary DOUBLE,
    churn BIGINT
);











----------------------------------------------------------------------------------
--procedure is updated in database
DROP PROCEDURE IF EXISTS sp_clean_and_validate_data;
DELIMITER $$
CREATE PROCEDURE sp_clean_and_validate_data()
BEGIN TRUNCATE TABLE bankchurn_clean;
    TRUNCATE TABLE reject_records;
    TRUNCATE TABLE duplicate_records;--error fixed  , truncate is imp before creating
    -----------------------------------------------------------------------------------
    --rows without customer id
    --rows with weird birdates , because age will be calulated
    --rows with age out of range 18-110
    INSERT INTO reject_records
    SELECT customer_id, nome, email, numero_conta, data_nascimento, cidade, country, gender, age, credit_score, tenure, balance, limite_credito, products_number, credit_card, active_member, estimated_salary, churn,
        CASE
            WHEN customer_id IS NULL THEN 'Missing Customer ID'
            WHEN data_nascimento IS NULL THEN 'Missing Birth Date'
            WHEN CASE WHEN data_nascimento LIKE '%/%/%' THEN STR_TO_DATE(data_nascimento, '%d/%m/%Y') 
                    WHEN data_nascimento LIKE '%-%-%' THEN STR_TO_DATE(data_nascimento, '%Y-%m-%d') 
                    ELSE NULL 
                 END IS NULL THEN 'Unparseable Birth Date Format'
            WHEN TIMESTAMPDIFF(YEAR, 
                    CASE 
                        WHEN data_nascimento LIKE '%/%/%' THEN STR_TO_DATE(data_nascimento, '%d/%m/%Y') 
                        WHEN data_nascimento LIKE '%-%-%' THEN STR_TO_DATE(data_nascimento, '%Y-%m-%d') 
                        ELSE NULL 
                    END, CURDATE()) < 18 THEN 'Age below 18'
            WHEN TIMESTAMPDIFF(YEAR, 
                    CASE 
                        WHEN data_nascimento LIKE '%/%/%' THEN STR_TO_DATE(data_nascimento, '%d/%m/%Y') 
                        WHEN data_nascimento LIKE '%-%-%' THEN STR_TO_DATE(data_nascimento, '%Y-%m-%d') 
                        ELSE NULL 
                    END, CURDATE()) > 110 THEN 'Age above 110'
            WHEN age = -1 THEN 'Staging Age is -1'
            ELSE 'Unknown Issue' --just to be safe
        END AS rejection_reason
    FROM bankChurn
    WHERE customer_id IS NULL 
       OR data_nascimento IS NULL
       OR CASE WHEN data_nascimento LIKE '%/%/%' THEN STR_TO_DATE(data_nascimento, '%d/%m/%Y') 
            WHEN data_nascimento LIKE '%-%-%' THEN STR_TO_DATE(data_nascimento, '%Y-%m-%d') 
            ELSE NULL 
          END IS NULL
       OR TIMESTAMPDIFF(YEAR, 
            CASE WHEN data_nascimento LIKE '%/%/%' THEN STR_TO_DATE(data_nascimento, '%d/%m/%Y') 
                WHEN data_nascimento LIKE '%-%-%' THEN STR_TO_DATE(data_nascimento, '%Y-%m-%d') 
                ELSE NULL 
            END, CURDATE()) < 18
       OR TIMESTAMPDIFF(YEAR, 
            CASE  WHEN data_nascimento LIKE '%/%/%' THEN STR_TO_DATE(data_nascimento, '%d/%m/%Y') 
                WHEN data_nascimento LIKE '%-%-%' THEN STR_TO_DATE(data_nascimento, '%Y-%m-%d') 
                ELSE NULL 
            END, CURDATE()) > 110
       OR age = -1;




    ------------------------------------------------------------------
    --temporary tables will be deleted after the connection is close
    DROP TEMPORARY TABLE IF EXISTS temp_valid_partitioned;--error fixed , drop before creating
    CREATE TEMPORARY TABLE temp_valid_partitioned AS
    SELECT customer_id, nome, email, numero_conta, data_nascimento, cidade, country, gender,age, credit_score, tenure, balance, limite_credito, products_number, credit_card, active_member, estimated_salary, churn,
        ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY (SELECT NULL)) as row_num
    FROM bankChurn
    WHERE customer_id IS NOT NULL
      AND data_nascimento IS NOT NULL
      AND CASE WHEN data_nascimento LIKE '%/%/%' THEN STR_TO_DATE(data_nascimento, '%d/%m/%Y') --STR_TO_DATE(data_nascimento, '%d/%m/%Y') is written by ai , check once 
            WHEN data_nascimento LIKE '%-%-%' THEN STR_TO_DATE(data_nascimento, '%Y-%m-%d') 
            ELSE NULL 
          END IS NOT NULL
      AND TIMESTAMPDIFF(YEAR, CASE 
                WHEN data_nascimento LIKE '%/%/%' THEN STR_TO_DATE(data_nascimento, '%d/%m/%Y') 
                WHEN data_nascimento LIKE '%-%-%' THEN STR_TO_DATE(data_nascimento, '%Y-%m-%d') 
                ELSE NULL 
            END, CURDATE()) BETWEEN 18 AND 110
      AND age != -1;
---------------------------------------------------------------------------------------
--making bankchurn_clean in which only rownumber not equal to 1 will be entered
    INSERT INTO duplicate_records
    SELECT 
        customer_id, nome, email, numero_conta, data_nascimento, cidade, country, gender, 
        age, credit_score, tenure, balance, limite_credito, products_number, credit_card, 
        active_member, estimated_salary, churn
    FROM temp_valid_partitioned
    WHERE row_num > 1;
---------------------------------------------------------------------------------------
--making bankchurn_clean in which only rownumber=1 will be entered
    INSERT INTO bankchurn_clean
    SELECT 
        CAST(customer_id AS SIGNED) AS customer_id, nome, email, numero_conta, data_nascimento, cidade, country, gender, age, credit_score, tenure, balance, limite_credito, products_number, credit_card, 
        active_member, estimated_salary, churn
    FROM temp_valid_partitioned
    WHERE row_num = 1;
    DROP TEMPORARY TABLE IF EXISTS temp_valid_partitioned;
END$$
DELIMITER ;
