-- ============================================================
-- HR Analytics SQL Queries — Employee Attrition Project
-- Author: Yousuf Patel | M.Sc. ISM Munich
-- ============================================================
-- These queries assume the IBM HR dataset is loaded into a
-- table called `hr_employees`. Adapt schema as needed.
-- ============================================================


-- 1. Overall attrition rate
SELECT
    COUNT(*)                                              AS total_employees,
    SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END)  AS churned,
    ROUND(
        100.0 * SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2
    )                                                     AS attrition_rate_pct
FROM hr_employees;


-- 2. Attrition rate by department
SELECT
    Department,
    COUNT(*)                                                          AS headcount,
    SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END)               AS churned,
    ROUND(
        100.0 * SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2
    )                                                                 AS attrition_rate_pct
FROM hr_employees
GROUP BY Department
ORDER BY attrition_rate_pct DESC;


-- 3. Attrition by job role
SELECT
    JobRole,
    COUNT(*)                                                          AS headcount,
    SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END)               AS churned,
    ROUND(
        100.0 * SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2
    )                                                                 AS attrition_rate_pct
FROM hr_employees
GROUP BY JobRole
ORDER BY attrition_rate_pct DESC;


-- 4. Overtime vs attrition (key SHAP driver)
SELECT
    OverTime,
    COUNT(*)                                                          AS headcount,
    SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END)               AS churned,
    ROUND(
        100.0 * SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2
    )                                                                 AS attrition_rate_pct
FROM hr_employees
GROUP BY OverTime;


-- 5. Promotion lag analysis
SELECT
    CASE
        WHEN YearsSinceLastPromotion = 0     THEN 'Promoted this year'
        WHEN YearsSinceLastPromotion <= 2    THEN '1-2 years ago'
        WHEN YearsSinceLastPromotion <= 5    THEN '3-5 years ago'
        ELSE '5+ years ago'
    END                                                               AS promotion_lag_bucket,
    COUNT(*)                                                          AS headcount,
    ROUND(AVG(JobSatisfaction), 2)                                    AS avg_job_satisfaction,
    SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END)               AS churned,
    ROUND(
        100.0 * SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2
    )                                                                 AS attrition_rate_pct
FROM hr_employees
GROUP BY promotion_lag_bucket
ORDER BY attrition_rate_pct DESC;


-- 6. Average tenure by attrition status
SELECT
    Attrition,
    ROUND(AVG(YearsAtCompany), 2)       AS avg_tenure_years,
    ROUND(AVG(TotalWorkingYears), 2)    AS avg_total_experience,
    ROUND(AVG(MonthlyIncome), 0)        AS avg_monthly_income,
    ROUND(AVG(JobSatisfaction), 2)      AS avg_job_satisfaction
FROM hr_employees
GROUP BY Attrition;


-- 7. High-risk employee segments (attrition AND overtime AND low satisfaction)
SELECT
    Department,
    JobRole,
    COUNT(*)                            AS segment_size,
    ROUND(AVG(MonthlyIncome), 0)        AS avg_income,
    ROUND(AVG(YearsAtCompany), 1)       AS avg_tenure
FROM hr_employees
WHERE
    OverTime = 'Yes'
    AND JobSatisfaction <= 2
    AND YearsSinceLastPromotion >= 3
GROUP BY Department, JobRole
HAVING COUNT(*) >= 5
ORDER BY segment_size DESC;


-- 8. Manager-level attrition risk (by manager distance proxy)
-- Using YearsWithCurrManager as a proxy for stability under current leadership
SELECT
    CASE
        WHEN YearsWithCurrManager <= 1  THEN 'New Manager (0-1yr)'
        WHEN YearsWithCurrManager <= 3  THEN 'Established (2-3yr)'
        ELSE 'Long-term (4+yr)'
    END                                                               AS manager_tenure_group,
    COUNT(*)                                                          AS headcount,
    ROUND(
        100.0 * SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2
    )                                                                 AS attrition_rate_pct
FROM hr_employees
GROUP BY manager_tenure_group
ORDER BY attrition_rate_pct DESC;
