# ML-Project
# Credit Default Prediction  💳

##  Data Source
The dataset used in this project is the **Default of Credit Card Clients** dataset from the **UCI Machine Learning Repository**. It was accessed via the `pycaret.datasets` module. It contains information on default payments, demographic factors, credit data, history of payment, and bill statements of credit card clients in Taiwan from April 2005 to September 2005.

---

##  Data Dictionary

| Column Name | Description | Values / Interpretation |
| :--- | :--- | :--- |
| **ID** | ID of each client | Unique Identifier |
| **LIMIT_BAL** | Amount of given credit (NT dollar) | Includes individual & family credit |
| **SEX** | Gender | 1 = Male, 2 = Female |
| **EDUCATION** | Education level | 1=grad school, 2=university, 3=high school, 4=others |
| **MARRIAGE** | Marital status | 1=married, 2=single, 3=others |
| **AGE** | Age in years | Numerical |
| **PAY_1 - 6** | Repayment status (Sept - April) | -1=pay duly, 1=delay 1 month, 2=delay 2 months... |
| **BILL_AMT1 - 6** | Amount of bill statement (Sept - April) | Amount in NT dollar |
| **PAY_AMT1 - 6** | Amount of previous payment (Sept - April) | Amount in NT dollar |
| **default** | **Target Variable** | 1 = Yes (Default), 0 = No |

---
