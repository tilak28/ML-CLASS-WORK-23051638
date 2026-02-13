import csv
import random

filename = "employees.csv"

fields = ["ID", "Age", "Salary", "Score", "Rating"]

records = []
for i in range(1, 11):
    record = {
        "ID": i,
        "Age": random.randint(20, 60),
        "Salary": random.randint(20000, 150000),
        "Score": random.randint(1, 100),
        "Rating": round(random.uniform(1, 5), 1)
    }
    records.append(record)

with open(filename, "w", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fields)
    writer.writeheader()
    writer.writerows(records)

print("CSV file 'employees.csv' created successfully with 10 records.")
