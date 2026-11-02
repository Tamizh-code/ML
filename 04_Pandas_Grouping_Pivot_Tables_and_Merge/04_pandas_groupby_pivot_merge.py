# Experiment 4: Grouping, pivot tables, merging, and joining
import pandas as pd

students = pd.DataFrame({
    "student_id": [1, 2, 3, 4, 5],
    "name": ["Anu", "Bala", "Cathy", "Deepak", "Ezhil"],
    "department": ["CSE", "CSE", "ECE", "ECE", "CSE"]
})
marks = pd.DataFrame({
    "student_id": [1, 2, 3, 4, 5],
    "subject": ["Python", "Python", "Python", "Python", "Python"],
    "score": [88, 76, 91, 69, 84]
})
attendance = pd.DataFrame({
    "student_id": [1, 2, 3, 4, 5],
    "attendance": [95, 82, 90, 75, 88]
})

merged = students.merge(marks, on="student_id").join(
    attendance.set_index("student_id"), on="student_id"
)
print("Merged and joined data:\n", merged)
print("\nDepartment summary:\n", merged.groupby("department")["score"].agg(["mean", "max", "min"]))
print("\nPivot table:\n", pd.pivot_table(
    merged, values="score", index="department", columns="subject", aggfunc="mean"
))
