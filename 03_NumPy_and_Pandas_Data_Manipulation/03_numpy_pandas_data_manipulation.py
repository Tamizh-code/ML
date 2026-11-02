import numpy as np
import pandas as pd

# Create dataset
df = pd.DataFrame({
    'Name': ['A', 'B', 'C', 'D'],
    'Marks': [80, 60, np.nan, 90],
    'Age': [20, 21, 19, 22]
})

# Handle missing values
df.fillna(df['Marks'].mean(), inplace=True)

# Filter rows
print("Filtered Data:")
print(df[df['Marks'] > 70])

# Modify column
df['Marks'] = df['Marks'] + 5

# Aggregate data
print("\nAverage Marks:", df['Marks'].mean())

print("\nFinal Dataset:")
print(df)
# # Experiment 3: NumPy and Pandas data manipulation
# import numpy as np
# import pandas as pd

# np.random.seed(42)
# df = pd.DataFrame({
#     "Name": ["Arun", "Bala", "Charan", "Divya", "Esha", "Farhan"],
#     "Department": ["CSE", "ECE", "CSE", "IT", "ECE", "CSE"],
#     "Marks": [85, 72, np.nan, 91, 68, 78],
#     "Attendance": [92, 80, 75, np.nan, 88, 95]
# })
# print("Original data:\n", df)
# df["Marks"] = df["Marks"].fillna(df["Marks"].mean())
# df["Attendance"] = df["Attendance"].fillna(df["Attendance"].median())
# df["Result"] = np.where(df["Marks"] >= 75, "Pass", "Needs Improvement")
# df["Bonus Marks"] = df["Marks"] + 5
# filtered = df[(df["Marks"] >= 75) & (df["Attendance"] >= 80)]
# print("\nFiltered rows:\n", filtered)
# print("\nDepartment aggregation:\n", df.groupby("Department")[["Marks", "Attendance"]].mean())
# print("\nNumPy marks array:", df["Marks"].to_numpy())
