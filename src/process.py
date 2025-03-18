import pandas as pd
import json

# File paths
TEST_CSV = "test_data.csv"  
TEST_JSON = "test_classification.json"  

# Instruction text
instruction_str = """Classify the following customer inquiry into one of the predefined topics based on its content. The topics are:
- **employee_benefits**: Programs and perks offered to employees beyond their salary.
- **employee_training**: Educational programs to enhance employees' skills.
- **other**: Miscellaneous topics.
- **payroll**: Managing employee salaries, wages, and deductions.
- **performance_management**: Assessing and improving employee performance.
- **talent_acquisition**: Attracting and hiring candidates.
- **tax_services**: Tax calculation, filing, and compliance.
- **time_and_attendance**: Tracking employees' working hours.
Classify the following inquiry:"""

# Read CSV and convert to JSON format
df_test = pd.read_csv(TEST_CSV)
data = [{"instruction": instruction_str, "input": row["message"], "output": row["topic"]} for _, row in df_test.iterrows()]

# Save as a JSON file
with open(TEST_JSON, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print(f"✅ Conversion completed, {len(data)} entries saved to {TEST_JSON}")