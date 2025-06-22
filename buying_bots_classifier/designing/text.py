import zipfile

file_path = "laptop.xlsx"  # Replace with your actual file path

if zipfile.is_zipfile(file_path):
    print("✅ File is a valid Excel .xlsx file.")
else:
    print("❌ File is not a valid Excel file (even if extension is .xlsx).")
