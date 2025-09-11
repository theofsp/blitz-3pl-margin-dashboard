import pandas as pd

# Function to convert Excel to Parquet
def convert_excel_to_parquet(excel_path, parquet_path):
    try:
        # Read Excel file
        df = pd.read_excel(excel_path)
        df.columns = df.columns.str.strip()
        # Convert 'Date Range' column to string if it exists
        if 'Date Range' in df.columns:
            df['Date Range'] = df['Date Range'].astype(str)
        # Save to Parquet, overwrite if exists
        df.to_parquet(parquet_path, index=False)
        print(f"Successfully converted '{excel_path}' to '{parquet_path}'")
        return True
    except FileNotFoundError:
        print(f"Excel file '{excel_path}' not found. Please ensure the file is in the same directory.")
        return False
    except Exception as e:
        print(f"Error converting Excel to Parquet: {str(e)}")
        return False

# File paths
excel_path = 'Ops Data Collection.xlsx'
parquet_path = 'Ops Data Collection.parquet'

# Run conversion
convert_excel_to_parquet(excel_path, parquet_path)