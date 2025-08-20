import os
import json
import pandas as pd
import psycopg
from psycopg.rows import dict_row
from dotenv import load_dotenv

load_dotenv()

# --- Configuration ---
# Load database connection details from environment variables
# These should be the same as your other Python scripts.
host = os.getenv("DB_HOST")
db   = os.getenv("DB_NAME", "postgres")
user = os.getenv("DB_USER")
pwd  = os.getenv("DB_PASS")
port = int(os.getenv("DB_PORT", "5432"))

# --- Main Functions ---

def load_json_data(file_name, metric_name):
    """
    Loads data from a JSON file downloaded from Blockchain.com.
    
    Args:
        file_name (str): The name of the JSON file to load.
        metric_name (str): The name of the key containing the data list.
        
    Returns:
        pandas.DataFrame: A DataFrame with 'ds' (date) and the metric's value.
    """
    print(f"Processing {file_name}...")
    try:
        with open(file_name, 'r') as f:
            data = json.load(f)
        
        # The data is a list of dictionaries, each with 'x' and 'y'
        df = pd.DataFrame(data[metric_name])
        
        # Convert the 'x' millisecond timestamp to a clean date
        # The unit='ms' tells pandas to interpret the number as milliseconds
        df['ds'] = pd.to_datetime(df['x'], unit='ms').dt.date
        
        # Rename the 'y' column to something meaningful
        df = df.rename(columns={'y': metric_name})
        
        # Keep only the essential columns
        return df[['ds', metric_name]]
        
    except FileNotFoundError:
        print(f"Error: {file_name} not found. Please make sure it's in the same directory.")
        return None
    except Exception as e:
        print(f"An error occurred processing {file_name}: {e}")
        return None

def main():
    """
    Main function to orchestrate loading, merging, and uploading data.
    """
    print("Starting ingestion of daily metrics...")

    # 1. Load data from each JSON file into a pandas DataFrame
    price_df = load_json_data('./pipeline/market-price.json', 'market-price')
    supply_df = load_json_data('./pipeline/total-bitcoins.json', 'total-bitcoins')
    volume_df = load_json_data('./pipeline/trade-volume.json', 'trade-volume')

    # Stop if any file failed to load
    if price_df is None or supply_df is None or volume_df is None:
        print("Aborting due to file loading errors.")
        return

    # 2. Merge the three DataFrames into one
    # We start with the price data and left-join the others onto it.
    print("Merging dataframes...")
    merged_df = pd.merge(price_df, supply_df, on='ds', how='left')
    merged_df = pd.merge(merged_df, volume_df, on='ds', how='left')
    
    # Add the symbol column, required by our database schema
    merged_df['symbol'] = 'BTC-USD'
    
    # Rename columns to match the database schema exactly
    merged_df = merged_df.rename(columns={
        'market-price': 'price_usd',
        'total-bitcoins': 'total_supply',
        'trade-volume': 'trade_volume_usd'
    })
    
    try:
        with psycopg.connect(host=host, dbname=db, user=user, password=pwd, port=port) as conn:
            with conn.cursor() as cur:
                print("Connected to database. Inserting data...")
                
                # Prepare the INSERT statement
                # ON CONFLICT...DO UPDATE is crucial. It means if we run the script
                # again, it will update existing records instead of failing.
                sql = """
                INSERT INTO public.bronze_daily_metrics (symbol, ds, price_usd, trade_volume_usd, total_supply)
                VALUES (%(symbol)s, %(ds)s, %(price_usd)s, %(trade_volume_usd)s, %(total_supply)s)
                ON CONFLICT (symbol, ds) DO UPDATE SET
                    price_usd = EXCLUDED.price_usd,
                    trade_volume_usd = EXCLUDED.trade_volume_usd,
                    total_supply = EXCLUDED.total_supply;
                """
                
                # Convert dataframe to a list of dictionaries for insertion
                data_to_insert = merged_df.to_dict(orient='records')
                
                # Execute the insert command for all rows
                cur.executemany(sql, data_to_insert)
                conn.commit()
                
                print(f"Successfully inserted or updated {len(data_to_insert)} rows.")

    except Exception as e:
        print(f"Database operation failed: {e}")

# --- Script Execution ---
if __name__ == "__main__":
    # This ensures the main() function runs only when the script is executed directly
    main()