import os
import json
import pandas as pd
import psycopg
from psycopg.rows import dict_row
from dotenv import load_dotenv

load_dotenv()

host = os.getenv("DB_HOST")
db   = os.getenv("DB_NAME", "postgres")
user = os.getenv("DB_USER")
pwd  = os.getenv("DB_PASS")
port = int(os.getenv("DB_PORT", "5432"))


def load_json_data(file_name, metric_name):
    print(f"Processing {file_name}...")
    try:
        with open(file_name, 'r') as f:
            data = json.load(f)
        
        df = pd.DataFrame(data[metric_name])
        
        df['ds'] = pd.to_datetime(df['x'], unit='ms').dt.date
        
        df = df.rename(columns={'y': metric_name})
        
        return df[['ds', metric_name]]
        
    except FileNotFoundError:
        print(f"Error: {file_name} not found. Please make sure it's in the same directory.")
        return None
    except Exception as e:
        print(f"An error occurred processing {file_name}: {e}")
        return None

def main():
    price_df = load_json_data('./pipeline/market-price.json', 'market-price')
    supply_df = load_json_data('./pipeline/total-bitcoins.json', 'total-bitcoins')
    volume_df = load_json_data('./pipeline/trade-volume.json', 'trade-volume')

    if price_df is None or supply_df is None or volume_df is None:
        print("Aborting due to file loading errors.")
        return

    print("Merging dataframes...")
    merged_df = pd.merge(price_df, supply_df, on='ds', how='left')
    merged_df = pd.merge(merged_df, volume_df, on='ds', how='left')
    
    merged_df['symbol'] = 'BTC-USD'
    
    merged_df = merged_df.rename(columns={
        'market-price': 'price_usd',
        'total-bitcoins': 'total_supply',
        'trade-volume': 'trade_volume_usd'
    })
    
    try:
        with psycopg.connect(host=host, dbname=db, user=user, password=pwd, port=port) as conn:
            with conn.cursor() as cur:
                print("Connected to database. Inserting data...")
    
                sql = """
                INSERT INTO public.bronze_daily_metrics (symbol, ds, price_usd, trade_volume_usd, total_supply)
                VALUES (%(symbol)s, %(ds)s, %(price_usd)s, %(trade_volume_usd)s, %(total_supply)s)
                ON CONFLICT (symbol, ds) DO UPDATE SET
                    price_usd = EXCLUDED.price_usd,
                    trade_volume_usd = EXCLUDED.trade_volume_usd,
                    total_supply = EXCLUDED.total_supply;
                """
                
                data_to_insert = merged_df.to_dict(orient='records')
                
                cur.executemany(sql, data_to_insert)
                conn.commit()
                
                print(f"Successfully inserted or updated {len(data_to_insert)} rows.")

    except Exception as e:
        print(f"Database operation failed: {e}")

if __name__ == "__main__":
    main()