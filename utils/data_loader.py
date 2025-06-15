import pandas as pd
import json
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
from twelvedata import TDClient
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self, api_key: Optional[str] = None):
        """Initialize DataLoader with optional Twelve Data API key."""
        self.api_key = api_key
        if api_key:
            self.td_client = TDClient(apikey=api_key)

    def load_ohlc_from_csv(self, file_path: str) -> pd.DataFrame:
        """
        Load 1-minute OHLC data from a CSV file.
        
        Args:
            file_path (str): Path to the CSV file
            
        Returns:
            pd.DataFrame: DataFrame with columns [timestamp, open, high, low, close, volume]
        """
        try:
            df = pd.read_csv(file_path)
            required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            
            # Ensure all required columns exist
            if not all(col in df.columns for col in required_columns):
                raise ValueError(f"CSV must contain columns: {required_columns}")
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')
            
            return df[required_columns]
        except Exception as e:
            logger.error(f"Error loading OHLC data from CSV: {str(e)}")
            raise

    async def load_ohlc_from_twelvedata(
        self,
        symbol: str,
        interval: str = "1min",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetch historical OHLC data from Twelve Data API.
        
        Args:
            symbol (str): Trading symbol (e.g., "NIFTY")
            interval (str): Time interval (default: "1min")
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
            
        Returns:
            pd.DataFrame: DataFrame with OHLC data
        """
        if not self.api_key:
            raise ValueError("Twelve Data API key is required")

        try:
            ts = self.td_client.time_series(
                symbol=symbol,
                interval=interval,
                start_date=start_date,
                end_date=end_date,
                outputsize=5000
            )
            
            df = await ts.as_pandas()
            df = df.rename(columns={
                'datetime': 'timestamp',
                'volume': 'volume'
            })
            
            return df.sort_values('timestamp')
        except Exception as e:
            logger.error(f"Error fetching data from Twelve Data: {str(e)}")
            raise

    def load_option_chain_snapshot(self, file_path: str) -> pd.DataFrame:
        """
        Load historical option chain data from a CSV or JSONL file.
        
        Args:
            file_path (str): Path to the data file
            
        Returns:
            pd.DataFrame: DataFrame with option chain data
        """
        try:
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith('.jsonl'):
                # Read JSONL file line by line
                data = []
                with open(file_path, 'r') as f:
                    for line in f:
                        data.append(json.loads(line))
                df = pd.DataFrame(data)
            else:
                raise ValueError("Unsupported file format. Use .csv or .jsonl")

            # Convert timestamp column
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])

            return df
        except Exception as e:
            logger.error(f"Error loading option chain data: {str(e)}")
            raise

    def convert_csv_to_jsonl(
        self,
        csv_file_path: str,
        jsonl_output_path: str,
        chunk_size: int = 10000
    ) -> None:
        """
        Convert intraday CSV data to JSONL format with proper timestamps.
        
        Args:
            csv_file_path (str): Input CSV file path
            jsonl_output_path (str): Output JSONL file path
            chunk_size (int): Number of rows to process at once
        """
        try:
            # Process the CSV file in chunks to handle large files
            for chunk_number, chunk in enumerate(pd.read_csv(csv_file_path, chunksize=chunk_size)):
                # Convert column names to lowercase
                chunk.columns = chunk.columns.str.lower()
                
                # Convert timestamp if present
                if 'timestamp' in chunk.columns:
                    chunk['timestamp'] = pd.to_datetime(chunk['timestamp'])
                    
                # Write to JSONL file
                mode = 'w' if chunk_number == 0 else 'a'
                with open(jsonl_output_path, mode) as f:
                    for _, row in chunk.iterrows():
                        json.dump(row.to_dict(), f)
                        f.write('\n')
                        
            logger.info(f"Successfully converted {csv_file_path} to {jsonl_output_path}")
        except Exception as e:
            logger.error(f"Error converting CSV to JSONL: {str(e)}")
            raise 