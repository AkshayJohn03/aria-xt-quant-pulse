  from dotenv import load_dotenv
  import os
  from pathlib import Path

  dotenv_path = Path(__file__).resolve().parent / ".env"
  load_dotenv(dotenv_path)
  print("API KEY:", os.getenv("ZERODHA_API_KEY"))
  print("API SECRET:", os.getenv("ZERODHA_API_SECRET"))
  print("ACCESS TOKEN:", os.getenv("ZERODHA_ACCESS_TOKEN"))