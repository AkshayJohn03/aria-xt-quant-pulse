import { ConfigManager } from './config';

export interface OHLCVData {
  timestamp: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface OptionChainData {
  strike: number;
  expiry: string;
  call: {
    ltp: number;
    volume: number;
    oi: number;
    iv: number;
    delta: number;
    gamma: number;
    theta: number;
    vega: number;
  };
  put: {
    ltp: number;
    volume: number;
    oi: number;
    iv: number;
    delta: number;
    gamma: number;
    theta: number;
    vega: number;
  };
}

export class DataFetcher {
  private config: ConfigManager;
  private baseURL: string;

  constructor() {
    this.config = new ConfigManager();
    this.baseURL = 'http://localhost:8000/api/v1';
  }

  async fetchLiveOHLCV(symbol: string = 'NIFTY50'): Promise<OHLCVData[] | null> {
    try {
      const response = await fetch(`${this.baseURL}/live-ohlcv?symbol=${symbol}`, {
        headers: {
          'Content-Type': 'application/json'
        }
      });

      if (!response.ok) {
        throw new Error(`Failed to fetch live OHLCV: ${response.statusText}`);
      }

      const result = await response.json();
      if (result.success) {
        return result.data;
      } else {
        throw new Error(result.error || 'Failed to fetch OHLCV data');
      }
    } catch (error) {
      console.error('Error fetching live OHLCV:', error);
      return null;
    }
  }

  async fetchHistoricalOHLCV(symbol: string, startDate: string, endDate: string): Promise<OHLCVData[] | null> {
    try {
      const response = await fetch(`${this.baseURL}/historical-ohlcv?symbol=${symbol}&start_date=${startDate}&end_date=${endDate}`, {
        headers: {
          'Content-Type': 'application/json'
        }
      });

      if (!response.ok) {
        throw new Error(`Failed to fetch historical OHLCV: ${response.statusText}`);
      }

      const result = await response.json();
      if (result.success) {
        return result.data;
      } else {
        throw new Error(result.error || 'Failed to fetch historical data');
      }
    } catch (error) {
      console.error('Error fetching historical OHLCV:', error);
      return null;
    }
  }

  async fetchOptionChain(url?: string): Promise<OptionChainData[] | null> {
    try {
      const endpoint = url ? `${this.baseURL}${url}` : `${this.baseURL}/option-chain`;
      const response = await fetch(endpoint, {
        headers: {
          'Content-Type': 'application/json'
        }
      });

      if (!response.ok) {
        throw new Error(`Failed to fetch option chain: ${response.statusText}`);
      }

      const result = await response.json();
      if (result.success) {
        return result.data;
      } else {
        throw new Error(result.error || 'Failed to fetch option chain');
      }
    } catch (error) {
      console.error('Error fetching option chain:', error);
      return null;
    }
  }

  async fetchMarketData(): Promise<any | null> {
    try {
      const response = await fetch(`${this.baseURL}/market-data`, {
        headers: {
          'Content-Type': 'application/json'
        }
      });

      if (!response.ok) {
        throw new Error(`Failed to fetch market data: ${response.statusText}`);
      }

      const result = await response.json();
      if (result.success) {
        return result.data;
      } else {
        throw new Error(result.error || 'Failed to fetch market data');
      }
    } catch (error) {
      console.error('Error fetching market data:', error);
      return null;
    }
  }

  private parseHistoricalData(data: any): OHLCVData[] {
    if (!data.values) return [];
    
    return data.values.map((item: any) => ({
      timestamp: new Date(item.datetime).getTime(),
      open: parseFloat(item.open),
      high: parseFloat(item.high),
      low: parseFloat(item.low),
      close: parseFloat(item.close),
      volume: parseInt(item.volume)
    }));
  }
}
