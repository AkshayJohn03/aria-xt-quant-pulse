
import { ConfigManager } from '../config';

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

  async fetchLiveOHLCV(symbol: string = 'NIFTY50'): Promise<OHLCVData[]> {
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
      return this.generateSimulatedOHLCV();
    }
  }

  async fetchHistoricalOHLCV(symbol: string, startDate: string, endDate: string): Promise<OHLCVData[]> {
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
      return this.generateSimulatedOHLCV();
    }
  }

  async fetchOptionChain(): Promise<OptionChainData[]> {
    try {
      const response = await fetch(`${this.baseURL}/option-chain`, {
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
      return this.generateSimulatedOptionChain();
    }
  }

  async fetchMarketData(): Promise<any> {
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
      return {
        nifty: {
          value: 19850.25 + (Math.random() - 0.5) * 100,
          change: (Math.random() - 0.5) * 50,
          percentChange: (Math.random() - 0.5) * 2
        },
        sensex: {
          value: 66589.93 + (Math.random() - 0.5) * 500,
          change: (Math.random() - 0.5) * 200,
          percentChange: (Math.random() - 0.5) * 1.5
        },
        marketStatus: 'OPEN',
        lastUpdate: new Date().toISOString(),
        aiSentiment: {
          direction: 'NEUTRAL',
          confidence: 50
        }
      };
    }
  }

  private generateSimulatedOHLCV(): OHLCVData[] {
    const data: OHLCVData[] = [];
    let price = 19850;
    const now = Date.now();

    for (let i = 100; i >= 0; i--) {
      const timestamp = now - i * 60000; // 1-minute intervals
      const open = price;
      const change = (Math.random() - 0.5) * 20;
      const close = open + change;
      const high = Math.max(open, close) + Math.random() * 10;
      const low = Math.min(open, close) - Math.random() * 10;
      const volume = Math.floor(Math.random() * 1000000) + 500000;

      data.push({
        timestamp,
        open,
        high,
        low,
        close,
        volume
      });

      price = close;
    }

    return data;
  }

  private generateSimulatedOptionChain(): OptionChainData[] {
    const basePrice = 19850;
    const strikes = [];

    for (let i = -5; i <= 5; i++) {
      const strike = Math.round((basePrice + i * 50) / 50) * 50;
      strikes.push({
        strike,
        expiry: '2024-06-27',
        call: {
          ltp: Math.max(1, basePrice - strike + Math.random() * 20),
          volume: Math.floor(Math.random() * 200000) + 50000,
          oi: Math.floor(Math.random() * 500000) + 100000,
          iv: 15 + Math.random() * 10,
          delta: Math.max(0, Math.min(1, (basePrice - strike) / 100 + 0.5)),
          gamma: 0.01 + Math.random() * 0.1,
          theta: -1 - Math.random() * 3,
          vega: 10 + Math.random() * 10
        },
        put: {
          ltp: Math.max(1, strike - basePrice + Math.random() * 20),
          volume: Math.floor(Math.random() * 150000) + 30000,
          oi: Math.floor(Math.random() * 400000) + 80000,
          iv: 16 + Math.random() * 8,
          delta: Math.max(-1, Math.min(0, (basePrice - strike) / 100 - 0.5)),
          gamma: 0.01 + Math.random() * 0.1,
          theta: -1 - Math.random() * 3,
          vega: 10 + Math.random() * 10
        }
      });
    }

    return strikes;
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
