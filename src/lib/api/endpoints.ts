
import { DataFetcher } from './dataFetcher';
import { ModelInterface } from '../ai/modelInterface';
import { RiskManager } from '../trading/riskManager';

export class AriaAPI {
  private dataFetcher: DataFetcher;
  private modelInterface: ModelInterface;
  private riskManager: RiskManager;
  private baseURL: string;

  constructor() {
    this.dataFetcher = new DataFetcher();
    this.modelInterface = new ModelInterface();
    this.riskManager = new RiskManager();
    this.baseURL = 'http://localhost:8000/api/v1';
  }

  async getMarketData() {
    try {
      const response = await fetch(`${this.baseURL}/market-data`, {
        headers: {
          'Content-Type': 'application/json'
        }
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      if (result.success) {
        return {
          success: true,
          data: result.data
        };
      } else {
        return {
          success: false,
          error: result.error || 'Failed to fetch market data'
        };
      }
    } catch (error) {
      console.error('Frontend API error:', error);
      return {
        success: false,
        error: error.message || 'Network error'
      };
    }
  }

  async getLiveOHLCV(symbol: string = 'NIFTY50') {
    try {
      const response = await fetch(`${this.baseURL}/live-ohlcv?symbol=${symbol}`, {
        headers: {
          'Content-Type': 'application/json'
        }
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      return {
        success: result.success || false,
        data: result.data || null,
        error: result.error || null
      };
    } catch (error) {
      return {
        success: false,
        error: error.message
      };
    }
  }

  async getOptionChain() {
    try {
      const response = await fetch(`${this.baseURL}/option-chain`, {
        headers: {
          'Content-Type': 'application/json'
        }
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      return {
        success: result.success || false,
        data: result.data || null,
        error: result.error || null
      };
    } catch (error) {
      return {
        success: false,
        error: error.message
      };
    }
  }

  async generatePrediction() {
    try {
      const response = await fetch(`${this.baseURL}/prediction`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        }
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      return {
        success: result.success || false,
        data: result.data || null,
        error: result.error || null
      };
    } catch (error) {
      return {
        success: false,
        error: error.message
      };
    }
  }

  async generateTradingSignal() {
    try {
      const response = await fetch(`${this.baseURL}/trading-signal`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        }
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      return {
        success: result.success || false,
        data: result.data || null,
        error: result.error || null
      };
    } catch (error) {
      return {
        success: false,
        error: error.message
      };
    }
  }

  async getPortfolio() {
    try {
      const response = await fetch(`${this.baseURL}/portfolio`, {
        headers: {
          'Content-Type': 'application/json'
        }
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      return {
        success: result.success || false,
        data: result.data || null,
        error: result.error || null
      };
    } catch (error) {
      return {
        success: false,
        error: error.message
      };
    }
  }

  async runBacktest(startDate: string, endDate: string, strategy: string = 'aria-lstm') {
    try {
      const response = await fetch(`${this.baseURL}/backtest`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          start_date: startDate,
          end_date: endDate,
          strategy
        })
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      return {
        success: result.success || false,
        data: result.data || null,
        error: result.error || null
      };
    } catch (error) {
      return {
        success: false,
        error: error.message
      };
    }
  }

  async getConnectionStatus() {
    try {
      const response = await fetch(`${this.baseURL}/connection-status`, {
        headers: {
          'Content-Type': 'application/json'
        }
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      return {
        success: result.success || false,
        data: result.data || null,
        error: result.error || null
      };
    } catch (error) {
      return {
        success: false,
        error: error.message
      };
    }
  }
}

// Initialize global API instance
export const ariaAPI = new AriaAPI();
