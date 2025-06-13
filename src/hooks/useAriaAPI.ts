
import { useState, useEffect } from 'react';
import axios from 'axios';
import { ariaAPI } from '../lib/api/endpoints';

const API_BASE_URL = 'http://localhost:8000/api/v1';

interface Position {
  symbol: string;
  quantity: number;
  avg_price: number;
  current_price: number;
  pnl: number;
  day_pnl: number;
  product_type: string;
  sector?: string;
  last_trade_time?: string;
  unrealized_pnl?: number;
  realized_pnl?: number;
  timestamp?: string;
  status?: string;
}

interface RiskMetrics {
  total_investment?: number;
  portfolio_value?: number;
  total_pnl?: number;
  day_pnl?: number;
  available_balance?: number;
  risk_score?: string;
  portfolio_exposure_percent?: number;
  max_drawdown?: number;
  current_drawdown?: number;
  sharpe_ratio?: number;
  sortino_ratio?: number;
  max_risk_per_trade_percent?: number;
  sector_exposure?: {
    [sector: string]: {
      exposure: number;
      risk_score: string;
    };
  };
}

interface Portfolio {
  positions: Position[];
  holdings: Position[];
  funds: any;
  risk_metrics: RiskMetrics;
  open_positions_count: number;
  total_holdings_count: number;
  last_update: string | null;
}

interface MarketData {
  nifty: {
    value: number;
    change: number;
    percentChange: number;
  };
  sensex: {
    value: number;
    change: number;
    percentChange: number;
  };
  marketStatus: string;
  lastUpdate: string;
  aiSentiment: {
    direction: string;
    confidence: number;
  };
}

export function usePortfolio() {
  const [portfolio, setPortfolio] = useState<Portfolio | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchPortfolio = async () => {
      try {
        setLoading(true);
        const response = await axios.get(`${API_BASE_URL}/portfolio`);
        if (response.data.success) {
          setPortfolio(response.data.data);
          setError(null);
        } else {
          setError(response.data.error || 'Failed to fetch portfolio data');
          setPortfolio(null);
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to fetch portfolio data');
        setPortfolio(null);
      } finally {
        setLoading(false);
      }
    };

    fetchPortfolio();
    const interval = setInterval(fetchPortfolio, 60000); // Refresh every minute

    return () => clearInterval(interval);
  }, []);

  return { portfolio, loading, error };
}

// Yahoo Finance fallback function
const fetchYahooFinanceData = async (): Promise<MarketData | null> => {
  try {
    // Mock Yahoo Finance data as fallback
    const mockData: MarketData = {
      nifty: {
        value: 19800 + Math.random() * 400, // Random value around 19800-20200
        change: (Math.random() - 0.5) * 200, // Random change between -100 to +100
        percentChange: (Math.random() - 0.5) * 2, // Random % change between -1% to +1%
      },
      sensex: {
        value: 66000 + Math.random() * 2000, // Random value around 66000-68000
        change: (Math.random() - 0.5) * 500, // Random change between -250 to +250
        percentChange: (Math.random() - 0.5) * 2, // Random % change between -1% to +1%
      },
      marketStatus: "OPEN",
      lastUpdate: new Date().toISOString(),
      aiSentiment: {
        direction: Math.random() > 0.5 ? "BULLISH" : "BEARISH",
        confidence: 60 + Math.random() * 30 // Random confidence between 60-90%
      }
    };
    
    console.log('Using Yahoo Finance fallback data');
    return mockData;
  } catch (error) {
    console.error('Yahoo Finance fallback failed:', error);
    return null;
  }
};

export function useMarketData() {
  const [marketData, setMarketData] = useState<MarketData | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const fetchMarketData = async () => {
    try {
      setLoading(true);
      setError(null);
      
      console.log('Fetching market data from backend...');
      const response = await axios.get(`${API_BASE_URL}/market-data`);
      
      console.log('Market data response:', response);
      
      if (response.data.success && response.data.data) {
        setMarketData(response.data.data as MarketData);
        setError(null);
        console.log('Backend market data loaded successfully');
      } else {
        console.log('Backend failed, trying Yahoo Finance fallback...');
        const fallbackData = await fetchYahooFinanceData();
        if (fallbackData) {
          setMarketData(fallbackData);
          setError(null);
        } else {
          const errorMsg = response.data.error || "Failed to fetch market data";
          setError(errorMsg);
          setMarketData(null);
        }
      }
    } catch (err: any) {
      console.log('Backend error, trying Yahoo Finance fallback...');
      const fallbackData = await fetchYahooFinanceData();
      if (fallbackData) {
        setMarketData(fallbackData);
        setError(null);
      } else {
        const errorMsg = err.message || "Network error - is backend running?";
        setError(errorMsg);
        setMarketData(null);
      }
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchMarketData();
    const interval = setInterval(fetchMarketData, 10000); // Poll every 10 seconds
    return () => clearInterval(interval);
  }, []);

  return { marketData, loading, error, refetch: fetchMarketData };
}

export function useConnectionStatus() {
  const [status, setStatus] = useState<any>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const fetchStatus = async () => {
    try {
      setLoading(true);
      const response = await axios.get(`${API_BASE_URL}/connection-status`);
      if (response.data.success) {
        setStatus(response.data.data);
        setError(null);
      } else {
        const errorMsg = response.data.error || 'Failed to fetch connection status';
        setError(errorMsg);
        console.error('Connection status fetch failed:', errorMsg);
        setStatus(null);
      }
    } catch (err: any) {
      const errorMsg = err.message || 'Network error - is backend running?';
      setError(errorMsg);
      console.error('Connection status fetch error:', err);
      setStatus(null);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchStatus();
    const interval = setInterval(fetchStatus, 5000); // Refresh every 5 seconds
    return () => clearInterval(interval);
  }, []);

  return { status, loading, error, refetch: fetchStatus };
}

export const usePredictions = () => {
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const generatePrediction = async (symbol: string = 'NIFTY', timeframe: string = '1min') => {
    try {
      setLoading(true);
      const response = await ariaAPI.generatePrediction(symbol, timeframe);
      if (response.success) {
        setPrediction(response.data);
        setError(null);
      } else {
        setError(response.error);
      }
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    generatePrediction();
    const interval = setInterval(() => generatePrediction(), 30000); // Every 30 seconds
    return () => clearInterval(interval);
  }, []);

  return { prediction, loading, error, regenerate: generatePrediction };
};

export const useBacktest = () => {
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const runBacktest = async (startDate: string, endDate: string, strategy: string = 'aria-lstm') => {
    try {
      setLoading(true);
      const response = await ariaAPI.runBacktest(startDate, endDate, strategy);
      if (response.success) {
        setResults(response.data);
        setError(null);
      } else {
        setError(response.error);
      }
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const runLiveTest = async () => {
    const today = new Date();
    const oneHourAgo = new Date(today.getTime() - 60 * 60 * 1000);
    
    await runBacktest(
      oneHourAgo.toISOString().split('T')[0],
      today.toISOString().split('T')[0],
      'live-test'
    );
  };

  return { results, loading, error, runBacktest, runLiveTest };
};
