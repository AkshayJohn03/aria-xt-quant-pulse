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
