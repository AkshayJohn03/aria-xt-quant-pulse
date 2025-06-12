
import { useState, useEffect } from 'react';
import { ariaAPI } from '../lib/api/endpoints';

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

interface PortfolioData {
  positions: Array<{
    symbol: string;
    quantity: number;
    avg_price: number;
    current_price: number;
    pnl: number;
    product_type: string;
    timestamp: string;
    status: string;
  }>;
  holdings: Array<any>;
  funds: {
    available_cash: number;
    free_margin: number;
    used_margin: number;
  };
  risk_metrics: {
    portfolio_value: number;
    total_investment: number;
    total_pnl: number;
    risk_score: string;
    max_drawdown: number;
    current_drawdown: number;
    sharpe_ratio: number;
    sortino_ratio: number;
    portfolio_exposure_percent: number;
    max_risk_per_trade_percent: number;
  };
  total_pnl: number;
  open_positions_count: number;
  last_update: string | null;
}

export const useMarketData = () => {
  const [marketData, setMarketData] = useState<MarketData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchMarketData = async () => {
    try {
      setLoading(true);
      setError(null);
      
      console.log('Fetching market data from backend...');
      const response = await ariaAPI.getMarketData();
      
      console.log('Market data response:', response);
      
      if (response.success && response.data) {
        setMarketData(response.data as MarketData);
        setError(null);
      } else {
        const errorMsg = response.error || "Failed to fetch market data";
        setError(errorMsg);
        console.error('Market data fetch failed:', errorMsg);
      }
    } catch (err: any) {
      const errorMsg = err.message || "Network error - is backend running?";
      setError(errorMsg);
      console.error('Market data fetch error:', err);
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
};

export const usePortfolio = () => {
  const [portfolio, setPortfolio] = useState<PortfolioData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchPortfolio = async () => {
    try {
      setLoading(true);
      setError(null);
      
      console.log('Fetching portfolio data from backend...');
      const response = await ariaAPI.getPortfolio();
      
      console.log('Portfolio response:', response);
      
      if (response.success && response.data) {
        setPortfolio(response.data as PortfolioData);
        setError(null);
      } else {
        const errorMsg = response.error || "Failed to fetch portfolio data";
        setError(errorMsg);
        console.error('Portfolio fetch failed:', errorMsg);
      }
    } catch (err: any) {
      const errorMsg = err.message || "Network error - is backend running?";
      setError(errorMsg);
      console.error('Portfolio fetch error:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchPortfolio();
    const interval = setInterval(fetchPortfolio, 15000); // Poll every 15 seconds
    return () => clearInterval(interval);
  }, []);

  return { portfolio, loading, error, refetch: fetchPortfolio };
};

export const useConnectionStatus = () => {
  const [status, setStatus] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchStatus = async () => {
    try {
      setLoading(true);
      setError(null);
      
      console.log('Fetching connection status from backend...');
      const response = await ariaAPI.getConnectionStatus();
      
      console.log('Connection status response:', response);
      
      if (response.success && response.data) {
        setStatus(response.data);
        setError(null);
      } else {
        const errorMsg = response.error || 'Failed to fetch connection status';
        setError(errorMsg);
        console.error('Connection status fetch failed:', errorMsg);
      }
    } catch (err: any) {
      const errorMsg = err.message || 'Network error - is backend running?';
      setError(errorMsg);
      console.error('Connection status fetch error:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchStatus();
    const interval = setInterval(fetchStatus, 30000); // Poll every 30 seconds
    return () => clearInterval(interval);
  }, []);

  return { status, loading, error, refetch: fetchStatus };
};

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
