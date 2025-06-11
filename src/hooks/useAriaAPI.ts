
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
    const interval = setInterval(fetchMarketData, 5000); // Poll every 5 seconds
    return () => clearInterval(interval);
  }, []);

  return { marketData, loading, error, refetch: fetchMarketData };
};

// Existing hooks remain the same for now, but will be updated later for real data
export const usePredictions = () => {
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const generatePrediction = async () => {
    try {
      setLoading(true);
      const response = await ariaAPI.generatePrediction();
      if (response.success) {
        setPrediction(response.data);
        setError(null);
      } else {
        setError(response.error);
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    generatePrediction();
    const interval = setInterval(generatePrediction, 30000); // Every 30 seconds
    return () => clearInterval(interval);
  }, []);

  return { prediction, loading, error, regenerate: generatePrediction };
};

export const usePortfolio = () => {
  const [portfolio, setPortfolio] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const fetchPortfolio = async () => {
    try {
      setLoading(true);
      const response = await ariaAPI.getPortfolio();
      if (response.success) {
        setPortfolio(response.data);
        setError(null);
      } else {
        setError(response.error);
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchPortfolio();
    const interval = setInterval(fetchPortfolio, 5000);
    return () => clearInterval(interval);
  }, []);

  return { portfolio, loading, error, refetch: fetchPortfolio };
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
    } catch (err) {
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

export const useConnectionStatus = () => {
  const [status, setStatus] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const fetchStatus = async () => {
    try {
      setLoading(true);
      const response = await ariaAPI.getConnectionStatus();
      if (response.success) {
        setStatus(response.data);
        setError(null);
      } else {
        // Handle the case where response might have an error property
        setError('Failed to fetch connection status');
      }
    } catch (err: any) {
      setError(err.message || 'Network error or unexpected response.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchStatus();
    const interval = setInterval(fetchStatus, 10000);
    return () => clearInterval(interval);
  }, []);

  return { status, loading, error, refetch: fetchStatus };
};
