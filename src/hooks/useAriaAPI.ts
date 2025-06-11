import { useState, useEffect } from 'react';
import { ariaAPI } from '../lib/api/endpoints';

// Define the expected MarketData interface
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
    direction: string; // e.g., "BULLISH", "BEARISH", "NEUTRAL"
    confidence: number; // Percentage
  };
}

export const useMarketData = () => {
  const [marketData, setMarketData] = useState<MarketData | null>(null); // Specify type
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null); // Specify type

  const fetchMarketData = async () => {
    try {
      setLoading(true);
      // This call will now expect the backend to return data matching MarketData interface
      const response = await ariaAPI.getMarketData(); 
      if (response.success) {
        setMarketData(response.data as MarketData); // Cast to MarketData
        setError(null);
      } else {
        setError(response.error || "An unknown error occurred."); // Handle potential undefined error
      }
    } catch (err: any) { // Catch any error type
      setError(err.message || "Network error or unexpected response.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchMarketData();
    const interval = setInterval(fetchMarketData, 3000); // Poll every 3 seconds
    return () => clearInterval(interval);
  }, []); // Empty dependency array means this runs once on mount

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
