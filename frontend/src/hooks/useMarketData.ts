import { useState, useEffect } from 'react';
import axios from 'axios';

interface MarketData {
  nifty: {
    value: number;
    change: number;
    percentChange: number;
  };
  banknifty: {
    value: number;
    change: number;
    percentChange: number;
  };
}

export const useMarketData = () => {
  const [marketData, setMarketData] = useState<MarketData | null>(null);
  const [isMarketOpen, setIsMarketOpen] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchMarketData = async () => {
      try {
        const response = await axios.get('/api/market-data');
        setMarketData(response.data);
        setError(null);
      } catch (err) {
        setError('Failed to fetch market data');
        console.error('Error fetching market data:', err);
      }
    };

    const fetchMarketStatus = async () => {
      try {
        const response = await axios.get('/api/market-status');
        setIsMarketOpen(response.data.is_open);
        setError(null);
      } catch (err) {
        setError('Failed to fetch market status');
        console.error('Error fetching market status:', err);
      }
    };

    fetchMarketData();
    fetchMarketStatus();

    const dataInterval = setInterval(fetchMarketData, 10000); // Refresh every 10 seconds
    const statusInterval = setInterval(fetchMarketStatus, 30000); // Refresh every 30 seconds

    return () => {
      clearInterval(dataInterval);
      clearInterval(statusInterval);
    };
  }, []);

  return { marketData, isMarketOpen, error };
}; 