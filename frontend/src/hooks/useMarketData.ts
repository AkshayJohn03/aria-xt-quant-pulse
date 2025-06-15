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
    let dataInterval: NodeJS.Timeout | null = null;
    let statusInterval: NodeJS.Timeout | null = null;
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

    if (!dataInterval) {
      dataInterval = setInterval(() => {
        console.log('Polling market data...');
        fetchMarketData();
      }, 30000); // Refresh every 30 seconds
    }
    if (!statusInterval) {
      statusInterval = setInterval(() => {
        console.log('Polling market status...');
        fetchMarketStatus();
      }, 30000); // Refresh every 30 seconds
    }

    return () => {
      if (dataInterval) clearInterval(dataInterval);
      if (statusInterval) clearInterval(statusInterval);
    };
  }, []);

  return { marketData, isMarketOpen, error };
}; 