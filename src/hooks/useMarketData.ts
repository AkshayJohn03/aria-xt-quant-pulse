
import { useState, useEffect } from 'react';
import axios from 'axios';

interface MarketData {
  nifty50: {
    value: number;
    change: number;
    percentChange: number;
    high: number;
    low: number;
    volume: number;
    timestamp: string | null;
    source: string;
  };
  banknifty: {
    value: number;
    change: number;
    percentChange: number;
    high: number;
    low: number;
    volume: number;
    timestamp: string | null;
    source: string;
  };
}

export const useMarketData = () => {
  const [marketData, setMarketData] = useState<MarketData | null>(null);
  const [isMarketOpen, setIsMarketOpen] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(true);

  const fetchMarketData = async () => {
    try {
      console.log('Fetching market data from backend...');
      const response = await axios.get('http://localhost:8000/api/v1/market-data', {
        timeout: 15000 // 15 second timeout for market data
      });
      
      console.log('Market data response:', response.data);
      
      if (response.data.success && response.data.data) {
        setMarketData(response.data.data);
        setError(null);
      } else {
        const errorMsg = response.data.error || 'Failed to fetch market data';
        setError(errorMsg);
        console.error('Market data fetch failed:', errorMsg);
        setMarketData(null);
      }
    } catch (err: any) {
      const errorMsg = err.response?.data?.error || err.message || 'Network error - is backend running?';
      setError(errorMsg);
      console.error('Market data fetch error:', err);
      setMarketData(null);
    } finally {
      setLoading(false);
    }
  };

  const fetchMarketStatus = async () => {
    try {
      const response = await axios.get('http://localhost:8000/api/v1/market-status', {
        timeout: 5000
      });
      if (response.data && typeof response.data.is_open === 'boolean') {
        setIsMarketOpen(response.data.is_open);
      }
    } catch (err) {
      console.error('Error fetching market status:', err);
    }
  };

  useEffect(() => {
    fetchMarketData();
    fetchMarketStatus();

    // Poll every 30 seconds for market data
    const dataInterval = setInterval(fetchMarketData, 30000);
    const statusInterval = setInterval(fetchMarketStatus, 60000); // Check market status every minute

    return () => {
      clearInterval(dataInterval);
      clearInterval(statusInterval);
    };
  }, []);

  return { marketData, isMarketOpen, error, loading, refetch: fetchMarketData };
};
