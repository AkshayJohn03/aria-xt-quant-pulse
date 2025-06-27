
import { useState, useEffect } from 'react';
import axios from 'axios';

interface MarketDataPoint {
  value: number;
  change: number;
  percentChange: number;
  high: number;
  low: number;
  volume: number;
  timestamp: string | null;
  source: string;
}

interface MarketData {
  nifty: MarketDataPoint;
  banknifty: MarketDataPoint;
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
        timeout: 10000
      });
      
      console.log('Market data response:', response.data);
      
      if (response.data.success && response.data.data) {
        const data = response.data.data;
        
        // Transform backend response to expected format
        const transformedData: MarketData = {
          nifty: data.nifty || {
            value: 24000,
            change: 0,
            percentChange: 0,
            high: 24000,
            low: 24000,
            volume: 0,
            timestamp: new Date().toISOString(),
            source: 'fallback'
          },
          banknifty: data.banknifty || {
            value: 51000,
            change: 0,
            percentChange: 0,
            high: 51000,
            low: 51000,
            volume: 0,
            timestamp: new Date().toISOString(),
            source: 'fallback'
          }
        };
        
        setMarketData(transformedData);
        setError(null);
        console.log('Market data updated:', transformedData);
      } else {
        const errorMsg = response.data.error || 'Failed to fetch market data';
        console.error('Market data fetch failed:', errorMsg);
        setError(`Backend: ${errorMsg}`);
      }
    } catch (err: any) {
      const errorMsg = err.code === 'ERR_NETWORK' 
        ? 'Backend server not running on http://localhost:8000. Please start the backend server.'
        : err.response?.data?.error || err.message || 'Failed to fetch market data';
      
      console.error('Market data fetch error:', err);
      setError(errorMsg);
    } finally {
      setLoading(false);
    }
  };

  const fetchMarketStatus = async () => {
    try {
      const response = await axios.get('http://localhost:8000/api/v1/market-status', {
        timeout: 5000
      });
      if (response.data.success && typeof response.data.data?.is_open === 'boolean') {
        setIsMarketOpen(response.data.data.is_open);
      }
    } catch (err) {
      console.error('Error fetching market status:', err);
      // Set market status based on IST time (9:15 AM to 3:30 PM on weekdays)
      const now = new Date();
      const istTime = new Date(now.getTime() + (5.5 * 60 * 60 * 1000));
      const day = istTime.getDay();
      const hour = istTime.getHours();
      const minute = istTime.getMinutes();
      const totalMinutes = hour * 60 + minute;
      
      const isWeekday = day >= 1 && day <= 5;
      const isMarketHours = totalMinutes >= 555 && totalMinutes <= 930; // 9:15 AM to 3:30 PM
      
      setIsMarketOpen(isWeekday && isMarketHours);
    }
  };

  useEffect(() => {
    fetchMarketData();
    fetchMarketStatus();

    // Poll every 30 seconds for market data
    const dataInterval = setInterval(fetchMarketData, 30000);
    const statusInterval = setInterval(fetchMarketStatus, 60000);

    return () => {
      clearInterval(dataInterval);
      clearInterval(statusInterval);
    };
  }, []);

  return { marketData, isMarketOpen, error, loading, refetch: fetchMarketData };
};
