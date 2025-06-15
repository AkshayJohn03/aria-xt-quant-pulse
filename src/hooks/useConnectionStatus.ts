
import { useState, useEffect } from 'react';
import axios from 'axios';

interface ConnectionStatus {
  zerodha: boolean;
  telegram: boolean;
  market_data: boolean;
  portfolio: boolean;
  options: boolean;
  risk_metrics: boolean;
  models: boolean;
  ollama: boolean;
  gemini: boolean;
  timestamp: string;
}

export const useConnectionStatus = () => {
  const [status, setStatus] = useState<ConnectionStatus | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const fetchStatus = async () => {
    try {
      console.log('Fetching connection status...');
      const response = await axios.get('http://localhost:8000/api/v1/connection-status', {
        timeout: 10000 // 10 second timeout
      });
      
      console.log('Connection status response:', response.data);
      
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
      const errorMsg = err.response?.data?.error || err.message || 'Network error - is backend running?';
      setError(errorMsg);
      console.error('Connection status fetch error:', err);
      setStatus(null);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchStatus();
    // Poll every 30 seconds instead of 5 seconds
    const interval = setInterval(fetchStatus, 30000);
    return () => clearInterval(interval);
  }, []);

  return { status, loading, error, refetch: fetchStatus };
};
