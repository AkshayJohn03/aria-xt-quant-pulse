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
  twelve_data: boolean;
  timestamp: string;
}

const API_BASE_URL = 'http://localhost:8000/api/v1';


export const useConnectionStatus = () => {
  const [status, setStatus] = useState<ConnectionStatus | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const fetchStatus = async () => {
    try {
      console.log('Fetching connection status...');
      const response = await axios.get(`${API_BASE_URL}/connection-status`, {
        timeout: 5000
      });
      
      console.log('Connection status response:', response.data);
      
      if (response.data.success) {
        setStatus(response.data.data);
        setError(null);
      } else {
        const errorMsg = response.data.error || 'Failed to fetch connection status';
        setError(errorMsg);
        console.error('Connection status fetch failed:', errorMsg);
      }
    } catch (err: any) {
      const errorMsg = 'Backend server not running on http://localhost:8000';
      setError(errorMsg);
      console.error('Connection status fetch error:', err);
      
      // Set fallback status when backend is not available
      setStatus({
        zerodha: false,
        telegram: false,
        market_data: false,
        portfolio: false,
        options: false,
        risk_metrics: false,
        models: false,
        ollama: false,
        gemini: false,
        twelve_data: false,
        timestamp: new Date().toISOString()
      });
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchStatus();
    // Poll every 30 seconds
    const interval = setInterval(fetchStatus, 30000);
    return () => clearInterval(interval);
  }, []);

  return { status, loading, error, refetch: fetchStatus };
}; 