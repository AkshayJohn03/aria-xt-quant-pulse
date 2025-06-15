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
  timestamp: string;
}

export const useConnectionStatus = () => {
  const [status, setStatus] = useState<ConnectionStatus | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchStatus = async () => {
      try {
        const response = await axios.get('/api/connection-status');
        if (response.data.success) {
          setStatus(response.data.data);
          setError(null);
        } else {
          setStatus(null);
          setError(response.data.error || 'Failed to fetch connection status');
        }
      } catch (err) {
        setStatus(null);
        setError('Failed to fetch connection status');
        console.error('Error fetching connection status:', err);
      }
    };

    fetchStatus();
    const interval = setInterval(fetchStatus, 30000); // Refresh every 30 seconds

    return () => clearInterval(interval);
  }, []);

  return { status, error };
}; 