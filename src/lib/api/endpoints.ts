interface ApiResponse<T> {
  success: boolean;
  data: T | null;
  error: string | null;
}

const API_BASE_URL = 'http://localhost:8000/api/v1'; // Ensure this matches your backend URL

const apiCall = async <T>(url: string, options?: RequestInit): Promise<ApiResponse<T>> => {
  try {
    const response = await fetch(url, options);
    // Even if backend returns 500, fetch might not throw an error.
    // We should check response.ok or response.status here.
    if (!response.ok) {
      let errorDetail = `HTTP error! status: ${response.status}`;
      try {
        const errorData = await response.json();
        errorDetail = errorData.error || errorData.detail || errorDetail;
      } catch (jsonError) {
        // If response is not JSON, use default error message
      }
      return { success: false, data: null, error: errorDetail };
    }
    
    // Attempt to parse the response as JSON.
    // The backend is now designed to always return { success, data, error }.
    const result: ApiResponse<T> = await response.json();
    return result; // Backend should provide success, data, error directly
    
  } catch (error: any) {
    console.error(`API call to ${url} failed:`, error);
    return { success: false, data: null, error: error.message || 'Network error' };
  }
};

export const ariaAPI = {
  // Config Endpoints
  getConfig: async (): Promise<ApiResponse<any>> => 
    apiCall(`${API_BASE_URL}/config`),

  updateConfig: async (updates: Record<string, any>): Promise<ApiResponse<any>> =>
    apiCall(`${API_BASE_URL}/config`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(updates),
    }),

  validateConfig: async (): Promise<ApiResponse<{ valid: boolean }>> =>
    apiCall(`${API_BASE_URL}/config/validate`),

  // Market Data Endpoints
  getMarketData: async (): Promise<ApiResponse<any>> =>
    apiCall(`${API_BASE_URL}/market-data`),

  getOhlcvData: async (symbol: string, timeframe: string = '1min', limit: number = 100): Promise<ApiResponse<any>> =>
    apiCall(`${API_BASE_URL}/ohlcv/${symbol}?timeframe=${timeframe}&limit=${limit}`),

  getOptionChain: async (symbol: string = 'NIFTY', expiry?: string): Promise<ApiResponse<any>> => {
    const url = expiry ? `${API_BASE_URL}/option-chain?symbol=${symbol}&expiry=${expiry}` : `${API_BASE_URL}/option-chain?symbol=${symbol}`;
    return apiCall(url);
  },

  // Model & Prediction Endpoints
  generatePrediction: async (symbol: string, timeframe: string = '1min'): Promise<ApiResponse<any>> =>
    apiCall(`${API_BASE_URL}/prediction`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ symbol, timeframe }),
    }),

  generateTradingSignal: async (symbol: string = 'NIFTY'): Promise<ApiResponse<any>> =>
    apiCall(`${API_BASE_URL}/trading-signal`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ symbol }),
    }),

  // Portfolio & Risk Management Endpoints
  getPortfolio: async (): Promise<ApiResponse<any>> =>
    apiCall(`${API_BASE_URL}/portfolio`),

  getPositions: async (status?: string): Promise<ApiResponse<any>> => {
    const url = status ? `${API_BASE_URL}/positions?status=${status}` : `${API_BASE_URL}/positions`;
    return apiCall(url);
  },

  getRiskMetrics: async (): Promise<ApiResponse<any>> =>
    apiCall(`${API_BASE_URL}/risk-metrics`),

  // Backtesting Endpoints
  runBacktest: async (startDate: string, endDate: string, symbol: string = 'NIFTY', strategy: string = 'aria-lstm'): Promise<ApiResponse<any>> =>
    apiCall(`${API_BASE_URL}/backtest`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ start_date: startDate, end_date: endDate, symbol, strategy }),
    }),

  runLiveBacktest: async (symbol: string = 'NIFTY', hours: number = 1, strategy: string = 'aria-lstm'): Promise<ApiResponse<any>> =>
    apiCall(`${API_BASE_URL}/backtest/live`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ symbol, hours, strategy }),
    }),

  // Connection Status Endpoint
  getConnectionStatus: async (): Promise<ApiResponse<any>> =>
    apiCall(`${API_BASE_URL}/connection-status`),
};