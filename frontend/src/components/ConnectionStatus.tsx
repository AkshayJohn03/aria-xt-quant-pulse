import React, { useEffect, useState } from 'react';
import { useConnectionStatus } from '../hooks/useConnectionStatus';
import { useMarketData } from '../hooks/useMarketData';
import { Badge } from './ui/badge';
import { Card } from './ui/card';
import { Clock, Wifi, WifiOff } from 'lucide-react';

export const ConnectionStatus: React.FC = () => {
  const { status, error } = useConnectionStatus();
  const { marketData, isMarketOpen, error: marketError } = useMarketData();
  const [currentTime, setCurrentTime] = useState<string>('');

  useEffect(() => {
    const timer = setInterval(() => {
      const now = new Date();
      const istTime = new Date(now.getTime() + (5.5 * 60 * 60 * 1000)); // Convert to IST
      setCurrentTime(istTime.toLocaleTimeString('en-US', { 
        hour12: false,
        timeZone: 'Asia/Kolkata'
      }));
    }, 1000);

    return () => clearInterval(timer);
  }, []);

  const getStatusColor = () => {
    if (error || marketError) return 'bg-red-500';
    if (!status) return 'bg-yellow-500';
    const allConnected = Object.entries(status).filter(([k]) => k !== 'timestamp').every(([, v]) => v === true);
    if (allConnected) return 'bg-green-500';
    return 'bg-yellow-500';
  };

  const getStatusText = () => {
    if (error || marketError) return 'Connection Error';
    if (!status) return 'No Data';
    const allConnected = Object.entries(status).filter(([k]) => k !== 'timestamp').every(([, v]) => v === true);
    if (allConnected) return 'Connected';
    return 'Partial Connection';
  };

  const moduleList = [
    { key: 'zerodha', label: 'Zerodha' },
    { key: 'telegram', label: 'Telegram' },
    { key: 'market_data', label: 'Market Data' },
    { key: 'portfolio', label: 'Portfolio' },
    { key: 'options', label: 'Options' },
    { key: 'risk_metrics', label: 'Risk Metrics' },
    { key: 'models', label: 'Models' },
  ];

  return (
    <Card className="p-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-2">
          {error || marketError ? (
            <WifiOff className="h-5 w-5 text-red-500" />
          ) : (
            <Wifi className="h-5 w-5 text-green-500" />
          )}
          <span className="font-medium">Connection Status</span>
        </div>
        <Badge className={getStatusColor()}>{getStatusText()}</Badge>
      </div>

      <div className="mt-4 space-y-2">
        {moduleList.map((mod) => (
          <div className="flex items-center justify-between" key={mod.key}>
            <span className="text-sm text-gray-500">{mod.label}</span>
            <span className="flex items-center space-x-1">
              <span className={`inline-block w-2 h-2 rounded-full ${status && status[mod.key] ? 'bg-green-500' : 'bg-red-500'}`}></span>
              <Badge variant={status && status[mod.key] ? "success" : "destructive"}>
                {status && status[mod.key] ? 'Connected' : 'Offline'}
              </Badge>
            </span>
          </div>
        ))}
        <div className="flex items-center justify-between">
          <span className="text-sm text-gray-500">Market Status</span>
          <span className="flex items-center space-x-1">
            <span className={`inline-block w-2 h-2 rounded-full ${isMarketOpen ? 'bg-green-500' : 'bg-red-500'}`}></span>
            <Badge variant={isMarketOpen ? "success" : "destructive"}>
              {isMarketOpen ? 'Open' : 'Closed'}
            </Badge>
          </span>
        </div>
        <div className="flex items-center justify-between">
          <span className="text-sm text-gray-500">IST Time</span>
          <div className="flex items-center space-x-1">
            <Clock className="h-4 w-4 text-gray-500" />
            <span className="text-sm font-mono">{currentTime}</span>
          </div>
        </div>
      </div>

      {marketData && (
        <div className="mt-4 space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-sm text-gray-500">NIFTY 50</span>
            <div className="text-right">
              <div className="font-medium">{marketData.nifty.value.toLocaleString()}</div>
              <div className={`text-sm ${marketData.nifty.percentChange >= 0 ? 'text-green-500' : 'text-red-500'}`}>
                {marketData.nifty.percentChange >= 0 ? '+' : ''}{marketData.nifty.percentChange.toFixed(2)}%
              </div>
            </div>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-sm text-gray-500">BANK NIFTY</span>
            <div className="text-right">
              <div className="font-medium">{marketData.banknifty.value.toLocaleString()}</div>
              <div className={`text-sm ${marketData.banknifty.percentChange >= 0 ? 'text-green-500' : 'text-red-500'}`}>
                {marketData.banknifty.percentChange >= 0 ? '+' : ''}{marketData.banknifty.percentChange.toFixed(2)}%
              </div>
            </div>
          </div>
        </div>
      )}

      {(error || marketError) && (
        <div className="mt-4 p-2 bg-red-50 text-red-500 text-sm rounded">
          {error || marketError}
        </div>
      )}
    </Card>
  );
}; 