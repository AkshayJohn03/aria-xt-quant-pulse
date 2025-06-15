
import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { 
  Wifi, 
  WifiOff, 
  Database, 
  Bot, 
  MessageSquare, 
  TrendingUp,
  AlertCircle,
  CheckCircle,
  RefreshCw,
  Brain
} from 'lucide-react';
import { useConnectionStatus } from '@/hooks/useConnectionStatus';

const ConnectionStatus = () => {
  const { status, loading, error, refetch } = useConnectionStatus();

  if (loading) {
    return (
      <Card className="bg-slate-800/50 border-slate-700">
        <CardHeader>
          <CardTitle className="text-white flex items-center">
            <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
            Connection Status
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {[...Array(6)].map((_, i) => (
              <div key={i} className="flex items-center justify-between animate-pulse">
                <div className="h-4 bg-slate-600 rounded w-1/2"></div>
                <div className="h-6 bg-slate-600 rounded w-16"></div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    );
  }

  if (error) {
    return (
      <Card className="bg-red-900/20 border-red-700">
        <CardHeader>
          <CardTitle className="text-white flex items-center">
            <AlertCircle className="h-4 w-4 mr-2 text-red-400" />
            Connection Status
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-red-400 text-sm mb-4">{error}</div>
          <Button onClick={refetch} size="sm" variant="outline" className="w-full">
            <RefreshCw className="h-4 w-4 mr-2" />
            Retry Connection
          </Button>
        </CardContent>
      </Card>
    );
  }

  const getStatusBadge = (isConnected: boolean) => {
    return (
      <Badge 
        variant="outline" 
        className={isConnected ? 'border-green-400 text-green-400' : 'border-red-400 text-red-400'}
      >
        {isConnected ? 'Connected' : 'Disconnected'}
      </Badge>
    );
  };

  return (
    <Card className="bg-slate-800/50 border-slate-700">
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-3">
        <CardTitle className="text-white flex items-center">
          <Wifi className="h-4 w-4 mr-2" />
          System Status
        </CardTitle>
        <Button onClick={refetch} size="sm" variant="ghost">
          <RefreshCw className="h-4 w-4" />
        </Button>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* API Connections */}
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <TrendingUp className="h-4 w-4 text-blue-400" />
              <span className="text-sm text-slate-300">Zerodha</span>
            </div>
            {getStatusBadge(status?.zerodha || false)}
          </div>

          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <Database className="h-4 w-4 text-purple-400" />
              <span className="text-sm text-slate-300">Market Data</span>
            </div>
            {getStatusBadge(status?.market_data || false)}
          </div>

          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <Database className="h-4 w-4 text-blue-400" />
              <span className="text-sm text-slate-300">Twelve Data</span>
            </div>
            {getStatusBadge(status?.twelve_data || false)}
          </div>

          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <Bot className="h-4 w-4 text-green-400" />
              <span className="text-sm text-slate-300">Gemini AI</span>
            </div>
            {getStatusBadge(status?.gemini || false)}
          </div>

          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <Brain className="h-4 w-4 text-orange-400" />
              <span className="text-sm text-slate-300">Ollama</span>
            </div>
            {getStatusBadge(status?.ollama || false)}
          </div>

          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <MessageSquare className="h-4 w-4 text-cyan-400" />
              <span className="text-sm text-slate-300">Telegram</span>
            </div>
            {getStatusBadge(status?.telegram || false)}
          </div>
        </div>

        {/* Last Update */}
        {status?.timestamp && (
          <div className="text-xs text-slate-500 text-center pt-2 border-t border-slate-700">
            Last update: {new Date(status.timestamp).toLocaleString('en-IN')}
          </div>
        )}
      </CardContent>
    </Card>
  );
};

export default ConnectionStatus;
