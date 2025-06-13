
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
import { useConnectionStatus } from '@/hooks/useAriaAPI';

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

  const connections = status?.connections || {};
  const brokerConnected = status?.broker_connected || false;
  const systemStatus = status?.system_status || {};
  const modelStatus = status?.model_status || {};

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
        {/* Broker Connection */}
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <TrendingUp className="h-4 w-4 text-blue-400" />
            <span className="text-sm text-slate-300">Zerodha Broker</span>
          </div>
          {getStatusBadge(brokerConnected)}
        </div>

        {/* API Connections */}
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <Database className="h-4 w-4 text-purple-400" />
              <span className="text-sm text-slate-300">Kite API</span>
            </div>
            {getStatusBadge(connections.zerodha)}
          </div>

          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <Database className="h-4 w-4 text-blue-400" />
              <span className="text-sm text-slate-300">Market Data</span>
            </div>
            {getStatusBadge(connections.twelve_data)}
          </div>

          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <Bot className="h-4 w-4 text-green-400" />
              <span className="text-sm text-slate-300">Gemini AI</span>
            </div>
            {getStatusBadge(connections.gemini)}
          </div>

          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <Brain className="h-4 w-4 text-orange-400" />
              <span className="text-sm text-slate-300">Ollama</span>
            </div>
            {getStatusBadge(connections.ollama)}
          </div>

          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <MessageSquare className="h-4 w-4 text-cyan-400" />
              <span className="text-sm text-slate-300">Telegram</span>
            </div>
            {getStatusBadge(connections.telegram)}
          </div>
        </div>

        {/* System Summary */}
        <div className="border-t border-slate-700 pt-3 mt-4">
          <div className="flex justify-between text-xs text-slate-400 mb-2">
            <span>Trading Engine:</span>
            <span className={systemStatus.is_running ? 'text-green-400' : 'text-red-400'}>
              {systemStatus.is_running ? 'Running' : 'Stopped'}
            </span>
          </div>
          <div className="flex justify-between text-xs text-slate-400 mb-2">
            <span>Active Trades:</span>
            <span className="text-white">{systemStatus.active_trades || 0}</span>
          </div>
          <div className="flex justify-between text-xs text-slate-400">
            <span>System Health:</span>
            <span className="text-green-400">{systemStatus.system_health || 'OK'}</span>
          </div>
        </div>

        {/* Last Update */}
        {status?.last_update && (
          <div className="text-xs text-slate-500 text-center pt-2 border-t border-slate-700">
            Last update: {new Date(status.last_update).toLocaleString('en-IN')}
          </div>
        )}
      </CardContent>
    </Card>
  );
};

export default ConnectionStatus;
