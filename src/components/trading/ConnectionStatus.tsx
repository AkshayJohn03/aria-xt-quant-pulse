
import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Wifi, WifiOff, Database, Brain, Shield, TrendingUp } from 'lucide-react';

const ConnectionStatus = () => {
  const [connections, setConnections] = useState({
    zerodhaAPI: { status: 'connected', lastUpdate: new Date() },
    ariaLSTM: { status: 'connected', lastUpdate: new Date() },
    xgboost: { status: 'connected', lastUpdate: new Date() },
    prophet: { status: 'connected', lastUpdate: new Date() },
    finbert: { status: 'connected', lastUpdate: new Date() },
    gemini: { status: 'connected', lastUpdate: new Date() },
    fallbackFilters: { status: 'connected', lastUpdate: new Date() },
    telegramBot: { status: 'connected', lastUpdate: new Date() }
  });

  const [systemLogs, setSystemLogs] = useState([
    { time: '14:32:15', event: 'Trade executed: NIFTY 20000 CE', type: 'success' },
    { time: '14:31:42', event: 'AI signal generated: BUY', type: 'info' },
    { time: '14:30:28', event: 'Data update received', type: 'info' },
    { time: '14:29:55', event: 'Trailing stop activated', type: 'warning' },
    { time: '14:28:33', event: 'Portfolio rebalanced', type: 'success' }
  ]);

  useEffect(() => {
    // Simulate periodic status updates
    const interval = setInterval(() => {
      setConnections(prev => {
        const updated = { ...prev };
        Object.keys(updated).forEach(key => {
          // Randomly update status (mostly connected)
          updated[key] = {
            status: Math.random() > 0.05 ? 'connected' : 'disconnected',
            lastUpdate: new Date()
          };
        });
        return updated;
      });

      // Add new log entry
      const events = [
        'Data refresh completed',
        'Model prediction updated',
        'Risk assessment complete',
        'Market scan finished',
        'Signal validation passed'
      ];
      
      setSystemLogs(prev => [
        {
          time: new Date().toLocaleTimeString(),
          event: events[Math.floor(Math.random() * events.length)],
          type: 'info'
        },
        ...prev.slice(0, 4)
      ]);
    }, 10000);

    return () => clearInterval(interval);
  }, []);

  const getStatusIcon = (status) => {
    return status === 'connected' ? (
      <Wifi className="h-4 w-4 text-green-400" />
    ) : (
      <WifiOff className="h-4 w-4 text-red-400" />
    );
  };

  const getStatusBadge = (status) => {
    return (
      <Badge variant={status === 'connected' ? 'default' : 'destructive'} className="text-xs">
        {status === 'connected' ? 'ONLINE' : 'OFFLINE'}
      </Badge>
    );
  };

  const connectionItems = [
    { key: 'zerodhaAPI', label: 'Zerodha API', icon: Database },
    { key: 'ariaLSTM', label: 'Aria-LSTM', icon: Brain },
    { key: 'xgboost', label: 'XGBoost', icon: TrendingUp },
    { key: 'prophet', label: 'Prophet', icon: TrendingUp },
    { key: 'finbert', label: 'FinBERT', icon: Brain },
    { key: 'gemini', label: 'Gemini AI', icon: Brain },
    { key: 'fallbackFilters', label: 'Fallback Filters', icon: Shield },
    { key: 'telegramBot', label: 'Telegram Bot', icon: Wifi }
  ];

  return (
    <div className="space-y-4">
      {/* Connection Status */}
      <Card className="bg-slate-800/50 border-slate-700">
        <CardHeader>
          <CardTitle className="text-white text-lg">System Status</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          {connectionItems.map(({ key, label, icon: Icon }) => (
            <div key={key} className="flex items-center justify-between p-2 bg-slate-700/30 rounded">
              <div className="flex items-center space-x-2">
                <Icon className="h-4 w-4 text-slate-400" />
                <span className="text-sm text-slate-300">{label}</span>
              </div>
              <div className="flex items-center space-x-2">
                {getStatusIcon(connections[key].status)}
                {getStatusBadge(connections[key].status)}
              </div>
            </div>
          ))}
        </CardContent>
      </Card>

      {/* Last Update */}
      <Card className="bg-slate-800/50 border-slate-700">
        <CardHeader>
          <CardTitle className="text-white text-lg">Last Updates</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-2">
            <div className="text-sm">
              <span className="text-slate-400">Data Feed:</span>
              <span className="text-white ml-2">{connections.zerodhaAPI.lastUpdate.toLocaleTimeString()}</span>
            </div>
            <div className="text-sm">
              <span className="text-slate-400">AI Models:</span>
              <span className="text-white ml-2">{connections.ariaLSTM.lastUpdate.toLocaleTimeString()}</span>
            </div>
            <div className="text-sm">
              <span className="text-slate-400">System Health:</span>
              <span className="text-green-400 ml-2">Optimal</span>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* System Logs */}
      <Card className="bg-slate-800/50 border-slate-700">
        <CardHeader>
          <CardTitle className="text-white text-lg">System Logs</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-2 max-h-64 overflow-y-auto">
            {systemLogs.map((log, index) => (
              <div key={index} className="flex items-start space-x-2 text-xs">
                <span className="text-slate-500 font-mono">{log.time}</span>
                <span className={`${
                  log.type === 'success' ? 'text-green-400' :
                  log.type === 'warning' ? 'text-yellow-400' :
                  'text-slate-300'
                }`}>
                  {log.event}
                </span>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* System Performance */}
      <Card className="bg-slate-800/50 border-slate-700">
        <CardHeader>
          <CardTitle className="text-white text-lg">Performance</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            <div className="flex justify-between">
              <span className="text-sm text-slate-400">CPU Usage</span>
              <span className="text-sm text-white">23%</span>
            </div>
            <div className="w-full bg-slate-700 rounded-full h-2">
              <div className="bg-green-400 h-2 rounded-full" style={{ width: '23%' }}></div>
            </div>
            
            <div className="flex justify-between">
              <span className="text-sm text-slate-400">Memory</span>
              <span className="text-sm text-white">67%</span>
            </div>
            <div className="w-full bg-slate-700 rounded-full h-2">
              <div className="bg-yellow-400 h-2 rounded-full" style={{ width: '67%' }}></div>
            </div>

            <div className="flex justify-between">
              <span className="text-sm text-slate-400">Latency</span>
              <span className="text-sm text-green-400">12ms</span>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default ConnectionStatus;
