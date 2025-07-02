
import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { ArrowUp, ArrowDown, Clock, DollarSign, TrendingUp, Bot } from 'lucide-react';

const TradeLog = () => {
  const [trades] = useState([
    {
      id: 1,
      timestamp: '2024-06-03 14:32:15',
      symbol: 'NIFTY 20000 CE',
      type: 'BUY',
      quantity: 50,
      entryPrice: 125.50,
      exitPrice: 138.75,
      pnl: 662.50,
      status: 'CLOSED',
      source: 'Aria-LSTM',
      confidence: 94,
      reason: 'Strong bullish signal with high volume confirmation'
    },
    {
      id: 2,
      timestamp: '2024-06-03 13:45:22',
      symbol: 'BANKNIFTY 45500 PE',
      type: 'BUY',
      quantity: 25,
      entryPrice: 89.20,
      exitPrice: null,
      pnl: -322.50,
      status: 'OPEN',
      source: 'XGBoost',
      confidence: 87,
      reason: 'Bearish divergence detected in RSI'
    },
    {
      id: 3,
      timestamp: '2024-06-03 12:18:30',
      symbol: 'RELIANCE 2800 CE',
      type: 'BUY',
      quantity: 100,
      entryPrice: 45.80,
      exitPrice: 52.15,
      pnl: 635.00,
      status: 'CLOSED',
      source: 'Fallback Filter',
      confidence: 92,
      reason: 'Breakout above resistance with volume spike'
    },
    {
      id: 4,
      timestamp: '2024-06-03 11:22:15',
      symbol: 'NIFTY 19800 PE',
      type: 'SELL',
      quantity: 75,
      entryPrice: 156.20,
      exitPrice: 142.35,
      pnl: 1038.75,
      status: 'CLOSED',
      source: 'Prophet + FinBERT',
      confidence: 89,
      reason: 'Sentiment analysis indicates market optimism'
    },
    {
      id: 5,
      timestamp: '2024-06-03 10:45:08',
      symbol: 'HDFC 1600 CE',
      type: 'BUY',
      quantity: 200,
      entryPrice: 28.90,
      exitPrice: 32.15,
      pnl: 650.00,
      status: 'CLOSED',
      source: 'Gemini Validated',
      confidence: 96,
      reason: 'AI consensus with trailing stop activation'
    }
  ]);

  const [signals] = useState([
    {
      id: 1,
      timestamp: '2024-06-03 14:35:42',
      symbol: 'NIFTY 20100 CE',
      signal: 'BUY',
      confidence: 91,
      source: 'Aria-LSTM',
      reason: 'Bullish momentum with volume confirmation',
      executed: false
    },
    {
      id: 2,
      timestamp: '2024-06-03 14:33:18',
      symbol: 'BANKNIFTY 45000 PE',
      signal: 'SELL',
      confidence: 85,
      source: 'XGBoost',
      reason: 'Overbought conditions detected',
      executed: true
    },
    {
      id: 3,
      timestamp: '2024-06-03 14:31:55',
      symbol: 'FINNIFTY 21500 CE',
      signal: 'HOLD',
      confidence: 78,
      source: 'Prophet',
      reason: 'Consolidation phase expected',
      executed: false
    }
  ]);

  const getStatusBadge = (status) => {
    const variants = {
      'OPEN': 'default',
      'CLOSED': 'secondary'
    };
    return <Badge variant={variants[status] || 'default'}>{status}</Badge>;
  };

  const getSignalBadge = (signal) => {
    const variants = {
      'BUY': 'default',
      'SELL': 'destructive',
      'HOLD': 'secondary'
    };
    return <Badge variant={variants[signal] || 'default'}>{signal}</Badge>;
  };

  const getPnLColor = (pnl) => {
    return pnl >= 0 ? 'text-green-400' : 'text-red-400';
  };

  const getSourceIcon = (source) => {
    if (source.includes('Aria') || source.includes('LSTM')) return <Bot className="h-4 w-4 text-purple-400" />;
    if (source.includes('Fallback')) return <TrendingUp className="h-4 w-4 text-blue-400" />;
    if (source.includes('Gemini')) return <Bot className="h-4 w-4 text-green-400" />;
    return <Bot className="h-4 w-4 text-slate-400" />;
  };

  return (
    <div className="space-y-6">
      {/* Trade Summary */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card className="bg-slate-800/50 border-slate-700">
          <CardContent className="p-4">
            <div className="flex items-center space-x-2">
              <DollarSign className="h-4 w-4 text-green-400" />
              <div>
                <div className="text-sm text-slate-400">Total P&L</div>
                <div className="text-xl font-bold text-green-400">+₹2,663</div>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="bg-slate-800/50 border-slate-700">
          <CardContent className="p-4">
            <div className="flex items-center space-x-2">
              <TrendingUp className="h-4 w-4 text-blue-400" />
              <div>
                <div className="text-sm text-slate-400">Win Rate</div>
                <div className="text-xl font-bold text-blue-400">80%</div>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="bg-slate-800/50 border-slate-700">
          <CardContent className="p-4">
            <div className="flex items-center space-x-2">
              <Clock className="h-4 w-4 text-yellow-400" />
              <div>
                <div className="text-sm text-slate-400">Avg Hold</div>
                <div className="text-xl font-bold text-yellow-400">2.3h</div>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="bg-slate-800/50 border-slate-700">
          <CardContent className="p-4">
            <div className="flex items-center space-x-2">
              <Bot className="h-4 w-4 text-purple-400" />
              <div>
                <div className="text-sm text-slate-400">AI Accuracy</div>
                <div className="text-xl font-bold text-purple-400">91%</div>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Recent Signals */}
      <Card className="bg-slate-800/50 border-slate-700">
        <CardHeader>
          <CardTitle className="text-white">Recent AI Signals</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {signals.map((signal) => (
              <div key={signal.id} className="flex items-center justify-between p-3 bg-slate-700/50 rounded-lg">
                <div className="flex items-center space-x-3">
                  {getSourceIcon(signal.source)}
                  <div>
                    <div className="font-semibold text-white">{signal.symbol}</div>
                    <div className="text-sm text-slate-400">{signal.reason}</div>
                  </div>
                </div>
                <div className="flex items-center space-x-4">
                  <div className="text-right">
                    <div className="text-sm text-slate-400">{signal.confidence}% confidence</div>
                    <div className="text-xs text-slate-500">{signal.source}</div>
                  </div>
                  {getSignalBadge(signal.signal)}
                  <Button 
                    size="sm" 
                    variant={signal.executed ? "secondary" : "default"}
                    disabled={signal.executed}
                  >
                    {signal.executed ? 'Executed' : 'Execute'}
                  </Button>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Trade History */}
      <Card className="bg-slate-800/50 border-slate-700">
        <CardHeader>
          <CardTitle className="text-white">Trade History</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {trades.map((trade) => (
              <div key={trade.id} className="p-4 bg-slate-700/50 rounded-lg border border-slate-600">
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center space-x-3">
                    {trade.type === 'BUY' ? (
                      <ArrowUp className="h-4 w-4 text-green-400" />
                    ) : (
                      <ArrowDown className="h-4 w-4 text-red-400" />
                    )}
                    <div>
                      <div className="font-semibold text-white">{trade.symbol}</div>
                      <div className="text-sm text-slate-400">{trade.timestamp}</div>
                    </div>
                  </div>
                  <div className="flex items-center space-x-3">
                    {getStatusBadge(trade.status)}
                    <div className="text-right">
                      <div className={`font-semibold ${getPnLColor(trade.pnl)}`}>
                        {trade.pnl >= 0 ? '+' : ''}₹{trade.pnl}
                      </div>
                      <div className="text-sm text-slate-400">
                        {trade.confidence}% confidence
                      </div>
                    </div>
                  </div>
                </div>
                
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                  <div>
                    <span className="text-slate-400">Quantity:</span>
                    <span className="text-white ml-1">{trade.quantity}</span>
                  </div>
                  <div>
                    <span className="text-slate-400">Entry:</span>
                    <span className="text-white ml-1">₹{trade.entryPrice}</span>
                  </div>
                  <div>
                    <span className="text-slate-400">Exit:</span>
                    <span className="text-white ml-1">
                      {trade.exitPrice ? `₹${trade.exitPrice}` : 'Open'}
                    </span>
                  </div>
                  <div>
                    <span className="text-slate-400">Source:</span>
                    <span className="text-white ml-1">{trade.source}</span>
                  </div>
                </div>
                
                <div className="mt-2 text-sm text-slate-300">
                  <span className="text-slate-400">Reason:</span> {trade.reason}
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default TradeLog;
