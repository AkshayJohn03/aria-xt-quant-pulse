import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Shield, AlertTriangle, Target, TrendingDown, Settings } from 'lucide-react';
import { usePortfolio } from '@/hooks/useAriaAPI';

const RiskManagement = () => {
  const { portfolio, loading, error } = usePortfolio();
  const riskMetrics = portfolio?.risk_metrics;

  if (loading) {
    return <div>Loading risk metrics...</div>;
  }
  if (error || !riskMetrics) {
    return <div>Error loading risk metrics: {error || 'No data'}</div>;
  }

  const [activeRisks] = useState([
    {
      id: 1,
      type: 'Position Size',
      level: 'Medium',
      description: 'NIFTY exposure at 45% of portfolio',
      action: 'Consider reducing position size'
    },
    {
      id: 2,
      type: 'Correlation Risk',
      level: 'Low',
      description: 'Bank Nifty and Nifty correlation at 0.82',
      action: 'Monitor correlation levels'
    },
    {
      id: 3,
      type: 'Time Decay',
      level: 'High',
      description: 'Options expiring in 2 days with high theta',
      action: 'Close or roll positions'
    }
  ]);

  const [stopLossOrders] = useState([
    {
      symbol: 'NIFTY 20000 CE',
      currentPrice: 138.75,
      stopLoss: 125.00,
      trailing: true,
      triggerPrice: 130.00
    },
    {
      symbol: 'BANKNIFTY 45500 PE',
      currentPrice: 76.30,
      stopLoss: 95.00,
      trailing: false,
      triggerPrice: null
    }
  ]);

  const getRiskLevelColor = (level) => {
    switch (level) {
      case 'Low': return 'bg-green-500';
      case 'Medium': return 'bg-yellow-500';
      case 'High': return 'bg-red-500';
      default: return 'bg-gray-500';
    }
  };

  const getRiskLevelBadge = (level) => {
    const variants = {
      'Low': 'default',
      'Medium': 'secondary',
      'High': 'destructive'
    };
    return <Badge variant={variants[level] || 'default'}>{level}</Badge>;
  };

  return (
    <div className="space-y-6">
      {/* Risk Overview */}
      <Card className="bg-slate-800/50 border-slate-700">
        <CardHeader>
          <CardTitle className="text-white flex items-center space-x-2">
            <Shield className="h-5 w-5 text-green-400" />
            <span>Risk Management Dashboard</span>
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="text-center p-3 bg-slate-700/50 rounded-lg">
              <div className="text-sm text-slate-400">Portfolio Risk</div>
              <div className="text-xl font-bold text-yellow-400">{riskMetrics.risk_score}</div>
              <div className="text-xs text-slate-500">Score: {riskMetrics.sharpe_ratio?.toFixed(2) ?? 'N/A'}</div>
            </div>
            <div className="text-center p-3 bg-slate-700/50 rounded-lg">
              <div className="text-sm text-slate-400">Max Drawdown</div>
              <div className="text-xl font-bold text-red-400">{riskMetrics.max_drawdown?.toFixed(2) ?? 'N/A'}%</div>
              <div className="text-xs text-slate-500">Historical</div>
            </div>
            <div className="text-center p-3 bg-slate-700/50 rounded-lg">
              <div className="text-sm text-slate-400">Current DD</div>
              <div className="text-xl font-bold text-orange-400">{riskMetrics.current_drawdown?.toFixed(2) ?? 'N/A'}%</div>
              <div className="text-xs text-slate-500">Live</div>
            </div>
            <div className="text-center p-3 bg-slate-700/50 rounded-lg">
              <div className="text-sm text-slate-400">Sharpe Ratio</div>
              <div className="text-xl font-bold text-green-400">{riskMetrics.sharpe_ratio?.toFixed(2) ?? 'N/A'}</div>
              <div className="text-xs text-slate-500">Risk-adjusted</div>
            </div>
          </div>

          {/* Exposure Meter */}
          <div className="space-y-2">
            <div className="flex justify-between text-sm">
              <span className="text-slate-400">Portfolio Exposure</span>
              <span className="text-white">{riskMetrics.portfolio_exposure_percent?.toFixed(2) ?? 'N/A'}%</span>
            </div>
            <Progress 
              value={riskMetrics.portfolio_exposure_percent ?? 0} 
              max={100}
              className="h-2"
            />
          </div>

          {/* Risk per Trade - removed, not available in backend risk metrics */}

        </CardContent>
      </Card>

      {/* Active Risk Alerts */}
      <Card className="bg-slate-800/50 border-slate-700">
        <CardHeader>
          <CardTitle className="text-white flex items-center space-x-2">
            <AlertTriangle className="h-5 w-5 text-yellow-400" />
            <span>Active Risk Alerts</span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {activeRisks.map((risk) => (
              <div key={risk.id} className="p-3 bg-slate-700/50 rounded-lg border border-slate-600">
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center space-x-2">
                    <div className={`w-3 h-3 rounded-full ${getRiskLevelColor(risk.level)}`}></div>
                    <h4 className="font-semibold text-white">{risk.type}</h4>
                  </div>
                  {getRiskLevelBadge(risk.level)}
                </div>
                <p className="text-sm text-slate-300 mb-2">{risk.description}</p>
                <div className="flex items-center justify-between">
                  <span className="text-xs text-blue-400">{risk.action}</span>
                  <Button size="sm" variant="outline">
                    Action
                  </Button>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Stop Loss Management */}
      <Card className="bg-slate-800/50 border-slate-700">
        <CardHeader>
          <CardTitle className="text-white flex items-center space-x-2">
            <TrendingDown className="h-5 w-5 text-red-400" />
            <span>Stop Loss Orders</span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {stopLossOrders.map((order, index) => (
              <div key={index} className="p-3 bg-slate-700/50 rounded-lg border border-slate-600">
                <div className="flex items-center justify-between mb-2">
                  <h4 className="font-semibold text-white">{order.symbol}</h4>
                  <Badge variant={order.trailing ? 'default' : 'secondary'}>
                    {order.trailing ? 'Trailing SL' : 'Fixed SL'}
                  </Badge>
                </div>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                  <div>
                    <span className="text-slate-400">Current:</span>
                    <span className="text-white ml-1">₹{order.currentPrice}</span>
                  </div>
                  <div>
                    <span className="text-slate-400">Stop Loss:</span>
                    <span className="text-red-400 ml-1">₹{order.stopLoss}</span>
                  </div>
                  <div>
                    <span className="text-slate-400">Trigger:</span>
                    <span className="text-yellow-400 ml-1">
                      {order.triggerPrice ? `₹${order.triggerPrice}` : 'N/A'}
                    </span>
                  </div>
                  <div className="flex justify-end">
                    <Button size="sm" variant="outline">
                      <Settings className="h-3 w-3 mr-1" />
                      Modify
                    </Button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Delta Neutral Position */}
      <Card className="bg-slate-800/50 border-slate-700">
        <CardHeader>
          <CardTitle className="text-white flex items-center space-x-2">
            <Target className="h-5 w-5 text-blue-400" />
            <span>Delta Neutral Analysis</span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="text-center p-4 bg-slate-700/50 rounded-lg">
              <div className="text-sm text-slate-400 mb-1">Portfolio Delta</div>
              <div className="text-2xl font-bold text-blue-400">+0.15</div>
              <div className="text-xs text-slate-500">Slightly bullish</div>
            </div>
            <div className="text-center p-4 bg-slate-700/50 rounded-lg">
              <div className="text-sm text-slate-400 mb-1">Gamma Exposure</div>
              <div className="text-2xl font-bold text-green-400">+0.08</div>
              <div className="text-xs text-slate-500">Positive gamma</div>
            </div>
            <div className="text-center p-4 bg-slate-700/50 rounded-lg">
              <div className="text-sm text-slate-400 mb-1">Theta Decay</div>
              <div className="text-2xl font-bold text-red-400">-₹280</div>
              <div className="text-xs text-slate-500">Daily decay</div>
            </div>
          </div>
          
          <div className="mt-4 p-3 bg-blue-500/20 border border-blue-500/30 rounded-lg">
            <div className="flex items-center space-x-2 mb-2">
              <Target className="h-4 w-4 text-blue-400" />
              <span className="text-blue-400 font-semibold">Delta Neutral Recommendation</span>
            </div>
            <p className="text-sm text-slate-300">
              Portfolio is 85% delta neutral. Consider selling 2 NIFTY 20050 CE contracts to achieve perfect neutrality.
            </p>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default RiskManagement;
