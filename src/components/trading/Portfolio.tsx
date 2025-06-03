
import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { ArrowUp, ArrowDown, TrendingUp, DollarSign, Target, Shield } from 'lucide-react';

const Portfolio = () => {
  const [positions] = useState([
    {
      id: 1,
      symbol: 'NIFTY 20000 CE',
      type: 'CALL',
      quantity: 50,
      avgPrice: 125.50,
      ltp: 138.75,
      pnl: 662.50,
      pnlPercent: 10.55,
      expiry: '2024-06-27'
    },
    {
      id: 2,
      symbol: 'BANKNIFTY 45500 PE',
      type: 'PUT',
      quantity: 25,
      avgPrice: 89.20,
      ltp: 76.30,
      pnl: -322.50,
      pnlPercent: -14.46,
      expiry: '2024-06-27'
    },
    {
      id: 3,
      symbol: 'RELIANCE 2800 CE',
      type: 'CALL',
      quantity: 100,
      avgPrice: 45.80,
      ltp: 52.15,
      pnl: 635.00,
      pnlPercent: 13.86,
      expiry: '2024-06-27'
    }
  ]);

  const totalInvestment = 26050;
  const currentValue = 27025;
  const totalPnL = currentValue - totalInvestment;
  const totalPnLPercent = (totalPnL / totalInvestment) * 100;

  return (
    <div className="space-y-6">
      {/* Portfolio Summary */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card className="bg-slate-800/50 border-slate-700">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium text-slate-300">Total Investment</CardTitle>
            <DollarSign className="h-4 w-4 text-blue-400" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-white">₹{totalInvestment.toLocaleString()}</div>
          </CardContent>
        </Card>

        <Card className="bg-slate-800/50 border-slate-700">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium text-slate-300">Current Value</CardTitle>
            <Target className="h-4 w-4 text-green-400" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-white">₹{currentValue.toLocaleString()}</div>
          </CardContent>
        </Card>

        <Card className="bg-slate-800/50 border-slate-700">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium text-slate-300">Total P&L</CardTitle>
            {totalPnL >= 0 ? <ArrowUp className="h-4 w-4 text-green-400" /> : <ArrowDown className="h-4 w-4 text-red-400" />}
          </CardHeader>
          <CardContent>
            <div className={`text-2xl font-bold ${totalPnL >= 0 ? 'text-green-400' : 'text-red-400'}`}>
              ₹{totalPnL.toFixed(2)}
            </div>
            <p className={`text-xs ${totalPnL >= 0 ? 'text-green-400' : 'text-red-400'}`}>
              {totalPnLPercent >= 0 ? '+' : ''}{totalPnLPercent.toFixed(2)}%
            </p>
          </CardContent>
        </Card>

        <Card className="bg-slate-800/50 border-slate-700">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium text-slate-300">Risk Score</CardTitle>
            <Shield className="h-4 w-4 text-yellow-400" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-yellow-400">Medium</div>
            <p className="text-xs text-slate-400">Delta Neutral: 85%</p>
          </CardContent>
        </Card>
      </div>

      {/* Active Positions */}
      <Card className="bg-slate-800/50 border-slate-700">
        <CardHeader>
          <CardTitle className="text-white">Active Positions</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {positions.map((position) => (
              <div key={position.id} className="flex items-center justify-between p-4 bg-slate-700/50 rounded-lg">
                <div className="flex-1">
                  <div className="flex items-center space-x-2">
                    <h3 className="font-semibold text-white">{position.symbol}</h3>
                    <Badge variant={position.type === 'CALL' ? 'default' : 'secondary'} className="text-xs">
                      {position.type}
                    </Badge>
                  </div>
                  <p className="text-sm text-slate-400">
                    Qty: {position.quantity} | Avg: ₹{position.avgPrice} | Exp: {position.expiry}
                  </p>
                </div>
                <div className="text-right">
                  <div className="text-lg font-semibold text-white">₹{position.ltp}</div>
                  <div className={`text-sm ${position.pnl >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                    {position.pnl >= 0 ? '+' : ''}₹{position.pnl} ({position.pnlPercent >= 0 ? '+' : ''}{position.pnlPercent}%)
                  </div>
                </div>
                <div className="ml-4">
                  <Button size="sm" variant="outline" className="mr-2">
                    Exit
                  </Button>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default Portfolio;
