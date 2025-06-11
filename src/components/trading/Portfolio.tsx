
import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { ArrowUp, ArrowDown, TrendingUp, DollarSign, Target, Shield, AlertCircle } from 'lucide-react';
import { usePortfolio } from '@/hooks/useAriaAPI';

const Portfolio = () => {
  const { portfolio, loading, error, refetch } = usePortfolio();

  if (loading) {
    return (
      <div className="space-y-6">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          {[...Array(4)].map((_, i) => (
            <Card key={i} className="bg-slate-800/50 border-slate-700">
              <CardHeader className="animate-pulse">
                <div className="h-4 bg-slate-600 rounded w-1/2"></div>
              </CardHeader>
              <CardContent className="animate-pulse">
                <div className="h-8 bg-slate-600 rounded w-3/4"></div>
              </CardContent>
            </Card>
          ))}
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="space-y-6">
        <Card className="bg-red-900/20 border-red-700">
          <CardHeader className="flex flex-row items-center space-y-0 pb-2">
            <AlertCircle className="h-4 w-4 text-red-400 mr-2" />
            <CardTitle className="text-red-400">Portfolio Error</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-red-300">{error}</p>
            <Button onClick={refetch} className="mt-4" variant="outline">
              Retry
            </Button>
          </CardContent>
        </Card>
      </div>
    );
  }

  // Use real data or fallback to defaults
  const positions = portfolio?.positions || [];
  const metrics = portfolio?.metrics || {};
  const funds = portfolio?.funds || {};

  const totalInvestment = metrics.total_value || 0;
  const totalPnL = metrics.total_pnl || 0;
  const totalPnLPercent = totalInvestment > 0 ? (totalPnL / totalInvestment) * 100 : 0;
  const availableFunds = funds.available_cash || 0;

  return (
    <div className="space-y-6">
      {/* Portfolio Summary */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card className="bg-slate-800/50 border-slate-700">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium text-slate-300">Available Funds</CardTitle>
            <DollarSign className="h-4 w-4 text-blue-400" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-white">₹{availableFunds.toLocaleString()}</div>
          </CardContent>
        </Card>

        <Card className="bg-slate-800/50 border-slate-700">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium text-slate-300">Portfolio Value</CardTitle>
            <Target className="h-4 w-4 text-green-400" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-white">₹{totalInvestment.toLocaleString()}</div>
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
            <CardTitle className="text-sm font-medium text-slate-300">Active Positions</CardTitle>
            <Shield className="h-4 w-4 text-yellow-400" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-yellow-400">{positions.length}</div>
            <p className="text-xs text-slate-400">Open Trades</p>
          </CardContent>
        </Card>
      </div>

      {/* Active Positions */}
      <Card className="bg-slate-800/50 border-slate-700">
        <CardHeader className="flex flex-row items-center justify-between">
          <CardTitle className="text-white">Active Positions</CardTitle>
          <Button onClick={refetch} variant="outline" size="sm">
            Refresh
          </Button>
        </CardHeader>
        <CardContent>
          {positions.length === 0 ? (
            <div className="text-center py-8">
              <p className="text-slate-400">No active positions</p>
            </div>
          ) : (
            <div className="space-y-4">
              {positions.map((position, index) => (
                <div key={index} className="flex items-center justify-between p-4 bg-slate-700/50 rounded-lg">
                  <div className="flex-1">
                    <div className="flex items-center space-x-2">
                      <h3 className="font-semibold text-white">{position.symbol}</h3>
                      <Badge variant={position.quantity > 0 ? 'default' : 'secondary'} className="text-xs">
                        {position.quantity > 0 ? 'LONG' : 'SHORT'}
                      </Badge>
                    </div>
                    <p className="text-sm text-slate-400">
                      Qty: {Math.abs(position.quantity)} | Avg: ₹{position.avg_price?.toFixed(2) || 'N/A'} | Product: {position.product || 'N/A'}
                    </p>
                  </div>
                  <div className="text-right">
                    <div className="text-lg font-semibold text-white">₹{position.current_price?.toFixed(2) || 'N/A'}</div>
                    <div className={`text-sm ${(position.pnl || 0) >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                      {(position.pnl || 0) >= 0 ? '+' : ''}₹{(position.pnl || 0).toFixed(2)}
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
          )}
        </CardContent>
      </Card>
    </div>
  );
};

export default Portfolio;
