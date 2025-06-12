import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { usePortfolio } from '@/hooks/useAriaAPI';
import { Badge } from '@/components/ui/badge';
import { TrendingUp, TrendingDown, AlertCircle, DollarSign, Wallet, BarChart3 } from 'lucide-react';

interface RiskMetrics {
  total_investment?: number;
  portfolio_value?: number;
  total_pnl?: number;
  day_pnl?: number;
  available_balance?: number;
  risk_score?: string;
  portfolio_exposure_percent?: number;
  max_drawdown?: number;
  current_drawdown?: number;
  sharpe_ratio?: number;
  sortino_ratio?: number;
  max_risk_per_trade_percent?: number;
  sector_exposure?: {
    [sector: string]: {
      exposure: number;
      risk_score: string;
    };
  };
}

interface Position {
  symbol: string;
  quantity: number;
  avg_price: number;
  current_price: number;
  pnl: number;
  day_pnl: number;
  product_type: string;
  sector?: string;
  last_trade_time?: string;
  unrealized_pnl?: number;
  realized_pnl?: number;
  timestamp?: string;
  status?: string;
}

const Portfolio: React.FC = () => {
  const { portfolio, loading, error } = usePortfolio();

  if (loading) {
    return (
      <Card className="bg-slate-800/50 border-slate-700 p-6 text-center text-white">
        <CardTitle className="text-xl">Loading Portfolio...</CardTitle>
        <CardContent className="mt-2 text-slate-400">Fetching your current positions and risk metrics.</CardContent>
      </Card>
    );
  }

  if (error) {
    return (
      <Card className="bg-red-900/50 border-red-700 p-6 text-center text-white">
        <CardTitle className="text-xl">Error Loading Portfolio</CardTitle>
        <CardContent className="mt-2 text-red-300">Error: {error}</CardContent>
        <CardContent className="mt-2 text-red-300">Please ensure backend is running and connected to broker.</CardContent>
      </Card>
    );
  }

  if (!portfolio) {
    return (
      <Card className="bg-slate-800/50 border-slate-700 p-6 text-center text-white">
        <CardTitle className="text-xl">No Portfolio Data Available</CardTitle>
        <CardContent className="mt-2 text-slate-400">Waiting for portfolio data from backend...</CardContent>
      </Card>
    );
  }

  // Safely access nested data with defaults and proper typing
  const riskMetrics: RiskMetrics = portfolio.risk_metrics || {};
  const totalInvestment = riskMetrics.total_investment || 0;
  const currentValue = riskMetrics.portfolio_value || 0;
  const totalPnl = riskMetrics.total_pnl || 0;
  const dayPnl = riskMetrics.day_pnl || 0;
  const availableBalance = riskMetrics.available_balance || 0;
  const riskScore = riskMetrics.risk_score || "N/A";
  const pnlPercent = totalInvestment !== 0 ? (totalPnl / totalInvestment * 100) : 0;
  const dayPnlPercent = currentValue !== 0 ? (dayPnl / currentValue * 100) : 0;
  const positions = portfolio.positions || [];
  const openPositionsCount = portfolio.open_positions_count || 0;
  const sectorExposure = riskMetrics.sector_exposure || {};

  return (
    <div className="space-y-6">
      {/* Portfolio Summary */}
      <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-6 gap-4">
        <Card className="bg-slate-800/50 border-slate-700 text-white">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium text-slate-300">Total Investment</CardTitle>
            <DollarSign className="h-4 w-4 text-blue-400" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">₹{totalInvestment.toFixed(2)}</div>
            <p className="text-xs text-slate-400">Invested Capital</p>
          </CardContent>
        </Card>

        <Card className="bg-slate-800/50 border-slate-700 text-white">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium text-slate-300">Current Value</CardTitle>
            <BarChart3 className="h-4 w-4 text-purple-400" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">₹{currentValue.toFixed(2)}</div>
            <p className="text-xs text-slate-400">Market Value</p>
          </CardContent>
        </Card>

        <Card className="bg-slate-800/50 border-slate-700 text-white">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium text-slate-300">Available Balance</CardTitle>
            <Wallet className="h-4 w-4 text-green-400" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">₹{availableBalance.toFixed(2)}</div>
            <p className="text-xs text-slate-400">Free Cash</p>
          </CardContent>
        </Card>

        <Card className="bg-slate-800/50 border-slate-700 text-white">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium text-slate-300">Total P&L</CardTitle>
            {totalPnl >= 0 ? (
              <TrendingUp className="h-4 w-4 text-green-400" />
            ) : (
              <TrendingDown className="h-4 w-4 text-red-400" />
            )}
          </CardHeader>
          <CardContent>
            <div className={`text-2xl font-bold ${totalPnl >= 0 ? 'text-green-400' : 'text-red-400'}`}>
              ₹{totalPnl.toFixed(2)}
            </div>
            <p className={`text-xs ${pnlPercent >= 0 ? 'text-green-400' : 'text-red-400'}`}>
              ({pnlPercent.toFixed(2)}%)
            </p>
          </CardContent>
        </Card>

        <Card className="bg-slate-800/50 border-slate-700 text-white">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium text-slate-300">Today's P&L</CardTitle>
            {dayPnl >= 0 ? (
              <TrendingUp className="h-4 w-4 text-green-400" />
            ) : (
              <TrendingDown className="h-4 w-4 text-red-400" />
            )}
          </CardHeader>
          <CardContent>
            <div className={`text-2xl font-bold ${dayPnl >= 0 ? 'text-green-400' : 'text-red-400'}`}>
              ₹{dayPnl.toFixed(2)}
            </div>
            <p className={`text-xs ${dayPnlPercent >= 0 ? 'text-green-400' : 'text-red-400'}`}>
              ({dayPnlPercent.toFixed(2)}%)
            </p>
          </CardContent>
        </Card>

        <Card className="bg-slate-800/50 border-slate-700 text-white">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium text-slate-300">Risk Score</CardTitle>
            <AlertCircle className="h-4 w-4 text-orange-400" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-orange-400">{riskScore}</div>
            <p className="text-xs text-slate-400">Exposure: {(riskMetrics.portfolio_exposure_percent || 0).toFixed(1)}%</p>
          </CardContent>
        </Card>
      </div>

      {/* Sector Risk Analysis */}
      <Card className="bg-slate-800/50 border-slate-700 text-white">
        <CardHeader>
          <CardTitle className="text-xl font-semibold text-slate-200">Sector Risk Analysis</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {Object.entries(sectorExposure).map(([sector, data]) => (
              <div key={sector} className="flex items-center justify-between p-4 bg-slate-700/30 rounded-lg">
                <div>
                  <div className="font-medium text-slate-200">{sector}</div>
                  <div className="text-sm text-slate-400">Exposure: {data.exposure.toFixed(1)}%</div>
                </div>
                <Badge 
                  variant="outline" 
                  className={
                    data.risk_score === 'HIGH' ? 'border-red-400 text-red-400' :
                    data.risk_score === 'MEDIUM' ? 'border-yellow-400 text-yellow-400' :
                    'border-green-400 text-green-400'
                  }
                >
                  {data.risk_score}
                </Badge>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Active Positions */}
      <Card className="bg-slate-800/50 border-slate-700 text-white">
        <CardHeader>
          <CardTitle className="text-xl font-semibold text-slate-200">Active Positions ({openPositionsCount})</CardTitle>
        </CardHeader>
        <CardContent>
          {positions.length > 0 ? (
            <Table>
              <TableHeader>
                <TableRow className="border-slate-700">
                  <TableHead className="text-slate-400">Symbol</TableHead>
                  <TableHead className="text-slate-400">Quantity</TableHead>
                  <TableHead className="text-slate-400">Avg. Price</TableHead>
                  <TableHead className="text-slate-400">LTP</TableHead>
                  <TableHead className="text-slate-400">Day P&L</TableHead>
                  <TableHead className="text-slate-400">Total P&L</TableHead>
                  <TableHead className="text-slate-400">Sector</TableHead>
                  <TableHead className="text-slate-400 text-right">Actions</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {positions.map((position: Position, index) => {
                  const quantity = position.quantity || 0;
                  const avgPrice = position.avg_price || 0;
                  const currentPrice = position.current_price || 0;
                  const pnl = position.pnl || 0;
                  const dayPnl = position.day_pnl || 0;
                  const pnlPercent = avgPrice !== 0 ? (pnl / (avgPrice * Math.abs(quantity)) * 100) : 0;
                  const dayPnlPercent = avgPrice !== 0 ? (dayPnl / (avgPrice * Math.abs(quantity)) * 100) : 0;
                  
                  return (
                    <TableRow key={`${position.symbol}-${index}`} className="border-slate-700">
                      <TableCell className="font-medium text-white">
                        {position.symbol}
                        <div className="text-xs text-slate-400">{position.product_type || 'N/A'}</div>
                      </TableCell>
                      <TableCell>
                        {quantity > 0 ? (
                          <Badge variant="outline" className="border-green-400 text-green-400">
                            {quantity} BUY
                          </Badge>
                        ) : (
                          <Badge variant="outline" className="border-red-400 text-red-400">
                            {Math.abs(quantity)} SELL
                          </Badge>
                        )}
                      </TableCell>
                      <TableCell>₹{avgPrice.toFixed(2)}</TableCell>
                      <TableCell>₹{currentPrice.toFixed(2)}</TableCell>
                      <TableCell>
                        <div className={`${dayPnl >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                          ₹{dayPnl.toFixed(2)}
                        </div>
                        <div className={`text-xs ${dayPnl >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                          ({dayPnl >= 0 ? '+' : ''}{dayPnlPercent.toFixed(2)}%)
                        </div>
                      </TableCell>
                      <TableCell>
                        <div className={`${pnl >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                          ₹{pnl.toFixed(2)}
                        </div>
                        <div className={`text-xs ${pnl >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                          ({pnl >= 0 ? '+' : ''}{pnlPercent.toFixed(2)}%)
                        </div>
                      </TableCell>
                      <TableCell>
                        <Badge variant="outline" className="border-blue-400 text-blue-400">
                          {position.sector || 'N/A'}
                        </Badge>
                      </TableCell>
                      <TableCell className="text-right">
                        <button className="px-3 py-1 bg-red-600 hover:bg-red-700 text-white rounded-md text-sm">
                          Exit
                        </button>
                      </TableCell>
                    </TableRow>
                  );
                })}
              </TableBody>
            </Table>
          ) : (
            <div className="text-center py-8">
              <p className="text-slate-400">No active positions found.</p>
              <p className="text-slate-500 text-sm mt-2">
                {portfolio.last_update ? `Last updated: ${new Date(portfolio.last_update).toLocaleString('en-IN')}` : 'No data available'}
              </p>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
};

export default Portfolio;
