import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { usePortfolio } from '@/hooks/useAriaAPI'; // Assuming usePortfolio is in this hook

// Define interfaces for portfolio data structure
interface Position {
  symbol: string;
  quantity: number;
  avg_price: number;
  current_price: number;
  pnl: number;
  product_type: string;
  timestamp: string;
  status: string;
}

interface RiskMetrics {
  portfolio_value: number;
  total_investment: number;
  total_pnl: number;
  risk_score: string;
  max_drawdown: number;
  current_drawdown: number;
  sharpe_ratio: number;
  sortino_ratio: number;
  portfolio_exposure_percent: number;
  max_risk_per_trade_percent: number;
}

interface PortfolioData {
  positions: Position[];
  risk_metrics: RiskMetrics;
  total_pnl: number; // This might be redundant if in risk_metrics, but keeping for now based on your previous structure
  open_positions_count: number;
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

  // Ensure portfolio data exists before accessing its properties
  const currentPortfolio: PortfolioData = portfolio || { 
    positions: [], 
    risk_metrics: {} as RiskMetrics, // Cast to empty RiskMetrics for default access
    total_pnl: 0, 
    open_positions_count: 0 
  };

  const totalInvestment = currentPortfolio.risk_metrics?.total_investment ?? 0;
  const currentValue = currentPortfolio.risk_metrics?.portfolio_value ?? 0;
  const totalPnl = currentPortfolio.risk_metrics?.total_pnl ?? 0;
  const riskScore = currentPortfolio.risk_metrics?.risk_score ?? "N/A";
  const pnlPercent = totalInvestment !== 0 ? (totalPnl / totalInvestment * 100) : 0;

  return (
    <div className="space-y-6">
      {/* Portfolio Summary */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card className="bg-slate-800/50 border-slate-700 text-white">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium text-slate-300">Total Investment</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">₹{totalInvestment.toFixed(2)}</div>
          </CardContent>
        </Card>
        <Card className="bg-slate-800/50 border-slate-700 text-white">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium text-slate-300">Current Value</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">₹{currentValue.toFixed(2)}</div>
          </CardContent>
        </Card>
        <Card className="bg-slate-800/50 border-slate-700 text-white">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium text-slate-300">Total P&L</CardTitle>
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
            <CardTitle className="text-sm font-medium text-slate-300">Risk Score</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-orange-400">{riskScore}</div>
            <p className="text-xs text-slate-400">Delta Neutral 35%</p> {/* This might be dynamic later */}
          </CardContent>
        </Card>
      </div>

      {/* Active Positions */}
      <Card className="bg-slate-800/50 border-slate-700 text-white">
        <CardHeader>
          <CardTitle className="text-xl font-semibold text-slate-200">Active Positions ({currentPortfolio.open_positions_count})</CardTitle>
        </CardHeader>
        <CardContent>
          {currentPortfolio.positions.length > 0 ? (
            <Table>
              <TableHeader>
                <TableRow className="border-slate-700">
                  <TableHead className="text-slate-400">Symbol</TableHead>
                  <TableHead className="text-slate-400">Quantity</TableHead>
                  <TableHead className="text-slate-400">Avg. Price</TableHead>
                  <TableHead className="text-slate-400">LTP</TableHead>
                  <TableHead className="text-slate-400">P&L</TableHead>
                  <TableHead className="text-slate-400 text-right">Actions</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {currentPortfolio.positions.map((position) => (
                  <TableRow key={position.symbol} className="border-slate-700">
                    <TableCell className="font-medium text-white">
                      {position.symbol} <span className="text-xs text-slate-400">({position.product_type})</span>
                    </TableCell>
                    <TableCell>
                      {position.quantity > 0 ? (
                        <span className="text-green-400">{position.quantity} BUY</span>
                      ) : (
                        <span className="text-red-400">{Math.abs(position.quantity)} SELL</span>
                      )}
                    </TableCell>
                    <TableCell>{position.avg_price.toFixed(2)}</TableCell>
                    <TableCell>{position.current_price.toFixed(2)}</TableCell>
                    <TableCell className={`${position.pnl >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                      ₹{position.pnl.toFixed(2)} ({position.pnl >= 0 ? '+' : ''}
                      {(position.pnl / (position.avg_price * Math.abs(position.quantity)) * 100).toFixed(2)}%)
                    </TableCell>
                    <TableCell className="text-right">
                      <button className="px-3 py-1 bg-red-600 hover:bg-red-700 text-white rounded-md text-sm">
                        Exit
                      </button>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          ) : (
            <p className="text-slate-400 text-center py-4">No active positions.</p>
          )}
        </CardContent>
      </Card>

      {/* Placeholder for Trade History or other portfolio related info */}
      {/* <Card className="bg-slate-800/50 border-slate-700 text-white">
        <CardHeader>
          <CardTitle className="text-xl font-semibold text-slate-200">Trade History</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-slate-400">Trade history will appear here.</p>
        </CardContent>
      </Card> */}
    </div>
  );
};

export default Portfolio;