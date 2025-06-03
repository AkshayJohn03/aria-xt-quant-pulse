
import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Badge } from '@/components/ui/badge';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar } from 'recharts';
import { PlayCircle, StopCircle, TrendingUp, Activity, Target, DollarSign } from 'lucide-react';

const BacktestingModule = () => {
  const [isRunning, setIsRunning] = useState(false);
  const [testType, setTestType] = useState('historical');
  const [strategy, setStrategy] = useState('aria-lstm');
  const [startDate, setStartDate] = useState('2024-01-01');
  const [endDate, setEndDate] = useState('2024-06-03');
  const [results, setResults] = useState(null);

  // Sample backtest results data
  const backtestData = [
    { date: '2024-01-01', portfolio: 100000, benchmark: 100000 },
    { date: '2024-01-15', portfolio: 102500, benchmark: 101200 },
    { date: '2024-02-01', portfolio: 98750, benchmark: 99800 },
    { date: '2024-02-15', portfolio: 105200, benchmark: 102100 },
    { date: '2024-03-01', portfolio: 108900, benchmark: 103500 },
    { date: '2024-03-15', portfolio: 112300, benchmark: 104200 },
    { date: '2024-04-01', portfolio: 115600, benchmark: 105800 },
    { date: '2024-04-15', portfolio: 118200, benchmark: 106500 },
    { date: '2024-05-01', portfolio: 121800, benchmark: 107200 },
    { date: '2024-05-15', portfolio: 125400, benchmark: 108900 },
    { date: '2024-06-01', portfolio: 128700, benchmark: 109500 }
  ];

  const tradeStats = [
    { metric: 'Total Trades', value: 247 },
    { metric: 'Win Rate', value: '68.4%' },
    { metric: 'Avg Win', value: '₹2,340' },
    { metric: 'Avg Loss', value: '₹1,120' },
    { metric: 'Max Drawdown', value: '8.2%' },
    { metric: 'Sharpe Ratio', value: '1.85' }
  ];

  const runBacktest = () => {
    setIsRunning(true);
    // Simulate backtest execution
    setTimeout(() => {
      setResults({
        totalReturn: 28.7,
        annualizedReturn: 22.3,
        volatility: 14.2,
        sharpeRatio: 1.85,
        maxDrawdown: 8.2,
        winRate: 68.4,
        totalTrades: 247
      });
      setIsRunning(false);
    }, 3000);
  };

  const runLiveTest = () => {
    setTestType('live');
    setIsRunning(true);
    // Simulate live data backtest
    setTimeout(() => {
      setResults({
        totalReturn: 15.2,
        annualizedReturn: 18.7,
        volatility: 12.8,
        sharpeRatio: 1.92,
        maxDrawdown: 5.4,
        winRate: 72.1,
        totalTrades: 89
      });
      setIsRunning(false);
    }, 2000);
  };

  return (
    <div className="space-y-6">
      {/* Backtest Configuration */}
      <Card className="bg-slate-800/50 border-slate-700">
        <CardHeader>
          <CardTitle className="text-white flex items-center space-x-2">
            <Activity className="h-5 w-5" />
            <span>Backtesting Configuration</span>
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            <div className="space-y-2">
              <Label htmlFor="strategy" className="text-slate-300">Strategy</Label>
              <Select value={strategy} onValueChange={setStrategy}>
                <SelectTrigger>
                  <SelectValue placeholder="Select strategy" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="aria-lstm">Aria-LSTM</SelectItem>
                  <SelectItem value="xgboost">XGBoost Ensemble</SelectItem>
                  <SelectItem value="prophet">Prophet + FinBERT</SelectItem>
                  <SelectItem value="hybrid">Hybrid AI Suite</SelectItem>
                  <SelectItem value="fallback">Fallback Filters</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2">
              <Label htmlFor="start-date" className="text-slate-300">Start Date</Label>
              <Input
                id="start-date"
                type="date"
                value={startDate}
                onChange={(e) => setStartDate(e.target.value)}
                className="bg-slate-700 border-slate-600"
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="end-date" className="text-slate-300">End Date</Label>
              <Input
                id="end-date"
                type="date"
                value={endDate}
                onChange={(e) => setEndDate(e.target.value)}
                className="bg-slate-700 border-slate-600"
              />
            </div>

            <div className="space-y-2">
              <Label className="text-slate-300">Capital</Label>
              <Input
                placeholder="₹1,00,000"
                className="bg-slate-700 border-slate-600"
              />
            </div>
          </div>

          <div className="flex space-x-4">
            <Button
              onClick={runBacktest}
              disabled={isRunning}
              className="bg-blue-600 hover:bg-blue-700"
            >
              {isRunning && testType === 'historical' ? (
                <StopCircle className="h-4 w-4 mr-2" />
              ) : (
                <PlayCircle className="h-4 w-4 mr-2" />
              )}
              {isRunning && testType === 'historical' ? 'Running...' : 'Run Historical Backtest'}
            </Button>

            <Button
              onClick={runLiveTest}
              disabled={isRunning}
              variant="outline"
              className="border-green-600 text-green-400 hover:bg-green-600/20"
            >
              {isRunning && testType === 'live' ? (
                <StopCircle className="h-4 w-4 mr-2" />
              ) : (
                <PlayCircle className="h-4 w-4 mr-2" />
              )}
              {isRunning && testType === 'live' ? 'Running...' : 'Run Live Data Test'}
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Results */}
      {results && (
        <>
          {/* Performance Metrics */}
          <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
            <Card className="bg-slate-800/50 border-slate-700">
              <CardContent className="p-4">
                <div className="text-sm text-slate-400">Total Return</div>
                <div className="text-xl font-bold text-green-400">+{results.totalReturn}%</div>
              </CardContent>
            </Card>
            <Card className="bg-slate-800/50 border-slate-700">
              <CardContent className="p-4">
                <div className="text-sm text-slate-400">Annual Return</div>
                <div className="text-xl font-bold text-green-400">+{results.annualizedReturn}%</div>
              </CardContent>
            </Card>
            <Card className="bg-slate-800/50 border-slate-700">
              <CardContent className="p-4">
                <div className="text-sm text-slate-400">Volatility</div>
                <div className="text-xl font-bold text-yellow-400">{results.volatility}%</div>
              </CardContent>
            </Card>
            <Card className="bg-slate-800/50 border-slate-700">
              <CardContent className="p-4">
                <div className="text-sm text-slate-400">Sharpe Ratio</div>
                <div className="text-xl font-bold text-blue-400">{results.sharpeRatio}</div>
              </CardContent>
            </Card>
            <Card className="bg-slate-800/50 border-slate-700">
              <CardContent className="p-4">
                <div className="text-sm text-slate-400">Max Drawdown</div>
                <div className="text-xl font-bold text-red-400">-{results.maxDrawdown}%</div>
              </CardContent>
            </Card>
            <Card className="bg-slate-800/50 border-slate-700">
              <CardContent className="p-4">
                <div className="text-sm text-slate-400">Win Rate</div>
                <div className="text-xl font-bold text-green-400">{results.winRate}%</div>
              </CardContent>
            </Card>
          </div>

          {/* Performance Chart */}
          <Card className="bg-slate-800/50 border-slate-700">
            <CardHeader>
              <CardTitle className="text-white">Portfolio Performance vs Benchmark</CardTitle>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={400}>
                <LineChart data={backtestData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis dataKey="date" stroke="#9CA3AF" />
                  <YAxis stroke="#9CA3AF" />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: '#1F2937',
                      border: '1px solid #374151',
                      borderRadius: '8px'
                    }}
                  />
                  <Line
                    type="monotone"
                    dataKey="portfolio"
                    stroke="#10B981"
                    strokeWidth={2}
                    name="Strategy"
                  />
                  <Line
                    type="monotone"
                    dataKey="benchmark"
                    stroke="#6B7280"
                    strokeWidth={2}
                    name="Benchmark"
                  />
                </LineChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>

          {/* Trade Statistics */}
          <Card className="bg-slate-800/50 border-slate-700">
            <CardHeader>
              <CardTitle className="text-white">Trade Statistics</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
                {tradeStats.map((stat, index) => (
                  <div key={index} className="text-center p-4 bg-slate-700/50 rounded-lg">
                    <div className="text-sm text-slate-400 mb-1">{stat.metric}</div>
                    <div className="text-lg font-bold text-white">{stat.value}</div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </>
      )}
    </div>
  );
};

export default BacktestingModule;
