
import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Badge } from '@/components/ui/badge';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { TrendingUp, TrendingDown, Target, Search } from 'lucide-react';

const OptionsChain = () => {
  const [selectedSymbol, setSelectedSymbol] = useState('NIFTY');
  const [selectedExpiry, setSelectedExpiry] = useState('2024-06-27');
  const [searchStrike, setSearchStrike] = useState('');
  
  const [optionsData, setOptionsData] = useState([
    {
      strike: 19800,
      call: {
        ltp: 145.50,
        change: 12.30,
        volume: 125000,
        oi: 450000,
        iv: 18.5,
        delta: 0.65,
        gamma: 0.05,
        theta: -2.1,
        vega: 15.2
      },
      put: {
        ltp: 38.75,
        change: -5.20,
        volume: 89000,
        oi: 380000,
        iv: 16.8,
        delta: -0.35,
        gamma: 0.05,
        theta: -1.8,
        vega: 14.8
      }
    },
    {
      strike: 19850,
      call: {
        ltp: 118.25,
        change: 8.90,
        volume: 198000,
        oi: 675000,
        iv: 17.2,
        delta: 0.58,
        gamma: 0.06,
        theta: -2.3,
        vega: 16.1
      },
      put: {
        ltp: 52.10,
        change: -3.40,
        volume: 156000,
        oi: 520000,
        iv: 17.5,
        delta: -0.42,
        gamma: 0.06,
        theta: -2.0,
        vega: 15.9
      }
    },
    {
      strike: 19900,
      call: {
        ltp: 95.80,
        change: 6.50,
        volume: 234000,
        oi: 890000,
        iv: 16.9,
        delta: 0.52,
        gamma: 0.07,
        theta: -2.5,
        vega: 17.0
      },
      put: {
        ltp: 68.45,
        change: -2.15,
        volume: 187000,
        oi: 745000,
        iv: 18.1,
        delta: -0.48,
        gamma: 0.07,
        theta: -2.2,
        vega: 16.8
      }
    },
    {
      strike: 19950,
      call: {
        ltp: 76.25,
        change: 4.80,
        volume: 156000,
        oi: 560000,
        iv: 17.8,
        delta: 0.45,
        gamma: 0.08,
        theta: -2.7,
        vega: 18.2
      },
      put: {
        ltp: 87.90,
        change: -1.20,
        volume: 143000,
        oi: 610000,
        iv: 19.2,
        delta: -0.55,
        gamma: 0.08,
        theta: -2.4,
        vega: 17.5
      }
    },
    {
      strike: 20000,
      call: {
        ltp: 59.15,
        change: 3.25,
        volume: 287000,
        oi: 1200000,
        iv: 18.5,
        delta: 0.38,
        gamma: 0.09,
        theta: -2.9,
        vega: 19.1
      },
      put: {
        ltp: 110.30,
        change: 0.85,
        volume: 201000,
        oi: 980000,
        iv: 20.1,
        delta: -0.62,
        gamma: 0.09,
        theta: -2.6,
        vega: 18.7
      }
    }
  ]);

  const [aiRecommendations, setAiRecommendations] = useState([
    {
      type: 'CALL',
      strike: 19900,
      reasoning: 'High volume, optimal delta, low premium',
      confidence: 92,
      targetPrice: 125.00,
      riskReward: '1:3.2'
    },
    {
      type: 'PUT',
      strike: 20000,
      reasoning: 'Strong support level, high OI',
      confidence: 87,
      targetPrice: 135.50,
      riskReward: '1:2.8'
    }
  ]);

  const getChangeColor = (change) => {
    return change >= 0 ? 'text-green-400' : 'text-red-400';
  };

  const formatNumber = (num) => {
    if (num >= 100000) {
      return (num / 100000).toFixed(1) + 'L';
    }
    if (num >= 1000) {
      return (num / 1000).toFixed(1) + 'K';
    }
    return num.toString();
  };

  return (
    <div className="space-y-6">
      {/* Options Chain Controls */}
      <Card className="bg-slate-800/50 border-slate-700">
        <CardHeader>
          <CardTitle className="text-white flex items-center space-x-2">
            <Target className="h-5 w-5" />
            <span>Options Chain Analysis</span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div className="space-y-2">
              <label className="text-sm text-slate-300">Symbol</label>
              <Select value={selectedSymbol} onValueChange={setSelectedSymbol}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="NIFTY">NIFTY</SelectItem>
                  <SelectItem value="BANKNIFTY">BANKNIFTY</SelectItem>
                  <SelectItem value="FINNIFTY">FINNIFTY</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2">
              <label className="text-sm text-slate-300">Expiry</label>
              <Select value={selectedExpiry} onValueChange={setSelectedExpiry}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="2024-06-27">27 Jun 2024</SelectItem>
                  <SelectItem value="2024-07-04">04 Jul 2024</SelectItem>
                  <SelectItem value="2024-07-11">11 Jul 2024</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2">
              <label className="text-sm text-slate-300">Search Strike</label>
              <Input
                placeholder="e.g., 20000"
                value={searchStrike}
                onChange={(e) => setSearchStrike(e.target.value)}
                className="bg-slate-700 border-slate-600"
              />
            </div>

            <div className="space-y-2">
              <label className="text-sm text-slate-300">Action</label>
              <Button className="w-full bg-blue-600 hover:bg-blue-700">
                <Search className="h-4 w-4 mr-2" />
                Refresh Chain
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* AI Recommendations */}
      <Card className="bg-slate-800/50 border-slate-700">
        <CardHeader>
          <CardTitle className="text-white">AI Recommendations</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {aiRecommendations.map((rec, index) => (
              <div key={index} className="p-4 bg-slate-700/50 rounded-lg border border-slate-600">
                <div className="flex items-center justify-between mb-2">
                  <Badge variant={rec.type === 'CALL' ? 'default' : 'secondary'}>
                    {rec.strike} {rec.type}
                  </Badge>
                  <div className="text-right">
                    <div className="text-sm text-green-400">{rec.confidence}% confidence</div>
                    <div className="text-xs text-slate-400">{rec.riskReward} R:R</div>
                  </div>
                </div>
                <p className="text-sm text-slate-300 mb-2">{rec.reasoning}</p>
                <div className="text-sm text-blue-400">Target: ₹{rec.targetPrice}</div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Options Chain Table */}
      <Card className="bg-slate-800/50 border-slate-700">
        <CardHeader>
          <CardTitle className="text-white">Options Chain - {selectedSymbol} ({selectedExpiry})</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-slate-600">
                  <th className="text-left py-2 px-2 text-slate-300">LTP</th>
                  <th className="text-left py-2 px-2 text-slate-300">Change</th>
                  <th className="text-left py-2 px-2 text-slate-300">Volume</th>
                  <th className="text-left py-2 px-2 text-slate-300">OI</th>
                  <th className="text-left py-2 px-2 text-slate-300">IV</th>
                  <th className="text-center py-2 px-4 text-white font-bold">STRIKE</th>
                  <th className="text-right py-2 px-2 text-slate-300">IV</th>
                  <th className="text-right py-2 px-2 text-slate-300">OI</th>
                  <th className="text-right py-2 px-2 text-slate-300">Volume</th>
                  <th className="text-right py-2 px-2 text-slate-300">Change</th>
                  <th className="text-right py-2 px-2 text-slate-300">LTP</th>
                </tr>
                <tr className="text-xs text-slate-400">
                  <th colSpan="5" className="text-center py-1 bg-green-500/20">CALLS</th>
                  <th className="py-1"></th>
                  <th colSpan="5" className="text-center py-1 bg-red-500/20">PUTS</th>
                </tr>
              </thead>
              <tbody>
                {optionsData.map((option, index) => (
                  <tr key={index} className="border-b border-slate-700 hover:bg-slate-700/30">
                    {/* CALL Side */}
                    <td className="py-2 px-2 text-white font-semibold">{option.call.ltp}</td>
                    <td className={`py-2 px-2 ${getChangeColor(option.call.change)}`}>
                      {option.call.change >= 0 ? '+' : ''}{option.call.change}
                    </td>
                    <td className="py-2 px-2 text-slate-300">{formatNumber(option.call.volume)}</td>
                    <td className="py-2 px-2 text-slate-300">{formatNumber(option.call.oi)}</td>
                    <td className="py-2 px-2 text-slate-300">{option.call.iv}%</td>
                    
                    {/* STRIKE */}
                    <td className="py-2 px-4 text-center text-white font-bold bg-slate-700/50">
                      {option.strike}
                    </td>
                    
                    {/* PUT Side */}
                    <td className="py-2 px-2 text-slate-300 text-right">{option.put.iv}%</td>
                    <td className="py-2 px-2 text-slate-300 text-right">{formatNumber(option.put.oi)}</td>
                    <td className="py-2 px-2 text-slate-300 text-right">{formatNumber(option.put.volume)}</td>
                    <td className={`py-2 px-2 text-right ${getChangeColor(option.put.change)}`}>
                      {option.put.change >= 0 ? '+' : ''}{option.put.change}
                    </td>
                    <td className="py-2 px-2 text-white font-semibold text-right">{option.put.ltp}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>

      {/* Greeks Analysis */}
      <Card className="bg-slate-800/50 border-slate-700">
        <CardHeader>
          <CardTitle className="text-white">Greeks Analysis</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <h4 className="text-white font-semibold mb-3">Call Options Greeks</h4>
              <div className="space-y-2">
                {optionsData.map((option, index) => (
                  <div key={index} className="flex justify-between items-center p-2 bg-slate-700/30 rounded">
                    <span className="text-slate-300">{option.strike}</span>
                    <div className="flex space-x-4 text-sm">
                      <span className="text-blue-400">Δ {option.call.delta}</span>
                      <span className="text-green-400">Γ {option.call.gamma}</span>
                      <span className="text-red-400">Θ {option.call.theta}</span>
                      <span className="text-purple-400">ν {option.call.vega}</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
            
            <div>
              <h4 className="text-white font-semibold mb-3">Put Options Greeks</h4>
              <div className="space-y-2">
                {optionsData.map((option, index) => (
                  <div key={index} className="flex justify-between items-center p-2 bg-slate-700/30 rounded">
                    <span className="text-slate-300">{option.strike}</span>
                    <div className="flex space-x-4 text-sm">
                      <span className="text-blue-400">Δ {option.put.delta}</span>
                      <span className="text-green-400">Γ {option.put.gamma}</span>
                      <span className="text-red-400">Θ {option.put.theta}</span>
                      <span className="text-purple-400">ν {option.put.vega}</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default OptionsChain;
