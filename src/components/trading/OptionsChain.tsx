import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { ArrowUp, ArrowDown, Filter, Download, RefreshCw } from 'lucide-react';

interface OptionData {
  strike: number;
  expiry: string;
  call: {
    ltp: number;
    volume: number;
    oi: number;
    iv: number;
    delta: number;
    gamma: number;
    theta: number;
    vega: number;
    change: number;
    changePct: number;
  };
  put: {
    ltp: number;
    volume: number;
    oi: number;
    iv: number;
    delta: number;
    gamma: number;
    theta: number;
    vega: number;
    change: number;
    changePct: number;
  };
}

const OptionsChain = () => {
  const [optionData, setOptionData] = useState<OptionData[]>([]);
  const [loading, setLoading] = useState(true);
  const [spotPrice] = useState(19850.25);
  const [filterStrike, setFilterStrike] = useState('');
  const [selectedExpiry, setSelectedExpiry] = useState('2024-06-27');

  // Mock data for demonstration
  useEffect(() => {
    const mockData: OptionData[] = [
      {
        strike: 19700,
        expiry: '2024-06-27',
        call: {
          ltp: 180.50,
          volume: 125000,
          oi: 450000,
          iv: 18.5,
          delta: 0.75,
          gamma: 0.08,
          theta: -2.5,
          vega: 12.3,
          change: 15.25,
          changePct: 9.22
        },
        put: {
          ltp: 35.75,
          volume: 85000,
          oi: 320000,
          iv: 16.8,
          delta: -0.25,
          gamma: 0.08,
          theta: -1.8,
          vega: 11.5,
          change: -8.50,
          changePct: -19.20
        }
      },
      {
        strike: 19750,
        expiry: '2024-06-27',
        call: {
          ltp: 145.20,
          volume: 140000,
          oi: 480000,
          iv: 17.8,
          delta: 0.65,
          gamma: 0.07,
          theta: -2.2,
          vega: 11.8,
          change: 12.50,
          changePct: 8.90
        },
        put: {
          ltp: 42.30,
          volume: 90000,
          oi: 330000,
          iv: 16.2,
          delta: -0.30,
          gamma: 0.07,
          theta: -1.6,
          vega: 11.0,
          change: -7.80,
          changePct: -17.50
        }
      },
      {
        strike: 19800,
        expiry: '2024-06-27',
        call: {
          ltp: 112.80,
          volume: 155000,
          oi: 510000,
          iv: 17.2,
          delta: 0.55,
          gamma: 0.06,
          theta: -1.9,
          vega: 11.2,
          change: 9.80,
          changePct: 8.65
        },
        put: {
          ltp: 51.90,
          volume: 95000,
          oi: 340000,
          iv: 15.7,
          delta: -0.35,
          gamma: 0.06,
          theta: -1.4,
          vega: 10.5,
          change: -6.20,
          changePct: -15.20
        }
      },
      {
        strike: 19850,
        expiry: '2024-06-27',
        call: {
          ltp: 85.40,
          volume: 170000,
          oi: 540000,
          iv: 16.7,
          delta: 0.45,
          gamma: 0.05,
          theta: -1.6,
          vega: 10.7,
          change: 7.40,
          changePct: 8.00
        },
        put: {
          ltp: 63.50,
          volume: 100000,
          oi: 350000,
          iv: 15.2,
          delta: -0.40,
          gamma: 0.05,
          theta: -1.2,
          vega: 10.0,
          change: -5.50,
          changePct: -13.80
        }
      },
      {
        strike: 19900,
        expiry: '2024-06-27',
        call: {
          ltp: 62.10,
          volume: 185000,
          oi: 570000,
          iv: 16.2,
          delta: 0.35,
          gamma: 0.04,
          theta: -1.3,
          vega: 10.2,
          change: 5.10,
          changePct: 7.50
        },
        put: {
          ltp: 78.10,
          volume: 105000,
          oi: 360000,
          iv: 14.7,
          delta: -0.45,
          gamma: 0.04,
          theta: -1.0,
          vega: 9.5,
          change: -4.80,
          changePct: -12.50
        }
      }
    ];

    setTimeout(() => {
      setOptionData(mockData);
      setLoading(false);
    }, 1000);
  }, []);

  const getChangeColor = (change: number): string => {
    return change >= 0 ? 'text-green-400' : 'text-red-400';
  };

  const formatNumber = (num: number): string => {
    if (num >= 100000) {
      return (num / 100000).toFixed(1) + 'L';
    }
    if (num >= 1000) {
      return (num / 1000).toFixed(1) + 'K';
    }
    return num.toFixed(2);
  };

  const filteredData = optionData.filter(option => 
    filterStrike === '' || option.strike.toString().includes(filterStrike)
  );

  const getMoneyness = (strike: number): string => {
    const diff = Math.abs(strike - spotPrice);
    if (diff <= 50) return 'ATM';
    return strike < spotPrice ? 'ITM' : 'OTM';
  };

  if (loading) {
    return (
      <Card className="bg-slate-900 border-slate-700">
        <CardContent className="p-6">
          <div className="flex items-center justify-center h-64">
            <RefreshCw className="h-8 w-8 animate-spin text-blue-400" />
            <span className="ml-2 text-slate-300">Loading option chain...</span>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className="bg-slate-900 border-slate-700">
      <CardHeader className="pb-4">
        <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4">
          <div>
            <CardTitle className="text-xl text-white mb-2">NIFTY50 Options Chain</CardTitle>
            <div className="flex items-center gap-4 text-sm text-slate-300">
              <span>Spot: ₹{spotPrice.toLocaleString()}</span>
              <Badge variant="outline" className="border-green-500 text-green-400">
                Live
              </Badge>
            </div>
          </div>
          
          <div className="flex flex-col sm:flex-row gap-2">
            <div className="flex items-center gap-2">
              <Filter className="h-4 w-4 text-slate-400" />
              <Input
                placeholder="Filter by strike"
                value={filterStrike}
                onChange={(e) => setFilterStrike(e.target.value)}
                className="w-32 bg-slate-800 border-slate-600 text-white"
              />
            </div>
            <Button size="sm" variant="outline" className="border-slate-600 text-slate-300">
              <Download className="h-4 w-4 mr-1" />
              Export
            </Button>
          </div>
        </div>
      </CardHeader>

      <CardContent>
        <Tabs defaultValue="chain" className="w-full">
          <TabsList className="grid w-full grid-cols-3 bg-slate-800">
            <TabsTrigger value="chain" className="data-[state=active]:bg-slate-700">Chain</TabsTrigger>
            <TabsTrigger value="greeks" className="data-[state=active]:bg-slate-700">Greeks</TabsTrigger>
            <TabsTrigger value="analysis" className="data-[state=active]:bg-slate-700">Analysis</TabsTrigger>
          </TabsList>

          <TabsContent value="chain" className="mt-4">
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-slate-700">
                    <th className="text-left p-2 text-slate-300">Calls</th>
                    <th className="text-center p-2 text-slate-300">Strike</th>
                    <th className="text-right p-2 text-slate-300">Puts</th>
                  </tr>
                </thead>
                <tbody>
                  {filteredData.map((option) => (
                    <tr key={option.strike} className="border-b border-slate-800 hover:bg-slate-800/50">
                      {/* Call Side */}
                      <td className="p-2">
                        <div className="grid grid-cols-4 gap-2 text-xs">
                          <div>
                            <div className="text-white font-medium">₹{option.call.ltp}</div>
                            <div className={getChangeColor(option.call.change)}>
                              {option.call.change > 0 ? '+' : ''}{option.call.change.toFixed(2)}
                            </div>
                          </div>
                          <div>
                            <div className="text-slate-300">{formatNumber(option.call.volume)}</div>
                            <div className="text-slate-500">Vol</div>
                          </div>
                          <div>
                            <div className="text-slate-300">{formatNumber(option.call.oi)}</div>
                            <div className="text-slate-500">OI</div>
                          </div>
                          <div>
                            <div className="text-slate-300">{option.call.iv.toFixed(1)}%</div>
                            <div className="text-slate-500">IV</div>
                          </div>
                        </div>
                      </td>

                      {/* Strike Price */}
                      <td className="p-2 text-center">
                        <div className="font-medium text-white">{option.strike}</div>
                        <Badge 
                          variant="outline" 
                          className={`text-xs mt-1 ${
                            getMoneyness(option.strike) === 'ATM' ? 'border-yellow-500 text-yellow-400' :
                            getMoneyness(option.strike) === 'ITM' ? 'border-green-500 text-green-400' :
                            'border-slate-500 text-slate-400'
                          }`}
                        >
                          {getMoneyness(option.strike)}
                        </Badge>
                      </td>

                      {/* Put Side */}
                      <td className="p-2">
                        <div className="grid grid-cols-4 gap-2 text-xs">
                          <div>
                            <div className="text-white font-medium">₹{option.put.ltp}</div>
                            <div className={getChangeColor(option.put.change)}>
                              {option.put.change > 0 ? '+' : ''}{option.put.change.toFixed(2)}
                            </div>
                          </div>
                          <div>
                            <div className="text-slate-300">{formatNumber(option.put.volume)}</div>
                            <div className="text-slate-500">Vol</div>
                          </div>
                          <div>
                            <div className="text-slate-300">{formatNumber(option.put.oi)}</div>
                            <div className="text-slate-500">OI</div>
                          </div>
                          <div>
                            <div className="text-slate-300">{option.put.iv.toFixed(1)}%</div>
                            <div className="text-slate-500">IV</div>
                          </div>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </TabsContent>

          <TabsContent value="greeks" className="mt-4">
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-slate-700">
                    <th className="text-left p-2 text-slate-300">Strike</th>
                    <th className="text-center p-2 text-slate-300">Call Delta</th>
                    <th className="text-center p-2 text-slate-300">Call Gamma</th>
                    <th className="text-center p-2 text-slate-300">Call Theta</th>
                    <th className="text-center p-2 text-slate-300">Put Delta</th>
                    <th className="text-center p-2 text-slate-300">Put Gamma</th>
                    <th className="text-center p-2 text-slate-300">Put Theta</th>
                  </tr>
                </thead>
                <tbody>
                  {filteredData.map((option) => (
                    <tr key={option.strike} className="border-b border-slate-800 hover:bg-slate-800/50">
                      <td className="p-2 font-medium text-white">{option.strike}</td>
                      <td className="p-2 text-center text-slate-300">{option.call.delta.toFixed(3)}</td>
                      <td className="p-2 text-center text-slate-300">{option.call.gamma.toFixed(3)}</td>
                      <td className="p-2 text-center text-red-400">{option.call.theta.toFixed(2)}</td>
                      <td className="p-2 text-center text-slate-300">{option.put.delta.toFixed(3)}</td>
                      <td className="p-2 text-center text-slate-300">{option.put.gamma.toFixed(3)}</td>
                      <td className="p-2 text-center text-red-400">{option.put.theta.toFixed(2)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </TabsContent>

          <TabsContent value="analysis" className="mt-4">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              <Card className="bg-slate-800 border-slate-700">
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm text-slate-300">Put-Call Ratio</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold text-white">0.87</div>
                  <div className="text-xs text-slate-400">Bullish sentiment</div>
                </CardContent>
              </Card>

              <Card className="bg-slate-800 border-slate-700">
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm text-slate-300">Max Pain</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold text-white">19,800</div>
                  <div className="text-xs text-slate-400">Expected expiry level</div>
                </CardContent>
              </Card>

              <Card className="bg-slate-800 border-slate-700">
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm text-slate-300">Volatility</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold text-white">17.2%</div>
                  <div className="text-xs text-green-400">↓ 2.1% from yesterday</div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
};

export default OptionsChain;
