import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { ArrowUp, ArrowDown, Filter, Download, RefreshCw } from 'lucide-react';
import { OptionChainData, DataFetcher } from '@/lib/api/dataFetcher';

const OptionsChain = () => {
  const [optionData, setOptionData] = useState<OptionChainData[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [spotPrice, setSpotPrice] = useState<number | null>(null);
  const [expiries, setExpiries] = useState<string[]>([]);
  const [selectedExpiry, setSelectedExpiry] = useState<string>('');
  const [filterStrike, setFilterStrike] = useState('');

  useEffect(() => {
    const fetchData = async () => {
      setLoading(true);
      setError(null);
      try {
        const fetcher = new DataFetcher();
        // Fetch option chain for selected expiry
        const url = selectedExpiry
          ? `/option-chain?symbol=NIFTY&expiry=${encodeURIComponent(selectedExpiry)}`
          : '/option-chain?symbol=NIFTY';
        const response = await fetcher.fetchOptionChain(url);
        if (response && Array.isArray(response)) {
          setOptionData(response);
          setError(null);
          // Extract expiries from data if available
          const expirySet = new Set(response.map(opt => opt.expiry));
          setExpiries(Array.from(expirySet));
          if (!selectedExpiry && expirySet.size > 0) {
            setSelectedExpiry(Array.from(expirySet)[0]);
          }
          // Set spot price if available
          if (response.length > 0 && response[0].call && typeof response[0].call.ltp === 'number') {
            setSpotPrice(response[0].call.ltp);
          }
        } else {
          setOptionData([]);
          setError('Failed to fetch option chain data from backend.');
        }
      } catch (err: any) {
        setOptionData([]);
        setError(err.message || 'Failed to fetch option chain data from backend.');
      }
      setLoading(false);
    };
    fetchData();
  }, [selectedExpiry]);

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
              <span>Spot: ₹{spotPrice?.toLocaleString()}</span>
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
            {/* Expiry Dropdown */}
            {expiries.length > 0 && (
              <select
                value={selectedExpiry}
                onChange={e => setSelectedExpiry(e.target.value)}
                className="w-40 bg-slate-800 border-slate-600 text-white rounded px-2 py-1"
              >
                {expiries.map(exp => (
                  <option key={exp} value={exp}>{exp}</option>
                ))}
              </select>
            )}
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
