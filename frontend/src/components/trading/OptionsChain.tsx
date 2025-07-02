
import { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Filter, RefreshCw, IndianRupee, AlertCircle } from 'lucide-react';
import axios from 'axios';

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
    total_cost?: number;
    affordable?: boolean;
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
    total_cost?: number;
    affordable?: boolean;
  };
}

interface OptionChainResponse {
  symbol: string;
  underlying_value: number;
  expiry_dates: string[];
  option_chain: OptionData[];
  available_funds?: number;
  source: string;
  timestamp: string;
}

const OptionsChain = () => {
  const [optionData, setOptionData] = useState<OptionData[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [spotPrice, setSpotPrice] = useState<number | null>(null);
  const [expiries, setExpiries] = useState<string[]>([]);
  const [selectedExpiry, setSelectedExpiry] = useState<string>('');
  const [filterStrike, setFilterStrike] = useState('');
  const [availableFunds, setAvailableFunds] = useState<number>(0);
  const [showAffordableOnly, setShowAffordableOnly] = useState(false);

  const fetchData = async () => {
    setLoading(true);
    setError(null);
    try {
      const url = selectedExpiry
        ? `http://localhost:8000/api/v1/option-chain?symbol=NIFTY&expiry=${encodeURIComponent(selectedExpiry)}`
        : 'http://localhost:8000/api/v1/option-chain?symbol=NIFTY';
      
      console.log('Fetching option chain from:', url);
      const response = await axios.get(url, { timeout: 30000 });
      
      if (response.data.success && response.data.data) {
        const data: OptionChainResponse = response.data.data;
        setOptionData(data.option_chain || []);
        setSpotPrice(data.underlying_value || null);
        setExpiries(data.expiry_dates || []);
        setAvailableFunds(data.available_funds || 100000);
        
        if (!selectedExpiry && data.expiry_dates && data.expiry_dates.length > 0) {
          setSelectedExpiry(data.expiry_dates[0]);
        }
        
        setError(null);
      } else {
        setError(response.data.error || 'Failed to fetch option chain data');
      }
    } catch (err: any) {
      console.error('Option chain fetch error:', err);
      const errorMessage = err.code === 'ERR_NETWORK' 
        ? 'Backend server not running on http://localhost:8000. Please start the backend server.'
        : err.response?.data?.error || err.message || 'Failed to fetch option chain data';
      setError(errorMessage);
      
      // Generate mock data for display when backend is not available
      if (err.code === 'ERR_NETWORK') {
        const mockData = generateMockOptionData();
        setOptionData(mockData);
        setSpotPrice(19850);
        setExpiries(['2025-01-02', '2025-01-09', '2025-01-16']);
        setAvailableFunds(100000);
        if (!selectedExpiry) {
          setSelectedExpiry('2025-01-02');
        }
      }
    }
    setLoading(false);
  };

  const generateMockOptionData = (): OptionData[] => {
    const strikes = [19600, 19650, 19700, 19750, 19800, 19850, 19900, 19950, 20000, 20050, 20100];
    return strikes.map(strike => ({
      strike,
      expiry: selectedExpiry || '2025-01-02',
      call: {
        ltp: Math.max(1, Math.random() * 150),
        volume: Math.floor(Math.random() * 100000),
        oi: Math.floor(Math.random() * 50000),
        iv: 15 + Math.random() * 20,
        delta: 0.1 + Math.random() * 0.8,
        gamma: Math.random() * 0.01,
        theta: -Math.random() * 5,
        vega: Math.random() * 10,
        total_cost: Math.floor((Math.max(1, Math.random() * 150)) * 50),
        affordable: true
      },
      put: {
        ltp: Math.max(1, Math.random() * 150),
        volume: Math.floor(Math.random() * 100000),
        oi: Math.floor(Math.random() * 50000),
        iv: 15 + Math.random() * 20,
        delta: -(0.1 + Math.random() * 0.8),
        gamma: Math.random() * 0.01,
        theta: -Math.random() * 5,
        vega: Math.random() * 10,
        total_cost: Math.floor((Math.max(1, Math.random() * 150)) * 50),
        affordable: true
      }
    }));
  };

  useEffect(() => {
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

  const getMoneyness = (strike: number): string => {
    if (!spotPrice) return 'Unknown';
    const diff = Math.abs(strike - spotPrice);
    if (diff <= 50) return 'ATM';
    return strike < spotPrice ? 'ITM' : 'OTM';
  };

  const filteredData = optionData.filter(option => {
    const strikeMatch = filterStrike === '' || option.strike.toString().includes(filterStrike);
    const affordableMatch = !showAffordableOnly || 
      (option.call.affordable || option.put.affordable);
    return strikeMatch && affordableMatch;
  });

  if (loading) {
    return (
      <Card className="bg-slate-900 border-slate-700">
        <CardContent className="p-6">
          <div className="flex items-center justify-center h-64">
            <RefreshCw className="h-8 w-8 animate-spin text-blue-400" />
            <span className="ml-2 text-slate-300">Loading option chain from NSE...</span>
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
              <span>Available Funds: ₹{formatNumber(availableFunds)}</span>
              <Badge variant="outline" className="border-green-500 text-green-400">
                {error?.includes('Backend server not running') ? 'Mock Data' : 'NSE Live'}
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
            
            <Button
              size="sm"
              variant={showAffordableOnly ? "default" : "outline"}
              onClick={() => setShowAffordableOnly(!showAffordableOnly)}
              className="border-slate-600 text-slate-300"
            >
              <IndianRupee className="h-4 w-4 mr-1" />
              Affordable Only
            </Button>
            
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
            
            <Button onClick={fetchData} size="sm" variant="outline" className="border-slate-600 text-slate-300">
              <RefreshCw className="h-4 w-4 mr-1" />
              Refresh
            </Button>
          </div>
        </div>
      </CardHeader>

      <CardContent>
        {error && (
          <div className="mb-4 p-3 bg-red-900/20 border border-red-700 rounded flex items-center gap-2">
            <AlertCircle className="h-4 w-4 text-red-400 flex-shrink-0" />
            <p className="text-red-400 text-sm">{error}</p>
          </div>
        )}

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
                            <div className={`font-medium ${option.call.affordable ? 'text-green-400' : 'text-white'}`}>
                              ₹{option.call.ltp.toFixed(2)}
                            </div>
                            {option.call.total_cost && (
                              <div className="text-slate-500">₹{formatNumber(option.call.total_cost)}</div>
                            )}
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
                            <div className="text-slate-300">{option.call.iv?.toFixed(1) || 0}%</div>
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
                            <div className={`font-medium ${option.put.affordable ? 'text-green-400' : 'text-white'}`}>
                              ₹{option.put.ltp.toFixed(2)}
                            </div>
                            {option.put.total_cost && (
                              <div className="text-slate-500">₹{formatNumber(option.put.total_cost)}</div>
                            )}
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
                            <div className="text-slate-300">{option.put.iv?.toFixed(1) || 0}%</div>
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
