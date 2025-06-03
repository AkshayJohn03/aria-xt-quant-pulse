
import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Area, AreaChart } from 'recharts';
import { Badge } from '@/components/ui/badge';
import { TrendingUp, Brain, BarChart3 } from 'lucide-react';

const PredictiveCharts = () => {
  const [chartData, setChartData] = useState([]);
  const [prediction, setPrediction] = useState({
    direction: 'BULLISH',
    confidence: 87.5,
    targetPrice: 20125,
    timeframe: '1H'
  });

  useEffect(() => {
    // Generate realistic OHLC data with predictions
    const generateData = () => {
      const data = [];
      let price = 19850;
      const now = new Date();
      
      for (let i = 30; i >= 0; i--) {
        const time = new Date(now.getTime() - i * 5 * 60 * 1000); // 5-minute intervals
        const change = (Math.random() - 0.5) * 20;
        price += change;
        
        data.push({
          time: time.toLocaleTimeString(),
          actual: price,
          predicted: i < 6 ? price + (Math.random() - 0.3) * 50 : null, // Predictions for last 6 points
          volume: Math.floor(Math.random() * 1000000) + 500000,
          confidence: i < 6 ? Math.random() * 20 + 80 : null
        });
      }
      return data;
    };

    setChartData(generateData());

    // Update data every 5 seconds
    const interval = setInterval(() => {
      setChartData(generateData());
    }, 5000);

    return () => clearInterval(interval);
  }, []);

  return (
    <div className="space-y-6">
      {/* Prediction Summary */}
      <Card className="bg-slate-800/50 border-slate-700">
        <CardHeader>
          <CardTitle className="text-white flex items-center space-x-2">
            <Brain className="h-5 w-5 text-purple-400" />
            <span>AI Prediction Summary</span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="text-center">
              <Badge variant={prediction.direction === 'BULLISH' ? 'default' : 'destructive'} className="mb-2">
                {prediction.direction}
              </Badge>
              <div className="text-sm text-slate-400">Market Direction</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-green-400">{prediction.confidence}%</div>
              <div className="text-sm text-slate-400">Confidence</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-blue-400">₹{prediction.targetPrice}</div>
              <div className="text-sm text-slate-400">Target Price</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-yellow-400">{prediction.timeframe}</div>
              <div className="text-sm text-slate-400">Timeframe</div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Price Chart with Predictions */}
      <Card className="bg-slate-800/50 border-slate-700">
        <CardHeader>
          <CardTitle className="text-white flex items-center space-x-2">
            <TrendingUp className="h-5 w-5 text-green-400" />
            <span>Live Price vs AI Predictions</span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <ResponsiveContainer width="100%" height={400}>
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis dataKey="time" stroke="#9CA3AF" />
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
                dataKey="actual"
                stroke="#10B981"
                strokeWidth={2}
                name="Live Price"
                dot={false}
              />
              <Line
                type="monotone"
                dataKey="predicted"
                stroke="#F59E0B"
                strokeWidth={2}
                strokeDasharray="5 5"
                name="AI Prediction"
                dot={{ fill: '#F59E0B', strokeWidth: 2, r: 4 }}
              />
            </LineChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>

      {/* Volume Analysis */}
      <Card className="bg-slate-800/50 border-slate-700">
        <CardHeader>
          <CardTitle className="text-white flex items-center space-x-2">
            <BarChart3 className="h-5 w-5 text-blue-400" />
            <span>Volume Analysis</span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <ResponsiveContainer width="100%" height={300}>
            <AreaChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis dataKey="time" stroke="#9CA3AF" />
              <YAxis stroke="#9CA3AF" />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#1F2937',
                  border: '1px solid #374151',
                  borderRadius: '8px'
                }}
              />
              <Area
                type="monotone"
                dataKey="volume"
                stroke="#3B82F6"
                fill="#3B82F6"
                fillOpacity={0.3}
                name="Volume"
              />
            </AreaChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>

      {/* Model Performance */}
      <Card className="bg-slate-800/50 border-slate-700">
        <CardHeader>
          <CardTitle className="text-white">Model Performance Metrics</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="text-center p-4 bg-slate-700/50 rounded-lg">
              <div className="text-sm text-slate-400 mb-1">Aria-LSTM</div>
              <div className="text-lg font-bold text-green-400">94.2%</div>
              <div className="text-xs text-slate-500">Accuracy</div>
            </div>
            <div className="text-center p-4 bg-slate-700/50 rounded-lg">
              <div className="text-sm text-slate-400 mb-1">XGBoost</div>
              <div className="text-lg font-bold text-blue-400">91.8%</div>
              <div className="text-xs text-slate-500">Accuracy</div>
            </div>
            <div className="text-center p-4 bg-slate-700/50 rounded-lg">
              <div className="text-sm text-slate-400 mb-1">Prophet</div>
              <div className="text-lg font-bold text-yellow-400">89.5%</div>
              <div className="text-xs text-slate-500">Accuracy</div>
            </div>
            <div className="text-center p-4 bg-slate-700/50 rounded-lg">
              <div className="text-sm text-slate-400 mb-1">FinBERT</div>
              <div className="text-lg font-bold text-purple-400">92.1%</div>
              <div className="text-xs text-slate-500">Sentiment</div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default PredictiveCharts;
