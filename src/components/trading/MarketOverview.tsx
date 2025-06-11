import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { ArrowUp, ArrowDown, TrendingUp, Activity, AlertCircle } from 'lucide-react';

interface MarketData {
  nifty: {
    value: number;
    change: number;
    percentChange: number;
  };
  sensex: {
    value: number;
    change: number;
    percentChange: number;
  };
  marketStatus: string;
  lastUpdate: string;
  aiSentiment: {
    direction: string; // e.g., "BULLISH", "BEARISH", "NEUTRAL"
    confidence: number; // Percentage
  };
}

interface MarketOverviewProps {
  marketData?: MarketData; // Make marketData optional to handle initial undefined/null state
  loading?: boolean;
  error?: string;
}

const MarketOverview: React.FC<MarketOverviewProps> = ({ marketData, loading = false, error }) => {
  if (loading) {
    return (
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {[...Array(4)].map((_, i) => (
          <Card key={i} className="bg-slate-800/50 border-slate-700">
            <CardHeader className="animate-pulse">
              <div className="h-4 bg-slate-600 rounded w-1/2"></div>
            </CardHeader>
            <CardContent className="animate-pulse">
              <div className="h-8 bg-slate-600 rounded w-3/4 mb-2"></div>
              <div className="h-3 bg-slate-600 rounded w-1/2"></div>
            </CardContent>
          </Card>
        ))}
      </div>
    );
  }

  if (error) {
    return (
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card className="bg-red-900/20 border-red-700 col-span-full">
          <CardHeader className="flex flex-row items-center space-y-0 pb-2">
            <AlertCircle className="h-4 w-4 text-red-400 mr-2" />
            <CardTitle className="text-red-400">Market Data Error</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-red-300">{error}</p>
            <p className="text-slate-400 text-sm mt-2">
              Please ensure the backend server is running on http://localhost:8000
            </p>
          </CardContent>
        </Card>
      </div>
    );
  }

  if (!marketData) {
    return (
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card className="bg-slate-800/50 border-slate-700 col-span-full text-center py-8">
          <CardHeader>
            <CardTitle className="text-white text-xl">No Market Data Available</CardTitle>
          </CardHeader>
          <CardContent className="text-slate-400">
            Waiting for market data from backend...
          </CardContent>
        </Card>
      </div>
    );
  }

  // Provide default/loading values if marketData is not yet available
  const currentNiftyValue = marketData?.nifty?.value ?? 0;
  const currentNiftyChange = marketData?.nifty?.change ?? 0;
  const currentNiftyPercentChange = marketData?.nifty?.percentChange ?? 0;

  const currentSensexValue = marketData?.sensex?.value ?? 0;
  const currentSensexChange = marketData?.sensex?.change ?? 0;
  const currentSensexPercentChange = marketData?.sensex?.percentChange ?? 0;

  const currentMarketStatus = marketData?.marketStatus ?? "Loading...";
  const currentLastUpdate = marketData?.lastUpdate ?? "N/A";

  const currentAiSentimentDirection = marketData?.aiSentiment?.direction ?? "NEUTRAL";
  const currentAiSentimentConfidence = marketData?.aiSentiment?.confidence ?? 0;

  // Determine change indicator for Nifty
  const niftyChangeIndicator = currentNiftyChange >= 0 ? ArrowUp : ArrowDown;
  const niftyChangeColor = currentNiftyChange >= 0 ? "text-green-400" : "text-red-400";

  // Determine change indicator for Sensex
  const sensexChangeIndicator = currentSensexChange >= 0 ? ArrowUp : ArrowDown;
  const sensexChangeColor = currentSensexChange >= 0 ? "text-green-400" : "text-red-400";

  // Determine sentiment badge color
  const sentimentBadgeColor = 
    currentAiSentimentDirection === "BULLISH" ? "bg-green-500/20 text-green-400 border-green-400" :
    currentAiSentimentDirection === "BEARISH" ? "bg-red-500/20 text-red-400 border-red-400" :
    "bg-gray-500/20 text-gray-400 border-gray-400";


  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
      {/* NIFTY 50 Card */}
      <Card className="bg-slate-800/50 border-slate-700">
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-medium text-slate-300">NIFTY 50</CardTitle>
          <TrendingUp className="h-4 w-4 text-green-400" /> {/* Icon remains static for now */}
        </CardHeader>
        <CardContent>
          <div className="text-2xl font-bold text-white">{currentNiftyValue.toFixed(2)}</div>
          <div className="flex items-center space-x-2 text-xs">
            {React.createElement(niftyChangeIndicator, { className: `h-3 w-3 ${niftyChangeColor}` })}
            <span className={niftyChangeColor}>
              {currentNiftyChange >= 0 ? '+' : ''}{currentNiftyChange.toFixed(2)}
            </span>
            <span className="text-slate-400">
              ({currentNiftyPercentChange >= 0 ? '+' : ''}{currentNiftyPercentChange.toFixed(2)}%)
            </span>
          </div>
        </CardContent>
      </Card>

      {/* SENSEX Card */}
      <Card className="bg-slate-800/50 border-slate-700">
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-medium text-slate-300">SENSEX</CardTitle>
          <Activity className="h-4 w-4 text-blue-400" /> {/* Icon remains static for now */}
        </CardHeader>
        <CardContent>
          <div className="text-2xl font-bold text-white">{currentSensexValue.toFixed(2)}</div>
          <div className="flex items-center space-x-2 text-xs">
            {React.createElement(sensexChangeIndicator, { className: `h-3 w-3 ${sensexChangeColor}` })}
            <span className={sensexChangeColor}>
              {currentSensexChange >= 0 ? '+' : ''}{currentSensexChange.toFixed(2)}
            </span>
            <span className="text-slate-400">
              ({currentSensexPercentChange >= 0 ? '+' : ''}{currentSensexPercentChange.toFixed(2)}%)
            </span>
          </div>
        </CardContent>
      </Card>

      {/* Market Status Card */}
      <Card className="bg-slate-800/50 border-slate-700">
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-medium text-slate-300">Market Status</CardTitle>
          <div className="h-2 w-2 bg-green-400 rounded-full"></div> {/* Static status light for now */}
        </CardHeader>
        <CardContent>
          <div className="text-2xl font-bold text-white">{currentMarketStatus}</div>
          <p className="text-xs text-slate-400">Last update: {currentLastUpdate}</p>
        </CardContent>
      </Card>

      {/* AI Sentiment Card */}
      <Card className="bg-slate-800/50 border-slate-700">
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-medium text-slate-300">AI Sentiment</CardTitle>
          <Badge variant="outline" className={sentimentBadgeColor}>
            {currentAiSentimentDirection.toUpperCase()}
          </Badge>
        </CardHeader>
        <CardContent>
          <div className={`text-2xl font-bold ${sentimentBadgeColor.includes('green') ? 'text-green-400' : sentimentBadgeColor.includes('red') ? 'text-red-400' : 'text-gray-400'}`}>
            {currentAiSentimentConfidence.toFixed(0)}%
          </div>
          <p className="text-xs text-slate-400">Confidence Score</p>
        </CardContent>
      </Card>
    </div>
  );
};

export default MarketOverview;
