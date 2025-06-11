import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { ArrowUp, ArrowDown, TrendingUp, Activity } from 'lucide-react';

interface MarketData {
  nifty: number;
  sensex: number;
  marketStatus: string;
  lastUpdate: string;
  // If 'qty' was ever part of this, consider adding it here,
  // but it's not present in your provided interface.
  // Assuming the error was about nifty/sensex being null initially.
}

interface MarketOverviewProps {
  marketData?: MarketData; // Make marketData optional to handle initial undefined/null state
}

const MarketOverview: React.FC<MarketOverviewProps> = ({ marketData }) => {
  // Provide default/loading values if marketData is not yet available
  const currentNifty = marketData?.nifty ?? 0;
  const currentSensex = marketData?.sensex ?? 0;
  const currentMarketStatus = marketData?.marketStatus ?? "Loading...";
  const currentLastUpdate = marketData?.lastUpdate ?? "N/A";

  // These are hardcoded in your component. If they should come from marketData, update MarketData interface
  const niftyChange = 145.30;
  const sensexChange = 287.65;

  // Basic check to see if marketData is undefined or null, and render a loading state
  if (!marketData) {
    return (
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card className="bg-slate-800/50 border-slate-700 col-span-full text-center py-8">
          <CardTitle className="text-white text-xl">Loading Market Data...</CardTitle>
          <CardContent className="text-slate-400 mt-2">
            Please ensure the backend is running and providing data.
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
      {/* NIFTY 50 Card */}
      <Card className="bg-slate-800/50 border-slate-700">
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-medium text-slate-300">NIFTY 50</CardTitle>
          <TrendingUp className="h-4 w-4 text-green-400" />
        </CardHeader>
        <CardContent>
          {/* Use currentNifty for rendering */}
          <div className="text-2xl font-bold text-white">{currentNifty.toFixed(2)}</div>
          <div className="flex items-center space-x-2 text-xs">
            <ArrowUp className="h-3 w-3 text-green-400" />
            <span className="text-green-400">+{niftyChange}</span>
            <span className="text-slate-400">(+0.73%)</span>
          </div>
        </CardContent>
      </Card>

      {/* SENSEX Card */}
      <Card className="bg-slate-800/50 border-slate-700">
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-medium text-slate-300">SENSEX</CardTitle>
          <Activity className="h-4 w-4 text-blue-400" />
        </CardHeader>
        <CardContent>
          {/* Use currentSensex for rendering */}
          <div className="text-2xl font-bold text-white">{currentSensex.toFixed(2)}</div>
          <div className="flex items-center space-x-2 text-xs">
            <ArrowUp className="h-3 w-3 text-green-400" />
            <span className="text-green-400">+{sensexChange}</span>
            <span className="text-slate-400">(+0.43%)</span>
          </div>
        </CardContent>
      </Card>

      {/* Market Status Card */}
      <Card className="bg-slate-800/50 border-slate-700">
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-medium text-slate-300">Market Status</CardTitle>
          <div className="h-2 w-2 bg-green-400 rounded-full"></div>
        </CardHeader>
        <CardContent>
          {/* Use currentMarketStatus and currentLastUpdate */}
          <div className="text-2xl font-bold text-white">{currentMarketStatus}</div>
          <p className="text-xs text-slate-400">Last update: {currentLastUpdate}</p>
        </CardContent>
      </Card>

      {/* AI Sentiment Card (static for now) */}
      <Card className="bg-slate-800/50 border-slate-700">
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-medium text-slate-300">AI Sentiment</CardTitle>
          <Badge variant="outline" className="bg-green-500/20 text-green-400 border-green-400">
            BULLISH
          </Badge>
        </CardHeader>
        <CardContent>
          <div className="text-2xl font-bold text-green-400">82%</div>
          <p className="text-xs text-slate-400">Confidence Score</p>
        </CardContent>
      </Card>
    </div>
  );
};

export default MarketOverview;