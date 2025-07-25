
import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import MarketOverview from '@/components/trading/MarketOverview';
import Portfolio from '@/components/trading/Portfolio';
import BacktestingModule from '@/components/trading/BacktestingModule';
import PredictiveCharts from '@/components/trading/PredictiveCharts';
import ConnectionStatus from '@/components/trading/ConnectionStatus';
import OptionsChain from '@/components/trading/OptionsChain';
import TradeLog from '@/components/trading/TradeLog';
import RiskManagement from '@/components/trading/RiskManagement';
import ConfigurationPanel from '@/components/trading/ConfigurationPanel';
import { useMarketData } from '@/hooks/useMarketData';

const Index = () => {
  const [activeTab, setActiveTab] = useState('dashboard');
  const { marketData, loading, error } = useMarketData();

  // Transform marketData to match MarketOverview expected format with proper null checking
  const transformedMarketData = marketData && marketData.nifty && marketData.banknifty ? {
    nifty: {
      value: marketData.nifty?.value || 0,
      change: marketData.nifty?.change || 0,
      percentChange: marketData.nifty?.percentChange || 0
    },
    sensex: {
      value: marketData.banknifty?.value || 0, // Using BANKNIFTY as SENSEX placeholder
      change: marketData.banknifty?.change || 0,
      percentChange: marketData.banknifty?.percentChange || 0
    },
    marketStatus: "OPEN", // You can add this to your backend response
    lastUpdate: marketData.nifty?.timestamp || new Date().toISOString(),
    aiSentiment: {
      direction: "NEUTRAL", // You can add this to your backend response
      confidence: 75 // You can add this to your backend response
    }
  } : null;

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-slate-800 text-white">
      <div className="container mx-auto p-4">
        {/* Header */}
        <div className="mb-6">
          <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">
            Aria-xT: Quant-AI Trading Engine
          </h1>
          <p className="text-slate-300 mt-2">Advanced Automated Trading System for Indian Markets</p>
        </div>

        <div className="grid grid-cols-1 xl:grid-cols-4 gap-6">
          {/* Main Content */}
          <div className="xl:col-span-3">
            <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-4">
              <TabsList className="grid w-full grid-cols-7 bg-slate-800/50">
                <TabsTrigger value="dashboard">Dashboard</TabsTrigger>
                <TabsTrigger value="portfolio">Portfolio</TabsTrigger>
                <TabsTrigger value="backtest">Backtest</TabsTrigger>
                <TabsTrigger value="charts">Charts</TabsTrigger>
                <TabsTrigger value="options">Options</TabsTrigger>
                <TabsTrigger value="trades">Trades</TabsTrigger>
                <TabsTrigger value="config">Config</TabsTrigger>
              </TabsList>

              <TabsContent value="dashboard" className="space-y-6">
                <MarketOverview marketData={transformedMarketData} loading={loading} error={error} />
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  <PredictiveCharts />
                  <RiskManagement />
                </div>
              </TabsContent>

              <TabsContent value="portfolio">
                <Portfolio />
              </TabsContent>

              <TabsContent value="backtest">
                <BacktestingModule />
              </TabsContent>

              <TabsContent value="charts">
                <PredictiveCharts />
              </TabsContent>

              <TabsContent value="options">
                <OptionsChain />
              </TabsContent>

              <TabsContent value="trades">
                <TradeLog />
              </TabsContent>

              <TabsContent value="config">
                <ConfigurationPanel />
              </TabsContent>
            </Tabs>
          </div>

          {/* Right Sidebar - Connection Status */}
          <div className="xl:col-span-1">
            <ConnectionStatus />
          </div>
        </div>
      </div>
    </div>
  );
};

export default Index;
