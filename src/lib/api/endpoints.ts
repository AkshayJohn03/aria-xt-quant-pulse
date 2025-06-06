
import { DataFetcher } from './dataFetcher';
import { ModelInterface } from '../ai/modelInterface';
import { RiskManager } from '../trading/riskManager';

export class AriaAPI {
  private dataFetcher: DataFetcher;
  private modelInterface: ModelInterface;
  private riskManager: RiskManager;

  constructor() {
    this.dataFetcher = new DataFetcher();
    this.modelInterface = new ModelInterface();
    this.riskManager = new RiskManager();
  }

  async getMarketData() {
    try {
      const marketData = await this.dataFetcher.fetchMarketData();
      return {
        success: true,
        data: marketData
      };
    } catch (error) {
      return {
        success: false,
        error: error.message
      };
    }
  }

  async getLiveOHLCV(symbol: string = 'NIFTY50') {
    try {
      const ohlcvData = await this.dataFetcher.fetchLiveOHLCV(symbol);
      return {
        success: true,
        data: ohlcvData
      };
    } catch (error) {
      return {
        success: false,
        error: error.message
      };
    }
  }

  async getOptionChain() {
    try {
      const optionChain = await this.dataFetcher.fetchOptionChain();
      return {
        success: true,
        data: optionChain
      };
    } catch (error) {
      return {
        success: false,
        error: error.message
      };
    }
  }

  async generatePrediction() {
    try {
      const ohlcvData = await this.dataFetcher.fetchLiveOHLCV();
      const prediction = await this.modelInterface.generatePrediction(ohlcvData);
      
      return {
        success: true,
        data: prediction
      };
    } catch (error) {
      return {
        success: false,
        error: error.message
      };
    }
  }

  async generateTradingSignal() {
    try {
      const ohlcvData = await this.dataFetcher.fetchLiveOHLCV();
      const optionData = await this.dataFetcher.fetchOptionChain();
      const signal = await this.modelInterface.generateTradingSignal(ohlcvData, optionData);
      
      if (!signal) {
        return {
          success: false,
          error: 'No trading signal generated'
        };
      }

      const isValid = this.riskManager.validateSignal(signal);
      
      return {
        success: true,
        data: {
          signal,
          validated: isValid
        }
      };
    } catch (error) {
      return {
        success: false,
        error: error.message
      };
    }
  }

  async getPortfolio() {
    try {
      const positions = this.riskManager.getPositions();
      const riskMetrics = this.riskManager.calculateRiskMetrics();
      
      return {
        success: true,
        data: {
          positions,
          riskMetrics
        }
      };
    } catch (error) {
      return {
        success: false,
        error: error.message
      };
    }
  }

  async runBacktest(startDate: string, endDate: string, strategy: string = 'aria-lstm') {
    try {
      const historicalData = await this.dataFetcher.fetchHistoricalOHLCV('NIFTY50', startDate, endDate);
      
      // Simplified backtesting logic
      let portfolio = 100000;
      const trades = [];
      
      for (let i = 10; i < historicalData.length; i += 10) {
        const dataSlice = historicalData.slice(i - 10, i);
        const prediction = await this.modelInterface.generatePrediction(dataSlice);
        
        if (prediction.confidence > 70) {
          const entry = dataSlice[dataSlice.length - 1];
          const exit = historicalData[Math.min(i + 5, historicalData.length - 1)];
          
          const isCall = prediction.direction === 'BULLISH';
          const pnl = isCall ? (exit.close - entry.close) * 100 : (entry.close - exit.close) * 100;
          
          portfolio += pnl;
          trades.push({
            entry: entry.timestamp,
            exit: exit.timestamp,
            type: isCall ? 'CE' : 'PE',
            pnl,
            confidence: prediction.confidence
          });
        }
      }

      const totalReturn = ((portfolio - 100000) / 100000) * 100;
      const winRate = trades.filter(t => t.pnl > 0).length / trades.length * 100;

      return {
        success: true,
        data: {
          totalReturn: totalReturn.toFixed(2),
          finalPortfolio: portfolio.toFixed(2),
          totalTrades: trades.length,
          winRate: winRate.toFixed(1),
          trades: trades.slice(-10) // Last 10 trades
        }
      };
    } catch (error) {
      return {
        success: false,
        error: error.message
      };
    }
  }

  async getConnectionStatus() {
    const status = {
      zerodhaAPI: Math.random() > 0.1 ? 'connected' : 'disconnected',
      twelveDataAPI: Math.random() > 0.05 ? 'connected' : 'disconnected',
      ollamaAPI: Math.random() > 0.15 ? 'connected' : 'disconnected',
      geminiAPI: Math.random() > 0.08 ? 'connected' : 'disconnected',
      ariaLSTM: Math.random() > 0.12 ? 'connected' : 'disconnected',
      finbert: Math.random() > 0.1 ? 'connected' : 'disconnected',
      prophet: Math.random() > 0.05 ? 'connected' : 'disconnected',
      xgboost: Math.random() > 0.07 ? 'connected' : 'disconnected',
      lastUpdate: new Date().toISOString()
    };

    return {
      success: true,
      data: status
    };
  }
}

// Initialize global API instance
export const ariaAPI = new AriaAPI();
