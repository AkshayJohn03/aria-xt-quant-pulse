import { ConfigManager } from '../config';
import { OHLCVData } from '../dataFetcher';

export interface TradingSignal {
  action: 'BUY' | 'SELL' | 'HOLD';
  confidence: number;
  symbol: string;
  price: number;
  quantity: number;
  stopLoss: number;
  target: number;
  reasoning: string;
  modelSource: string;
}

export interface ModelPrediction {
  direction: 'BULLISH' | 'BEARISH' | 'NEUTRAL';
  confidence: number;
  targetPrice: number;
  timeframe: string;
  probability: number;
}

export class ModelInterface {
  private config: ConfigManager;

  constructor() {
    this.config = new ConfigManager();
  }

  async queryOllama(prompt: string): Promise<any> {
    try {
      const { ollamaUrl } = this.config.getConfig().models;
      
      const response = await fetch(`${ollamaUrl}/api/generate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          model: 'qwen2.5:0.5b',
          prompt,
          stream: false
        })
      });

      if (!response.ok) {
        throw new Error(`Ollama API error: ${response.statusText}`);
      }

      const result = await response.json();
      return result.response;
    } catch (error) {
      console.error('Error querying Ollama:', error);
      return null;
    }
  }

  async queryGemini(prompt: string): Promise<any> {
    try {
      const { apiKey } = this.config.getConfig().apis.gemini;
      
      const response = await fetch('https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'x-goog-api-key': apiKey
        },
        body: JSON.stringify({
          contents: [{
            parts: [{ text: prompt }]
          }]
        })
      });

      if (!response.ok) {
        throw new Error(`Gemini API error: ${response.statusText}`);
      }

      const result = await response.json();
      return result.candidates[0]?.content?.parts[0]?.text;
    } catch (error) {
      console.error('Error querying Gemini:', error);
      return null;
    }
  }

  async generatePrediction(ohlcvData: OHLCVData[]): Promise<ModelPrediction> {
    try {
      // Simulate Aria-LSTM prediction
      const latestPrice = ohlcvData[ohlcvData.length - 1]?.close || 19850;
      const priceChange = (Math.random() - 0.4) * 100; // Slight bullish bias
      const targetPrice = latestPrice + priceChange;
      
      const direction = priceChange > 0 ? 'BULLISH' : priceChange < -20 ? 'BEARISH' : 'NEUTRAL';
      const confidence = Math.min(95, 70 + Math.abs(priceChange) / 2);

      return {
        direction,
        confidence,
        targetPrice,
        timeframe: '1H',
        probability: confidence / 100
      };
    } catch (error) {
      console.error('Error generating prediction:', error);
      return {
        direction: 'NEUTRAL',
        confidence: 50,
        targetPrice: 19850,
        timeframe: '1H',
        probability: 0.5
      };
    }
  }

  async generateTradingSignal(ohlcvData: OHLCVData[], optionData: any[]): Promise<TradingSignal | null> {
    try {
      const prediction = await this.generatePrediction(ohlcvData);
      const { confidenceThreshold } = this.config.getConfig().models;

      if (prediction.confidence < confidenceThreshold * 100) {
        return null; // Low confidence, no signal
      }

      const latestPrice = ohlcvData[ohlcvData.length - 1]?.close || 19850;
      
      // Find suitable option based on prediction
      const suitableOption = this.findSuitableOption(optionData, prediction, latestPrice);
      
      if (!suitableOption) {
        return null;
      }

      const action = prediction.direction === 'BULLISH' ? 'BUY' : 'SELL';
      const stopLoss = suitableOption.price * (action === 'BUY' ? 0.8 : 1.2);
      const target = suitableOption.price * (action === 'BUY' ? 1.5 : 0.7);

      return {
        action,
        confidence: prediction.confidence,
        symbol: suitableOption.symbol,
        price: suitableOption.price,
        quantity: this.calculateQuantity(suitableOption.price),
        stopLoss,
        target,
        reasoning: `${prediction.direction} prediction with ${prediction.confidence}% confidence`,
        modelSource: 'Aria-LSTM'
      };
    } catch (error) {
      console.error('Error generating trading signal:', error);
      return null;
    }
  }

  private findSuitableOption(optionData: any[], prediction: ModelPrediction, currentPrice: number): any {
    // Simplified option selection logic
    const isCall = prediction.direction === 'BULLISH';
    const atmStrike = Math.round(currentPrice / 50) * 50;
    
    const suitable = optionData.find(option => 
      Math.abs(option.strike - atmStrike) <= 100
    );

    if (suitable) {
      return {
        symbol: `NIFTY ${suitable.strike} ${isCall ? 'CE' : 'PE'}`,
        price: isCall ? suitable.call.ltp : suitable.put.ltp,
        strike: suitable.strike,
        type: isCall ? 'CE' : 'PE'
      };
    }

    return null;
  }

  private calculateQuantity(price: number): number {
    const { capitalAllocation, maxRiskPerTrade } = this.config.getConfig().trading;
    const maxRiskAmount = capitalAllocation * (maxRiskPerTrade / 100);
    return Math.floor(maxRiskAmount / price);
  }
}
