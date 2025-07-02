import { TradingSignal } from '../ai/modelInterface';
import { ConfigManager } from '../config';

export interface Position {
  id: string;
  symbol: string;
  type: 'CE' | 'PE';
  quantity: number;
  entryPrice: number;
  currentPrice: number;
  stopLoss: number;
  target: number;
  trailingStop: number;
  pnl: number;
  status: 'OPEN' | 'CLOSED';
  entryTime: number;
}

export interface RiskMetrics {
  totalExposure: number;
  maxDrawdown: number;
  currentDrawdown: number;
  portfolioRisk: 'LOW' | 'MEDIUM' | 'HIGH';
  riskScore: number;
}

export class RiskManager {
  private config: ConfigManager;
  private positions: Map<string, Position>;

  constructor() {
    this.config = new ConfigManager();
    this.positions = new Map();
  }

  validateSignal(signal: TradingSignal): boolean {
    const config = this.config.getConfig();
    
    // Check position limits
    if (this.positions.size >= config.trading.maxPositions) {
      console.log('Position limit reached');
      return false;
    }

    // Check risk per trade
    const riskAmount = signal.quantity * signal.price;
    const maxRisk = config.trading.capitalAllocation * (config.trading.maxRiskPerTrade / 100);
    
    if (riskAmount > maxRisk) {
      console.log('Risk per trade exceeded');
      return false;
    }

    // Check confidence threshold
    if (signal.confidence < config.models.confidenceThreshold * 100) {
      console.log('Confidence threshold not met');
      return false;
    }

    return true;
  }

  addPosition(signal: TradingSignal): string {
    const positionId = `${signal.symbol}_${Date.now()}`;
    const position: Position = {
      id: positionId,
      symbol: signal.symbol,
      type: signal.symbol.includes('CE') ? 'CE' : 'PE',
      quantity: signal.quantity,
      entryPrice: signal.price,
      currentPrice: signal.price,
      stopLoss: signal.stopLoss,
      target: signal.target,
      trailingStop: signal.stopLoss,
      pnl: 0,
      status: 'OPEN',
      entryTime: Date.now()
    };

    this.positions.set(positionId, position);
    return positionId;
  }

  updatePosition(positionId: string, currentPrice: number): void {
    const position = this.positions.get(positionId);
    if (!position) return;

    position.currentPrice = currentPrice;
    position.pnl = (currentPrice - position.entryPrice) * position.quantity;

    // Update trailing stop
    const trailingPercent = this.config.getConfig().trading.trailingStopPercent / 100;
    
    if (position.type === 'CE' && currentPrice > position.entryPrice) {
      const newTrailingStop = currentPrice * (1 - trailingPercent);
      position.trailingStop = Math.max(position.trailingStop, newTrailingStop);
    } else if (position.type === 'PE' && currentPrice < position.entryPrice) {
      const newTrailingStop = currentPrice * (1 + trailingPercent);
      position.trailingStop = Math.min(position.trailingStop, newTrailingStop);
    }
  }

  checkExitConditions(positionId: string): boolean {
    const position = this.positions.get(positionId);
    if (!position) return false;

    // Check stop loss
    if (position.type === 'CE' && position.currentPrice <= position.trailingStop) {
      return true;
    }
    
    if (position.type === 'PE' && position.currentPrice >= position.trailingStop) {
      return true;
    }

    // Check target
    if (position.type === 'CE' && position.currentPrice >= position.target) {
      return true;
    }
    
    if (position.type === 'PE' && position.currentPrice <= position.target) {
      return true;
    }

    return false;
  }

  closePosition(positionId: string): Position | null {
    const position = this.positions.get(positionId);
    if (!position) return null;

    position.status = 'CLOSED';
    this.positions.set(positionId, position);
    return position;
  }

  getPositions(): Position[] {
    return Array.from(this.positions.values());
  }

  getOpenPositions(): Position[] {
    return this.getPositions().filter(p => p.status === 'OPEN');
  }

  calculateRiskMetrics(): RiskMetrics {
    const openPositions = this.getOpenPositions();
    const totalExposure = openPositions.reduce((sum, pos) => 
      sum + (pos.quantity * pos.currentPrice), 0
    );

    const totalPnL = openPositions.reduce((sum, pos) => sum + pos.pnl, 0);
    const capitalAllocation = this.config.getConfig().trading.capitalAllocation;
    
    const currentDrawdown = Math.max(0, (-totalPnL / capitalAllocation) * 100);
    const exposurePercent = (totalExposure / capitalAllocation) * 100;

    let portfolioRisk: 'LOW' | 'MEDIUM' | 'HIGH' = 'LOW';
    let riskScore = 0;

    if (exposurePercent > 80 || currentDrawdown > 10) {
      portfolioRisk = 'HIGH';
      riskScore = 8;
    } else if (exposurePercent > 50 || currentDrawdown > 5) {
      portfolioRisk = 'MEDIUM';
      riskScore = 5;
    } else {
      riskScore = 2;
    }

    return {
      totalExposure: exposurePercent,
      maxDrawdown: 15, // Historical max
      currentDrawdown,
      portfolioRisk,
      riskScore
    };
  }
}
