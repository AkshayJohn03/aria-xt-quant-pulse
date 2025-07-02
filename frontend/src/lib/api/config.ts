
export interface AriaConfig {
  apis: {
    zerodha: {
      apiKey: string;
      apiSecret: string;
      accessToken: string;
    };
    twelveData: {
      apiKey: string;
    };
    gemini: {
      apiKey: string;
    };
    telegram: {
      botToken: string;
      chatId: string;
    };
  };
  trading: {
    maxRiskPerTrade: number;
    trailingStopPercent: number;
    maxPositions: number;
    capitalAllocation: number;
  };
  models: {
    ollamaUrl: string;
    modelPath: string;
    confidenceThreshold: number;
  };
}

export const defaultConfig: AriaConfig = {
  apis: {
    zerodha: {
      apiKey: '',
      apiSecret: '',
      accessToken: ''
    },
    twelveData: {
      apiKey: ''
    },
    gemini: {
      apiKey: ''
    },
    telegram: {
      botToken: '',
      chatId: ''
    }
  },
  trading: {
    maxRiskPerTrade: 2.5,
    trailingStopPercent: 5.0,
    maxPositions: 5,
    capitalAllocation: 100000
  },
  models: {
    ollamaUrl: 'http://localhost:11434',
    modelPath: '/models/runtime/',
    confidenceThreshold: 0.75
  }
};

export class ConfigManager {
  private config: AriaConfig;

  constructor() {
    this.config = this.loadConfig();
  }

  private loadConfig(): AriaConfig {
    try {
      const stored = localStorage.getItem('aria-config');
      if (stored) {
        return { ...defaultConfig, ...JSON.parse(stored) };
      }
    } catch (error) {
      console.error('Failed to load config:', error);
    }
    return defaultConfig;
  }

  public saveConfig(config: AriaConfig): void {
    try {
      localStorage.setItem('aria-config', JSON.stringify(config));
      this.config = config;
    } catch (error) {
      console.error('Failed to save config:', error);
    }
  }

  public getConfig(): AriaConfig {
    return this.config;
  }

  public updateConfig(updates: Partial<AriaConfig>): void {
    this.config = { ...this.config, ...updates };
    this.saveConfig(this.config);
  }
}
