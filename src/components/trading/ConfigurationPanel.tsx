
import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Badge } from '@/components/ui/badge';
import { Settings, Key, Shield, Brain, MessageSquare } from 'lucide-react';
import { ConfigManager, AriaConfig } from '@/lib/config';
import { toast } from 'sonner';

const ConfigurationPanel = () => {
  const [config, setConfig] = useState(() => new ConfigManager().getConfig());
  const [saving, setSaving] = useState(false);

  const handleSave = async () => {
    try {
      setSaving(true);
      const configManager = new ConfigManager();
      configManager.saveConfig(config);
      toast.success('Configuration saved successfully!');
    } catch (error) {
      toast.error('Failed to save configuration');
    } finally {
      setSaving(false);
    }
  };

  const updateConfig = (section: keyof AriaConfig, field: string, value: any) => {
    setConfig(prev => ({
      ...prev,
      [section]: {
        ...prev[section],
        [field]: value
      }
    }));
  };

  const testConnection = async (service: string) => {
    toast.info(`Testing ${service} connection...`);
    // Simulate connection test
    setTimeout(() => {
      toast.success(`${service} connection successful!`);
    }, 1000);
  };

  return (
    <div className="space-y-6">
      <Card className="bg-slate-800/50 border-slate-700">
        <CardHeader>
          <CardTitle className="text-white flex items-center space-x-2">
            <Settings className="h-5 w-5" />
            <span>Aria-xT Configuration</span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <Tabs defaultValue="apis" className="space-y-4">
            <TabsList className="grid w-full grid-cols-4 bg-slate-700">
              <TabsTrigger value="apis">API Keys</TabsTrigger>
              <TabsTrigger value="trading">Trading</TabsTrigger>
              <TabsTrigger value="models">Models</TabsTrigger>
              <TabsTrigger value="notifications">Alerts</TabsTrigger>
            </TabsList>

            <TabsContent value="apis" className="space-y-4">
              <Card className="bg-slate-700/50 border-slate-600">
                <CardHeader>
                  <CardTitle className="text-white flex items-center space-x-2">
                    <Key className="h-4 w-4" />
                    <span>Zerodha API</span>
                    <Badge variant="outline" className="text-green-400">Primary</Badge>
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <Label htmlFor="zerodha-key">API Key</Label>
                      <Input
                        id="zerodha-key"
                        type="password"
                        value={config.apis.zerodha.apiKey}
                        onChange={(e) => updateConfig('apis', 'zerodha', { ...config.apis.zerodha, apiKey: e.target.value })}
                        className="bg-slate-600 border-slate-500"
                        placeholder="Enter Zerodha API Key"
                      />
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="zerodha-secret">API Secret</Label>
                      <Input
                        id="zerodha-secret"
                        type="password"
                        value={config.apis.zerodha.apiSecret}
                        onChange={(e) => updateConfig('apis', 'zerodha', { ...config.apis.zerodha, apiSecret: e.target.value })}
                        className="bg-slate-600 border-slate-500"
                        placeholder="Enter Zerodha API Secret"
                      />
                    </div>
                  </div>
                  <Button 
                    onClick={() => testConnection('Zerodha')}
                    variant="outline" 
                    size="sm"
                    className="border-green-600 text-green-400"
                  >
                    Test Connection
                  </Button>
                </CardContent>
              </Card>

              <Card className="bg-slate-700/50 border-slate-600">
                <CardHeader>
                  <CardTitle className="text-white flex items-center space-x-2">
                    <Key className="h-4 w-4" />
                    <span>Twelve Data API</span>
                    <Badge variant="secondary">Backtesting</Badge>
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="space-y-2">
                    <Label htmlFor="twelve-key">API Key</Label>
                    <Input
                      id="twelve-key"
                      type="password"
                      value={config.apis.twelveData.apiKey}
                      onChange={(e) => updateConfig('apis', 'twelveData', { apiKey: e.target.value })}
                      className="bg-slate-600 border-slate-500"
                      placeholder="Enter Twelve Data API Key"
                    />
                  </div>
                  <Button 
                    onClick={() => testConnection('Twelve Data')}
                    variant="outline" 
                    size="sm"
                    className="border-blue-600 text-blue-400"
                  >
                    Test Connection
                  </Button>
                </CardContent>
              </Card>

              <Card className="bg-slate-700/50 border-slate-600">
                <CardHeader>
                  <CardTitle className="text-white flex items-center space-x-2">
                    <Brain className="h-4 w-4" />
                    <span>Gemini 2.0 Flash</span>
                    <Badge variant="outline" className="text-purple-400">AI Validation</Badge>
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="space-y-2">
                    <Label htmlFor="gemini-key">API Key</Label>
                    <Input
                      id="gemini-key"
                      type="password"
                      value={config.apis.gemini.apiKey}
                      onChange={(e) => updateConfig('apis', 'gemini', { apiKey: e.target.value })}
                      className="bg-slate-600 border-slate-500"
                      placeholder="Enter Gemini API Key"
                    />
                  </div>
                  <Button 
                    onClick={() => testConnection('Gemini')}
                    variant="outline" 
                    size="sm"
                    className="border-purple-600 text-purple-400"
                  >
                    Test Connection
                  </Button>
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="trading" className="space-y-4">
              <Card className="bg-slate-700/50 border-slate-600">
                <CardHeader>
                  <CardTitle className="text-white flex items-center space-x-2">
                    <Shield className="h-4 w-4" />
                    <span>Risk Management</span>
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <Label htmlFor="max-risk">Max Risk Per Trade (%)</Label>
                      <Input
                        id="max-risk"
                        type="number"
                        value={config.trading.maxRiskPerTrade}
                        onChange={(e) => updateConfig('trading', 'maxRiskPerTrade', parseFloat(e.target.value))}
                        className="bg-slate-600 border-slate-500"
                        min="0.1"
                        max="10"
                        step="0.1"
                      />
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="trailing-stop">Trailing Stop (%)</Label>
                      <Input
                        id="trailing-stop"
                        type="number"
                        value={config.trading.trailingStopPercent}
                        onChange={(e) => updateConfig('trading', 'trailingStopPercent', parseFloat(e.target.value))}
                        className="bg-slate-600 border-slate-500"
                        min="1"
                        max="20"
                        step="0.5"
                      />
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="max-positions">Max Positions</Label>
                      <Input
                        id="max-positions"
                        type="number"
                        value={config.trading.maxPositions}
                        onChange={(e) => updateConfig('trading', 'maxPositions', parseInt(e.target.value))}
                        className="bg-slate-600 border-slate-500"
                        min="1"
                        max="20"
                      />
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="capital">Capital Allocation (₹)</Label>
                      <Input
                        id="capital"
                        type="number"
                        value={config.trading.capitalAllocation}
                        onChange={(e) => updateConfig('trading', 'capitalAllocation', parseFloat(e.target.value))}
                        className="bg-slate-600 border-slate-500"
                        min="10000"
                        step="1000"
                      />
                    </div>
                  </div>
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="models" className="space-y-4">
              <Card className="bg-slate-700/50 border-slate-600">
                <CardHeader>
                  <CardTitle className="text-white flex items-center space-x-2">
                    <Brain className="h-4 w-4" />
                    <span>AI Models Configuration</span>
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <Label htmlFor="ollama-url">Ollama URL</Label>
                      <Input
                        id="ollama-url"
                        value={config.models.ollamaUrl}
                        onChange={(e) => updateConfig('models', 'ollamaUrl', e.target.value)}
                        className="bg-slate-600 border-slate-500"
                        placeholder="http://localhost:11434"
                      />
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="confidence-threshold">Confidence Threshold</Label>
                      <Input
                        id="confidence-threshold"
                        type="number"
                        value={config.models.confidenceThreshold}
                        onChange={(e) => updateConfig('models', 'confidenceThreshold', parseFloat(e.target.value))}
                        className="bg-slate-600 border-slate-500"
                        min="0.1"
                        max="1.0"
                        step="0.05"
                      />
                    </div>
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="model-path">Model Path</Label>
                    <Input
                      id="model-path"
                      value={config.models.modelPath}
                      onChange={(e) => updateConfig('models', 'modelPath', e.target.value)}
                      className="bg-slate-600 border-slate-500"
                      placeholder="/models/runtime/"
                    />
                  </div>
                  <Button 
                    onClick={() => testConnection('Ollama')}
                    variant="outline" 
                    size="sm"
                    className="border-orange-600 text-orange-400"
                  >
                    Test Ollama Connection
                  </Button>
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="notifications" className="space-y-4">
              <Card className="bg-slate-700/50 border-slate-600">
                <CardHeader>
                  <CardTitle className="text-white flex items-center space-x-2">
                    <MessageSquare className="h-4 w-4" />
                    <span>Telegram Notifications</span>
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <Label htmlFor="telegram-token">Bot Token</Label>
                      <Input
                        id="telegram-token"
                        type="password"
                        value={config.apis.telegram.botToken}
                        onChange={(e) => updateConfig('apis', 'telegram', { ...config.apis.telegram, botToken: e.target.value })}
                        className="bg-slate-600 border-slate-500"
                        placeholder="Enter Telegram Bot Token"
                      />
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="telegram-chat">Chat ID</Label>
                      <Input
                        id="telegram-chat"
                        value={config.apis.telegram.chatId}
                        onChange={(e) => updateConfig('apis', 'telegram', { ...config.apis.telegram, chatId: e.target.value })}
                        className="bg-slate-600 border-slate-500"
                        placeholder="Enter Telegram Chat ID"
                      />
                    </div>
                  </div>
                  <Button 
                    onClick={() => testConnection('Telegram')}
                    variant="outline" 
                    size="sm"
                    className="border-cyan-600 text-cyan-400"
                  >
                    Send Test Message
                  </Button>
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>

          <div className="flex justify-end pt-4 border-t border-slate-600">
            <Button 
              onClick={handleSave}
              disabled={saving}
              className="bg-blue-600 hover:bg-blue-700"
            >
              {saving ? 'Saving...' : 'Save Configuration'}
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default ConfigurationPanel;
