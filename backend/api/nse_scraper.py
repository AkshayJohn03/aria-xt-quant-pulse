
import asyncio
import aiohttp
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class NSEOptionChainScraper:
    def __init__(self):
        self.base_url = "https://www.nseindia.com"
        self.option_chain_url = f"{self.base_url}/api/option-chain-indices"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
        self.session = None

    async def get_session(self):
        if not self.session:
            self.session = aiohttp.ClientSession()
        return self.session

    async def close_session(self):
        if self.session:
            await self.session.close()
            self.session = None

    async def get_cookies(self):
        """Get cookies by visiting NSE website first"""
        try:
            session = await self.get_session()
            async with session.get(self.base_url, headers=self.headers) as response:
                return session.cookie_jar
        except Exception as e:
            logger.error(f"Error getting cookies: {e}")
            return None

    async def fetch_option_chain(self, symbol: str = "NIFTY") -> Optional[Dict[str, Any]]:
        """Fetch option chain data from NSE"""
        try:
            # Get cookies first
            await self.get_cookies()
            
            session = await self.get_session()
            params = {'symbol': symbol}
            
            async with session.get(self.option_chain_url, 
                                 headers=self.headers, 
                                 params=params,
                                 timeout=30) as response:
                
                if response.status == 200:
                    data = await response.json()
                    return self.parse_option_chain(data)
                else:
                    logger.error(f"NSE API returned status {response.status}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error fetching option chain from NSE: {e}")
            return None

    def parse_option_chain(self, raw_data: Dict) -> Dict[str, Any]:
        """Parse NSE option chain data"""
        try:
            records = raw_data.get('records', {})
            data = records.get('data', [])
            expiry_dates = records.get('expiryDates', [])
            
            if not data:
                return {}

            # Get current spot price
            underlying_value = records.get('underlyingValue', 0)
            
            # Parse option chain data
            option_chain = []
            for item in data:
                strike_price = item.get('strikePrice', 0)
                expiry_date = item.get('expiryDate', '')
                
                call_data = item.get('CE', {})
                put_data = item.get('PE', {})
                
                option_chain.append({
                    'strike': strike_price,
                    'expiry': expiry_date,
                    'call': {
                        'ltp': call_data.get('lastPrice', 0),
                        'volume': call_data.get('totalTradedVolume', 0),
                        'oi': call_data.get('openInterest', 0),
                        'change': call_data.get('change', 0),
                        'pChange': call_data.get('pChange', 0),
                        'bid_qty': call_data.get('bidQty', 0),
                        'bid_price': call_data.get('bidprice', 0),
                        'ask_qty': call_data.get('askQty', 0),
                        'ask_price': call_data.get('askPrice', 0),
                        'iv': call_data.get('impliedVolatility', 0),
                        'delta': call_data.get('delta', 0),
                        'gamma': call_data.get('gamma', 0),
                        'theta': call_data.get('theta', 0),
                        'vega': call_data.get('vega', 0)
                    },
                    'put': {
                        'ltp': put_data.get('lastPrice', 0),
                        'volume': put_data.get('totalTradedVolume', 0),
                        'oi': put_data.get('openInterest', 0),
                        'change': put_data.get('change', 0),
                        'pChange': put_data.get('pChange', 0),
                        'bid_qty': put_data.get('bidQty', 0),
                        'bid_price': put_data.get('bidprice', 0),
                        'ask_qty': put_data.get('askQty', 0),
                        'ask_price': put_data.get('askPrice', 0),
                        'iv': put_data.get('impliedVolatility', 0),
                        'delta': put_data.get('delta', 0),
                        'gamma': put_data.get('gamma', 0),
                        'theta': put_data.get('theta', 0),
                        'vega': put_data.get('vega', 0)
                    }
                })
            
            return {
                'symbol': raw_data.get('records', {}).get('underlyingValue', 0),
                'underlying_value': underlying_value,
                'expiry_dates': expiry_dates,
                'option_chain': option_chain,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error parsing option chain data: {e}")
            return {}

    def filter_by_budget(self, option_chain: List[Dict], available_funds: float) -> List[Dict]:
        """Filter options that are within budget"""
        affordable_options = []
        
        for option in option_chain:
            call_price = option['call'].get('ltp', 0)
            put_price = option['put'].get('ltp', 0)
            
            # Calculate lot size (usually 25 for NIFTY, 15 for BANKNIFTY)
            lot_size = 25  # Default for NIFTY
            
            call_cost = call_price * lot_size
            put_cost = put_price * lot_size
            
            option['call']['total_cost'] = call_cost
            option['put']['total_cost'] = put_cost
            option['call']['affordable'] = call_cost <= available_funds
            option['put']['affordable'] = put_cost <= available_funds
            
            if call_cost <= available_funds or put_cost <= available_funds:
                affordable_options.append(option)
        
        return affordable_options
