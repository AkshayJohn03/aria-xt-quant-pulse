document.addEventListener('DOMContentLoaded', () => {
    const backtestForm = document.getElementById('backtestForm');
    const loadingOverlay = document.getElementById('loading');
    const resultsSection = document.getElementById('results');

    backtestForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        // Show loading overlay
        loadingOverlay.style.display = 'flex';
        
        // Get form data
        const formData = new FormData(backtestForm);
        const data = {
            symbol: formData.get('symbol'),
            start_date: formData.get('start_date'),
            end_date: formData.get('end_date'),
            strategy_profile: formData.get('strategy_profile'),
            mode: formData.get('mode'),
            initial_capital: parseFloat(formData.get('initial_capital')),
            max_trades_per_day: parseInt(formData.get('max_trades_per_day')),
            risk_per_trade: parseFloat(formData.get('risk_per_trade')) / 100  // Convert percentage to decimal
        };
        
        try {
            // Send request to backend
            const response = await fetch('/backtest', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(data)
            });
            
            if (!response.ok) {
                throw new Error('Backtesting failed');
            }
            
            const results = await response.json();
            
            // Update metrics
            updateMetrics(results.metrics);
            
            // Update charts
            updateCharts(results.trades);
            
            // Update trade log
            updateTradeLog(results.trades);
            
            // Show results section
            resultsSection.classList.remove('hidden');
            
        } catch (error) {
            console.error('Error:', error);
            alert('An error occurred while running the backtest. Please try again.');
        } finally {
            // Hide loading overlay
            loadingOverlay.style.display = 'none';
        }
    });
});

function updateMetrics(metrics) {
    // Update metrics display
    document.getElementById('totalTrades').textContent = metrics.total_trades;
    document.getElementById('winRate').textContent = `${metrics.win_rate}%`;
    document.getElementById('totalPnL').textContent = `₹${metrics.total_pnl.toLocaleString()}`;
    document.getElementById('maxDrawdown').textContent = `${metrics.max_drawdown}%`;
    document.getElementById('sharpeRatio').textContent = metrics.sharpe_ratio.toFixed(2);
    document.getElementById('avgProfitPerTrade').textContent = `₹${metrics.avg_profit_per_trade.toLocaleString()}`;
    document.getElementById('avgTradeDuration').textContent = metrics.avg_trade_duration;
}

function updateCharts(trades) {
    // Prepare data for equity curve
    const equityData = calculateEquityCurve(trades);
    
    // Create equity curve chart
    Plotly.newPlot('equityCurve', [{
        x: equityData.timestamps,
        y: equityData.equity,
        type: 'scatter',
        mode: 'lines',
        name: 'Equity',
        line: {
            color: '#3B82F6'
        }
    }], {
        title: 'Equity Curve',
        xaxis: {
            title: 'Time'
        },
        yaxis: {
            title: 'Equity (₹)'
        },
        showlegend: false
    });
    
    // Prepare data for drawdown chart
    const drawdownData = calculateDrawdown(equityData.equity);
    
    // Create drawdown chart
    Plotly.newPlot('drawdownChart', [{
        x: equityData.timestamps,
        y: drawdownData,
        type: 'scatter',
        mode: 'lines',
        name: 'Drawdown',
        fill: 'tozeroy',
        line: {
            color: '#EF4444'
        }
    }], {
        title: 'Drawdown',
        xaxis: {
            title: 'Time'
        },
        yaxis: {
            title: 'Drawdown (%)',
            autorange: 'reversed'  // Invert y-axis for drawdown
        },
        showlegend: false
    });
}

function calculateEquityCurve(trades) {
    // Sort trades by entry time
    const sortedTrades = [...trades].sort((a, b) => 
        new Date(a.entry_time) - new Date(b.entry_time)
    );
    
    let equity = 100000;  // Initial capital
    const timestamps = [];
    const equityValues = [];
    
    // Add initial point
    timestamps.push(new Date(sortedTrades[0].entry_time));
    equityValues.push(equity);
    
    // Calculate equity curve
    sortedTrades.forEach(trade => {
        equity += trade.pnl;
        timestamps.push(new Date(trade.exit_time));
        equityValues.push(equity);
    });
    
    return {
        timestamps: timestamps,
        equity: equityValues
    };
}

function calculateDrawdown(equity) {
    let peak = equity[0];
    const drawdown = [];
    
    equity.forEach(value => {
        if (value > peak) {
            peak = value;
        }
        const dd = ((peak - value) / peak) * 100;
        drawdown.push(dd);
    });
    
    return drawdown;
}

function updateTradeLog(trades) {
    const tradeLog = document.getElementById('tradeLog');
    tradeLog.innerHTML = '';  // Clear existing rows
    
    trades.forEach(trade => {
        const row = document.createElement('tr');
        
        // Format timestamps
        const entryTime = new Date(trade.entry_time).toLocaleString();
        const exitTime = new Date(trade.exit_time).toLocaleString();
        
        // Format P&L with color
        const pnlColor = trade.pnl >= 0 ? 'text-green-600' : 'text-red-600';
        
        row.innerHTML = `
            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-900">${entryTime}</td>
            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-900">${exitTime}</td>
            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-900">${trade.signal_type}</td>
            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-900">${trade.strike}</td>
            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-900">${trade.option_type}</td>
            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-900">₹${trade.entry_price}</td>
            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-900">₹${trade.exit_price}</td>
            <td class="px-6 py-4 whitespace-nowrap text-sm ${pnlColor} font-semibold">₹${trade.pnl.toLocaleString()}</td>
            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-900">${trade.exit_reason}</td>
        `;
        
        tradeLog.appendChild(row);
    });
} 