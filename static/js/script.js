let autoRefreshInterval = null;
let lastUpdateTime = null;

function formatTime(dateString) {
    const date = new Date(dateString);
    return date.toLocaleTimeString();
}

function updatePageData(data) {
    // Update current price
    document.getElementById('current-price').textContent = data.current_price;
    
    // Update prediction
    const predictionEl = document.getElementById('prediction');
    predictionEl.textContent = data.prediction;
    predictionEl.className = `prediction-value ${data.prediction.toLowerCase()}`;
    
    // Update confidence
    const confidence = data.confidence || 50;
    document.getElementById('confidence-percentage').textContent = `${confidence}%`;
    document.getElementById('confidence-fill').style.width = `${confidence}%`;
    
    // Update action signal
    const actionEl = document.getElementById('action-signal');
    const actionTextEl = document.getElementById('action-text');
    const actionSubtextEl = document.getElementById('action-subtext');
    
    actionTextEl.textContent = data.action;
    actionEl.className = `action-signal ${data.action.toLowerCase()}-signal`;
    
    let subtext = '';
    if (data.action === 'BUY') {
        subtext = 'Strong upward movement predicted';
    } else if (data.action === 'SELL') {
        subtext = 'Strong downward movement predicted';
    } else {
        subtext = 'Market conditions uncertain, waiting for clearer signal';
    }
    actionSubtextEl.textContent = subtext;
    
    // Update indicators
    if (data.indicators) {
        for (const [key, value] of Object.entries(data.indicators)) {
            const el = document.getElementById(`indicator-${key.toLowerCase()}`);
            if (el) {
                el.textContent = value;
            }
        }
    }
    
    // Update chart
    if (data.chart_data) {
        try {
            const chartData = JSON.parse(data.chart_data);
            Plotly.react('price-chart', chartData.data, chartData.layout);
        } catch (e) {
            console.error('Error updating chart:', e);
        }
    }
    
    // Update timestamp
    if (data.timestamp) {
        document.getElementById('last-updated').textContent = 
            `Last updated: ${formatTime(data.timestamp)}`;
        lastUpdateTime = new Date(data.timestamp);
    }
    
    // Show/hide loading
    document.getElementById('loading').style.display = 'none';
    document.getElementById('content').style.display = 'block';
}

function fetchPredictions() {
    document.getElementById('loading').style.display = 'block';
    
    fetch('/api/predictions')
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            updatePageData(data);
            
            // Show success message if data is fresh
            const now = new Date();
            const dataTime = new Date(data.timestamp);
            const diffMinutes = (now - dataTime) / (1000 * 60);
            
            if (diffMinutes < 2) {
                showAlert('Data updated successfully!', 'success');
            }
        })
        .catch(error => {
            console.error('Error fetching predictions:', error);
            showAlert('Error updating data. Retrying...', 'warning');
            document.getElementById('loading').style.display = 'none';
            
            // Retry after 5 seconds
            setTimeout(fetchPredictions, 5000);
        });
}

function showAlert(message, type) {
    const alertEl = document.getElementById('alert');
    alertEl.textContent = message;
    alertEl.className = `alert alert-${type}`;
    alertEl.style.display = 'block';
    
    setTimeout(() => {
        alertEl.style.display = 'none';
    }, 5000);
}

function startAutoRefresh() {
    // Clear existing interval
    if (autoRefreshInterval) {
        clearInterval(autoRefreshInterval);
    }
    
    // Fetch immediately
    fetchPredictions();
    
    // Set up interval for every 60 seconds
    autoRefreshInterval = setInterval(fetchPredictions, 60000);
    
    // Update countdown every second
    setInterval(updateCountdown, 1000);
}

function updateCountdown() {
    if (!lastUpdateTime) return;
    
    const now = new Date();
    const nextUpdate = new Date(lastUpdateTime.getTime() + 60000);
    const secondsLeft = Math.max(0, Math.floor((nextUpdate - now) / 1000));
    
    document.getElementById('countdown').textContent = 
        `Next update in: ${secondsLeft} seconds`;
}

function checkHealth() {
    fetch('/api/health')
        .then(response => response.json())
        .then(data => {
            console.log('Health check:', data);
        })
        .catch(error => {
            console.error('Health check failed:', error);
        });
}

// Initialize when page loads
document.addEventListener('DOMContentLoaded', function() {
    // Start auto-refresh
    startAutoRefresh();
    
    // Initial health check
    checkHealth();
    
    // Set up manual refresh button
    document.getElementById('refresh-btn').addEventListener('click', function() {
        fetchPredictions();
        showAlert('Manual refresh initiated...', 'success');
    });
    
    // Check connection every 30 seconds
    setInterval(checkHealth, 30000);
});