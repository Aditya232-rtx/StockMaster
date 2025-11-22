// Global state
let alerts = [];
let filteredAlerts = [];
let charts = {};
let socket = null;

// API Configuration
const API_BASE_URL = 'http://127.0.0.1:8100';

// Risk level colors
const RISK_COLORS = {
    CRITICAL: 'border-red-500 bg-red-900',
    HIGH: 'border-orange-500 bg-orange-900',
    MEDIUM: 'border-yellow-500 bg-yellow-900',
    LOW: 'border-blue-500 bg-blue-900'
};

const RISK_BADGE_COLORS = {
    CRITICAL: 'bg-red-600 text-white',
    HIGH: 'bg-orange-600 text-white',
    MEDIUM: 'bg-yellow-600 text-black',
    LOW: 'bg-blue-600 text-white'
};

// Initialize application
document.addEventListener('DOMContentLoaded', function() {
    initializeCharts();
    loadInitialData();
    setupEventListeners();
    initializeWebSocket();
});

// Initialize WebSocket connection
function initializeWebSocket() {
    // Note: WebSocket would need to be implemented on the backend
    // For now, we'll simulate real-time updates with polling
    setInterval(() => {
        fetchLatestAlerts();
    }, 5000); // Poll every 5 seconds
}

// Setup event listeners
function setupEventListeners() {
    document.getElementById('filter-risk').addEventListener('change', filterAlerts);
    document.getElementById('bulk-actions').addEventListener('click', showBulkActions);
    document.getElementById('close-modal').addEventListener('click', closeModal);
    
    // Close modal when clicking outside
    document.getElementById('alert-modal').addEventListener('click', function(e) {
        if (e.target === this) {
            closeModal();
        }
    });
}

// Load initial data from API
async function loadInitialData() {
    try {
        const response = await fetch(`${API_BASE_URL}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                contamination: 0.05,
                n_estimators: 100,
                random_state: 42,
                top_n: 50
            })
        });
        
        if (!response.ok) {
            throw new Error('Failed to fetch data');
        }
        
        const data = await response.json();
        
        // Transform API data to alert format
        alerts = data.top_anomalies.map((item, index) => ({
            id: `alert-${index}`,
            user: `User_${index % 10}`, // Simulated user
            product: item.product_sku || 'Unknown',
            adjustment: item.quantity_change || 0,
            value: item.anomaly_score * 1000, // Simulated value
            riskLevel: item.risk_level || 'MEDIUM',
            timestamp: item.date || new Date().toISOString(),
            location: item.location || 'Warehouse',
            transactionType: item.transaction_type || 'adjustment',
            status: 'pending',
            action: 'Supervisor review triggered'
        }));
        
        filteredAlerts = [...alerts];
        renderAlerts();
        updateStats();
        updateCharts();
        updateTopUsers();
        
    } catch (error) {
        console.error('Error loading initial data:', error);
        // Load demo data if API fails
        loadDemoData();
    }
}

// Load demo data for testing
function loadDemoData() {
    const demoAlerts = [
        {
            id: 'alert-1',
            user: 'John_Doe',
            product: 'iPhone_14',
            adjustment: -15,
            value: 10500,
            riskLevel: 'CRITICAL',
            timestamp: new Date().toISOString(),
            location: 'Warehouse_A',
            transactionType: 'adjustment',
            status: 'pending',
            action: 'Supervisor review triggered'
        },
        {
            id: 'alert-2',
            user: 'Jane_Smith',
            product: 'MacBook_Pro',
            adjustment: -5,
            value: 7500,
            riskLevel: 'HIGH',
            timestamp: new Date(Date.now() - 300000).toISOString(),
            location: 'Warehouse_B',
            transactionType: 'adjustment',
            status: 'pending',
            action: 'Manager review required'
        },
        {
            id: 'alert-3',
            user: 'Mike_Johnson',
            product: 'iPad_Air',
            adjustment: -8,
            value: 3200,
            riskLevel: 'MEDIUM',
            timestamp: new Date(Date.now() - 600000).toISOString(),
            location: 'Warehouse_A',
            transactionType: 'adjustment',
            status: 'investigated',
            action: 'Under investigation'
        }
    ];
    
    alerts = demoAlerts;
    filteredAlerts = [...alerts];
    renderAlerts();
    updateStats();
    updateCharts();
    updateTopUsers();
}

// Fetch latest alerts
async function fetchLatestAlerts() {
    try {
        // Simulate new alert
        const newAlert = {
            id: `alert-${Date.now()}`,
            user: `User_${Math.floor(Math.random() * 10)}`,
            product: `Product_${Math.floor(Math.random() * 100)}`,
            adjustment: Math.floor(Math.random() * 20) - 10,
            value: Math.floor(Math.random() * 10000),
            riskLevel: ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW'][Math.floor(Math.random() * 4)],
            timestamp: new Date().toISOString(),
            location: `Warehouse_${String.fromCharCode(65 + Math.floor(Math.random() * 3))}`,
            transactionType: 'adjustment',
            status: 'pending',
            action: 'Supervisor review triggered'
        };
        
        alerts.unshift(newAlert);
        filteredAlerts = [...alerts];
        renderAlerts();
        updateStats();
        updateCharts();
        updateTopUsers();
        
    } catch (error) {
        console.error('Error fetching latest alerts:', error);
    }
}

// Render alerts in the feed
function renderAlerts() {
    const feedContainer = document.getElementById('alert-feed');
    feedContainer.innerHTML = '';
    
    filteredAlerts.forEach((alert, index) => {
        const alertCard = createAlertCard(alert);
        alertCard.classList.add('alert-slide-in');
        feedContainer.appendChild(alertCard);
    });
}

// Create alert card element
function createAlertCard(alert) {
    const card = document.createElement('div');
    card.className = `p-4 border-l-4 ${RISK_COLORS[alert.riskLevel]} mb-4 cursor-pointer hover:bg-gray-700 transition-colors`;
    card.onclick = () => showAlertDetails(alert);
    
    card.innerHTML = `
        <div class="flex items-start justify-between">
            <div class="flex-1">
                <div class="flex items-center space-x-2 mb-2">
                    <span class="text-lg font-bold">🚨 ANOMALY DETECTED!</span>
                    <span class="px-2 py-1 rounded text-xs font-semibold ${RISK_BADGE_COLORS[alert.riskLevel]}">
                        ${alert.riskLevel}
                    </span>
                </div>
                <div class="text-sm space-y-1">
                    <div class="flex items-center space-x-4">
                        <span><strong>User:</strong> ${alert.user}</span>
                        <span><strong>Product:</strong> ${alert.product}</span>
                    </div>
                    <div class="flex items-center space-x-4">
                        <span><strong>Adjustment:</strong> ${alert.adjustment} units</span>
                        <span><strong>Value:</strong> $${alert.value.toLocaleString()}</span>
                    </div>
                    <div class="flex items-center space-x-4">
                        <span><strong>Location:</strong> ${alert.location}</span>
                        <span><strong>Time:</strong> ${new Date(alert.timestamp).toLocaleTimeString()}</span>
                    </div>
                    <div class="mt-2">
                        <span class="text-yellow-400"><strong>Action:</strong> ${alert.action}</span>
                    </div>
                </div>
            </div>
            <div class="ml-4 flex flex-col space-y-2">
                <button onclick="event.stopPropagation(); handleAlertAction('${alert.id}', 'approve')" class="bg-green-600 hover:bg-green-700 px-3 py-1 rounded text-sm">
                    Approve
                </button>
                <button onclick="event.stopPropagation(); handleAlertAction('${alert.id}', 'reject')" class="bg-red-600 hover:bg-red-700 px-3 py-1 rounded text-sm">
                    Reject
                </button>
                <button onclick="event.stopPropagation(); handleAlertAction('${alert.id}', 'investigate')" class="bg-blue-600 hover:bg-blue-700 px-3 py-1 rounded text-sm">
                    Investigate
                </button>
            </div>
        </div>
    `;
    
    return card;
}

// Handle alert actions
function handleAlertAction(alertId, action) {
    const alert = alerts.find(a => a.id === alertId);
    if (alert) {
        alert.status = action === 'investigate' ? 'investigated' : action;
        renderAlerts();
        updateStats();
        
        // Show notification
        showNotification(`Alert ${action}d successfully`, 'success');
    }
}

// Show alert details modal
function showAlertDetails(alert) {
    const modal = document.getElementById('alert-modal');
    const modalContent = document.getElementById('modal-content');
    
    modalContent.innerHTML = `
        <div class="space-y-6">
            <div class="bg-gray-700 rounded-lg p-4">
                <h4 class="font-semibold mb-3">Alert Information</h4>
                <div class="grid grid-cols-2 gap-4 text-sm">
                    <div><strong>Alert ID:</strong> ${alert.id}</div>
                    <div><strong>Risk Level:</strong> <span class="px-2 py-1 rounded text-xs ${RISK_BADGE_COLORS[alert.riskLevel]}">${alert.riskLevel}</span></div>
                    <div><strong>User:</strong> ${alert.user}</div>
                    <div><strong>Product:</strong> ${alert.product}</div>
                    <div><strong>Adjustment:</strong> ${alert.adjustment} units</div>
                    <div><strong>Value:</strong> $${alert.value.toLocaleString()}</div>
                    <div><strong>Location:</strong> ${alert.location}</div>
                    <div><strong>Transaction Type:</strong> ${alert.transactionType}</div>
                    <div><strong>Timestamp:</strong> ${new Date(alert.timestamp).toLocaleString()}</div>
                    <div><strong>Status:</strong> ${alert.status}</div>
                </div>
            </div>
            
            <div class="bg-gray-700 rounded-lg p-4">
                <h4 class="font-semibold mb-3">User History & Patterns</h4>
                <div class="text-sm text-gray-300">
                    <p>User ${alert.user} has had ${Math.floor(Math.random() * 10) + 1} previous alerts in the last 30 days.</p>
                    <p>Typical adjustment range: -5 to +5 units</p>
                    <p>This adjustment is ${Math.abs(alert.adjustment) > 10 ? 'unusual' : 'within normal range'} for this user.</p>
                </div>
            </div>
            
            <div class="bg-gray-700 rounded-lg p-4">
                <h4 class="font-semibold mb-3">Similar Past Incidents</h4>
                <div class="space-y-2 text-sm">
                    ${generateSimilarIncidents(alert)}
                </div>
            </div>
            
            <div class="bg-gray-700 rounded-lg p-4">
                <h4 class="font-semibold mb-3">Action Audit Trail</h4>
                <div class="space-y-2 text-sm">
                    <div class="flex justify-between">
                        <span>${new Date(alert.timestamp).toLocaleString()}</span>
                        <span>Alert detected</span>
                    </div>
                    <div class="flex justify-between">
                        <span>${new Date().toLocaleString()}</span>
                        <span>Viewed by supervisor</span>
                    </div>
                </div>
            </div>
            
            <div class="flex justify-end space-x-3">
                <button onclick="handleAlertAction('${alert.id}', 'approve'); closeModal();" class="bg-green-600 hover:bg-green-700 px-4 py-2 rounded">
                    Approve
                </button>
                <button onclick="handleAlertAction('${alert.id}', 'reject'); closeModal();" class="bg-red-600 hover:bg-red-700 px-4 py-2 rounded">
                    Reject
                </button>
                <button onclick="handleAlertAction('${alert.id}', 'investigate'); closeModal();" class="bg-blue-600 hover:bg-blue-700 px-4 py-2 rounded">
                    Mark as Investigated
                </button>
            </div>
        </div>
    `;
    
    modal.classList.remove('hidden');
}

// Generate similar incidents
function generateSimilarIncidents(alert) {
    const incidents = [];
    for (let i = 0; i < 3; i++) {
        const date = new Date(Date.now() - Math.random() * 30 * 24 * 60 * 60 * 1000);
        incidents.push(`
            <div class="flex justify-between">
                <span>${date.toLocaleDateString()}</span>
                <span>Similar adjustment on ${alert.product}</span>
            </div>
        `);
    }
    return incidents.join('');
}

// Close modal
function closeModal() {
    document.getElementById('alert-modal').classList.add('hidden');
}

// Filter alerts
function filterAlerts() {
    const riskLevel = document.getElementById('filter-risk').value;
    
    if (riskLevel) {
        filteredAlerts = alerts.filter(alert => alert.riskLevel === riskLevel);
    } else {
        filteredAlerts = [...alerts];
    }
    
    renderAlerts();
}

// Show bulk actions
function showBulkActions() {
    showNotification('Bulk actions feature coming soon', 'info');
}

// Update statistics
function updateStats() {
    const stats = {
        total: alerts.length,
        critical: alerts.filter(a => a.riskLevel === 'CRITICAL').length,
        high: alerts.filter(a => a.riskLevel === 'HIGH').length,
        pending: alerts.filter(a => a.status === 'pending').length
    };
    
    document.getElementById('total-alerts').textContent = stats.total;
    document.getElementById('critical-alerts').textContent = stats.critical;
    document.getElementById('high-alerts').textContent = stats.high;
    document.getElementById('pending-alerts').textContent = stats.pending;
}

// Initialize charts
function initializeCharts() {
    // Risk Distribution Pie Chart
    const riskCtx = document.getElementById('risk-chart').getContext('2d');
    charts.risk = new Chart(riskCtx, {
        type: 'doughnut',
        data: {
            labels: ['Critical', 'High', 'Medium', 'Low'],
            datasets: [{
                data: [0, 0, 0, 0],
                backgroundColor: ['#dc2626', '#ea580c', '#ca8a04', '#2563eb'],
                borderWidth: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: { color: '#fff' }
                }
            }
        }
    });
    
    // Trends Line Chart
    const trendsCtx = document.getElementById('trends-chart').getContext('2d');
    charts.trends = new Chart(trendsCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Alerts',
                data: [],
                borderColor: '#dc2626',
                backgroundColor: 'rgba(220, 38, 38, 0.1)',
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    ticks: { color: '#fff' },
                    grid: { color: '#374151' }
                },
                y: {
                    ticks: { color: '#fff' },
                    grid: { color: '#374151' }
                }
            },
            plugins: {
                legend: {
                    labels: { color: '#fff' }
                }
            }
        }
    });
}

// Update charts with current data
function updateCharts() {
    // Update risk distribution
    const riskCounts = {
        CRITICAL: alerts.filter(a => a.riskLevel === 'CRITICAL').length,
        HIGH: alerts.filter(a => a.riskLevel === 'HIGH').length,
        MEDIUM: alerts.filter(a => a.riskLevel === 'MEDIUM').length,
        LOW: alerts.filter(a => a.riskLevel === 'LOW').length
    };
    
    charts.risk.data.datasets[0].data = Object.values(riskCounts);
    charts.risk.update();
    
    // Update trends (last 24 hours)
    const now = new Date();
    const labels = [];
    const data = [];
    
    for (let i = 23; i >= 0; i--) {
        const hour = new Date(now - i * 60 * 60 * 1000);
        labels.push(hour.getHours() + ':00');
        data.push(Math.floor(Math.random() * 10)); // Simulated data
    }
    
    charts.trends.data.labels = labels;
    charts.trends.data.datasets[0].data = data;
    charts.trends.update();
}

// Update top users
function updateTopUsers() {
    const userCounts = {};
    alerts.forEach(alert => {
        userCounts[alert.user] = (userCounts[alert.user] || 0) + 1;
    });
    
    const topUsers = Object.entries(userCounts)
        .sort(([,a], [,b]) => b - a)
        .slice(0, 5);
    
    const container = document.getElementById('top-users');
    container.innerHTML = topUsers.map(([user, count]) => `
        <div class="flex justify-between items-center">
            <span class="text-sm">${user}</span>
            <span class="bg-gray-700 px-2 py-1 rounded text-xs">${count} alerts</span>
        </div>
    `).join('');
}

// Show notification
function showNotification(message, type = 'info') {
    const colors = {
        success: 'bg-green-600',
        error: 'bg-red-600',
        info: 'bg-blue-600',
        warning: 'bg-yellow-600'
    };
    
    const notification = document.createElement('div');
    notification.className = `fixed top-4 right-4 ${colors[type]} text-white px-4 py-2 rounded-lg shadow-lg z-50 alert-slide-in`;
    notification.textContent = message;
    
    document.body.appendChild(notification);
    
    setTimeout(() => {
        notification.remove();
    }, 3000);
}
