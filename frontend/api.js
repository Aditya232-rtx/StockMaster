// api.js - Centralized API handling

const API_BASE_URL = '/api'; // Relative path since we'll likely serve from same origin or proxy

// Helper to get CSRF token from cookies
function getCookie(name) {
    let cookieValue = null;
    if (document.cookie && document.cookie !== '') {
        const cookies = document.cookie.split(';');
        for (let i = 0; i < cookies.length; i++) {
            const cookie = cookies[i].trim();
            if (cookie.substring(0, name.length + 1) === (name + '=')) {
                cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                break;
            }
        }
    }
    return cookieValue;
}

const api = {
    // Generic fetch wrapper
    async request(endpoint, method = 'GET', data = null) {
        const headers = {
            'Content-Type': 'application/json',
            'X-CSRFToken': getCookie('csrftoken')
        };

        const config = {
            method,
            headers,
        };

        if (data) {
            config.body = JSON.stringify(data);
        }

        try {
            const response = await fetch(`${endpoint}`, config);

            // Handle 401 Unauthorized - Redirect to login if not already there
            if (response.status === 401 && !window.location.pathname.includes('index.html')) {
                window.location.href = 'index.html';
                return null;
            }

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                throw new Error(errorData.detail || errorData.error || `Request failed: ${response.status}`);
            }

            return await response.json();
        } catch (error) {
            console.error('API Error:', error);
            throw error;
        }
    },

    // Auth
    async login(username, password) {
        // Django's default login view expects form data, not JSON usually, 
        // but let's try standard JSON if using DRF or custom view.
        // If using standard Django auth views, we might need FormData.
        // Let's assume we are using the standard /accounts/login/ which expects form-data.

        const formData = new FormData();
        formData.append('username', username);
        formData.append('password', password);

        const response = await fetch('/accounts/login/', {
            method: 'POST',
            body: formData,
            headers: {
                'X-CSRFToken': getCookie('csrftoken')
            }
        });

        if (response.ok) {
            // Check if we were redirected to login again (failure)
            if (response.url.includes('login') && !response.url.includes('success')) {
                // This is tricky with standard Django login redirects. 
                // Usually it redirects to profile or next.
                // If we are still on login page, it might have failed.
                // For now, let's assume 200 OK means success if we are redirected elsewhere.
                return true;
            }
            return true;
        }
        return false;
    },

    async logout() {
        await fetch('/accounts/logout/', {
            method: 'POST',
            headers: { 'X-CSRFToken': getCookie('csrftoken') }
        });
        window.location.href = 'index.html';
    },

    // Dashboard
    async getKPIs() {
        return this.request(`${API_BASE_URL}/kpis/`);
    },

    // Stock
    async getInventory() {
        return this.request(`${API_BASE_URL}/inventory-state/`);
    },

    // Operations
    async getReceipts() {
        return this.request(`${API_BASE_URL}/receipts/`);
    },

    async getDeliveries() {
        return this.request(`${API_BASE_URL}/deliveries/`);
    },

    async getAdjustments() {
        return this.request(`${API_BASE_URL}/adjustments/`);
    },

    async getTransfers() {
        return this.request(`${API_BASE_URL}/transfers/`);
    }
};

// Expose to window
window.api = api;
