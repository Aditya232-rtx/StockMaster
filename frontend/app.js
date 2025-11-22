const API_BASE = ""; // same origin

const productForm = document.querySelector("#productForm");
const vendorForm = document.querySelector("#vendorForm");
const productIdInput = document.querySelector("#productId");
const demandResult = document.querySelector("#demandResult");
const priceResult = document.querySelector("#priceResult");
const recommendationResult = document.querySelector("#recommendationResult");
const healthStatus = document.querySelector("#healthStatus");
const inputTemplate = document.querySelector("#inputTemplate");

const predictDemandBtn = document.querySelector("#predictDemand");
const predictPriceBtn = document.querySelector("#predictPrice");
const recommendationBtn = document.querySelector("#getRecommendation");
const loadSampleBtn = document.querySelector("#loadSample");

let cachedSampleProduct = {};
let cachedSampleVendor = {};

function setHealthStatus(text, state = "ok") {
    healthStatus.textContent = text;
    healthStatus.className = `chip chip--${state}`;
}

async function checkHealth() {
    try {
        const res = await fetch(`${API_BASE}/api/health`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        setHealthStatus("Online", "success");
    } catch (err) {
        setHealthStatus("Offline", "error");
        console.error("Health check failed", err);
    }
}

function renderFeatureInputs(form, data) {
    form.innerHTML = "";
    const entries = Object.entries(data);
    if (entries.length === 0) {
        form.innerHTML = "<p class='muted'>No features available.</p>";
        return;
    }
    entries.forEach(([key, value]) => {
        const node = inputTemplate.content.cloneNode(true);
        const label = node.querySelector(".input-group__label");
        const input = node.querySelector("input");

        label.textContent = key;
        input.value = value ?? "";
        input.dataset.key = key;

        form.appendChild(node);
    });
}

function getFormValues(form) {
    const inputs = form.querySelectorAll("input[data-key]");
    const features = {};
    inputs.forEach((input) => {
        const key = input.dataset.key;
        const value = input.value;
        if (value === "" || value === null || value === undefined) {
            throw new Error(`Feature '${key}' is required.`);
        }
        const numeric = Number(value);
        if (Number.isNaN(numeric)) {
            throw new Error(`Feature '${key}' must be numeric.`);
        }
        features[key] = numeric;
    });
    return features;
}

function formatJson(content) {
    return JSON.stringify(content, null, 2);
}

function setBusy(button, busy) {
    button.disabled = busy;
    button.classList.toggle("is-busy", busy);
}

async function loadSampleData() {
    setBusy(loadSampleBtn, true);
    try {
        const res = await fetch(`${API_BASE}/api/sample-data`);
        const payload = await res.json();
        if (payload.status !== "success") {
            throw new Error(payload.message || "Failed to load sample data");
        }
        cachedSampleProduct = payload.product_features;
        cachedSampleVendor = payload.vendor_features;
        renderFeatureInputs(productForm, cachedSampleProduct);
        renderFeatureInputs(vendorForm, cachedSampleVendor);
    } catch (err) {
        console.error(err);
        alert(err.message || "Unable to load sample data.");
    } finally {
        setBusy(loadSampleBtn, false);
    }
}

async function callApi(endpoint, body) {
    const res = await fetch(`${API_BASE}${endpoint}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
    });
    const payload = await res.json();
    if (!res.ok || payload.status !== "success") {
        throw new Error(payload.message || "Request failed");
    }
    return payload;
}

predictDemandBtn.addEventListener("click", async () => {
    try {
        setBusy(predictDemandBtn, true);
        const productFeatures = getFormValues(productForm);
        const payload = await callApi("/api/predict/demand", { product_features: productFeatures });
        demandResult.textContent = `Predicted Demand: ${payload.prediction.toFixed(2)}`;
    } catch (err) {
        demandResult.textContent = `Error: ${err.message}`;
    } finally {
        setBusy(predictDemandBtn, false);
    }
});

predictPriceBtn.addEventListener("click", async () => {
    try {
        setBusy(predictPriceBtn, true);
        const vendorFeatures = getFormValues(vendorForm);
        const payload = await callApi("/api/predict/vendor-price", { vendor_features: vendorFeatures });
        priceResult.textContent = `Predicted Vendor Price: ${payload.prediction.toFixed(2)}`;
    } catch (err) {
        priceResult.textContent = `Error: ${err.message}`;
    } finally {
        setBusy(predictPriceBtn, false);
    }
});

recommendationBtn.addEventListener("click", async () => {
    try {
        setBusy(recommendationBtn, true);
        const productFeatures = getFormValues(productForm);
        const vendorFeatures = getFormValues(vendorForm);
        const productId = productIdInput.value.trim() || "UNKNOWN";
        const payload = await callApi("/api/recommendation", {
            product_id: productId,
            product_features: productFeatures,
            vendor_features: vendorFeatures,
        });
        recommendationResult.textContent = formatJson(payload.recommendation);
    } catch (err) {
        recommendationResult.textContent = `Error: ${err.message}`;
    } finally {
        setBusy(recommendationBtn, false);
    }
});

loadSampleBtn.addEventListener("click", loadSampleData);

window.addEventListener("load", async () => {
    await Promise.all([checkHealth(), loadSampleData()]);
});
