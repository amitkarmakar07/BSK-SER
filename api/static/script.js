// API Base URL - Change this to your deployed URL
const API_BASE_URL = window.location.origin;

// Global variables
let currentMode = 'phone';
let currentCitizenId = null;
let allServices = [];
let allDistricts = [];

// Initialize the app
document.addEventListener('DOMContentLoaded', async () => {
    await loadInitialData();
});

// Load initial data (services and districts)
async function loadInitialData() {
    try {
        // Load services
        const servicesResponse = await fetch(`${API_BASE_URL}/api/services`);
        const servicesData = await servicesResponse.json();
        allServices = servicesData.services;
        
        // Populate service dropdowns
        populateServiceDropdown('serviceSelect', allServices);
        populateServiceDropdown('manualServiceSelect', allServices);
        
        // Load districts
        const districtsResponse = await fetch(`${API_BASE_URL}/api/districts`);
        const districtsData = await districtsResponse.json();
        allDistricts = districtsData.districts;
        
        // Populate district dropdown
        populateDistrictDropdown('districtSelect', allDistricts);
        
    } catch (error) {
        console.error('Error loading initial data:', error);
        showError('Failed to load initial data. Please refresh the page.');
    }
}

// Populate service dropdown
function populateServiceDropdown(elementId, services) {
    const select = document.getElementById(elementId);
    select.innerHTML = '<option value="">-- Select a Service --</option>';
    
    services.forEach(service => {
        const option = document.createElement('option');
        option.value = service.service_id;
        option.textContent = `${service.service_id} - ${service.service_name}`;
        select.appendChild(option);
    });
}

// Populate district dropdown
function populateDistrictDropdown(elementId, districts) {
    const select = document.getElementById(elementId);
    select.innerHTML = '<option value="">-- Select District --</option>';
    
    districts.forEach(district => {
        const option = document.createElement('option');
        option.value = district.district_id;
        option.textContent = district.district_name;
        select.appendChild(option);
    });
}

// Switch between phone and manual mode
function switchMode(mode) {
    currentMode = mode;
    
    const phoneMode = document.getElementById('phoneMode');
    const manualMode = document.getElementById('manualMode');
    const phoneBtn = document.getElementById('phoneBtn');
    const manualBtn = document.getElementById('manualBtn');
    const recommendationsSection = document.getElementById('recommendationsSection');
    
    if (mode === 'phone') {
        phoneMode.style.display = 'block';
        manualMode.style.display = 'none';
        phoneBtn.classList.add('active');
        manualBtn.classList.remove('active');
    } else {
        phoneMode.style.display = 'none';
        manualMode.style.display = 'block';
        phoneBtn.classList.remove('active');
        manualBtn.classList.add('active');
    }
    
    // Hide recommendations when switching modes
    recommendationsSection.style.display = 'none';
    hideError();
}

// Search citizen by phone number
async function searchByPhone() {
    const phoneInput = document.getElementById('phoneInput');
    const phone = phoneInput.value.trim();
    
    if (!phone) {
        showError('Please enter a phone number');
        return;
    }
    
    showLoading(true);
    hideError();
    
    try {
        const response = await fetch(`${API_BASE_URL}/api/citizen/phone/${phone}`);
        
        if (!response.ok) {
            throw new Error('Citizen not found');
        }
        
        const data = await response.json();
        
        if (data.citizens && data.citizens.length > 0) {
            const citizen = data.citizens[0]; // Take first citizen
            currentCitizenId = citizen.citizen_id;
            
            // Display citizen details
            displayCitizenInfo(citizen);
            
            // Load services used by citizen
            await loadCitizenServices(citizen.citizen_id);
            
            // Show citizen details section
            document.getElementById('citizenDetails').style.display = 'block';
            document.getElementById('recommendationsSection').style.display = 'none';
        }
        
    } catch (error) {
        console.error('Error searching citizen:', error);
        showError('No citizen found with this phone number. Please try another number or use Manual Entry mode.');
        document.getElementById('citizenDetails').style.display = 'none';
    } finally {
        showLoading(false);
    }
}

// Display citizen information
function displayCitizenInfo(citizen) {
    const citizenInfo = document.getElementById('citizenInfo');
    
    citizenInfo.innerHTML = `
        <div class="info-item">
            <div class="info-label">Name</div>
            <div class="info-value">${citizen.name || '--'}</div>
        </div>
        <div class="info-item">
            <div class="info-label">Gender</div>
            <div class="info-value">${citizen.gender || '--'}</div>
        </div>
        <div class="info-item">
            <div class="info-label">Age</div>
            <div class="info-value">${citizen.age !== null ? citizen.age : '--'}</div>
        </div>
        <div class="info-item">
            <div class="info-label">Caste</div>
            <div class="info-value">${citizen.caste || '--'}</div>
        </div>
        <div class="info-item">
            <div class="info-label">Religion</div>
            <div class="info-value">${citizen.religion || '--'}</div>
        </div>
    `;
}

// Load services used by citizen
async function loadCitizenServices(citizenId) {
    try {
        const response = await fetch(`${API_BASE_URL}/api/citizen/${citizenId}/services`);
        const data = await response.json();
        
        const servicesUsed = document.getElementById('servicesUsed');
        
        if (data.services && data.services.length > 0) {
            let tableHTML = `
                <p class="services-count">Total Unique Services Used: <strong>${data.total_count}</strong></p>
                <table class="service-table">
                    <thead>
                        <tr>
                            <th>Service ID</th>
                            <th>Service Name</th>
                            <th>Times Used</th>
                        </tr>
                    </thead>
                    <tbody>
            `;
            
            data.services.slice(0, 20).forEach(service => {
                tableHTML += `
                    <tr>
                        <td>${service.service_id}</td>
                        <td>${service.service_name}</td>
                        <td>${service.count}</td>
                    </tr>
                `;
            });
            
            tableHTML += `
                    </tbody>
                </table>
            `;
            
            if (data.services.length > 20) {
                tableHTML += `<p class="hint">Showing top 20 of ${data.services.length} services</p>`;
            }
            
            servicesUsed.innerHTML = tableHTML;
        } else {
            servicesUsed.innerHTML = '<p class="no-recommendations">No services found for this citizen.</p>';
        }
        
    } catch (error) {
        console.error('Error loading citizen services:', error);
        servicesUsed.innerHTML = '<p class="no-recommendations">Unable to load services.</p>';
    }
}

// Get recommendations
async function getRecommendations(mode) {
    hideError();
    showLoading(true);
    
    try {
        let response;
        
        if (mode === 'phone') {
            const serviceId = document.getElementById('serviceSelect').value;
            
            if (!currentCitizenId) {
                showError('Please search for a citizen first');
                showLoading(false);
                return;
            }
            
            response = await fetch(`${API_BASE_URL}/api/recommend/phone`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    citizen_id: currentCitizenId,
                    selected_service_id: serviceId ? parseInt(serviceId) : null
                })
            });
            
        } else {
            // Manual mode
            const districtId = document.getElementById('districtSelect').value;
            const gender = document.getElementById('genderSelect').value;
            const age = document.getElementById('ageInput').value;
            const caste = document.getElementById('casteSelect').value;
            const religion = document.getElementById('religionSelect').value;
            const serviceId = document.getElementById('manualServiceSelect').value;
            
            // Validate inputs
            if (!districtId || !gender || !age || !caste || !religion || !serviceId) {
                showError('Please fill in all required fields');
                showLoading(false);
                return;
            }
            
            response = await fetch(`${API_BASE_URL}/api/recommend/manual`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    district_id: parseInt(districtId),
                    gender: gender,
                    caste: caste,
                    age: parseInt(age),
                    religion: religion,
                    selected_service_id: parseInt(serviceId)
                })
            });
        }
        
        if (!response.ok) {
            throw new Error('Failed to get recommendations');
        }
        
        const data = await response.json();
        displayRecommendations(data);
        
        // Scroll to recommendations
        document.getElementById('recommendationsSection').scrollIntoView({ 
            behavior: 'smooth', 
            block: 'start' 
        });
        
    } catch (error) {
        console.error('Error getting recommendations:', error);
        showError('Failed to get recommendations. Please try again.');
    } finally {
        showLoading(false);
    }
}

// Display recommendations
function displayRecommendations(data) {
    const recommendationsSection = document.getElementById('recommendationsSection');
    recommendationsSection.style.display = 'block';
    recommendationsSection.classList.add('fade-in');
    
    // District Recommendations
    const districtRecs = document.getElementById('districtRecs');
    if (data.district_recommendations && data.district_recommendations.length > 0) {
        districtRecs.innerHTML = `
            <ul class="recommendation-list">
                ${data.district_recommendations.map(service => `<li>${service}</li>`).join('')}
            </ul>
        `;
    } else {
        districtRecs.innerHTML = '<p class="no-recommendations">No district recommendations available</p>';
    }
    
    // Demographic Recommendations
    const demographicRecs = document.getElementById('demographicRecs');
    if (data.demographic_recommendations && data.demographic_recommendations.length > 0) {
        demographicRecs.innerHTML = `
            <ul class="recommendation-list">
                ${data.demographic_recommendations.map(service => `<li>${service}</li>`).join('')}
            </ul>
        `;
    } else {
        demographicRecs.innerHTML = '<p class="no-recommendations">No demographic recommendations available</p>';
    }
    
    // Content-based Recommendations
    const contentRecs = document.getElementById('contentRecs');
    if (data.content_recommendations && Object.keys(data.content_recommendations).length > 0) {
        let contentHTML = '';
        for (const [serviceName, recommendations] of Object.entries(data.content_recommendations)) {
            contentHTML += `
                <div class="service-group">
                    <div class="service-group-title">${serviceName}</div>
                    <ul class="recommendation-list">
                        ${recommendations.map(service => `<li>${service}</li>`).join('')}
                    </ul>
                </div>
            `;
        }
        contentRecs.innerHTML = contentHTML;
    } else {
        contentRecs.innerHTML = '<p class="no-recommendations">No content-based recommendations available</p>';
    }
}

// Show/hide loading spinner
function showLoading(show) {
    const spinner = document.getElementById('loadingSpinner');
    spinner.style.display = show ? 'block' : 'none';
}

// Show error message
function showError(message) {
    const errorDiv = document.getElementById('errorMessage');
    errorDiv.textContent = message;
    errorDiv.style.display = 'block';
    errorDiv.classList.add('fade-in');
    
    // Auto-hide after 5 seconds
    setTimeout(() => {
        hideError();
    }, 5000);
}

// Hide error message
function hideError() {
    const errorDiv = document.getElementById('errorMessage');
    errorDiv.style.display = 'none';
}
