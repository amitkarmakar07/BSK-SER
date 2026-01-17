# Bangla Sahayata Kendra - API Deployment Guide

## Overview
This is the FastAPI backend and HTML/CSS/JS frontend for the Bangla Sahayata Kendra Service Recommendation System.

## Project Structure
```
api/
├── main.py                 # FastAPI application
├── requirements.txt        # Python dependencies
├── Procfile               # For Heroku/Railway deployment
├── runtime.txt            # Python version
├── static/                # Frontend files
│   ├── index.html        # Main HTML page
│   ├── styles.css        # Styling
│   └── script.js         # JavaScript logic
└── README.md             # This file
```

## Local Development

### Prerequisites
- Python 3.8 or higher
- Virtual environment (recommended)

### Setup Instructions

1. **Install Dependencies**
   ```bash
   cd api
   pip install -r requirements.txt
   ```

2. **Run the Application**
   ```bash
   python main.py
   ```
   
   Or using uvicorn directly:
   ```bash
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

3. **Access the Application**
   - Frontend: http://localhost:8000
   - API Documentation: http://localhost:8000/docs
   - Alternative API Docs: http://localhost:8000/redoc

## API Endpoints

### Health Check
- **GET** `/api/health` - Check if API is running

### Districts & Services
- **GET** `/api/districts` - Get all districts
- **GET** `/api/services` - Get all services

### Citizen Operations
- **GET** `/api/citizen/phone/{phone}` - Search citizen by phone
- **GET** `/api/citizen/{citizen_id}/services` - Get services used by citizen

### Recommendations
- **POST** `/api/recommend/phone` - Get recommendations by phone search
  ```json
  {
    "citizen_id": "GRPA_12369567",
    "selected_service_id": 124
  }
  ```

- **POST** `/api/recommend/manual` - Get recommendations by manual entry
  ```json
  {
    "district_id": 1,
    "gender": "Male",
    "caste": "General",
    "age": 30,
    "religion": "Hindu",
    "selected_service_id": 124
  }
  ```

## Deployment

### Deploy to Railway

1. **Install Railway CLI**
   ```bash
   npm i -g @railway/cli
   ```

2. **Login and Deploy**
   ```bash
   railway login
   railway init
   railway up
   ```

### Deploy to Heroku

1. **Install Heroku CLI**
   ```bash
   # Follow instructions at https://devcenter.heroku.com/articles/heroku-cli
   ```

2. **Deploy**
   ```bash
   heroku create bangla-sahayata-kendra
   git add .
   git commit -m "Deploy API"
   git push heroku main
   ```

### Deploy to Render

1. Create a new Web Service on Render
2. Connect your GitHub repository
3. Set the following:
   - **Build Command**: `pip install -r api/requirements.txt`
   - **Start Command**: `cd api && uvicorn main:app --host 0.0.0.0 --port $PORT`
   - **Root Directory**: `api`

### Deploy to DigitalOcean App Platform

1. Create a new App
2. Connect GitHub repository
3. Configure:
   - **Source Directory**: `api`
   - **Build Command**: `pip install -r requirements.txt`
   - **Run Command**: `uvicorn main:app --host 0.0.0.0 --port 8080`

## Environment Variables

No environment variables are required for basic operation. All data is loaded from CSV files in the `data/` directory.

Optional variables:
- `PORT` - Port to run the server on (default: 8000)
- `HOST` - Host to bind to (default: 0.0.0.0)

## Data Files Required

Ensure these files exist in the `data/` directory:
- `grouped_df.csv`
- `services.csv`
- `final_df.csv`
- `cluster_service_map.pkl`
- `service_id_with_name.csv`
- `ml_citizen_master.csv`
- `ml_provision.csv`
- `district_top_services.csv`
- `service_with_domains.csv`
- `openai_similarity_matrix.csv`

## Frontend Features

- **Phone Number Search**: Search citizens by phone number
- **Manual Entry**: Enter demographic details manually
- **Three Types of Recommendations**:
  - District-based recommendations
  - Demographic recommendations
  - Content-based recommendations
- **Responsive Design**: Works on desktop, tablet, and mobile
- **Professional UI**: Clean, modern interface with Government of West Bengal branding

## API Response Format

All recommendations follow this structure:
```json
{
  "district_recommendations": ["Service 1", "Service 2", ...],
  "demographic_recommendations": ["Service A", "Service B", ...],
  "content_recommendations": {
    "Service Name": ["Similar 1", "Similar 2", ...]
  }
}
```

## Error Handling

The API returns appropriate HTTP status codes:
- `200` - Success
- `404` - Resource not found (citizen, service)
- `422` - Validation error (invalid input)
- `500` - Server error

## Security Considerations

- CORS is enabled for all origins (configure appropriately for production)
- Citizen names are masked for privacy
- Birth and death services are filtered out
- Input validation on all endpoints

## Performance

- Data is loaded once on startup for fast responses
- Pandas DataFrames cached in memory
- Recommendation calculations are optimized

## Support

For issues or questions:
- Check API documentation at `/docs`
- Review this README
- Check the main project documentation

---

**Government of West Bengal - Bangla Sahayata Kendra**
*Powered by AI-driven Service Recommendation System*
