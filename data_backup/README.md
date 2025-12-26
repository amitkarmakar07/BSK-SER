# Data Backup - Category 2 and 3 Files

This folder contains **Category 2** (System-provided) and **Category 3** (Pre-generated) files that need to be copied to the main `data/` folder.

## 📋 **Files to Copy to data/ folder**

### **Category 2: System-Provided Configuration Files** *(Required)*
These files are provided by our development team and must be copied to `data/` folder:

- **`services.csv`** - Service eligibility criteria (age, religion, gender, caste filters)
  - Defines who can access which services based on demographic criteria
  - Used for filtering inappropriate service recommendations

- **`service_with_domains.csv`** - Service descriptions for similarity matching
  - Enhanced service descriptions with domain information
  - Used by content-based recommendation engine

### **Category 3: Pre-Generated File** *(Cost-Saving)*
- **`openai_similarity_matrix.csv`** - Pre-computed service similarity matrix
  - **💰 Cost Saving**: Use this pre-generated similarity matrix to save OpenAI API costs
  - Contains embeddings already computed for all services
  - Avoids need to regenerate embeddings during first setup

## 🚀 **Setup Instructions**

1. **Copy all files from this folder to `data/` folder:**
   ```bash
   cp data_backup/services.csv data/
   cp data_backup/service_with_domains.csv data/
   cp data_backup/openai_similarity_matrix.csv data/
   ```

2. **For Windows users:**
   ```cmd
   copy data_backup\services.csv data\
   copy data_backup\service_with_domains.csv data\
   copy data_backup\openai_similarity_matrix.csv data\
   ```

3. **Verify files are in place:**
   - Check that `data/services.csv` exists
   - Check that `data/service_with_domains.csv` exists  
   - Check that `data/openai_similarity_matrix.csv` exists

## ⚠️ **Important Notes**

- **Do NOT modify these files** unless you understand the system requirements
- The `openai_similarity_matrix.csv` is large (~50MB+) but saves significant API costs
- If you regenerate the similarity matrix, it will overwrite the cost-saving version
- These files work with the flexible data loading system (CSV-first, database-fallback)

## 🔄 **File Updates**

- **services.csv**: Update when service eligibility criteria change
- **service_with_domains.csv**: Update when new services are added or descriptions change
- **openai_similarity_matrix.csv**: Regenerate only when services change significantly

