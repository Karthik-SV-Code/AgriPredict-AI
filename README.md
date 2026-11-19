# AgriPredict AI 🌱
### Crop Yield Prediction Using Machine Learning

AgriPredict AI is a simple web application that predicts crop yield using agricultural and climate data.

## Features

- **Crop yield prediction** using crop, region, year, rainfall, temperature and pesticide usage.
- **What-if simulator** to compare predictions after changing rainfall, temperature or pesticide inputs.
- **Model performance dashboard** displaying R², MAE and RMSE.

## Tech Stack

Python · Pandas · NumPy · Scikit-learn · Flask · HTML · CSS

## Machine Learning

The project uses **RandomForestRegressor** to predict crop yield (`hg/ha_yield`). Crop and region names are converted into numerical features using One-Hot Encoding. The data is split into 80% training and 20% testing sets, and the model is evaluated using R², MAE and RMSE.

## Run Locally

1. Clone the repository and open the project folder:

   ```bash
   git clone https://github.com/Karthik-SV-Code/AgriPredict-AI-.git
   
   cd AgriPredict_AI
   ```

2. Create and activate a virtual environment (recommended):

   ```bash
   python -m venv venv
   ```

   **Windows**
   ```bash
   venv\Scripts\activate
   ```

   **macOS / Linux**
   ```bash
   source venv/bin/activate
   ```

3. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

4. Train the model:

   ```bash
   python train_model.py
   ```

5. Start the web application:

   ```bash
   python app.py
   ```

6. Open **http://127.0.0.1:5001** in browser.

Training creates the saved model and evaluation metrics inside the `models/` folder.

## Project Structure

```text
AgriPredict_AI/
├── dataset/
│   └── yield_df.csv
├── models/                 # Generated after training
├── templates/
│   └── index.html
├── static/
│   └── style.css
├── train_model.py
├── app.py
└── requirements.txt
```

## What-if simulator

The what-if simulator shows how the model's prediction changes when inputs are adjusted.
