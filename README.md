# Aria-XT Quant Pulse

A modular, research-driven quantitative trading framework for Indian markets, featuring advanced deep learning models, robust data pipelines, and a fully integrated backtesting and live trading stack.

---

## Table of Contents
1. [Quantitative Methodology](#quantitative-methodology)
2. [Dataset & Data Pipeline](#dataset--data-pipeline)
3. [Feature Engineering](#feature-engineering)
4. [Model Architecture](#model-architecture)
5. [Backtesting Framework](#backtesting-framework)
6. [Inference & Live Trading](#inference--live-trading)
7. [Setup & Deployment](#setup--deployment)

---

## Quantitative Methodology

- **Ensemble Approach:**
  - Combines deep learning (CNN-LSTM-Attention), XGBoost, FinBERT sentiment, and Prophet time series models for robust signal generation.
- **Signal Types:**
  - **Classification:** Directional signals (Buy Call, Hold, Buy Put)
  - **Regression:** Magnitude of expected price movement
- **Risk Management:**
  - Dynamic stop-loss/take-profit, volatility regime detection, and position sizing.
- **Market Context:**
  - Incorporates trend, volatility, volume proxies, and sentiment for adaptive decision-making.

---

## Dataset & Data Pipeline

### Data Source
- **Kaggle NIFTY 50 Minute Data**  
  [NIFTY 50 minute data on Kaggle](https://www.kaggle.com/datasets/debashis74017/nifty-50-minute-data)
- **Usage:**
  - Only the minute-wise OHLC data for different NIFTY 50 symbols was used.
  - **Volume data is not present** in the dataset.

### Volume Proxy Approach
- Since volume is missing, the following indicators are used as proxies for volatility and momentum:
  - **ATR (Average True Range)**
  - **Bollinger Band Width**
  - **Rate of Change (ROC)**
  - **Supertrend**
  - Other derived features (see below)

### Data Pipeline
- **Steps:**
  1. Download and parse minute-wise OHLC data for all NIFTY 50 symbols.
  2. Compute technical indicators and engineered features (see next section).
  3. Normalize features using MinMaxScaler.
  4. Generate rolling sequences for time series modeling (lookback window).
  5. Store processed data and metadata for reproducibility.

---

## Feature Engineering

The following features are used as model inputs (from metadata):

| Feature Name         | Description                                  |
|---------------------|----------------------------------------------|
| open                | Open price                                   |
| high                | High price                                   |
| low                 | Low price                                    |
| close               | Close price                                  |
| rsi                 | Relative Strength Index                      |
| macd                | MACD value                                   |
| macd_signal         | MACD signal line                             |
| bb_upper            | Bollinger Band upper                         |
| bb_middle           | Bollinger Band middle                        |
| bb_lower            | Bollinger Band lower                         |
| atr                 | Average True Range (volatility proxy)        |
| bb_width            | Bollinger Band Width (volatility proxy)      |
| roc                 | Rate of Change (momentum proxy)              |
| supertrend          | Supertrend indicator (trend/volatility)      |
| ...                 | (Add any additional features from metadata)  |

> **Note:** The actual feature list is dynamically loaded from the model metadata (`chunk_metadata_xaat.pkl`).

---

## Model Architecture

- **AriaXaTModel:**
  - **Input:** 3D tensor (batch, sequence_length, num_features)
  - **CNN Block:** 1D convolutions for per-timestep feature extraction
  - **LSTM Block:** Multi-layer LSTM for temporal dependencies
  - **Attention:** Soft attention over LSTM outputs
  - **Heads:**
    - Classification (direction: up/hold/down)
    - Regression (magnitude of move)
  - **Training:**
    - Combined loss (weighted sum of classification and regression)
    - Early stopping, learning rate scheduling, checkpointing
    - Hyperparameter tuning with Optuna

---

## Backtesting Framework

- **Custom Backtest Loop:**
  - Loads processed data, model, scaler, and metadata
  - Simulates trading logic based on model predictions (entry/exit, position sizing)
  - Logs trades, cash, and performance metrics
- **Metrics:**
  - Final cash, trade count, win/loss ratio, and detailed trade logs
- **How to Run:**
  - `python -m src.analysis.backtest_strategy`
  - Edit config paths as needed for your environment

---

## Inference & Live Trading

- **Inference Pipeline:**
  - Loads model, scaler, and metadata
  - Preprocesses incoming data slices, scales features, and generates predictions
  - Returns class (signal), confidence, and regression output
- **Integration:**
  - APIs for real-time prediction (REST, WebSocket, or direct Python calls)
  - Ready for integration with broker APIs (e.g., Zerodha Kite Connect)
  - Modular for extension to options, futures, or multi-asset strategies

---

## Setup & Deployment

- **Requirements:**
  - Python 3.9+
  - See `requirements.txt` for all dependencies
- **Installation:**
  1. Clone the repo
  2. Install dependencies: `pip install -r requirements.txt`
  3. Prepare data and models as per the pipeline
- **Directory Structure:**

| Directory/File                | Purpose                                 |
|------------------------------|-----------------------------------------|
| `src/models/`                 | Model definitions                       |
| `src/analysis/`               | Backtest and hyperparameter tuning      |
| `src/inference/`              | Inference utilities                     |
| `aria_xat_training.py`        | Training pipeline                       |
| `backend/`                    | API and integration code                |
| `aria/data/converted/`        | Processed data and model artifacts      |

---

For detailed usage, customization, or research collaboration, please refer to the code comments or contact the maintainers.

The only requirement is having Node.js & npm installed - [install with nvm](https://github.com/nvm-sh/nvm#installing-and-updating)

Follow these steps:

```sh
# Step 1: Clone the repository using the project's Git URL.
git clone <YOUR_GIT_URL>

# Step 2: Navigate to the project directory.
cd <YOUR_PROJECT_NAME>

# Step 3: Install the necessary dependencies.
npm i

# Step 4: Start the development server with auto-reloading and an instant preview.
npm run dev
```

**Edit a file directly in GitHub**

- Navigate to the desired file(s).
- Click the "Edit" button (pencil icon) at the top right of the file view.
- Make your changes and commit the changes.

**Use GitHub Codespaces**

- Navigate to the main page of your repository.
- Click on the "Code" button (green button) near the top right.
- Select the "Codespaces" tab.
- Click on "New codespace" to launch a new Codespace environment.
- Edit files directly within the Codespace and commit and push your changes once you're done.

## What technologies are used for this project?

This project is built with:

- Vite
- TypeScript
- React
- shadcn-ui
- Tailwind CSS

How to run frontend:

- cd frontend
- npm run dev
