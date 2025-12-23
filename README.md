# 📈 Stock Predictor - AI-Powered Weekly Predictions with Risk Management

An advanced LSTM-based stock prediction system with dynamic risk management, market regime detection, and intelligent position sizing. Uses deep learning to predict weekly stock movements with comprehensive risk analysis.

---

## 🚨 **CRITICAL DISCLAIMER**

```
⚠️ EXPERIMENTAL SOFTWARE - NOT FINANCIAL ADVICE ⚠️

This model is in ALPHA/TESTING phase:
- Predictions show 54-63% probability (barely above random chance)
- NO real-world validation or backtesting completed
- Shows potential bullish bias (100% UP predictions in testing)
- NOT recommended for real money trading without extensive testing

📊 PAPER TRADE ONLY for 3-6 months minimum
💸 Never risk money you cannot afford to lose
🧠 YOU are responsible for all trading decisions
🚫 This is NOT financial advice - consult a licensed professional
```

---

## ✨ **Features**

### 🎯 **Core Capabilities**
- ✅ **Weekly Predictions** - Uses LSTM neural networks for 1-week ahead forecasts
- ✅ **Latest Data** - Automatically uses most recent date from your CSV files
- ✅ **Dynamic Thresholds** - Adjusts probability requirements based on market conditions
- ✅ **Risk Management** - ATR-based stop losses, targets, and risk-reward ratios
- ✅ **Market Regime Detection** - Identifies bull/bear/sideways/mixed markets
- ✅ **CSV Logging** - Tracks all predictions for performance analysis
- ✅ **Conservative Approach** - Rejects low-confidence trades (only ~17% trade rate)

### 📊 **Technical Features**
- Multi-timeframe analysis (daily data)
- 15+ technical indicators (RSI, ADX, ATR, VWAP, EMAs, etc.)
- Volatility-adjusted position sizing
- Smart stop-loss placement using ATR
- Expected return and max loss calculations

---

## 📋 **Requirements**

### **Python Version**
- Python 3.8 - 3.10 (tested on 3.10)

### **Dependencies**
```bash
tensorflow==2.15.0
numpy==1.24.3
pandas==2.0.3
scikit-learn==1.3.0
yfinance==0.2.28
```

---

## 🚀 **Installation**

### **1. Clone Repository**
```bash
git clone https://github.com/yourusername/stock-predictor.git
cd stock-predictor
```

### **2. Create Virtual Environment**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### **3. Install Dependencies**
```bash
pip install -r requirements.txt
```

### **4. Verify Setup**
```bash
python predict.py --check
```

Expected output:
```
✅ SETUP CHECK
   ✅ models/stock_model_fixed.keras
✅ Ready! Run: python predict.py --portfolio
```

---

## 📂 **Project Structure**

```
stock-predictor/
├── predict.py                 # Main prediction script
├── train_fixed.py            # Model training script
├── requirements.txt          # Python dependencies
├── models/
│   └── stock_model_fixed.keras   # Trained LSTM model
├── data/
│   ├── AAPL.csv              # Apple stock data
│   ├── MSFT.csv              # Microsoft stock data
│   ├── GOOGL.csv             # Google stock data
│   ├── AMZN.csv              # Amazon stock data
│   ├── NVDA.csv              # NVIDIA stock data
│   └── TSLA.csv              # Tesla stock data
├── predictions_log.csv       # Prediction history (auto-generated)
└── README.md                 # This file
```

---

## 💻 **Usage**

### **Quick Start**

#### **Single Stock Prediction**
```bash
python predict.py -s AAPL
```

#### **Multiple Stocks**
```bash
python predict.py -s AAPL MSFT GOOGL
```

#### **Portfolio Analysis (Default 6 Stocks)**
```bash
python predict.py --portfolio
```

#### **Disable CSV Logging**
```bash
python predict.py -s AAPL --no-log
```

### **Command-Line Options**

| Option | Description | Example |
|--------|-------------|---------|
| `-s, --stocks` | Stock symbols to analyze | `-s AAPL MSFT` |
| `-p, --portfolio` | Analyze default portfolio | `--portfolio` |
| `--check` | Verify setup and model | `--check` |
| `--no-log` | Don't log predictions to CSV | `--no-log` |

---

## 📊 **Understanding the Output**

### **Sample Prediction Output**

```
🔮 AAPL - WEEKLY PREDICTION (Latest Data: 2025-12-22)

CURRENT SITUATION
Price:        $272.12
Data Date:    2025-12-22
Direction:    📈 UP (Probability: 62.0%)
Confidence:   🟢 HIGH
Threshold:    57.0% (dynamic)
Volatility:   0.84% | ATR: 1.63%
Market:       📈 BULL STRONG

TARGETS & RISK MANAGEMENT
Entry:             $272.12
Target Range:      $274.33 - $278.76
Expected Return:   +1.63%
Stop Loss:         $267.69
Max Loss:          -1.63%
Risk-Reward:       1.50:1

DECISION & REASONING
Action: ⚡ CAUTION BUY
Signal: VALID
  ✅ Probability above threshold (62.0% > 57.0%)
  ⚠️ Poor risk-reward (1.50:1 < 1.5:1)
  ✅ Aligned with bull market

⚠️ WARNINGS:
  • Risk-reward below 1.5:1
```

### **Understanding Actions**

| Action | Meaning | What to Do |
|--------|---------|------------|
| 🟢 **BUY** | High confidence UP signal | Consider buying (paper trade first!) |
| 🔴 **SELL** | High confidence DOWN signal | Consider selling/shorting |
| ⚡ **CAUTION BUY/SELL** | Valid signal with warnings | Proceed with extra caution |
| ⏸️ **WAIT** | Multiple warnings present | Skip this trade |
| ❌ **NO TRADE** | Probability below threshold | Do not trade |

### **Signal Strength**

| Strength | Probability Range | Confidence Level |
|----------|------------------|------------------|
| 🟢 **VERY HIGH** | 10%+ above threshold | Strong signal |
| 🟢 **HIGH** | 5-10% above threshold | Good signal |
| 🟡 **MEDIUM** | 0-5% above threshold | Marginal signal |
| 🟠 **LOW** | -5-0% below threshold | Weak signal |
| 🔴 **VERY LOW** | 5%+ below threshold | Very weak signal |

---

## 🎯 **Dynamic Threshold System**

The model adjusts probability thresholds based on market conditions:

### **Base Threshold: 60%**

### **Volatility Adjustments**
- Very High (>4%): +8% → **68% required**
- High (2-4%): +5% → **65% required**
- Moderate (1-2%): +3% → **63% required**
- Low (<1%): +0% → **60% required**

### **Market Regime Adjustments**
- 📈 **BULL STRONG**: -3% (more aggressive)
- 📈 **BULL**: -2%
- 📉 **BEAR**: +5% (more conservative)
- ⚖️ **SIDEWAYS**: +3%
- 🔄 **MIXED**: +2%

### **Example:**
```
AAPL: BULL STRONG + Low Volatility
= 60% base - 3% (bull) + 0% (low vol) = 57% threshold ✅

MSFT: BEAR + Moderate Volatility
= 60% base + 5% (bear) + 3% (vol) = 68% threshold ⚠️
```

---

## 📈 **CSV Data Format**

### **Required CSV Structure**

Your CSV files must be located in: `data/SYMBOL.csv`

Example: `data/AAPL.csv`

```csv
date,open,high,low,close,volume
2025-12-20,270.50,275.00,269.00,273.67,50000000
2025-12-21,273.50,276.00,271.00,274.20,48000000
2025-12-22,274.00,277.00,272.00,272.12,52000000
```

### **Required Columns**
- `date` - Trading date (YYYY-MM-DD format)
- `open` - Opening price
- `high` - Daily high price
- `low` - Daily low price
- `close` - Closing price
- `volume` - Trading volume

### **Important Notes**
- ✅ Model automatically uses the **LATEST date** in your CSV
- ✅ Update CSV daily for most recent predictions
- ✅ Minimum 200 rows recommended for accurate predictions
- ✅ Column names are case-insensitive

---

## 📊 **CSV Prediction Logging**

All predictions are automatically logged to `predictions_log.csv`:

### **Logged Fields**
- Timestamp (when prediction was made)
- Symbol
- Price Date (date of data used)
- Current Price
- Probability, Direction, Confidence
- Targets, Stop Loss, Risk-Reward
- Market Regime, Volatility, ATR
- Dynamic Threshold
- Action, Signal Strength
- Warnings

### **Using the Log**
```python
import pandas as pd

# Load prediction history
df = pd.read_csv('predictions_log.csv')

# Calculate actual win rate (requires manual entry of outcomes)
# Track: Did the prediction match reality within 1 week?
```

---

## 🧪 **Testing & Validation**

### ⚠️ **CRITICAL: Paper Trade First!**

Before risking real money, you **MUST**:

1. **Paper Trade for 3-6 Months**
   - Track all predictions
   - Record actual outcomes
   - Calculate real win rate

2. **Backtest Historical Data**
   - Test on 2023-2024 data
   - Calculate performance metrics
   - Verify profitability

3. **Validate Calibration**
   - Does 62% probability = 62% actual wins?
   - Check if model is optimistic/pessimistic

### **Performance Metrics to Track**

| Metric | Target | Formula |
|--------|--------|---------|
| Win Rate | ≥55% | Wins / Total Trades |
| Avg R:R | ≥1.3:1 | Avg Win / Avg Loss |
| Max Drawdown | <20% | Worst peak-to-valley loss |
| Sharpe Ratio | >1.0 | (Return - Risk-Free) / Std Dev |
| Profit Factor | >1.5 | Gross Profit / Gross Loss |

---

## ⚠️ **Known Limitations**

### **Current Issues**
1. ❌ **Low Predictive Power**: 54-63% probability (barely above 50%)
2. ❌ **Bullish Bias**: All test predictions were UP (6/6)
3. ❌ **No Backtesting**: Zero historical validation
4. ❌ **Narrow Range**: Probabilities clustered 54-63%
5. ❌ **High Rejection Rate**: 83% of signals rejected (5/6)

### **What This Means**
- Model is **uncertain** about most predictions
- Needs extensive real-world testing
- Conservative by design (good for risk management)
- May miss opportunities due to strict thresholds

---

## 🔧 **Model Training**

### **Training Your Own Model**

```bash
python train_fixed.py
```

### **Training Configuration**
- Architecture: LSTM with 3 output heads
- Features: 15 technical indicators
- Sequence Length: 60 days
- Outputs:
  - Daily direction (up/down)
  - 3-day direction
  - Weekly direction (primary)

### **Features Used**
1. ATR Percentage
2. Volatility (20-day)
3. Trend Strength (price/MA ratio)
4. Rate of Change (10-day)
5. Volume Ratio
6. SMA deviation (7-day)
7. EMA deviation (7-day)
8. RSI (14-day normalized)
9. Volume Trend (7-day/30-day)
10. Weekly Return (5-day)
11. Weekly Volatility (5-day)
12. EMA Difference (20-day/50-day)
13. ADX (14-day normalized)
14. Price vs VWAP
15. Market Trend (SPY alignment)

---

## 📊 **Risk Management Best Practices**

### **Position Sizing Rules**
```
Max Risk per Trade: 1-2% of account
Max Position Size: 10% of account
Stop Loss: ATR-based (typically 1.0x ATR)
Take Profit: 1.5-2.0x ATR
```

### **Trade Management**
1. ✅ Only trade signals with ≥65% probability
2. ✅ Respect stop losses (no moving them!)
3. ✅ Take partial profits at first target
4. ✅ Trail stop on remaining position
5. ✅ Never risk more than 2% on single trade
6. ✅ Limit to 3-5 positions maximum

### **When NOT to Trade**
- ❌ Probability below dynamic threshold
- ❌ Market regime conflicts with signal
- ❌ High volatility + low probability
- ❌ Multiple warnings present
- ❌ During major news events
- ❌ When emotionally compromised

---

## 🐛 **Troubleshooting**

### **Common Issues**

#### **Problem: "No data returned for AAPL"**
```bash
# Solution: Check CSV file location
# Expected: data/AAPL.csv
ls data/AAPL.csv

# If missing, download or place CSV in correct location
```

#### **Problem: "Date showing 1970-01-01"**
```bash
# Solution: Date column not parsed correctly
# Check CSV has 'date' column with format: YYYY-MM-DD
# Example: 2025-12-22
```

#### **Problem: "unsupported operand type(s) for -: 'str' and 'str'"**
```bash
# Solution: Numeric columns are strings
# Ensure CSV has numeric values without quotes
# Bad:  "272.12"
# Good: 272.12
```

#### **Problem: "Model not found"**
```bash
# Solution: Train the model first
python train_fixed.py

# Or check model location
python predict.py --check
```

#### **Problem: All predictions are "NO TRADE"**
```bash
# This is NORMAL! Model is conservative
# Typical trade rate: 10-20%
# Only trades high-confidence setups
```

---

## 📚 **Additional Resources**

### **Learning Materials**
- [LSTM Networks Explained](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [Technical Analysis Basics](https://www.investopedia.com/technical-analysis-4689657)
- [Risk Management Principles](https://www.investopedia.com/terms/r/riskmanagement.asp)
- [Position Sizing Calculator](https://www.investopedia.com/terms/p/positionsizing.asp)

### **Trading Psychology**
- Start with paper trading (fake money)
- Track every trade in a journal
- Accept losses as part of the process
- Never chase losses with bigger bets
- Take breaks after big wins/losses

---

## 🤝 **Contributing**

Contributions are welcome! Focus areas:

1. **Backtesting Framework** - Test on historical data
2. **Model Improvements** - Increase predictive accuracy
3. **Feature Engineering** - Add new technical indicators
4. **Performance Metrics** - Sharpe ratio, max drawdown tracking
5. **Documentation** - Improve guides and examples

---

## 📄 **License**

MIT License - See LICENSE file for details

---

## ⚖️ **Legal Disclaimer**

```
IMPORTANT LEGAL NOTICE

This software is provided "AS IS" for educational and research purposes only.

NO WARRANTY: The software is provided without warranty of any kind, 
express or implied, including but not limited to merchantability, 
fitness for a particular purpose, or non-infringement.

NO FINANCIAL ADVICE: This tool does NOT provide financial, investment, 
trading, or any other advice. All predictions are experimental and 
should not be relied upon for making financial decisions.

RISK OF LOSS: Trading stocks involves substantial risk of loss. 
You may lose some or all of your invested capital. Never trade with 
money you cannot afford to lose.

PAST PERFORMANCE: Past performance is not indicative of future results. 
Historical testing does not guarantee future profitability.

PERSONAL RESPONSIBILITY: You are solely responsible for all trading 
decisions and their consequences. The creators and contributors of 
this software are not liable for any losses incurred.

CONSULT PROFESSIONALS: Always consult with licensed financial advisors, 
accountants, and legal professionals before making investment decisions.

By using this software, you acknowledge that you have read, understood, 
and agree to these terms.
```

---

## 📧 **Support**

- 🐛 **Bug Reports**: Open an issue on GitHub
- 💡 **Feature Requests**: Submit via GitHub Issues
- 📖 **Documentation**: Check README and code comments
- 💬 **Questions**: Use GitHub Discussions

---

## 🎯 **Development Roadmap**

### **Phase 1: Validation (Current)**
- [ ] 3-6 months paper trading
- [ ] Historical backtesting (2022-2024)
- [ ] Model calibration verification
- [ ] Performance metrics dashboard

### **Phase 2: Enhancement**
- [ ] Sentiment analysis integration
- [ ] Multi-timeframe predictions
- [ ] Portfolio optimization
- [ ] Risk-adjusted position sizing

### **Phase 3: Production**
- [ ] Real-time data integration
- [ ] Automated trade execution (optional)
- [ ] Mobile app interface
- [ ] Advanced analytics dashboard

---

## 📊 **Performance Goals**

### **Target Metrics (Post-Validation)**
- Win Rate: ≥60%
- Average R:R: ≥1.5:1
- Max Drawdown: <15%
- Sharpe Ratio: >1.5
- Profit Factor: >2.0
- Trade Frequency: 2-4 signals per week

### **Current Metrics (Pre-Validation)**
- Win Rate: Unknown (needs testing)
- Average R:R: 1.50:1 (theoretical)
- Max Drawdown: Unknown
- Sharpe Ratio: Unknown
- Profit Factor: Unknown
- Trade Frequency: ~1 per 6 stocks analyzed

---

## 🙏 **Acknowledgments**

- TensorFlow team for deep learning framework
- yfinance for market data access
- scikit-learn for preprocessing tools
- The trading community for domain knowledge

---

## 📝 **Changelog**

### **Version 1.0.0 (2025-12-23)**
- ✅ Initial release
- ✅ LSTM model with weekly predictions
- ✅ Dynamic threshold system
- ✅ Market regime detection
- ✅ CSV logging
- ✅ Risk management framework
- ✅ Automated latest data selection

---

**Remember: The best investment is in your education. Learn, test, and validate before risking real capital!** 🎓📈

---

*Last Updated: December 23, 2025*
