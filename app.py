import sys
from pathlib import Path
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime
import pandas as pd
import numpy as np

# Add project root to path
ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))

# Import your enhanced prediction module
from predict import predict_stock_enhanced, log_to_csv

# Initialize session state
if "predictions" not in st.session_state:
    st.session_state.predictions = {}
if "last_analysis_time" not in st.session_state:
    st.session_state.last_analysis_time = None

# Page config
st.set_page_config(
    page_title="📈 Stock Predictor - Enhanced v2",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Modern Glassmorphism CSS
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .hero-section {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(20px);
        border-radius: 24px;
        padding: 2.5rem 2rem;
        margin-bottom: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    }
    
    .hero-title {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #fff 0%, #e0e7ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .hero-subtitle {
        font-size: 1rem;
        color: rgba(255, 255, 255, 0.8);
        font-weight: 300;
    }
    
    .glass-card {
        background: rgba(255, 255, 255, 0.12);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        padding: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    
    .metric-card {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(15px);
        border-radius: 16px;
        padding: 1.2rem;
        border: 1px solid rgba(255, 255, 255, 0.2);
        text-align: center;
    }
    
    .metric-label {
        font-size: 0.8rem;
        color: rgba(255, 255, 255, 0.7);
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.5rem;
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #fff;
    }
    
    .metric-subtitle {
        font-size: 0.75rem;
        color: rgba(255, 255, 255, 0.6);
        margin-top: 0.3rem;
    }
    
    .action-badge {
        display: inline-block;
        padding: 0.8rem 2rem;
        border-radius: 30px;
        font-weight: 600;
        font-size: 1.1rem;
        margin: 1rem 0;
        backdrop-filter: blur(10px);
        border: 2px solid rgba(255, 255, 255, 0.3);
    }
    
    .action-strong-buy {
        background: rgba(34, 197, 94, 0.4);
        color: #fff;
        box-shadow: 0 4px 15px rgba(34, 197, 94, 0.3);
    }
    
    .action-buy, .action-cautious-buy {
        background: rgba(34, 197, 94, 0.3);
        color: #fff;
    }
    
    .action-strong-sell {
        background: rgba(239, 68, 68, 0.4);
        color: #fff;
        box-shadow: 0 4px 15px rgba(239, 68, 68, 0.3);
    }
    
    .action-sell, .action-cautious-sell {
        background: rgba(239, 68, 68, 0.3);
        color: #fff;
    }
    
    .action-wait {
        background: rgba(156, 163, 175, 0.3);
        color: #fff;
    }
    
    .action-no-trade {
        background: rgba(107, 114, 128, 0.3);
        color: #fff;
    }
    
    .warning-box {
        background: rgba(251, 191, 36, 0.2);
        border-left: 4px solid #fbbf24;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        color: rgba(255, 255, 255, 0.9);
    }
    
    .info-box {
        background: rgba(59, 130, 246, 0.2);
        border-left: 4px solid #3b82f6;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        color: rgba(255, 255, 255, 0.9);
    }
    
    .success-box {
        background: rgba(34, 197, 94, 0.2);
        border-left: 4px solid #22c55e;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        color: rgba(255, 255, 255, 0.9);
    }
    
    .score-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.2rem;
        margin: 0.5rem 0;
    }
    
    .score-excellent {
        background: rgba(34, 197, 94, 0.3);
        color: #22c55e;
        border: 2px solid rgba(34, 197, 94, 0.5);
    }
    
    .score-good {
        background: rgba(59, 130, 246, 0.3);
        color: #3b82f6;
        border: 2px solid rgba(59, 130, 246, 0.5);
    }
    
    .score-marginal {
        background: rgba(251, 191, 36, 0.3);
        color: #fbbf24;
        border: 2px solid rgba(251, 191, 36, 0.5);
    }
    
    .score-weak {
        background: rgba(239, 68, 68, 0.3);
        color: #ef4444;
        border: 2px solid rgba(239, 68, 68, 0.5);
    }
    
    .stButton button {
        background: rgba(255, 255, 255, 0.2);
        border: 1px solid rgba(255, 255, 255, 0.3);
        border-radius: 12px;
        color: #fff;
        font-weight: 600;
        padding: 0.8rem 2rem;
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }
    
    .stButton button:hover {
        background: rgba(255, 255, 255, 0.3);
        transform: translateY(-2px);
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
    }
    
    [data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(20px);
    }
    
    .progress-bar {
        height: 8px;
        background: rgba(255, 255, 255, 0.2);
        border-radius: 10px;
        overflow: hidden;
        margin-top: 0.5rem;
    }
    
    .progress-fill {
        height: 100%;
        border-radius: 10px;
        transition: width 0.3s ease;
    }
</style>
""", unsafe_allow_html=True)

# Hero Section
st.markdown("""
<div class="hero-section">
    <div class="hero-title">📈 Stock Predictor Pro v2</div>
    <div class="hero-subtitle">
        Enhanced LSTM • Adaptive Thresholds • Weighted Scoring • Market Regime Detection • R:R ≥1.5:1
    </div>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### 📊 Stock Selection")
    
    # Default portfolio stocks
    default_stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "AMD"]
    
    analysis_mode = st.radio(
        "Analysis Mode",
        ["Single Stock", "Portfolio Analysis"],
        help="Analyze one stock or multiple stocks for comparison"
    )
    
    if analysis_mode == "Single Stock":
        ticker = st.selectbox(
            "Select Symbol",
            default_stocks + ["Custom"],
            index=0
        )
        
        if ticker == "Custom":
            ticker = st.text_input("Enter Symbol", "").upper()
        
        analyze_stocks = [ticker] if ticker else []
    else:
        analyze_stocks = st.multiselect(
            "Select Stocks",
            default_stocks,
            default=default_stocks[:4],
            help="Select stocks for comparative analysis"
        )
    
    st.markdown("---")
    
    # Analysis button
    if st.button("🚀 Generate Predictions", use_container_width=True):
        if not analyze_stocks:
            st.error("Please select at least one stock")
        else:
            with st.spinner("🔮 Running Enhanced LSTM Model..."):
                st.session_state.predictions = {}
                progress_bar = st.progress(0)
                
                for idx, stock in enumerate(analyze_stocks):
                    try:
                        pred = predict_stock_enhanced(stock)
                        st.session_state.predictions[stock] = pred
                        progress_bar.progress((idx + 1) / len(analyze_stocks))
                    except Exception as e:
                        st.error(f"Error predicting {stock}: {str(e)}")
                
                if st.session_state.predictions:
                    st.session_state.last_analysis_time = datetime.now()
                    st.success(f"✅ Analyzed {len(st.session_state.predictions)} stocks!")
                    
                    # Auto-log to CSV
                    try:
                        log_to_csv(list(st.session_state.predictions.values()))
                        st.info("📊 Logged to predictions_log.csv")
                    except Exception as e:
                        st.warning(f"Logging failed: {e}")
    
    st.markdown("---")
    st.markdown("### ℹ️ Enhanced v2 Features")
    st.markdown("""
    **✨ New in v2:**
    - 🎯 Adaptive thresholds per stock
    - 📊 Weighted signal scoring (0-100)
    - 🔍 Enhanced market regime detection
    - ⚖️ Guaranteed R:R ≥ 1.5:1
    - 📈 Trend consistency analysis
    
    **⚠️ Always paper trade first!**
    """)

# Main Content Tabs
tab_predictions, tab_comparison, tab_analysis, tab_logs = st.tabs([
    "🔮 Predictions",
    "📊 Comparison Table",
    "📈 Technical Charts", 
    "📋 History Log"
])

# Tab 1: Individual Predictions
with tab_predictions:
    if not st.session_state.predictions:
        st.info("👆 Select stocks in the sidebar and click **Generate Predictions** to see results")
    else:
        # Show analysis time
        if st.session_state.last_analysis_time:
            st.caption(f"🕐 Analysis Time: {st.session_state.last_analysis_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Portfolio overview if multiple stocks
        if len(st.session_state.predictions) > 1:
            st.subheader("📊 Portfolio Overview")
            
            cols = st.columns(5)
            total = len(st.session_state.predictions)
            
            buy_signals = sum(1 for p in st.session_state.predictions.values() 
                            if "BUY" in p.action)
            sell_signals = sum(1 for p in st.session_state.predictions.values() 
                             if "SELL" in p.action)
            wait_signals = sum(1 for p in st.session_state.predictions.values()
                              if "WAIT" in p.action)
            no_trade = sum(1 for p in st.session_state.predictions.values()
                          if "NO TRADE" in p.action)
            
            avg_score = np.mean([p.signal_score for p in st.session_state.predictions.values()])
            avg_rr = np.mean([p.risk_reward for p in st.session_state.predictions.values()])
            
            with cols[0]:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Total Stocks</div>
                    <div class="metric-value">{total}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with cols[1]:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">🟢 Buy/Sell</div>
                    <div class="metric-value">{buy_signals + sell_signals}</div>
                    <div class="metric-subtitle">{(buy_signals + sell_signals)/total*100:.0f}%</div>
                </div>
                """, unsafe_allow_html=True)
            
            with cols[2]:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">⏸️ Wait/Hold</div>
                    <div class="metric-value">{wait_signals + no_trade}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with cols[3]:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Avg Score</div>
                    <div class="metric-value">{avg_score:.0f}</div>
                    <div class="metric-subtitle">out of 100</div>
                </div>
                """, unsafe_allow_html=True)
            
            with cols[4]:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Avg R:R</div>
                    <div class="metric-value">{avg_rr:.2f}:1</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
        
        # Individual stock predictions
        for symbol, pred in st.session_state.predictions.items():
            st.subheader(f"📈 {symbol} - Enhanced Prediction")
            
            col_left, col_right = st.columns([2, 1])
            
            with col_left:
                # Top metrics row
                cols = st.columns(4)
                
                with cols[0]:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Current Price</div>
                        <div class="metric-value">${pred.current_price:.2f}</div>
                        <div class="metric-subtitle">{pred.price_date}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with cols[1]:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Direction</div>
                        <div class="metric-value">{pred.week_direction}</div>
                        <div class="metric-subtitle">{pred.week_prob_up*100:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with cols[2]:
                    # Score badge with color
                    if pred.signal_score >= 75:
                        score_class = "score-excellent"
                    elif pred.signal_score >= 65:
                        score_class = "score-good"
                    elif pred.signal_score >= 55:
                        score_class = "score-marginal"
                    else:
                        score_class = "score-weak"
                    
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Signal Score</div>
                        <div class="metric-value">{pred.signal_score:.0f}/100</div>
                        <div class="metric-subtitle">{pred.signal_strength}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with cols[3]:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Confidence</div>
                        <div class="metric-value" style="font-size: 1.3rem;">{pred.confidence}</div>
                        <div class="metric-subtitle">Thresh: {pred.adaptive_threshold*100:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Score breakdown
                st.markdown("### 📊 Score Breakdown")
                breakdown_cols = st.columns(4)
                
                for idx, (component, score) in enumerate(pred.score_breakdown.items()):
                    with breakdown_cols[idx]:
                        component_label = component.replace('_', ' ').title()
                        max_score = {'probability': 40, 'risk_reward': 25, 
                                    'market_alignment': 20, 'volatility': 15}[component]
                        percentage = (score / max_score) * 100
                        
                        st.markdown(f"""
                        <div class="glass-card" style="padding: 0.8rem;">
                            <strong>{component_label}</strong><br>
                            <span style="font-size: 1.5rem; color: #fff;">{score:.0f}</span>
                            <span style="font-size: 0.9rem; color: rgba(255,255,255,0.6);">/{max_score}</span>
                            <div class="progress-bar">
                                <div class="progress-fill" style="width: {percentage}%; background: {'#22c55e' if percentage >= 75 else '#fbbf24' if percentage >= 50 else '#ef4444'};"></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Risk Management Section
                st.markdown("### 🎯 Risk Management")
                risk_cols = st.columns(3)
                
                with risk_cols[0]:
                    st.markdown(f"""
                    <div class="glass-card">
                        <strong>Entry & Targets:</strong><br>
                        Entry: <span style="color: #3b82f6;">${pred.current_price:.2f}</span><br>
                        Target: <span style="color: #22c55e;">${pred.target_low:.2f} - ${pred.target_high:.2f}</span><br>
                        Expected: <span style="color: #22c55e;">+{pred.expected_return:.2f}%</span>
                    </div>
                    """, unsafe_allow_html=True)
                
                with risk_cols[1]:
                    rr_color = "#22c55e" if pred.risk_reward >= 2.0 else "#fbbf24" if pred.risk_reward >= 1.5 else "#ef4444"
                    st.markdown(f"""
                    <div class="glass-card">
                        <strong>Risk Parameters:</strong><br>
                        Stop Loss: <span style="color: #ef4444;">${pred.stop_loss:.2f}</span><br>
                        Max Loss: <span style="color: #ef4444;">-{pred.max_loss:.2f}%</span><br>
                        R:R: <span style="color: {rr_color};">{pred.risk_reward:.2f}:1</span>
                    </div>
                    """, unsafe_allow_html=True)
                
                with risk_cols[2]:
                    st.markdown(f"""
                    <div class="glass-card">
                        <strong>Market Context:</strong><br>
                        Regime: {pred.market_regime}<br>
                        Volatility: {pred.volatility*100:.2f}% ({pred.volatility_regime})<br>
                        ATR: {pred.atr_pct:.2f}%
                    </div>
                    """, unsafe_allow_html=True)
                
                # Action
                action_class = pred.action.lower().replace(" ", "-").replace("🟢", "").replace("🔴", "").replace("⚡", "").replace("⏸️", "").replace("❌", "").strip()
                
                st.markdown(f"""
                <div style="text-align: center; margin: 1.5rem 0;">
                    <div class="action-badge action-{action_class}">
                        {pred.action}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col_right:
                # Threshold adaptation info
                st.markdown("### 🎚️ Adaptive Threshold")
                st.markdown(f"""
                <div class="info-box">
                    <strong>Base:</strong> 58.0%<br>
                    <strong>Vol Adj:</strong> {pred.threshold_breakdown['vol_adjustment']*100:+.1f}%<br>
                    <strong>Regime Adj:</strong> {pred.threshold_breakdown['regime_adjustment']*100:+.1f}%<br>
                    <strong>Final:</strong> {pred.adaptive_threshold*100:.1f}%<br>
                    <strong>Trend Consistency:</strong> {pred.threshold_breakdown.get('trend_consistency', 0)*100:.0f}%
                </div>
                """, unsafe_allow_html=True)
                
                # Reasoning
                st.markdown("### 🧠 Analysis")
                for reason in pred.reasoning:
                    if "✅" in reason:
                        st.markdown(f'<div class="success-box">{reason}</div>', 
                                  unsafe_allow_html=True)
                    elif "⚠️" in reason or "❌" in reason:
                        st.markdown(f'<div class="warning-box">{reason}</div>', 
                                  unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="info-box">{reason}</div>', 
                                  unsafe_allow_html=True)
                
                # Warnings
                if pred.warnings:
                    st.markdown("### ⚠️ Warnings")
                    for warning in pred.warnings:
                        st.markdown(f'<div class="warning-box">• {warning}</div>', 
                                  unsafe_allow_html=True)
            
            st.markdown("---")

# Tab 2: Comparison Table
with tab_comparison:
    if not st.session_state.predictions:
        st.info("Generate predictions to see comparison table")
    else:
        st.subheader("📊 Comparative Analysis Table")
        
        # Create comparison dataframe
        comparison_data = []
        for symbol, p in st.session_state.predictions.items():
            comparison_data.append({
                'Symbol': symbol,
                'Price': f"${p.current_price:.2f}",
                'Direction': p.week_direction.replace('📈', '').replace('📉', '').strip(),
                'Probability': f"{p.week_prob_up*100:.1f}%",
                'Threshold': f"{p.adaptive_threshold*100:.1f}%",
                'Score': f"{p.signal_score:.0f}/100",
                'Signal': p.signal_strength,
                'Confidence': p.confidence.replace('🟢', '').replace('🟡', '').replace('🟠', '').replace('🔴', '').strip(),
                'Target Range': f"${p.target_low:.0f}-${p.target_high:.0f}",
                'Expected Return': f"+{p.expected_return:.1f}%",
                'R:R': f"{p.risk_reward:.2f}:1",
                'Market Regime': p.market_regime.replace('🚀', '').replace('📈', '').replace('📉', '').replace('⚖️', '').replace('🔄', '').replace('⚡', '').strip(),
                'Action': p.action.replace('🟢', '').replace('🔴', '').replace('⚡', '').replace('⏸️', '').replace('❌', '').strip()
            })
        
        df_comparison = pd.DataFrame(comparison_data)
        
        # Apply styling
        def highlight_score(val):
            if 'Score' in val.name:
                score = int(val.split('/')[0])
                if score >= 75:
                    return ['background-color: rgba(34, 197, 94, 0.2)'] * len(val)
                elif score >= 65:
                    return ['background-color: rgba(59, 130, 246, 0.2)'] * len(val)
                elif score >= 55:
                    return ['background-color: rgba(251, 191, 36, 0.2)'] * len(val)
            return [''] * len(val)
        
        st.dataframe(df_comparison, use_container_width=True, hide_index=True)
        
        # Download button
        csv = df_comparison.to_csv(index=False)
        st.download_button(
            label="📥 Download Comparison CSV",
            data=csv,
            file_name=f"stock_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
        
        # Best opportunity
        if any("BUY" in p.action or "SELL" in p.action for p in st.session_state.predictions.values()):
            trade_preds = [p for p in st.session_state.predictions.values() 
                          if "BUY" in p.action or "SELL" in p.action]
            best = max(trade_preds, key=lambda x: x.signal_score)
            
            st.success(f"🏆 **Best Opportunity:** {best.symbol} (Score: {best.signal_score:.0f}, R:R: {best.risk_reward:.2f}:1, {best.action})")

# Tab 3: Technical Analysis
with tab_analysis:
    if not st.session_state.predictions:
        st.info("Generate predictions first to see technical charts")
    else:
        selected_stock = st.selectbox(
            "Select Stock for Charts",
            list(st.session_state.predictions.keys())
        )
        
        if selected_stock:
            pred = st.session_state.predictions[selected_stock]
            
            st.subheader(f"📊 {selected_stock} - Technical Charts")
            
            try:
                # Load stock data
                csv_paths = [
                    Path(f"data/stock_data/{selected_stock}.csv"),
                    Path(f"data/{selected_stock}.csv"),
                ]
                
                df = None
                for csv_path in csv_paths:
                    if csv_path.exists():
                        df = pd.read_csv(csv_path)
                        break
                
                if df is not None:
                    df.columns = df.columns.str.lower()
                    
                    # Find date column
                    date_col = None
                    for col in ['date', 'datetime', 'timestamp']:
                        if col in df.columns:
                            date_col = col
                            break
                    
                    if date_col:
                        df[date_col] = pd.to_datetime(df[date_col])
                        df = df.set_index(date_col).sort_index()
                    
                    # Display recent data (90 days)
                    df_recent = df.tail(90)
                    
                    # Price chart with targets
                    st.markdown("### 📈 Price Chart with Targets (Last 90 Days)")
                    
                    fig = go.Figure()
                    
                    # Candlestick
                    fig.add_trace(go.Candlestick(
                        x=df_recent.index,
                        open=df_recent['open'],
                        high=df_recent['high'],
                        low=df_recent['low'],
                        close=df_recent['close'],
                        name='OHLC'
                    ))
                    
                    # Current price line
                    fig.add_hline(y=pred.current_price, line_dash="solid", 
                                 line_color="white", annotation_text="Current",
                                 annotation_position="right")
                    
                    # Target high
                    fig.add_hline(y=pred.target_high, line_dash="dash", 
                                 line_color="lime", annotation_text="Target High",
                                 annotation_position="right")
                    
                    # Target low
                    fig.add_hline(y=pred.target_low, line_dash="dash", 
                                 line_color="lightgreen", annotation_text="Target Low",
                                 annotation_position="right")
                    
                    # Stop loss
                    fig.add_hline(y=pred.stop_loss, line_dash="dash", 
                                 line_color="red", annotation_text="Stop Loss",
                                 annotation_position="right")
                    
                    fig.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(255,255,255,0.05)',
                        font=dict(color='white'),
                        xaxis_rangeslider_visible=False,
                        height=500,
                        showlegend=True
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Volume chart
                    st.markdown("### 📊 Trading Volume")
                    fig_vol = go.Figure(data=[go.Bar(
                        x=df_recent.index,
                        y=df_recent['volume'],
                        marker_color='rgba(139, 92, 246, 0.6)',
                        name='Volume'
                    )])
                    
                    fig_vol.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(255,255,255,0.05)',
                        font=dict(color='white'),
                        height=250,
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig_vol, use_container_width=True)
                    
                    # Price statistics
                    st.markdown("### 📈 Price Statistics")
                    stats_cols = st.columns(4)
                    
                    week_change = ((df_recent['close'].iloc[-1] - df_recent['close'].iloc[-5]) / df_recent['close'].iloc[-5] * 100) if len(df_recent) >= 5 else 0
                    month_change = ((df_recent['close'].iloc[-1] - df_recent['close'].iloc[-20]) / df_recent['close'].iloc[-20] * 100) if len(df_recent) >= 20 else 0
                    
                    with stats_cols[0]:
                        st.metric("52W High", f"${df_recent['high'].max():.2f}")
                    
                    with stats_cols[1]:
                        st.metric("52W Low", f"${df_recent['low'].min():.2f}")
                    
                    with stats_cols[2]:
                        st.metric("1W Change", f"{week_change:+.2f}%")
                    
                    with stats_cols[3]:
                        st.metric("1M Change", f"{month_change:+.2f}%")
                    
                else:
                    st.warning(f"Could not load chart data for {selected_stock}")
                    
            except Exception as e:
                st.error(f"Error loading charts: {str(e)}")

# Tab 4: History Log
with tab_logs:
    st.subheader("📋 Prediction History")
    
    try:
        log_path = Path("predictions_log.csv")
        if log_path.exists():
            log_df = pd.read_csv(log_path)
            
            # Show summary
            st.markdown("### 📊 Log Summary")
            summary_cols = st.columns(4)
            
            with summary_cols[0]:
                st.metric("Total Predictions", len(log_df))
            
            with summary_cols[1]:
                unique_stocks = log_df['symbol'].nunique() if 'symbol' in log_df.columns else 0
                st.metric("Unique Stocks", unique_stocks)
            
            with summary_cols[2]:
                if 'action' in log_df.columns:
                    trade_signals = log_df['action'].str.contains('BUY|SELL', case=False, na=False).sum()
                    st.metric("Trade Signals", trade_signals)
                else:
                    st.metric("Trade Signals", "N/A")
            
            with summary_cols[3]:
                if 'signal_score' in log_df.columns:
                    avg_score = log_df['signal_score'].mean()
                    st.metric("Avg Score", f"{avg_score:.1f}/100")
                else:
                    st.metric("Avg Score", "N/A")
            
            st.markdown("---")
            
            # Filters
            st.markdown("### 🔍 Filters")
            filter_cols = st.columns(3)
            
            with filter_cols[0]:
                if 'symbol' in log_df.columns:
                    selected_symbols = st.multiselect(
                        "Filter by Symbol",
                        options=sorted(log_df['symbol'].unique()),
                        default=[]
                    )
                else:
                    selected_symbols = []
            
            with filter_cols[1]:
                if 'action' in log_df.columns:
                    selected_actions = st.multiselect(
                        "Filter by Action",
                        options=sorted(log_df['action'].unique()),
                        default=[]
                    )
                else:
                    selected_actions = []
            
            with filter_cols[2]:
                show_rows = st.selectbox(
                    "Show Rows",
                    options=[25, 50, 100, "All"],
                    index=1
                )
            
            # Apply filters
            filtered_df = log_df.copy()
            if selected_symbols:
                filtered_df = filtered_df[filtered_df['symbol'].isin(selected_symbols)]
            if selected_actions:
                filtered_df = filtered_df[filtered_df['action'].isin(selected_actions)]
            
            # Display dataframe
            st.markdown("### 📄 Log Data")
            if show_rows == "All":
                st.dataframe(filtered_df, use_container_width=True, height=400)
            else:
                st.dataframe(filtered_df.tail(int(show_rows)), use_container_width=True, height=400)
            
            # Download buttons
            st.markdown("### 📥 Download Options")
            download_cols = st.columns(2)
            
            with download_cols[0]:
                csv = filtered_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Filtered Log",
                    data=csv,
                    file_name=f"predictions_log_filtered_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with download_cols[1]:
                csv_full = log_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Full Log",
                    data=csv_full,
                    file_name=f"predictions_log_full_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            # Performance insights (if enough data)
            if len(log_df) >= 10 and 'signal_score' in log_df.columns:
                st.markdown("---")
                st.markdown("### 📊 Performance Insights")
                
                insight_cols = st.columns(3)
                
                with insight_cols[0]:
                    if 'action' in log_df.columns:
                        action_counts = log_df['action'].value_counts()
                        st.markdown("**Action Distribution:**")
                        for action, count in action_counts.head(5).items():
                            st.text(f"{action}: {count} ({count/len(log_df)*100:.1f}%)")
                
                with insight_cols[1]:
                    st.markdown("**Score Distribution:**")
                    excellent = (log_df['signal_score'] >= 75).sum()
                    good = ((log_df['signal_score'] >= 65) & (log_df['signal_score'] < 75)).sum()
                    marginal = ((log_df['signal_score'] >= 55) & (log_df['signal_score'] < 65)).sum()
                    weak = (log_df['signal_score'] < 55).sum()
                    
                    st.text(f"Excellent (≥75): {excellent} ({excellent/len(log_df)*100:.1f}%)")
                    st.text(f"Good (65-74): {good} ({good/len(log_df)*100:.1f}%)")
                    st.text(f"Marginal (55-64): {marginal} ({marginal/len(log_df)*100:.1f}%)")
                    st.text(f"Weak (<55): {weak} ({weak/len(log_df)*100:.1f}%)")
                
                with insight_cols[2]:
                    if 'symbol' in log_df.columns:
                        st.markdown("**Most Analyzed:**")
                        top_stocks = log_df['symbol'].value_counts().head(5)
                        for stock, count in top_stocks.items():
                            st.text(f"{stock}: {count} times")
        else:
            st.info("No prediction log found. Generate predictions to create a log.")
            st.markdown("""
            <div class="info-box">
                <strong>📝 Log File Location:</strong> predictions_log.csv<br>
                The log file will be automatically created in the project root directory 
                when you generate your first prediction.
            </div>
            """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error loading log: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem; color: rgba(255,255,255,0.6);">
    <p style="font-size: 1.1rem; margin-bottom: 0.5rem;">
        <strong>Stock Predictor Pro v2.0</strong> | Enhanced LSTM Neural Network
    </p>
    <p style="font-size: 0.9rem; margin-bottom: 0.5rem;">
        ✨ Features: Adaptive Thresholds • Weighted Scoring • Market Regime Detection • R:R ≥1.5:1
    </p>
    <p style="font-size: 0.85rem; color: rgba(255,255,255,0.5);">
        ⚠️ <strong>CRITICAL DISCLAIMER:</strong> For Educational & Research Purposes Only | Not Financial Advice
    </p>
    <p style="font-size: 0.8rem; color: rgba(255,255,255,0.4); margin-top: 1rem;">
        Always paper trade extensively before risking real capital | Consult licensed financial professionals for investment advice
    </p>
</div>
""", unsafe_allow_html=True)

# Sidebar footer
with st.sidebar:
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 1rem; color: rgba(255,255,255,0.5); font-size: 0.75rem;">
        <p><strong>⚠️ DISCLAIMER</strong></p>
        <p>This tool is for educational purposes only. Not financial advice. 
        Always paper trade first and consult professionals.</p>
    </div>
    """, unsafe_allow_html=True)
