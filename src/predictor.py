import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import tensorflow as tf
from typing import Dict

from config import Config
from data_loader import fetch_stock_data, get_current_price
from feature_engineer import create_technical_indicators, create_targets, build_feature_matrix, make_sequences
from decision_engine import make_trading_decision, PredictionResult, result_to_dict

# Calibration temperatures (tune with --calibrate)
TEMP_TOM, TEMP_WEEK = 1.5, 1.2
_VAL_ACC_TOM, _VAL_ACC_WEEK = 0.597, 0.674

def calibrate(p: float, temp: float = 1.0) -> float:
    """Apply temperature scaling"""
    p = np.clip(p, 1e-7, 1-1e-7)
    return float(1 / (1 + np.exp(-np.log(p / (1-p)) / temp)))

def _load_model():
    if not Config.MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found. Run: python train.py")
    model = tf.keras.models.load_model(Config.MODEL_PATH)
    
    # Get expected input shape from model
    expected_shape = model.input_shape
    print(f"📐 Model expects: {expected_shape}")
    
    return model

def _get_sequence(symbol: str, model):
    """Build latest sequence matching model's expected shape"""
    df = fetch_stock_data(symbol, use_cache=False)
    df = create_technical_indicators(df)
    df = create_targets(df)
    X, y_tom, y_week, y_tr, y_wr, _ = build_feature_matrix(df)
    
    # Get model's expected shape
    expected_shape = model.input_shape  # (None, timesteps, features)
    expected_timesteps = expected_shape[1] if expected_shape[1] is not None else Config.SEQUENCE_LENGTH
    expected_features = expected_shape[2] if expected_shape[2] is not None else X.shape[1]
    
    print(f"   📊 Data shape: {X.shape} (rows, features)")
    print(f"   🎯 Model needs: timesteps={expected_timesteps}, features={expected_features}")
    
    # Check if features match
    if X.shape[1] != expected_features:
        print(f"   ⚠️  Feature mismatch: Data has {X.shape[1]}, model needs {expected_features}")
        
        # Try to adjust features
        if X.shape[1] > expected_features:
            # Too many features - trim to match
            X = X[:, :expected_features]
            print(f"   ✂️  Trimmed to {expected_features} features")
        else:
            # Too few features - pad with zeros
            padding = np.zeros((X.shape[0], expected_features - X.shape[1]))
            X = np.concatenate([X, padding], axis=1)
            print(f"   ➕ Padded to {expected_features} features")
    
    # Create sequences with correct length
    X_seq, _, _, _, _ = make_sequences(X, y_tom, y_week, y_tr, y_wr, expected_timesteps)
    
    if len(X_seq) == 0:
        raise ValueError(f"Need {expected_timesteps}+ days for {symbol}, only have {len(X)} days")
    
    final_shape = X_seq[-1:].shape
    print(f"   ✅ Final sequence shape: {final_shape}")
    
    return X_seq[-1:]

def predict_for_symbol(symbol: str) -> Dict:
    """Make calibrated prediction"""
    symbol = symbol.upper()
    if symbol not in Config.SUPPORTED_STOCKS:
        print(f"⚠️  {symbol} not in config, but trying anyway...")
    
    print(f"🔄 Predicting {symbol}...")
    model = _load_model()
    
    try:
        X = _get_sequence(symbol, model)
    except Exception as e:
        print(f"❌ Error building sequence: {e}")
        raise
    
    print(f"   🤖 Running model prediction...")
    pred = model.predict(X, verbose=0)
    
    # Handle different model output formats
    if isinstance(pred, list):
        # Multiple outputs
        tom_raw = float(pred[0][0, 0])
        week_raw = float(pred[1][0, 0]) if len(pred) > 1 else tom_raw
        tom_ret = float(pred[2][0, 0]) if len(pred) > 2 else 0.0
        week_ret = float(pred[3][0, 0]) if len(pred) > 3 else 0.0
    else:
        # Single output
        tom_raw = float(pred[0, 0])
        week_raw = tom_raw
        tom_ret = 0.0
        week_ret = 0.0
    
    # Calibrate
    tom_cal = calibrate(tom_raw, TEMP_TOM)
    week_cal = calibrate(week_raw, TEMP_WEEK)
    
    price = get_current_price(symbol)
    
    print(f"📊 Raw: {tom_raw:.1%} → Calibrated: {tom_cal:.1%} ✨")
    print(f"📈 Price: ${price:.2f}")
    
    result = make_trading_decision(tom_cal, week_cal, tom_ret, week_ret, price, _VAL_ACC_TOM, _VAL_ACC_WEEK)
    result.symbol = symbol
    return result_to_dict(result)

def ui_predict_for_symbol(symbol: str) -> PredictionResult:
    """Streamlit version"""
    symbol = symbol.upper()
    model = _load_model()
    X = _get_sequence(symbol, model)
    pred = model.predict(X, verbose=0)
    
    if isinstance(pred, list):
        tom_cal = calibrate(float(pred[0][0, 0]), TEMP_TOM)
        week_cal = calibrate(float(pred[1][0, 0]), TEMP_WEEK)
        tom_ret = float(pred[2][0, 0]) if len(pred) > 2 else 0.0
        week_ret = float(pred[3][0, 0]) if len(pred) > 3 else 0.0
    else:
        tom_cal = calibrate(float(pred[0, 0]), TEMP_TOM)
        week_cal = tom_cal
        tom_ret = 0.0
        week_ret = 0.0
    
    result = make_trading_decision(
        tom_cal, week_cal, tom_ret, week_ret,
        get_current_price(symbol), _VAL_ACC_TOM, _VAL_ACC_WEEK
    )
    result.symbol = symbol
    return result

def find_optimal_temperatures():
    """Find best temperatures"""
    from sklearn.metrics import brier_score_loss
    print("="*70 + "\nCALIBRATING...\n" + "="*70)
    
    model = _load_model()
    expected_shape = model.input_shape
    expected_timesteps = expected_shape[1] if expected_shape[1] is not None else Config.SEQUENCE_LENGTH
    expected_features = expected_shape[2] if expected_shape[2] is not None else 15
    
    all_X, all_tom, all_week = [], [], []
    for sym in Config.SUPPORTED_STOCKS:
        try:
            print(f"Loading {sym}...")
            df = fetch_stock_data(sym)
            df = create_technical_indicators(df)
            df = create_targets(df)
            X, yt, yw, ytr, ywr, _ = build_feature_matrix(df)
            
            # Adjust features to match model
            if X.shape[1] != expected_features:
                if X.shape[1] > expected_features:
                    X = X[:, :expected_features]
                else:
                    padding = np.zeros((X.shape[0], expected_features - X.shape[1]))
                    X = np.concatenate([X, padding], axis=1)
            
            all_X.append(X)
            all_tom.append(yt)
            all_week.append(yw)
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            continue
    
    if not all_X:
        return print("❌ No data")
    
    X = np.concatenate(all_X)
    yt = np.concatenate(all_tom)
    yw = np.concatenate(all_week)
    
    X_seq, yt_seq, yw_seq, _, _ = make_sequences(X, yt, yw, np.zeros(len(yt)), np.zeros(len(yw)), expected_timesteps)
    
    split = int(len(X_seq) * (1 - Config.VALIDATION_SPLIT))
    X_val, yt_val, yw_val = X_seq[split:], yt_seq[split:], yw_seq[split:]
    
    print(f"Val set: {len(X_val):,} samples")
    
    pred = model.predict(X_val, verbose=0)
    
    if isinstance(pred, list):
        raw_tom = pred[0].flatten()
        raw_week = pred[1].flatten() if len(pred) > 1 else raw_tom
    else:
        raw_tom = pred.flatten()
        raw_week = raw_tom
    
    # Find best temps
    best_t_tom = min(np.linspace(0.5, 3, 50), key=lambda t: brier_score_loss(yt_val, [calibrate(p, t) for p in raw_tom]))
    best_t_week = min(np.linspace(0.5, 3, 50), key=lambda t: brier_score_loss(yw_val, [calibrate(p, t) for p in raw_week]))
    
    print(f"\n{'='*70}\n✅ RESULTS\n{'='*70}")
    print(f"TEMP_TOM = {best_t_tom:.3f}\nTEMP_WEEK = {best_t_week:.3f}")
    print(f"{'='*70}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        try:
            if sys.argv[1] == "--calibrate":
                find_optimal_temperatures()
            else:
                result = predict_for_symbol(sys.argv[1])
                print(f"\n{'='*70}")
                print(f"PREDICTION: {result['prediction']}")
                print(f"{'='*70}")
        except Exception as e:
            print(f"\n❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    else:
        print("Usage:\n  python src/predictor.py AAPL\n  python src/predictor.py --calibrate")
