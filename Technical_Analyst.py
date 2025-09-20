"""
Technical Analysis Module

This module provides comprehensive technical analysis functionality for stocks,
inspired by the TradingAgents project. It includes technical indicators calculation,
signal detection, pattern recognition, and automated trading recommendations.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from typing import Dict, List, Tuple, Optional, Any, Union
import json
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

@dataclass
class TechnicalSignal:
    """Data class for technical analysis signals."""
    indicator: str
    signal_type: str  # 'BUY', 'SELL', 'HOLD', 'NEUTRAL'
    strength: float  # 0-1 scale
    description: str
    value: float
    timestamp: datetime

class TechnicalAnalyst:
    """
    A comprehensive technical analysis class that provides technical indicators,
    signals, and trading insights for stocks.
    """
    
    def __init__(self, ticker: str, period: str = "1y"):
        """
        Initialize the TechnicalAnalyst with a stock ticker.
        
        Args:
            ticker (str): Stock ticker symbol (e.g., 'AAPL', 'GOOGL')
            period (str): Period for historical data ('1y', '2y', '5y', '6mo', etc.)
        """
        self.ticker = ticker.upper()
        self.period = period
        self.stock = yf.Ticker(self.ticker)
        self.data = None
        self.indicators = {}
        self.signals = []
        self._load_data()
    
    def _load_data(self):
        """Load historical stock data."""
        try:
            # Get historical data
            self.data = self.stock.history(period=self.period)
            
            if self.data.empty:
                raise ValueError(f"No data found for ticker {self.ticker}")
            
            # Ensure we have required columns
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in required_columns:
                if col not in self.data.columns:
                    raise ValueError(f"Missing required column: {col}")
            
            print(f"✅ Loaded {len(self.data)} days of data for {self.ticker}")
            
        except Exception as e:
            print(f"Error loading data for {self.ticker}: {e}")
            raise
    
    def calculate_sma(self, window: int = 20, column: str = 'Close') -> pd.Series:
        """
        Calculate Simple Moving Average.
        
        Args:
            window (int): Period for moving average
            column (str): Column to calculate SMA for
            
        Returns:
            pd.Series: SMA values
        """
        return self.data[column].rolling(window=window).mean()
    
    def calculate_ema(self, window: int = 20, column: str = 'Close') -> pd.Series:
        """
        Calculate Exponential Moving Average.
        
        Args:
            window (int): Period for moving average
            column (str): Column to calculate EMA for
            
        Returns:
            pd.Series: EMA values
        """
        return self.data[column].ewm(span=window).mean()
    
    def calculate_rsi(self, window: int = 14, column: str = 'Close') -> pd.Series:
        """
        Calculate Relative Strength Index.
        
        Args:
            window (int): Period for RSI calculation
            column (str): Column to calculate RSI for
            
        Returns:
            pd.Series: RSI values
        """
        delta = self.data[column].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_macd(self, fast: int = 12, slow: int = 26, signal: int = 9, column: str = 'Close') -> Dict[str, pd.Series]:
        """
        Calculate MACD (Moving Average Convergence Divergence).
        
        Args:
            fast (int): Fast EMA period
            slow (int): Slow EMA period
            signal (int): Signal line EMA period
            column (str): Column to calculate MACD for
            
        Returns:
            Dict containing MACD line, signal line, and histogram
        """
        ema_fast = self.calculate_ema(fast, column)
        ema_slow = self.calculate_ema(slow, column)
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }
    
    def calculate_bollinger_bands(self, window: int = 20, num_std: float = 2, column: str = 'Close') -> Dict[str, pd.Series]:
        """
        Calculate Bollinger Bands.
        
        Args:
            window (int): Period for moving average
            num_std (float): Number of standard deviations
            column (str): Column to calculate Bollinger Bands for
            
        Returns:
            Dict containing upper band, middle band (SMA), and lower band
        """
        sma = self.calculate_sma(window, column)
        std = self.data[column].rolling(window=window).std()
        
        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)
        
        return {
            'upper': upper_band,
            'middle': sma,
            'lower': lower_band
        }
    
    def calculate_atr(self, window: int = 14) -> pd.Series:
        """
        Calculate Average True Range (ATR).
        
        Args:
            window (int): Period for ATR calculation
            
        Returns:
            pd.Series: ATR values
        """
        high_low = self.data['High'] - self.data['Low']
        high_close_prev = np.abs(self.data['High'] - self.data['Close'].shift(1))
        low_close_prev = np.abs(self.data['Low'] - self.data['Close'].shift(1))
        
        true_range = np.maximum(high_low, np.maximum(high_close_prev, low_close_prev))
        atr = true_range.rolling(window=window).mean()
        
        return atr
    
    def calculate_stochastic(self, k_window: int = 14, d_window: int = 3) -> Dict[str, pd.Series]:
        """
        Calculate Stochastic Oscillator.
        
        Args:
            k_window (int): Period for %K calculation
            d_window (int): Period for %D calculation
            
        Returns:
            Dict containing %K and %D values
        """
        lowest_low = self.data['Low'].rolling(window=k_window).min()
        highest_high = self.data['High'].rolling(window=k_window).max()
        
        k_percent = 100 * ((self.data['Close'] - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_window).mean()
        
        return {
            'k_percent': k_percent,
            'd_percent': d_percent
        }
    
    def calculate_williams_r(self, window: int = 14) -> pd.Series:
        """
        Calculate Williams %R.
        
        Args:
            window (int): Period for Williams %R calculation
            
        Returns:
            pd.Series: Williams %R values
        """
        highest_high = self.data['High'].rolling(window=window).max()
        lowest_low = self.data['Low'].rolling(window=window).min()
        
        williams_r = -100 * ((highest_high - self.data['Close']) / (highest_high - lowest_low))
        
        return williams_r
    
    def calculate_obv(self) -> pd.Series:
        """
        Calculate On-Balance Volume (OBV).
        
        Returns:
            pd.Series: OBV values
        """
        obv = pd.Series(index=self.data.index, dtype=float)
        obv.iloc[0] = self.data['Volume'].iloc[0]
        
        for i in range(1, len(self.data)):
            if self.data['Close'].iloc[i] > self.data['Close'].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + self.data['Volume'].iloc[i]
            elif self.data['Close'].iloc[i] < self.data['Close'].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - self.data['Volume'].iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        
        return obv
    
    def calculate_vwap(self) -> pd.Series:
        """
        Calculate Volume Weighted Average Price (VWAP).
        
        Returns:
            pd.Series: VWAP values
        """
        typical_price = (self.data['High'] + self.data['Low'] + self.data['Close']) / 3
        vwap = (typical_price * self.data['Volume']).cumsum() / self.data['Volume'].cumsum()
        
        return vwap
    
    def calculate_all_indicators(self) -> Dict[str, Any]:
        """
        Calculate all technical indicators.
        
        Returns:
            Dict containing all calculated indicators
        """
        print("📊 Calculating technical indicators...")
        
        indicators = {}
        
        # Moving Averages
        indicators['sma_10'] = self.calculate_sma(10)
        indicators['sma_20'] = self.calculate_sma(20)
        indicators['sma_50'] = self.calculate_sma(50)
        indicators['sma_200'] = self.calculate_sma(200)
        
        indicators['ema_10'] = self.calculate_ema(10)
        indicators['ema_20'] = self.calculate_ema(20)
        indicators['ema_50'] = self.calculate_ema(50)
        
        # Momentum Indicators
        indicators['rsi'] = self.calculate_rsi()
        indicators['rsi_9'] = self.calculate_rsi(9)
        indicators['rsi_21'] = self.calculate_rsi(21)
        
        # MACD
        macd_data = self.calculate_macd()
        indicators.update({
            'macd': macd_data['macd'],
            'macd_signal': macd_data['signal'],
            'macd_histogram': macd_data['histogram']
        })
        
        # Bollinger Bands
        bb_data = self.calculate_bollinger_bands()
        indicators.update({
            'bb_upper': bb_data['upper'],
            'bb_middle': bb_data['middle'],
            'bb_lower': bb_data['lower']
        })
        
        # Volatility
        indicators['atr'] = self.calculate_atr()
        indicators['atr_21'] = self.calculate_atr(21)
        
        # Stochastic
        stoch_data = self.calculate_stochastic()
        indicators.update({
            'stoch_k': stoch_data['k_percent'],
            'stoch_d': stoch_data['d_percent']
        })
        
        # Williams %R
        indicators['williams_r'] = self.calculate_williams_r()
        
        # Volume Indicators
        indicators['obv'] = self.calculate_obv()
        indicators['vwap'] = self.calculate_vwap()
        
        # Price-based indicators
        indicators['price_change'] = self.data['Close'].pct_change()
        indicators['price_change_5d'] = self.data['Close'].pct_change(5)
        indicators['volume_sma'] = self.calculate_sma(20, 'Volume')
        
        self.indicators = indicators
        return indicators
    
    def detect_signals(self) -> List[TechnicalSignal]:
        """
        Detect trading signals based on technical indicators.
        
        Returns:
            List of TechnicalSignal objects
        """
        if not self.indicators:
            self.calculate_all_indicators()
        
        signals = []
        latest_date = self.data.index[-1]
        
        # Get latest values
        latest_close = self.data['Close'].iloc[-1]
        latest_rsi = self.indicators['rsi'].iloc[-1]
        latest_macd = self.indicators['macd'].iloc[-1]
        latest_macd_signal = self.indicators['macd_signal'].iloc[-1]
        latest_bb_upper = self.indicators['bb_upper'].iloc[-1]
        latest_bb_lower = self.indicators['bb_lower'].iloc[-1]
        latest_stoch_k = self.indicators['stoch_k'].iloc[-1]
        
        # RSI Signals
        if latest_rsi < 30:
            signals.append(TechnicalSignal(
                indicator='RSI',
                signal_type='BUY',
                strength=min((30 - latest_rsi) / 10, 1.0),
                description=f'RSI oversold at {latest_rsi:.1f}',
                value=latest_rsi,
                timestamp=latest_date
            ))
        elif latest_rsi > 70:
            signals.append(TechnicalSignal(
                indicator='RSI',
                signal_type='SELL',
                strength=min((latest_rsi - 70) / 10, 1.0),
                description=f'RSI overbought at {latest_rsi:.1f}',
                value=latest_rsi,
                timestamp=latest_date
            ))
        
        # MACD Signals
        if latest_macd > latest_macd_signal and self.indicators['macd'].iloc[-2] <= self.indicators['macd_signal'].iloc[-2]:
            signals.append(TechnicalSignal(
                indicator='MACD',
                signal_type='BUY',
                strength=0.7,
                description='MACD bullish crossover',
                value=latest_macd - latest_macd_signal,
                timestamp=latest_date
            ))
        elif latest_macd < latest_macd_signal and self.indicators['macd'].iloc[-2] >= self.indicators['macd_signal'].iloc[-2]:
            signals.append(TechnicalSignal(
                indicator='MACD',
                signal_type='SELL',
                strength=0.7,
                description='MACD bearish crossover',
                value=latest_macd - latest_macd_signal,
                timestamp=latest_date
            ))
        
        # Bollinger Bands Signals
        if latest_close < latest_bb_lower:
            signals.append(TechnicalSignal(
                indicator='Bollinger Bands',
                signal_type='BUY',
                strength=0.6,
                description='Price below lower Bollinger Band',
                value=(latest_bb_lower - latest_close) / latest_close * 100,
                timestamp=latest_date
            ))
        elif latest_close > latest_bb_upper:
            signals.append(TechnicalSignal(
                indicator='Bollinger Bands',
                signal_type='SELL',
                strength=0.6,
                description='Price above upper Bollinger Band',
                value=(latest_close - latest_bb_upper) / latest_close * 100,
                timestamp=latest_date
            ))
        
        # Moving Average Signals
        sma_50 = self.indicators['sma_50'].iloc[-1]
        sma_200 = self.indicators['sma_200'].iloc[-1]
        
        if latest_close > sma_50 > sma_200:
            signals.append(TechnicalSignal(
                indicator='Moving Averages',
                signal_type='BUY',
                strength=0.5,
                description='Price above SMA 50 and SMA 200 (Golden Cross setup)',
                value=(latest_close - sma_50) / sma_50 * 100,
                timestamp=latest_date
            ))
        elif latest_close < sma_50 < sma_200:
            signals.append(TechnicalSignal(
                indicator='Moving Averages',
                signal_type='SELL',
                strength=0.5,
                description='Price below SMA 50 and SMA 200 (Death Cross setup)',
                value=(sma_50 - latest_close) / latest_close * 100,
                timestamp=latest_date
            ))
        
        # Stochastic Signals
        if latest_stoch_k < 20:
            signals.append(TechnicalSignal(
                indicator='Stochastic',
                signal_type='BUY',
                strength=0.4,
                description=f'Stochastic oversold at {latest_stoch_k:.1f}',
                value=latest_stoch_k,
                timestamp=latest_date
            ))
        elif latest_stoch_k > 80:
            signals.append(TechnicalSignal(
                indicator='Stochastic',
                signal_type='SELL',
                strength=0.4,
                description=f'Stochastic overbought at {latest_stoch_k:.1f}',
                value=latest_stoch_k,
                timestamp=latest_date
            ))
        
        self.signals = signals
        return signals
    
    def calculate_support_resistance(self, window: int = 20) -> Dict[str, float]:
        """
        Calculate support and resistance levels.
        
        Args:
            window (int): Window for calculating support/resistance
            
        Returns:
            Dict containing support and resistance levels
        """
        recent_data = self.data.tail(window * 3)
        
        # Find local maxima and minima
        highs = recent_data['High'].rolling(window=window, center=True).max()
        lows = recent_data['Low'].rolling(window=window, center=True).min()
        
        resistance_levels = highs[recent_data['High'] == highs].dropna().values
        support_levels = lows[recent_data['Low'] == lows].dropna().values
        
        current_price = self.data['Close'].iloc[-1]
        
        # Find closest levels
        resistance = np.min(resistance_levels[resistance_levels > current_price]) if len(resistance_levels[resistance_levels > current_price]) > 0 else None
        support = np.max(support_levels[support_levels < current_price]) if len(support_levels[support_levels < current_price]) > 0 else None
        
        return {
            'resistance': resistance,
            'support': support,
            'current_price': current_price
        }
    
    def generate_technical_score(self) -> Dict[str, Any]:
        """
        Generate overall technical analysis score.
        
        Returns:
            Dict containing technical score and breakdown
        """
        if not self.signals:
            self.detect_signals()
        
        # Calculate weighted score based on signals
        total_buy_strength = sum(signal.strength for signal in self.signals if signal.signal_type == 'BUY')
        total_sell_strength = sum(signal.strength for signal in self.signals if signal.signal_type == 'SELL')
        
        buy_signals = len([s for s in self.signals if s.signal_type == 'BUY'])
        sell_signals = len([s for s in self.signals if s.signal_type == 'SELL'])
        
        # Calculate net score (-10 to +10)
        net_strength = total_buy_strength - total_sell_strength
        net_signals = buy_signals - sell_signals
        
        technical_score = (net_strength * 5) + (net_signals * 1)
        technical_score = max(-10, min(10, technical_score))  # Clamp to -10, +10
        
        # Determine recommendation
        if technical_score >= 6:
            recommendation = "STRONG BUY"
        elif technical_score >= 3:
            recommendation = "BUY"
        elif technical_score >= -2:
            recommendation = "HOLD"
        elif technical_score >= -5:
            recommendation = "SELL"
        else:
            recommendation = "STRONG SELL"
        
        return {
            'score': technical_score,
            'recommendation': recommendation,
            'buy_signals': buy_signals,
            'sell_signals': sell_signals,
            'total_buy_strength': total_buy_strength,
            'total_sell_strength': total_sell_strength,
            'confidence': min(abs(technical_score) / 10, 1.0)
        }
    
    def get_current_trend(self) -> Dict[str, Any]:
        """
        Analyze current price trend.
        
        Returns:
            Dict containing trend analysis
        """
        if not self.indicators:
            self.calculate_all_indicators()
        
        latest_close = self.data['Close'].iloc[-1]
        sma_20 = self.indicators['sma_20'].iloc[-1]
        sma_50 = self.indicators['sma_50'].iloc[-1]
        sma_200 = self.indicators['sma_200'].iloc[-1]
        
        # Short-term trend
        if latest_close > sma_20:
            short_term = "Bullish"
        else:
            short_term = "Bearish"
        
        # Medium-term trend
        if latest_close > sma_50:
            medium_term = "Bullish"
        else:
            medium_term = "Bearish"
        
        # Long-term trend
        if latest_close > sma_200:
            long_term = "Bullish"
        else:
            long_term = "Bearish"
        
        # Overall trend
        trends = [short_term, medium_term, long_term]
        bullish_count = trends.count("Bullish")
        
        if bullish_count >= 2:
            overall_trend = "Bullish"
        elif bullish_count <= 1:
            overall_trend = "Bearish"
        else:
            overall_trend = "Neutral"
        
        return {
            'short_term': short_term,
            'medium_term': medium_term,
            'long_term': long_term,
            'overall': overall_trend,
            'sma_alignment': sma_20 > sma_50 > sma_200
        }
    
    def generate_comprehensive_report(self) -> str:
        """
        Generate comprehensive technical analysis report.
        
        Returns:
            Formatted string report
        """
        # Calculate all indicators and signals
        self.calculate_all_indicators()
        self.detect_signals()
        
        # Get analysis components
        technical_score = self.generate_technical_score()
        trend_analysis = self.get_current_trend()
        support_resistance = self.calculate_support_resistance()
        
        report = []
        report.append("=" * 80)
        report.append(f"TECHNICAL ANALYSIS REPORT: {self.ticker}")
        report.append("=" * 80)
        report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Analysis Period: {self.period} ({len(self.data)} trading days)")
        report.append("")
        
        # Current Price Information
        latest_data = self.data.iloc[-1]
        price_change = latest_data['Close'] - self.data['Close'].iloc[-2]
        price_change_pct = (price_change / self.data['Close'].iloc[-2]) * 100
        
        report.append("📈 CURRENT PRICE ACTION")
        report.append("-" * 40)
        report.append(f"Current Price: ${latest_data['Close']:.2f}")
        report.append(f"Daily Change: ${price_change:+.2f} ({price_change_pct:+.2f}%)")
        report.append(f"Volume: {latest_data['Volume']:,.0f}")
        report.append(f"High: ${latest_data['High']:.2f}")
        report.append(f"Low: ${latest_data['Low']:.2f}")
        report.append("")
        
        # Technical Score
        report.append("⭐ TECHNICAL ANALYSIS SCORE")
        report.append("-" * 40)
        report.append(f"Overall Score: {technical_score['score']:.1f}/10")
        report.append(f"Recommendation: {technical_score['recommendation']}")
        report.append(f"Confidence Level: {technical_score['confidence']:.2f}")
        report.append(f"Buy Signals: {technical_score['buy_signals']}")
        report.append(f"Sell Signals: {technical_score['sell_signals']}")
        report.append("")
        
        # Trend Analysis
        report.append("📊 TREND ANALYSIS")
        report.append("-" * 40)
        report.append(f"Overall Trend: {trend_analysis['overall']}")
        report.append(f"Short-term (20-day): {trend_analysis['short_term']}")
        report.append(f"Medium-term (50-day): {trend_analysis['medium_term']}")
        report.append(f"Long-term (200-day): {trend_analysis['long_term']}")
        report.append(f"SMA Alignment: {'Bullish' if trend_analysis['sma_alignment'] else 'Bearish'}")
        report.append("")
        
        # Key Technical Indicators
        report.append("🔍 KEY TECHNICAL INDICATORS")
        report.append("-" * 40)
        
        latest_indicators = {
            'RSI (14)': self.indicators['rsi'].iloc[-1],
            'MACD': self.indicators['macd'].iloc[-1],
            'MACD Signal': self.indicators['macd_signal'].iloc[-1],
            'Stochastic %K': self.indicators['stoch_k'].iloc[-1],
            'Williams %R': self.indicators['williams_r'].iloc[-1],
            'ATR': self.indicators['atr'].iloc[-1]
        }
        
        for indicator, value in latest_indicators.items():
            if pd.notna(value):
                report.append(f"{indicator}: {value:.2f}")
        report.append("")
        
        # Moving Averages
        report.append("📈 MOVING AVERAGES")
        report.append("-" * 40)
        current_price = self.data['Close'].iloc[-1]
        
        mas = {
            'SMA 10': self.indicators['sma_10'].iloc[-1],
            'SMA 20': self.indicators['sma_20'].iloc[-1],
            'SMA 50': self.indicators['sma_50'].iloc[-1],
            'SMA 200': self.indicators['sma_200'].iloc[-1],
            'EMA 20': self.indicators['ema_20'].iloc[-1]
        }
        
        for ma_name, ma_value in mas.items():
            if pd.notna(ma_value):
                distance = ((current_price - ma_value) / ma_value) * 100
                position = "Above" if distance > 0 else "Below"
                report.append(f"{ma_name}: ${ma_value:.2f} ({position} by {abs(distance):.1f}%)")
        report.append("")
        
        # Support and Resistance
        report.append("🎯 SUPPORT & RESISTANCE LEVELS")
        report.append("-" * 40)
        if support_resistance['resistance']:
            resistance_distance = ((support_resistance['resistance'] - current_price) / current_price) * 100
            report.append(f"Resistance: ${support_resistance['resistance']:.2f} (+{resistance_distance:.1f}%)")
        else:
            report.append("Resistance: No clear level identified")
            
        if support_resistance['support']:
            support_distance = ((current_price - support_resistance['support']) / current_price) * 100
            report.append(f"Support: ${support_resistance['support']:.2f} (-{support_distance:.1f}%)")
        else:
            report.append("Support: No clear level identified")
        report.append("")
        
        # Technical Signals
        if self.signals:
            report.append("🚨 ACTIVE TECHNICAL SIGNALS")
            report.append("-" * 40)
            
            # Group signals by type
            buy_signals = [s for s in self.signals if s.signal_type == 'BUY']
            sell_signals = [s for s in self.signals if s.signal_type == 'SELL']
            
            if buy_signals:
                report.append("Buy Signals:")
                for signal in sorted(buy_signals, key=lambda x: x.strength, reverse=True):
                    strength_stars = "★" * int(signal.strength * 5)
                    report.append(f"  • {signal.indicator}: {signal.description} {strength_stars}")
                report.append("")
            
            if sell_signals:
                report.append("Sell Signals:")
                for signal in sorted(sell_signals, key=lambda x: x.strength, reverse=True):
                    strength_stars = "★" * int(signal.strength * 5)
                    report.append(f"  • {signal.indicator}: {signal.description} {strength_stars}")
                report.append("")
        else:
            report.append("🚨 ACTIVE TECHNICAL SIGNALS")
            report.append("-" * 40)
            report.append("No significant technical signals detected")
            report.append("")
        
        # Volume Analysis
        report.append("📊 VOLUME ANALYSIS")
        report.append("-" * 40)
        current_volume = latest_data['Volume']
        avg_volume = self.indicators['volume_sma'].iloc[-1]
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
        
        report.append(f"Current Volume: {current_volume:,.0f}")
        report.append(f"20-day Avg Volume: {avg_volume:,.0f}")
        report.append(f"Volume Ratio: {volume_ratio:.2f}x")
        
        if volume_ratio > 1.5:
            report.append("Volume Status: High volume - significant interest")
        elif volume_ratio < 0.7:
            report.append("Volume Status: Low volume - limited interest")
        else:
            report.append("Volume Status: Normal volume levels")
        report.append("")
        
        # Risk Assessment
        report.append("⚠️  TECHNICAL RISK ASSESSMENT")
        report.append("-" * 40)
        
        risk_factors = []
        
        # Volatility risk
        atr_pct = (self.indicators['atr'].iloc[-1] / current_price) * 100
        if atr_pct > 5:
            risk_factors.append(f"High volatility: ATR at {atr_pct:.1f}% of price")
        
        # Overbought/oversold risk
        rsi_current = self.indicators['rsi'].iloc[-1]
        if rsi_current > 80:
            risk_factors.append(f"Extremely overbought: RSI at {rsi_current:.1f}")
        elif rsi_current < 20:
            risk_factors.append(f"Extremely oversold: RSI at {rsi_current:.1f}")
        
        # Trend reversal risk
        if len([s for s in self.signals if s.signal_type == 'SELL']) > 2:
            risk_factors.append("Multiple bearish signals detected - potential trend reversal")
        
        if not risk_factors:
            risk_factors.append("No major technical risk factors identified")
        
        for risk in risk_factors:
            report.append(f"• {risk}")
        report.append("")
        
        # Trading Recommendations
        report.append("🎯 TRADING RECOMMENDATIONS")
        report.append("-" * 40)
        
        if technical_score['score'] >= 5:
            report.append("Primary Recommendation: STRONG BUY")
            report.append("• Multiple bullish signals present")
            report.append("• Consider entering long positions")
            report.append("• Set stop-loss below recent support levels")
        elif technical_score['score'] >= 2:
            report.append("Primary Recommendation: BUY")
            report.append("• Bullish bias with some caution")
            report.append("• Consider partial position entry")
            report.append("• Monitor for confirmation signals")
        elif technical_score['score'] >= -2:
            report.append("Primary Recommendation: HOLD")
            report.append("• Mixed technical signals")
            report.append("• Wait for clearer directional bias")
            report.append("• Monitor key levels for breakout/breakdown")
        elif technical_score['score'] >= -5:
            report.append("Primary Recommendation: SELL")
            report.append("• Bearish technical outlook")
            report.append("• Consider reducing positions")
            report.append("• Watch for oversold bounce opportunities")
        else:
            report.append("Primary Recommendation: STRONG SELL")
            report.append("• Multiple bearish signals present")
            report.append("• Consider exiting long positions")
            report.append("• Potential short opportunities")
        
        report.append("")
        
        # Key Levels to Watch
        report.append("🔍 KEY LEVELS TO WATCH")
        report.append("-" * 40)
        
        # Calculate key levels
        bb_upper = self.indicators['bb_upper'].iloc[-1]
        bb_lower = self.indicators['bb_lower'].iloc[-1]
        sma_50 = self.indicators['sma_50'].iloc[-1]
        
        report.append(f"Immediate Resistance: ${bb_upper:.2f} (Bollinger Upper)")
        report.append(f"Immediate Support: ${bb_lower:.2f} (Bollinger Lower)")
        report.append(f"Key MA Level: ${sma_50:.2f} (50-day SMA)")
        
        if support_resistance['resistance']:
            report.append(f"Major Resistance: ${support_resistance['resistance']:.2f}")
        if support_resistance['support']:
            report.append(f"Major Support: ${support_resistance['support']:.2f}")
        
        report.append("")
        report.append("=" * 80)
        report.append("Disclaimer: This technical analysis is for informational purposes only.")
        report.append("Technical indicators can give false signals and should be combined with")
        report.append("fundamental analysis and risk management. Always conduct your own research.")
        report.append("=" * 80)
        
        return "\n".join(report)


def analyze_stock_technical(ticker: str, period: str = "1y", save_to_file: bool = False, filename: Optional[str] = None) -> str:
    """
    Main function to perform comprehensive technical analysis on a stock.
    
    Args:
        ticker (str): Stock ticker symbol
        period (str): Analysis period ('1y', '2y', '6mo', etc.)
        save_to_file (bool): Whether to save the report to a file
        filename (str, optional): Custom filename for the report
    
    Returns:
        str: Comprehensive technical analysis report
    """
    try:
        analyst = TechnicalAnalyst(ticker, period)
        report = analyst.generate_comprehensive_report()
        
        if save_to_file:
            if not filename:
                filename = f"{ticker}_technical_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"Report saved to: {filename}")
        
        return report
    
    except Exception as e:
        error_msg = f"Error analyzing {ticker}: {str(e)}"
        print(error_msg)
        return error_msg


def compare_technical_analysis(tickers: List[str], period: str = "1y") -> pd.DataFrame:
    """
    Compare technical analysis across multiple stocks.
    
    Args:
        tickers (List[str]): List of stock ticker symbols
        period (str): Analysis period
    
    Returns:
        pandas.DataFrame: Comparison table of technical metrics
    """
    comparison_data = []
    
    for ticker in tickers:
        try:
            analyst = TechnicalAnalyst(ticker, period)
            analyst.calculate_all_indicators()
            analyst.detect_signals()
            
            technical_score = analyst.generate_technical_score()
            trend_analysis = analyst.get_current_trend()
            
            latest_close = analyst.data['Close'].iloc[-1]
            rsi = analyst.indicators['rsi'].iloc[-1]
            macd = analyst.indicators['macd'].iloc[-1]
            
            row_data = {
                "Ticker": ticker,
                "Price": f"${latest_close:.2f}",
                "Technical Score": f"{technical_score['score']:.1f}/10",
                "Recommendation": technical_score['recommendation'],
                "Overall Trend": trend_analysis['overall'],
                "RSI": f"{rsi:.1f}" if pd.notna(rsi) else "N/A",
                "MACD": f"{macd:.3f}" if pd.notna(macd) else "N/A",
                "Buy Signals": technical_score['buy_signals'],
                "Sell Signals": technical_score['sell_signals']
            }
            comparison_data.append(row_data)
            
        except Exception as e:
            print(f"Error analyzing {ticker}: {e}")
            comparison_data.append({"Ticker": ticker, "Error": str(e)})
    
    return pd.DataFrame(comparison_data)


# Example usage and testing
if __name__ == "__main__":
    print("Technical Analysis Module - Example Usage")
    print("=" * 50)
    
    # Example 1: Single stock technical analysis
    print("\n1. Analyzing Apple (AAPL) technical indicators...")
    try:
        apple_report = analyze_stock_technical("AAPL", period="6mo")
        print(apple_report[:1000] + "...\n[Report truncated for display]")
    except Exception as e:
        print(f"Error: {e}")
    
    # Example 2: Compare technical analysis across multiple stocks
    print("\n2. Comparing technical analysis for tech stocks...")
    try:
        comparison = compare_technical_analysis(["AAPL", "MSFT", "GOOGL"], period="3mo")
        print(comparison.to_string(index=False))
    except Exception as e:
        print(f"Error: {e}")
    
    # Example 3: Custom technical analyst usage
    print("\n3. Creating custom technical analyst object...")
    try:
        analyst = TechnicalAnalyst("TSLA", period="6mo")
        analyst.calculate_all_indicators()
        signals = analyst.detect_signals()
        technical_score = analyst.generate_technical_score()
        
        print(f"Company: {analyst.ticker}")
        print(f"Data Points: {len(analyst.data)} days")
        print(f"Active Signals: {len(signals)}")
        print(f"Technical Score: {technical_score['score']:.1f}/10 ({technical_score['recommendation']})")
        
        if signals:
            print("Recent Signals:")
            for signal in signals[:3]:  # Show top 3 signals
                print(f"  • {signal.indicator}: {signal.description}")
                
    except Exception as e:
        print(f"Error: {e}")
