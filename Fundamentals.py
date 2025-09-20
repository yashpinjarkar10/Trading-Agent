"""
Fundamental Analysis Module

This module provides comprehensive fundamental analysis functionality for stocks,
inspired by the TradingAgents project. It includes financial metrics calculation,
ratio analysis, and automated report generation.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from typing import Dict, List, Tuple, Optional, Any
import json
import warnings

# Suppress yfinance warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)

class FundamentalAnalyst:
    """
    A comprehensive fundamental analysis class that provides financial metrics,
    ratios, and analysis reports for stocks.
    """
    
    def __init__(self, ticker: str):
        """
        Initialize the FundamentalAnalyst with a stock ticker.
        
        Args:
            ticker (str): Stock ticker symbol (e.g., 'AAPL', 'GOOGL')
        """
        self.ticker = ticker.upper()
        self.stock = yf.Ticker(self.ticker)
        self.info = None
        self.financials = None
        self.balance_sheet = None
        self.cash_flow = None
        self._load_data()
    
    def _load_data(self):
        """Load basic stock data and financial statements."""
        try:
            self.info = self.stock.info
            self.financials = self.stock.financials
            self.balance_sheet = self.stock.balance_sheet
            self.cash_flow = self.stock.cashflow
        except Exception as e:
            print(f"Warning: Could not load all data for {self.ticker}: {e}")
    
    def get_company_info(self) -> Dict[str, Any]:
        """
        Get basic company information.
        
        Returns:
            Dict containing company information
        """
        if not self.info:
            return {}
        
        return {
            "Company Name": self.info.get("shortName", "N/A"),
            "Industry": self.info.get("industry", "N/A"),
            "Sector": self.info.get("sector", "N/A"),
            "Country": self.info.get("country", "N/A"),
            "Website": self.info.get("website", "N/A"),
            "Business Summary": self.info.get("longBusinessSummary", "N/A")[:500] + "..." if self.info.get("longBusinessSummary") else "N/A",
            "Market Cap": self.info.get("marketCap", "N/A"),
            "Enterprise Value": self.info.get("enterpriseValue", "N/A"),
            "Full Time Employees": self.info.get("fullTimeEmployees", "N/A")
        }
    
    def get_key_financial_metrics(self) -> Dict[str, Any]:
        """
        Calculate key financial metrics and ratios.
        
        Returns:
            Dict containing financial metrics
        """
        metrics = {}
        
        if not self.info:
            return metrics
        
        # Valuation Metrics
        metrics["PE Ratio"] = self.info.get("trailingPE", "N/A")
        metrics["Forward PE"] = self.info.get("forwardPE", "N/A")
        metrics["PEG Ratio"] = self.info.get("pegRatio", "N/A")
        metrics["Price to Book"] = self.info.get("priceToBook", "N/A")
        metrics["Price to Sales"] = self.info.get("priceToSalesTrailing12Months", "N/A")
        metrics["EV/Revenue"] = self.info.get("enterpriseToRevenue", "N/A")
        metrics["EV/EBITDA"] = self.info.get("enterpriseToEbitda", "N/A")
        
        # Profitability Metrics
        metrics["Profit Margin"] = self.info.get("profitMargins", "N/A")
        metrics["Operating Margin"] = self.info.get("operatingMargins", "N/A")
        metrics["Return on Assets"] = self.info.get("returnOnAssets", "N/A")
        metrics["Return on Equity"] = self.info.get("returnOnEquity", "N/A")
        
        # Growth Metrics
        metrics["Revenue Growth"] = self.info.get("revenueGrowth", "N/A")
        metrics["Earnings Growth"] = self.info.get("earningsGrowth", "N/A")
        
        # Financial Health
        metrics["Current Ratio"] = self.info.get("currentRatio", "N/A")
        metrics["Quick Ratio"] = self.info.get("quickRatio", "N/A")
        metrics["Debt to Equity"] = self.info.get("debtToEquity", "N/A")
        metrics["Total Debt"] = self.info.get("totalDebt", "N/A")
        metrics["Total Cash"] = self.info.get("totalCash", "N/A")
        
        # Dividend Information
        metrics["Dividend Yield"] = self.info.get("dividendYield", "N/A")
        metrics["Payout Ratio"] = self.info.get("payoutRatio", "N/A")
        
        return metrics
    
    def calculate_custom_ratios(self) -> Dict[str, Any]:
        """
        Calculate additional custom financial ratios from financial statements.
        
        Returns:
            Dict containing custom calculated ratios
        """
        ratios = {}
        
        try:
            if self.financials is not None and not self.financials.empty:
                latest_financials = self.financials.iloc[:, 0]  # Most recent year
                
                # Revenue and Profitability Analysis
                total_revenue = latest_financials.get("Total Revenue", 0)
                gross_profit = latest_financials.get("Gross Profit", 0)
                operating_income = latest_financials.get("Operating Income", 0)
                net_income = latest_financials.get("Net Income", 0)
                
                if total_revenue and total_revenue != 0:
                    ratios["Gross Margin"] = (gross_profit / total_revenue) * 100 if gross_profit else "N/A"
                    ratios["Operating Margin"] = (operating_income / total_revenue) * 100 if operating_income else "N/A"
                    ratios["Net Margin"] = (net_income / total_revenue) * 100 if net_income else "N/A"
            
            if self.balance_sheet is not None and not self.balance_sheet.empty:
                latest_balance = self.balance_sheet.iloc[:, 0]  # Most recent year
                
                # Liquidity Ratios
                current_assets = latest_balance.get("Current Assets", 0)
                current_liabilities = latest_balance.get("Current Liabilities", 0)
                
                if current_liabilities and current_liabilities != 0:
                    ratios["Current Ratio (Calculated)"] = current_assets / current_liabilities if current_assets else "N/A"
                
                # Leverage Ratios
                total_debt = latest_balance.get("Total Debt", 0)
                total_equity = latest_balance.get("Total Stockholder Equity", 0)
                
                if total_equity and total_equity != 0:
                    ratios["Debt-to-Equity (Calculated)"] = total_debt / total_equity if total_debt else "N/A"
        
        except Exception as e:
            print(f"Error calculating custom ratios: {e}")
        
        return ratios
    
    def get_analyst_recommendations(self) -> Dict[str, Any]:
        """
        Get analyst recommendations and target prices.
        
        Returns:
            Dict containing analyst data
        """
        recommendations = {}
        
        try:
            # Get analyst recommendations
            analyst_recs = self.stock.recommendations
            if analyst_recs is not None and not analyst_recs.empty:
                latest_rec = analyst_recs.iloc[0]  # Most recent recommendation
                recommendations["Latest Recommendation"] = {
                    "Strong Buy": latest_rec.get("strongBuy", 0),
                    "Buy": latest_rec.get("buy", 0),
                    "Hold": latest_rec.get("hold", 0),
                    "Sell": latest_rec.get("sell", 0),
                    "Strong Sell": latest_rec.get("strongSell", 0)
                }
            
            # Target price information
            if self.info:
                recommendations["Target Mean Price"] = self.info.get("targetMeanPrice", "N/A")
                recommendations["Target High Price"] = self.info.get("targetHighPrice", "N/A")
                recommendations["Target Low Price"] = self.info.get("targetLowPrice", "N/A")
                recommendations["Recommendation Mean"] = self.info.get("recommendationMean", "N/A")
                recommendations["Number of Analyst Opinions"] = self.info.get("numberOfAnalystOpinions", "N/A")
        
        except Exception as e:
            print(f"Error getting analyst recommendations: {e}")
        
        return recommendations
    
    def analyze_financial_trends(self, years: int = 3) -> Dict[str, Any]:
        """
        Analyze financial trends over multiple years.
        
        Args:
            years (int): Number of years to analyze
            
        Returns:
            Dict containing trend analysis
        """
        trends = {}
        
        try:
            if self.financials is not None and not self.financials.empty:
                # Limit to available years
                available_years = min(years, len(self.financials.columns))
                
                if available_years >= 2:
                    # Revenue trend
                    revenue_data = self.financials.loc["Total Revenue"].iloc[:available_years]
                    if not revenue_data.empty and len(revenue_data) >= 2:
                        revenue_growth = ((revenue_data.iloc[0] - revenue_data.iloc[-1]) / revenue_data.iloc[-1]) * 100
                        trends["Revenue Growth (Annual)"] = f"{revenue_growth:.2f}%"
                    
                    # Net Income trend
                    if "Net Income" in self.financials.index:
                        income_data = self.financials.loc["Net Income"].iloc[:available_years]
                        if not income_data.empty and len(income_data) >= 2:
                            income_growth = ((income_data.iloc[0] - income_data.iloc[-1]) / income_data.iloc[-1]) * 100
                            trends["Net Income Growth (Annual)"] = f"{income_growth:.2f}%"
        
        except Exception as e:
            print(f"Error analyzing financial trends: {e}")
        
        return trends
    
    def generate_investment_score(self) -> Dict[str, Any]:
        """
        Generate a simple investment score based on key metrics.
        
        Returns:
            Dict containing investment score and reasoning
        """
        score = 0
        max_score = 10
        reasons = []
        
        if not self.info:
            return {"Score": "N/A", "Max Score": max_score, "Reasons": ["Insufficient data"]}
        
        # PE Ratio scoring (1 point)
        pe_ratio = self.info.get("trailingPE")
        if pe_ratio and 10 <= pe_ratio <= 25:
            score += 1
            reasons.append("Reasonable P/E ratio")
        elif pe_ratio and pe_ratio > 25:
            reasons.append("High P/E ratio - potentially overvalued")
        
        # PEG Ratio scoring (1 point)
        peg_ratio = self.info.get("pegRatio")
        if peg_ratio and peg_ratio < 1:
            score += 1
            reasons.append("Good PEG ratio - growth at reasonable price")
        
        # Profit Margin scoring (1 point)
        profit_margin = self.info.get("profitMargins")
        if profit_margin and profit_margin > 0.1:  # 10%
            score += 1
            reasons.append("Strong profit margins")
        
        # ROE scoring (1 point)
        roe = self.info.get("returnOnEquity")
        if roe and roe > 0.15:  # 15%
            score += 1
            reasons.append("Strong return on equity")
        
        # Current Ratio scoring (1 point)
        current_ratio = self.info.get("currentRatio")
        if current_ratio and current_ratio > 1.0:
            score += 1
            reasons.append("Good liquidity position")
        
        # Debt to Equity scoring (1 point)
        debt_to_equity = self.info.get("debtToEquity")
        if debt_to_equity is not None and debt_to_equity < 50:  # Less than 50%
            score += 1
            reasons.append("Manageable debt levels")
        
        # Revenue Growth scoring (1 point)
        revenue_growth = self.info.get("revenueGrowth")
        if revenue_growth and revenue_growth > 0.05:  # 5%
            score += 1
            reasons.append("Positive revenue growth")
        
        # Dividend Yield scoring (1 point)
        dividend_yield = self.info.get("dividendYield")
        if dividend_yield and dividend_yield > 0:
            score += 1
            reasons.append("Pays dividends")
        
        # Analyst recommendation scoring (1 point)
        rec_mean = self.info.get("recommendationMean")
        if rec_mean and rec_mean <= 2.5:  # Buy to Hold range
            score += 1
            reasons.append("Positive analyst sentiment")
        
        # Market cap stability (1 point)
        market_cap = self.info.get("marketCap")
        if market_cap and market_cap > 1e9:  # > $1B
            score += 1
            reasons.append("Large market cap - stability")
        
        return {
            "Score": score,
            "Max Score": max_score,
            "Percentage": f"{(score/max_score)*100:.1f}%",
            "Reasons": reasons
        }
    
    def generate_comprehensive_report(self) -> str:
        """
        Generate a comprehensive fundamental analysis report.
        
        Returns:
            Formatted string report
        """
        report = []
        report.append("=" * 80)
        report.append(f"FUNDAMENTAL ANALYSIS REPORT: {self.ticker}")
        report.append("=" * 80)
        report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Company Information
        company_info = self.get_company_info()
        report.append("📊 COMPANY OVERVIEW")
        report.append("-" * 40)
        for key, value in company_info.items():
            if key != "Business Summary":
                report.append(f"{key}: {value}")
        
        if company_info.get("Business Summary") != "N/A":
            report.append(f"Business Summary: {company_info['Business Summary']}")
        report.append("")
        
        # Key Financial Metrics
        metrics = self.get_key_financial_metrics()
        report.append("💰 KEY FINANCIAL METRICS")
        report.append("-" * 40)
        
        valuation_metrics = ["PE Ratio", "Forward PE", "PEG Ratio", "Price to Book", "Price to Sales", "EV/Revenue", "EV/EBITDA"]
        profitability_metrics = ["Profit Margin", "Operating Margin", "Return on Assets", "Return on Equity"]
        growth_metrics = ["Revenue Growth", "Earnings Growth"]
        health_metrics = ["Current Ratio", "Quick Ratio", "Debt to Equity", "Total Debt", "Total Cash"]
        dividend_metrics = ["Dividend Yield", "Payout Ratio"]
        
        report.append("Valuation Metrics:")
        for metric in valuation_metrics:
            if metric in metrics:
                value = metrics[metric]
                if isinstance(value, float):
                    value = f"{value:.2f}"
                report.append(f"  {metric}: {value}")
        
        report.append("\nProfitability Metrics:")
        for metric in profitability_metrics:
            if metric in metrics:
                value = metrics[metric]
                if isinstance(value, float):
                    if metric in ["Profit Margin", "Operating Margin"]:
                        value = f"{value*100:.2f}%"
                    else:
                        value = f"{value:.2f}"
                report.append(f"  {metric}: {value}")
        
        report.append("\nGrowth Metrics:")
        for metric in growth_metrics:
            if metric in metrics:
                value = metrics[metric]
                if isinstance(value, float):
                    value = f"{value*100:.2f}%"
                report.append(f"  {metric}: {value}")
        
        report.append("\nFinancial Health:")
        for metric in health_metrics:
            if metric in metrics:
                value = metrics[metric]
                if isinstance(value, float) and metric not in ["Total Debt", "Total Cash"]:
                    value = f"{value:.2f}"
                elif isinstance(value, (int, float)) and metric in ["Total Debt", "Total Cash"]:
                    value = f"${value:,.0f}"
                report.append(f"  {metric}: {value}")
        
        report.append("\nDividend Information:")
        for metric in dividend_metrics:
            if metric in metrics:
                value = metrics[metric]
                if isinstance(value, float):
                    if metric == "Dividend Yield":
                        value = f"{value*100:.2f}%"
                    else:
                        value = f"{value:.2f}"
                report.append(f"  {metric}: {value}")
        
        report.append("")
        
        # Custom Ratios
        custom_ratios = self.calculate_custom_ratios()
        if custom_ratios:
            report.append("🔍 CALCULATED RATIOS")
            report.append("-" * 40)
            for key, value in custom_ratios.items():
                if isinstance(value, float):
                    if "Margin" in key:
                        value = f"{value:.2f}%"
                    else:
                        value = f"{value:.2f}"
                report.append(f"{key}: {value}")
            report.append("")
        
        # Financial Trends
        trends = self.analyze_financial_trends()
        if trends:
            report.append("📈 FINANCIAL TRENDS")
            report.append("-" * 40)
            for key, value in trends.items():
                report.append(f"{key}: {value}")
            report.append("")
        
        # Analyst Recommendations
        analyst_data = self.get_analyst_recommendations()
        if analyst_data:
            report.append("👥 ANALYST RECOMMENDATIONS")
            report.append("-" * 40)
            
            if "Latest Recommendation" in analyst_data:
                rec = analyst_data["Latest Recommendation"]
                total_recs = sum(rec.values())
                if total_recs > 0:
                    report.append("Analyst Sentiment Distribution:")
                    for rating, count in rec.items():
                        percentage = (count / total_recs) * 100
                        report.append(f"  {rating}: {count} ({percentage:.1f}%)")
                    report.append("")
            
            target_metrics = ["Target Mean Price", "Target High Price", "Target Low Price", 
                            "Recommendation Mean", "Number of Analyst Opinions"]
            for metric in target_metrics:
                if metric in analyst_data:
                    value = analyst_data[metric]
                    if isinstance(value, float) and "Price" in metric:
                        value = f"${value:.2f}"
                    report.append(f"{metric}: {value}")
            report.append("")
        
        # Investment Score
        investment_score = self.generate_investment_score()
        report.append("⭐ INVESTMENT SCORE")
        report.append("-" * 40)
        report.append(f"Overall Score: {investment_score['Score']}/{investment_score['Max Score']} ({investment_score['Percentage']})")
        report.append("")
        report.append("Score Factors:")
        for reason in investment_score['Reasons']:
            report.append(f"  ✓ {reason}")
        report.append("")
        
        # Summary and Recommendation
        report.append("📝 SUMMARY & RECOMMENDATION")
        report.append("-" * 40)
        
        score_pct = (investment_score['Score'] / investment_score['Max Score']) * 100
        
        if score_pct >= 70:
            recommendation = "STRONG BUY"
            explanation = "The stock shows strong fundamentals across multiple metrics."
        elif score_pct >= 50:
            recommendation = "BUY/HOLD"
            explanation = "The stock shows solid fundamentals with some areas of concern."
        elif score_pct >= 30:
            recommendation = "HOLD/NEUTRAL"
            explanation = "The stock shows mixed fundamentals requiring careful consideration."
        else:
            recommendation = "CAUTION"
            explanation = "The stock shows weak fundamentals and may pose investment risks."
        
        report.append(f"Recommendation: {recommendation}")
        report.append(f"Explanation: {explanation}")
        report.append("")
        
        # Risk Factors
        report.append("⚠️  KEY RISK FACTORS TO CONSIDER")
        report.append("-" * 40)
        
        risk_factors = []
        
        # Check for high valuation
        pe_ratio = self.info.get("trailingPE") if self.info else None
        if pe_ratio and pe_ratio > 30:
            risk_factors.append("High P/E ratio suggests potential overvaluation")
        
        # Check for high debt
        debt_to_equity = self.info.get("debtToEquity") if self.info else None
        if debt_to_equity and debt_to_equity > 100:
            risk_factors.append("High debt-to-equity ratio indicates financial leverage risk")
        
        # Check for negative growth
        revenue_growth = self.info.get("revenueGrowth") if self.info else None
        if revenue_growth and revenue_growth < 0:
            risk_factors.append("Negative revenue growth indicates declining business")
        
        # Check for low profitability
        profit_margin = self.info.get("profitMargins") if self.info else None
        if profit_margin and profit_margin < 0.05:  # Less than 5%
            risk_factors.append("Low profit margins may indicate operational challenges")
        
        if not risk_factors:
            risk_factors.append("No major red flags identified in the analysis")
        
        for risk in risk_factors:
            report.append(f"  • {risk}")
        
        report.append("")
        report.append("=" * 80)
        report.append("Disclaimer: This analysis is for informational purposes only and should not be")
        report.append("considered as investment advice. Please conduct your own research and consult")
        report.append("with financial professionals before making investment decisions.")
        report.append("=" * 80)
        
        return "\n".join(report)


def analyze_stock_fundamentals(ticker: str, save_to_file: bool = False, filename: Optional[str] = None) -> str:
    """
    Main function to perform comprehensive fundamental analysis on a stock.
    
    Args:
        ticker (str): Stock ticker symbol
        save_to_file (bool): Whether to save the report to a file
        filename (str, optional): Custom filename for the report
    
    Returns:
        str: Comprehensive analysis report
    """
    try:
        analyst = FundamentalAnalyst(ticker)
        report = analyst.generate_comprehensive_report()
        
        if save_to_file:
            if not filename:
                filename = f"{ticker}_fundamental_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"Report saved to: {filename}")
        
        return report
    
    except Exception as e:
        error_msg = f"Error analyzing {ticker}: {str(e)}"
        print(error_msg)
        return error_msg


def get_current_date():
    """Get current date in YYYY-MM-DD format."""
    return date.today().strftime("%Y-%m-%d")


def compare_stocks(tickers: List[str]) -> pd.DataFrame:
    """
    Compare fundamental metrics across multiple stocks.
    
    Args:
        tickers (List[str]): List of stock ticker symbols
    
    Returns:
        pandas.DataFrame: Comparison table of key metrics
    """
    comparison_data = []
    
    for ticker in tickers:
        try:
            analyst = FundamentalAnalyst(ticker)
            metrics = analyst.get_key_financial_metrics()
            score_data = analyst.generate_investment_score()
            
            row_data = {
                "Ticker": ticker,
                "PE Ratio": metrics.get("PE Ratio", "N/A"),
                "PEG Ratio": metrics.get("PEG Ratio", "N/A"),
                "Profit Margin": metrics.get("Profit Margin", "N/A"),
                "ROE": metrics.get("Return on Equity", "N/A"),
                "Revenue Growth": metrics.get("Revenue Growth", "N/A"),
                "Debt to Equity": metrics.get("Debt to Equity", "N/A"),
                "Investment Score": f"{score_data['Score']}/{score_data['Max Score']}"
            }
            comparison_data.append(row_data)
            
        except Exception as e:
            print(f"Error analyzing {ticker}: {e}")
            comparison_data.append({"Ticker": ticker, "Error": str(e)})
    
    return pd.DataFrame(comparison_data)


# Example usage and testing
if __name__ == "__main__":
    print("Fundamental Analysis Module - Example Usage")
    print("=" * 50)
    
    # Example 1: Single stock analysis
    print("\n1. Analyzing Apple (AAPL)...")
    try:
        apple_report = analyze_stock_fundamentals("AAPL")
        print(apple_report[:1000] + "...\n[Report truncated for display]")
    except Exception as e:
        print(f"Error: {e}")
    
    # Example 2: Compare multiple stocks
    print("\n2. Comparing multiple tech stocks...")
    try:
        comparison = compare_stocks(["AAPL", "MSFT", "GOOGL"])
        print(comparison.to_string(index=False))
    except Exception as e:
        print(f"Error: {e}")
    
    # Example 3: Custom analysis
    print("\n3. Creating custom analyst object...")
    try:
        analyst = FundamentalAnalyst("TSLA")
        company_info = analyst.get_company_info()
        print(f"Company: {company_info.get('Company Name')}")
        print(f"Sector: {company_info.get('Sector')}")
        print(f"Market Cap: {company_info.get('Market Cap')}")
        
        investment_score = analyst.generate_investment_score()
        print(f"Investment Score: {investment_score['Score']}/{investment_score['Max Score']} ({investment_score['Percentage']})")
    except Exception as e:
        print(f"Error: {e}")
