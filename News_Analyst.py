"""
News Analysis Module

This module provides comprehensive news analysis functionality for stocks,
inspired by the TradingAgents project. It includes news fetching from multiple sources,
sentiment analysis, and automated report generation for trading insights.
"""

import requests
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from typing import Dict, List, Tuple, Optional, Any, Union
import json
import warnings
import time
import random
import re
from urllib.parse import quote_plus
from bs4 import BeautifulSoup
import feedparser

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

class NewsAnalyst:
    """
    A comprehensive news analysis class that provides news fetching, sentiment analysis,
    and trading insights for stocks.
    """
    
    def __init__(self, ticker: str):
        """
        Initialize the NewsAnalyst with a stock ticker.
        
        Args:
            ticker (str): Stock ticker symbol (e.g., 'AAPL', 'GOOGL')
        """
        self.ticker = ticker.upper()
        self.stock = yf.Ticker(self.ticker)
        self.company_name = self._get_company_name()
        
        # News sentiment keywords for analysis
        self.positive_keywords = [
            'surge', 'rally', 'gain', 'profit', 'growth', 'beat', 'exceed', 'strong', 
            'positive', 'bullish', 'breakthrough', 'success', 'record', 'high', 
            'upgrade', 'outperform', 'buy', 'optimistic', 'boom', 'soar', 'rise'
        ]
        
        self.negative_keywords = [
            'fall', 'drop', 'decline', 'loss', 'crash', 'plunge', 'weak', 'negative', 
            'bearish', 'downgrade', 'sell', 'concern', 'worry', 'risk', 'miss', 
            'disappoint', 'struggle', 'challenge', 'crisis', 'recession', 'cut'
        ]
        
        self.neutral_keywords = [
            'stable', 'maintain', 'hold', 'unchanged', 'steady', 'flat', 'sideways',
            'mixed', 'neutral', 'balanced', 'cautious', 'wait', 'monitor'
        ]
    
    def _get_company_name(self) -> str:
        """Get company name for news searches."""
        try:
            info = self.stock.info
            return info.get('shortName', self.ticker)
        except:
            return self.ticker
    
    def _make_request_with_retry(self, url: str, headers: Dict[str, str], max_retries: int = 3) -> Optional[requests.Response]:
        """Make HTTP request with retry logic and rate limiting."""
        for attempt in range(max_retries):
            try:
                # Random delay to avoid rate limiting
                time.sleep(random.uniform(1, 3))
                response = requests.get(url, headers=headers, timeout=10)
                
                if response.status_code == 200:
                    return response
                elif response.status_code == 429:  # Rate limited
                    time.sleep(random.uniform(5, 10))
                    continue
                else:
                    print(f"Request failed with status {response.status_code}")
                    
            except Exception as e:
                print(f"Request attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(random.uniform(2, 5))
        
        return None
    
    def get_yahoo_finance_news(self, max_articles: int = 20) -> List[Dict[str, Any]]:
        """
        Fetch news from Yahoo Finance for the specific ticker.
        
        Args:
            max_articles (int): Maximum number of articles to fetch
            
        Returns:
            List of news articles
        """
        news_articles = []
        
        try:
            # Get news from yfinance
            news = self.stock.news
            
            for article in news[:max_articles]:
                processed_article = {
                    'title': article.get('title', ''),
                    'link': article.get('link', ''),
                    'published': datetime.fromtimestamp(article.get('providerPublishTime', 0)),
                    'publisher': article.get('publisher', ''),
                    'summary': article.get('summary', ''),
                    'source': 'Yahoo Finance',
                    'relevance_score': self._calculate_relevance_score(article.get('title', '') + ' ' + article.get('summary', ''))
                }
                news_articles.append(processed_article)
                
        except Exception as e:
            print(f"Error fetching Yahoo Finance news: {e}")
        
        return news_articles
    
    def get_google_news(self, days_back: int = 7, max_articles: int = 20) -> List[Dict[str, Any]]:
        """
        Fetch news from Google News using RSS feeds.
        
        Args:
            days_back (int): Number of days to look back for news
            max_articles (int): Maximum number of articles to fetch
            
        Returns:
            List of news articles
        """
        news_articles = []
        
        try:
            # Create search queries for both ticker and company name
            search_queries = [self.ticker, self.company_name]
            
            for query in search_queries:
                # Google News RSS feed URL
                encoded_query = quote_plus(f"{query} stock news")
                url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"
                
                try:
                    feed = feedparser.parse(url)
                    
                    cutoff_date = datetime.now() - timedelta(days=days_back)
                    
                    for entry in feed.entries[:max_articles]:
                        try:
                            # Parse publication date
                            pub_date = datetime(*entry.published_parsed[:6]) if hasattr(entry, 'published_parsed') else datetime.now()
                            
                            if pub_date >= cutoff_date:
                                article = {
                                    'title': entry.get('title', ''),
                                    'link': entry.get('link', ''),
                                    'published': pub_date,
                                    'publisher': entry.get('source', {}).get('title', 'Unknown'),
                                    'summary': entry.get('summary', ''),
                                    'source': 'Google News',
                                    'relevance_score': self._calculate_relevance_score(entry.get('title', '') + ' ' + entry.get('summary', ''))
                                }
                                news_articles.append(article)
                                
                        except Exception as e:
                            print(f"Error processing Google News entry: {e}")
                            continue
                            
                except Exception as e:
                    print(f"Error fetching Google News feed for {query}: {e}")
                    continue
                
        except Exception as e:
            print(f"Error in Google News fetching: {e}")
        
        # Remove duplicates based on title similarity
        unique_articles = self._remove_duplicate_articles(news_articles)
        
        # Sort by relevance score and date
        unique_articles.sort(key=lambda x: (x['relevance_score'], x['published']), reverse=True)
        
        return unique_articles[:max_articles]
    
    def get_financial_news_feeds(self, max_articles: int = 15) -> List[Dict[str, Any]]:
        """
        Fetch news from major financial news RSS feeds.
        
        Args:
            max_articles (int): Maximum number of articles per feed
            
        Returns:
            List of news articles
        """
        news_articles = []
        
        # Major financial news RSS feeds
        feeds = {
            'Reuters Business': 'https://feeds.reuters.com/reuters/businessNews',
            'MarketWatch': 'https://feeds.marketwatch.com/marketwatch/marketpulse/',
            'Bloomberg': 'https://feeds.bloomberg.com/markets/news.rss',
            'Yahoo Finance': 'https://feeds.finance.yahoo.com/rss/2.0/headline',
            'CNBC': 'https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=10000664'
        }
        
        for source_name, feed_url in feeds.items():
            try:
                feed = feedparser.parse(feed_url)
                
                for entry in feed.entries[:max_articles]:
                    try:
                        # Check if article is relevant to our stock
                        content = (entry.get('title', '') + ' ' + entry.get('summary', '')).lower()
                        
                        if (self.ticker.lower() in content or 
                            self.company_name.lower() in content or
                            any(keyword in content for keyword in ['market', 'stock', 'trading', 'earnings', 'financial'])):
                            
                            pub_date = datetime(*entry.published_parsed[:6]) if hasattr(entry, 'published_parsed') else datetime.now()
                            
                            article = {
                                'title': entry.get('title', ''),
                                'link': entry.get('link', ''),
                                'published': pub_date,
                                'publisher': source_name,
                                'summary': entry.get('summary', ''),
                                'source': f'{source_name} RSS',
                                'relevance_score': self._calculate_relevance_score(content)
                            }
                            news_articles.append(article)
                            
                    except Exception as e:
                        print(f"Error processing {source_name} entry: {e}")
                        continue
                        
            except Exception as e:
                print(f"Error fetching {source_name} feed: {e}")
                continue
        
        return news_articles
    
    def _calculate_relevance_score(self, text: str) -> float:
        """
        Calculate relevance score for news articles based on ticker and company mentions.
        
        Args:
            text (str): Article text to analyze
            
        Returns:
            float: Relevance score (0-10)
        """
        text_lower = text.lower()
        score = 0.0
        
        # Direct ticker mention (high relevance)
        if self.ticker.lower() in text_lower:
            score += 5.0
        
        # Company name mention
        if self.company_name.lower() in text_lower:
            score += 3.0
        
        # Financial keywords
        financial_keywords = ['earnings', 'revenue', 'profit', 'stock', 'shares', 'market', 'trading', 'financial']
        for keyword in financial_keywords:
            if keyword in text_lower:
                score += 0.5
        
        # Industry-specific terms (basic implementation)
        industry_keywords = ['technology', 'healthcare', 'finance', 'energy', 'consumer', 'industrial']
        for keyword in industry_keywords:
            if keyword in text_lower:
                score += 0.3
        
        return min(score, 10.0)  # Cap at 10
    
    def _remove_duplicate_articles(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate articles based on title similarity."""
        unique_articles = []
        seen_titles = set()
        
        for article in articles:
            # Simple deduplication based on first 50 characters of title
            title_key = article['title'][:50].lower().strip()
            
            if title_key not in seen_titles:
                seen_titles.add(title_key)
                unique_articles.append(article)
        
        return unique_articles
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment of text using keyword-based approach.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            Dict containing sentiment analysis results
        """
        text_lower = text.lower()
        
        positive_count = sum(1 for word in self.positive_keywords if word in text_lower)
        negative_count = sum(1 for word in self.negative_keywords if word in text_lower)
        neutral_count = sum(1 for word in self.neutral_keywords if word in text_lower)
        
        total_sentiment_words = positive_count + negative_count + neutral_count
        
        if total_sentiment_words == 0:
            sentiment_score = 0.0
            sentiment_label = "Neutral"
        else:
            # Calculate weighted sentiment score (-1 to 1)
            sentiment_score = (positive_count - negative_count) / max(total_sentiment_words, 1)
            
            if sentiment_score > 0.3:
                sentiment_label = "Positive"
            elif sentiment_score < -0.3:
                sentiment_label = "Negative"
            else:
                sentiment_label = "Neutral"
        
        return {
            'sentiment_score': sentiment_score,
            'sentiment_label': sentiment_label,
            'positive_keywords_count': positive_count,
            'negative_keywords_count': negative_count,
            'neutral_keywords_count': neutral_count,
            'confidence': min(total_sentiment_words / 10.0, 1.0)  # Confidence based on keyword density
        }
    
    def get_comprehensive_news_analysis(self, days_back: int = 7, max_articles: int = 50) -> Dict[str, Any]:
        """
        Get comprehensive news analysis combining multiple sources.
        
        Args:
            days_back (int): Number of days to look back
            max_articles (int): Maximum total articles to analyze
            
        Returns:
            Dict containing comprehensive news analysis
        """
        print(f"🔍 Fetching news for {self.ticker} ({self.company_name})...")
        
        # Fetch news from multiple sources
        yahoo_news = self.get_yahoo_finance_news(max_articles=15)
        google_news = self.get_google_news(days_back=days_back, max_articles=15)
        financial_feeds = self.get_financial_news_feeds(max_articles=10)
        
        # Combine all news sources
        all_articles = yahoo_news + google_news + financial_feeds
        
        # Remove duplicates and sort by relevance
        unique_articles = self._remove_duplicate_articles(all_articles)
        unique_articles.sort(key=lambda x: (x['relevance_score'], x['published']), reverse=True)
        
        # Limit to max_articles
        unique_articles = unique_articles[:max_articles]
        
        print(f"📰 Analyzing {len(unique_articles)} relevant articles...")
        
        # Analyze sentiment for each article
        analyzed_articles = []
        overall_sentiment_scores = []
        
        for article in unique_articles:
            text_for_analysis = f"{article['title']} {article['summary']}"
            sentiment = self.analyze_sentiment(text_for_analysis)
            
            article['sentiment'] = sentiment
            analyzed_articles.append(article)
            
            # Weight sentiment by relevance score for overall calculation
            weighted_score = sentiment['sentiment_score'] * (article['relevance_score'] / 10.0)
            overall_sentiment_scores.append(weighted_score)
        
        # Calculate overall sentiment
        if overall_sentiment_scores:
            overall_sentiment = np.mean(overall_sentiment_scores)
            overall_confidence = np.mean([article['sentiment']['confidence'] for article in analyzed_articles])
        else:
            overall_sentiment = 0.0
            overall_confidence = 0.0
        
        # Determine overall sentiment label
        if overall_sentiment > 0.2:
            overall_label = "Positive"
        elif overall_sentiment < -0.2:
            overall_label = "Negative"
        else:
            overall_label = "Neutral"
        
        # Generate insights
        insights = self._generate_news_insights(analyzed_articles, overall_sentiment)
        
        return {
            'ticker': self.ticker,
            'company_name': self.company_name,
            'analysis_date': datetime.now(),
            'articles_analyzed': len(analyzed_articles),
            'days_back': days_back,
            'articles': analyzed_articles,
            'overall_sentiment': {
                'score': overall_sentiment,
                'label': overall_label,
                'confidence': overall_confidence
            },
            'insights': insights,
            'source_breakdown': self._get_source_breakdown(analyzed_articles)
        }
    
    def _generate_news_insights(self, articles: List[Dict[str, Any]], overall_sentiment: float) -> List[str]:
        """Generate trading insights based on news analysis."""
        insights = []
        
        if not articles:
            insights.append("No relevant news articles found for analysis")
            return insights
        
        # Sentiment-based insights
        positive_articles = [a for a in articles if a['sentiment']['sentiment_label'] == 'Positive']
        negative_articles = [a for a in articles if a['sentiment']['sentiment_label'] == 'Negative']
        
        if len(positive_articles) > len(negative_articles) * 1.5:
            insights.append(f"Strong positive news momentum with {len(positive_articles)} positive vs {len(negative_articles)} negative articles")
        elif len(negative_articles) > len(positive_articles) * 1.5:
            insights.append(f"Concerning negative news trend with {len(negative_articles)} negative vs {len(positive_articles)} positive articles")
        else:
            insights.append("Mixed news sentiment - monitor for directional changes")
        
        # Recent news analysis (last 24 hours)
        recent_cutoff = datetime.now() - timedelta(hours=24)
        recent_articles = [a for a in articles if a['published'] >= recent_cutoff]
        
        if recent_articles:
            recent_sentiment = np.mean([a['sentiment']['sentiment_score'] for a in recent_articles])
            if abs(recent_sentiment) > 0.3:
                direction = "positive" if recent_sentiment > 0 else "negative"
                insights.append(f"Recent 24h news shows {direction} sentiment shift - potential short-term impact")
        
        # High-relevance articles
        high_relevance = [a for a in articles if a['relevance_score'] >= 7.0]
        if high_relevance:
            insights.append(f"{len(high_relevance)} high-relevance articles found - direct stock impact likely")
        
        # Volume analysis
        if len(articles) > 20:
            insights.append("High news volume detected - increased market attention expected")
        elif len(articles) < 5:
            insights.append("Low news volume - limited market impact expected")
        
        return insights
    
    def _get_source_breakdown(self, articles: List[Dict[str, Any]]) -> Dict[str, int]:
        """Get breakdown of articles by source."""
        source_counts = {}
        for article in articles:
            source = article['source']
            source_counts[source] = source_counts.get(source, 0) + 1
        return source_counts
    
    def generate_news_report(self, days_back: int = 7, max_articles: int = 50) -> str:
        """
        Generate a comprehensive news analysis report.
        
        Args:
            days_back (int): Number of days to analyze
            max_articles (int): Maximum articles to include
            
        Returns:
            Formatted string report
        """
        analysis = self.get_comprehensive_news_analysis(days_back=days_back, max_articles=max_articles)
        
        report = []
        report.append("=" * 80)
        report.append(f"NEWS ANALYSIS REPORT: {self.ticker}")
        report.append("=" * 80)
        report.append(f"Company: {analysis['company_name']}")
        report.append(f"Analysis Period: {days_back} days back from {analysis['analysis_date'].strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Articles Analyzed: {analysis['articles_analyzed']}")
        report.append("")
        
        # Overall Sentiment
        sentiment = analysis['overall_sentiment']
        report.append("📈 OVERALL NEWS SENTIMENT")
        report.append("-" * 40)
        report.append(f"Sentiment Score: {sentiment['score']:.3f} (Range: -1 to +1)")
        report.append(f"Sentiment Label: {sentiment['label']}")
        report.append(f"Confidence Level: {sentiment['confidence']:.2f}")
        
        # Sentiment interpretation
        if sentiment['score'] > 0.5:
            interpretation = "Strong positive sentiment - bullish news environment"
        elif sentiment['score'] > 0.2:
            interpretation = "Moderately positive sentiment - cautiously optimistic"
        elif sentiment['score'] > -0.2:
            interpretation = "Neutral sentiment - balanced news coverage"
        elif sentiment['score'] > -0.5:
            interpretation = "Moderately negative sentiment - some concerns present"
        else:
            interpretation = "Strong negative sentiment - bearish news environment"
        
        report.append(f"Interpretation: {interpretation}")
        report.append("")
        
        # Key Insights
        report.append("🔍 KEY INSIGHTS & TRADING IMPLICATIONS")
        report.append("-" * 40)
        for i, insight in enumerate(analysis['insights'], 1):
            report.append(f"{i}. {insight}")
        report.append("")
        
        # Source Breakdown
        report.append("📊 NEWS SOURCE BREAKDOWN")
        report.append("-" * 40)
        for source, count in analysis['source_breakdown'].items():
            percentage = (count / analysis['articles_analyzed']) * 100
            report.append(f"{source}: {count} articles ({percentage:.1f}%)")
        report.append("")
        
        # Recent High-Impact Articles
        recent_articles = sorted(analysis['articles'], key=lambda x: x['published'], reverse=True)[:5]
        high_relevance_articles = [a for a in recent_articles if a['relevance_score'] >= 6.0]
        
        if high_relevance_articles:
            report.append("🗞️  RECENT HIGH-IMPACT ARTICLES")
            report.append("-" * 40)
            for i, article in enumerate(high_relevance_articles, 1):
                report.append(f"{i}. {article['title']}")
                report.append(f"   Publisher: {article['publisher']}")
                report.append(f"   Published: {article['published'].strftime('%Y-%m-%d %H:%M')}")
                report.append(f"   Sentiment: {article['sentiment']['sentiment_label']} ({article['sentiment']['sentiment_score']:.2f})")
                report.append(f"   Relevance: {article['relevance_score']:.1f}/10")
                if article['summary']:
                    summary = article['summary'][:200] + "..." if len(article['summary']) > 200 else article['summary']
                    report.append(f"   Summary: {summary}")
                report.append("")
        
        # Sentiment Distribution
        positive_count = len([a for a in analysis['articles'] if a['sentiment']['sentiment_label'] == 'Positive'])
        negative_count = len([a for a in analysis['articles'] if a['sentiment']['sentiment_label'] == 'Negative'])
        neutral_count = len([a for a in analysis['articles'] if a['sentiment']['sentiment_label'] == 'Neutral'])
        
        report.append("📊 SENTIMENT DISTRIBUTION")
        report.append("-" * 40)
        if analysis['articles_analyzed'] > 0:
            report.append(f"Positive Articles: {positive_count} ({(positive_count/analysis['articles_analyzed'])*100:.1f}%)")
            report.append(f"Negative Articles: {negative_count} ({(negative_count/analysis['articles_analyzed'])*100:.1f}%)")
            report.append(f"Neutral Articles: {neutral_count} ({(neutral_count/analysis['articles_analyzed'])*100:.1f}%)")
        report.append("")
        
        # Trading Recommendation
        report.append("🎯 NEWS-BASED TRADING RECOMMENDATION")
        report.append("-" * 40)
        
        if sentiment['score'] > 0.4 and sentiment['confidence'] > 0.6:
            recommendation = "POSITIVE NEWS MOMENTUM - Consider bullish positions"
        elif sentiment['score'] < -0.4 and sentiment['confidence'] > 0.6:
            recommendation = "NEGATIVE NEWS MOMENTUM - Consider bearish positions or exit"
        elif abs(sentiment['score']) < 0.2:
            recommendation = "NEUTRAL NEWS ENVIRONMENT - Monitor for changes"
        else:
            recommendation = "MIXED SIGNALS - Exercise caution and wait for clarity"
        
        report.append(f"Recommendation: {recommendation}")
        report.append("")
        
        # Risk Factors
        report.append("⚠️  NEWS-BASED RISK FACTORS")
        report.append("-" * 40)
        
        risk_factors = []
        
        if negative_count > positive_count * 2:
            risk_factors.append("High negative news ratio - potential downward pressure")
        
        if sentiment['confidence'] < 0.3:
            risk_factors.append("Low confidence in sentiment analysis - limited news data")
        
        recent_negative = [a for a in recent_articles[:3] if a['sentiment']['sentiment_label'] == 'Negative']
        if len(recent_negative) >= 2:
            risk_factors.append("Recent negative news concentration - immediate impact possible")
        
        if analysis['articles_analyzed'] < 5:
            risk_factors.append("Limited news coverage - may not reflect full market sentiment")
        
        if not risk_factors:
            risk_factors.append("No major news-based risk factors identified")
        
        for risk in risk_factors:
            report.append(f"• {risk}")
        
        report.append("")
        report.append("=" * 80)
        report.append("Disclaimer: This news analysis is for informational purposes only. News sentiment")
        report.append("can change rapidly and should be combined with fundamental and technical analysis")
        report.append("for making investment decisions. Always conduct your own research.")
        report.append("=" * 80)
        
        return "\n".join(report)


def analyze_stock_news(ticker: str, days_back: int = 7, max_articles: int = 50, save_to_file: bool = False, filename: Optional[str] = None) -> str:
    """
    Main function to perform comprehensive news analysis on a stock.
    
    Args:
        ticker (str): Stock ticker symbol
        days_back (int): Number of days to look back for news
        max_articles (int): Maximum number of articles to analyze
        save_to_file (bool): Whether to save the report to a file
        filename (str, optional): Custom filename for the report
    
    Returns:
        str: Comprehensive news analysis report
    """
    try:
        analyst = NewsAnalyst(ticker)
        report = analyst.generate_news_report(days_back=days_back, max_articles=max_articles)
        
        if save_to_file:
            if not filename:
                filename = f"{ticker}_news_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"Report saved to: {filename}")
        
        return report
    
    except Exception as e:
        error_msg = f"Error analyzing news for {ticker}: {str(e)}"
        print(error_msg)
        return error_msg


def compare_news_sentiment(tickers: List[str], days_back: int = 7) -> pd.DataFrame:
    """
    Compare news sentiment across multiple stocks.
    
    Args:
        tickers (List[str]): List of stock ticker symbols
        days_back (int): Number of days to analyze
    
    Returns:
        pandas.DataFrame: Comparison table of news sentiment
    """
    comparison_data = []
    
    for ticker in tickers:
        try:
            analyst = NewsAnalyst(ticker)
            analysis = analyst.get_comprehensive_news_analysis(days_back=days_back, max_articles=30)
            
            row_data = {
                "Ticker": ticker,
                "Company": analysis['company_name'],
                "Articles": analysis['articles_analyzed'],
                "Sentiment Score": f"{analysis['overall_sentiment']['score']:.3f}",
                "Sentiment Label": analysis['overall_sentiment']['label'],
                "Confidence": f"{analysis['overall_sentiment']['confidence']:.2f}",
                "Positive Articles": len([a for a in analysis['articles'] if a['sentiment']['sentiment_label'] == 'Positive']),
                "Negative Articles": len([a for a in analysis['articles'] if a['sentiment']['sentiment_label'] == 'Negative'])
            }
            comparison_data.append(row_data)
            
        except Exception as e:
            print(f"Error analyzing {ticker}: {e}")
            comparison_data.append({"Ticker": ticker, "Error": str(e)})
    
    return pd.DataFrame(comparison_data)


# Example usage and testing
if __name__ == "__main__":
    print("News Analysis Module - Example Usage")
    print("=" * 50)
    
    # Example 1: Single stock news analysis
    print("\n1. Analyzing Apple (AAPL) news...")
    try:
        apple_report = analyze_stock_news("AAPL", days_back=3, max_articles=20)
        print(apple_report[:1000] + "...\n[Report truncated for display]")
    except Exception as e:
        print(f"Error: {e}")
    
    # Example 2: Compare news sentiment across multiple stocks
    print("\n2. Comparing news sentiment for tech stocks...")
    try:
        comparison = compare_news_sentiment(["AAPL", "MSFT", "GOOGL"], days_back=5)
        print(comparison.to_string(index=False))
    except Exception as e:
        print(f"Error: {e}")
    
    # Example 3: Custom news analyst usage
    print("\n3. Creating custom news analyst object...")
    try:
        analyst = NewsAnalyst("TSLA")
        analysis = analyst.get_comprehensive_news_analysis(days_back=2, max_articles=15)
        print(f"Company: {analysis['company_name']}")
        print(f"Articles Analyzed: {analysis['articles_analyzed']}")
        print(f"Overall Sentiment: {analysis['overall_sentiment']['label']} ({analysis['overall_sentiment']['score']:.3f})")
        print(f"Key Insights: {len(analysis['insights'])} insights generated")
    except Exception as e:
        print(f"Error: {e}")
