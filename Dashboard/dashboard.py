import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np
from collections import Counter
import os
from datetime import datetime, timedelta
import io

# --- Page Configuration ---
st.set_page_config(
    page_title="Reddit Sentiment & Topic Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for better styling ---
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
    .insight-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# --- Enhanced Data Loading ---
@st.cache_data
def load_data():
    """Loads the final analyzed data from the CSV file."""
    path = '../data/processed/final_analyzed_data.csv'
    if not os.path.exists(path):
        st.error("Error: final_analyzed_data.csv not found! Please run the analysis notebook first.")
        return None
    
    df = pd.read_csv(path)
    df['created_utc'] = pd.to_datetime(df['created_utc'])
    
    # Add derived columns for analysis
    df['hour'] = df['created_utc'].dt.hour
    df['day_of_week'] = df['created_utc'].dt.day_name()
    df['date'] = df['created_utc'].dt.date
    df['sentiment_category'] = pd.cut(df['sentiment'], 
                                    bins=[-1, -0.05, 0.05, 1], 
                                    labels=['Negative', 'Neutral', 'Positive'])
    
    return df

# --- Word Cloud Generation ---
@st.cache_data
def generate_wordcloud(text, colormap='viridis', max_words=100):
    """Generate word cloud from text."""
    if not text or len(text.strip()) == 0:
        return None
    
    wordcloud = WordCloud(
        width=800, 
        height=400,
        background_color='white',
        max_words=max_words,
        colormap=colormap,
        relative_scaling=0.5,
        random_state=42
    ).generate(text)
    
    return wordcloud

# --- Advanced Analytics Functions ---
@st.cache_data
def calculate_insights(df):
    """Calculate detailed insights from the data."""
    insights = {}
    
    for subreddit in df['subreddit'].unique():
        sub_data = df[df['subreddit'] == subreddit]
        
        insights[subreddit] = {
            'total_comments': len(sub_data),
            'avg_sentiment': sub_data['sentiment'].mean(),
            'median_sentiment': sub_data['sentiment'].median(),
            'sentiment_std': sub_data['sentiment'].std(),
            'positive_pct': (sub_data['sentiment'] > 0.05).sum() / len(sub_data) * 100,
            'negative_pct': (sub_data['sentiment'] < -0.05).sum() / len(sub_data) * 100,
            'neutral_pct': ((sub_data['sentiment'] >= -0.05) & (sub_data['sentiment'] <= 0.05)).sum() / len(sub_data) * 100,
            'avg_comment_length': sub_data['comment_body'].str.len().mean(),
            'most_active_hour': sub_data['hour'].mode().iloc[0] if not sub_data['hour'].mode().empty else 0,
            'most_active_day': sub_data['day_of_week'].mode().iloc[0] if not sub_data['day_of_week'].mode().empty else 'Unknown',
            'top_topic': sub_data[sub_data['topic_id'] != -1]['topic_name'].mode().iloc[0] if len(sub_data[sub_data['topic_id'] != -1]) > 0 else 'No topics',
            'topic_diversity': sub_data['topic_id'].nunique() - 1  # Exclude -1
        }
    
    return insights

# --- Main App ---
def main():
    # Load data
    df = load_data()
    
    if df is None:
        st.stop()
    
    # --- Header ---
    st.markdown('<h1 class="main-header">🚀 Reddit Sentiment & Topic Co-evolution Tracker</h1>', unsafe_allow_html=True)
    st.markdown("### Comparative Analysis: r/technology vs. r/startups")
    st.markdown("---")
    
    # --- Sidebar Configuration ---
    with st.sidebar:
        st.header("🎛️ Dashboard Controls")
        
        # Subreddit Selection
        all_subreddits = ["All"] + sorted(df['subreddit'].unique().tolist())
        selected_subreddit = st.selectbox("📱 Select Subreddit", all_subreddits)
        
        # Date Range Selection
        min_date = df['created_utc'].min().date()
        max_date = df['created_utc'].max().date()
        selected_date_range = st.date_input(
            "📅 Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
        
        # Topic Selection
        if 'topic_name' in df.columns:
            available_topics = ["All Topics"] + sorted(df[df['topic_id'] != -1]['topic_name'].unique().tolist())
            selected_topics = st.multiselect(
                "🧠 Filter by Topics",
                available_topics,
                default=["All Topics"]
            )
        
        # Sentiment Range
        sentiment_range = st.slider(
            "😊 Sentiment Range",
            min_value=-1.0,
            max_value=1.0,
            value=(-1.0, 1.0),
            step=0.1
        )
        
        # Analysis Options
        st.subheader("📊 Analysis Options")
        show_outliers = st.checkbox("Include Topic Outliers", value=False)
        min_comment_length = st.slider("Min Comment Length", 0, 500, 0)
        
        # Download Section
        st.subheader("💾 Data Export")
        if st.button("Download Filtered Data"):
            # Apply filters to get download data
            download_df = apply_filters(df, selected_subreddit, selected_date_range, 
                                     selected_topics, sentiment_range, show_outliers, min_comment_length)
            csv = download_df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"reddit_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )
    
    # --- Apply Filters ---
    filtered_df = apply_filters(df, selected_subreddit, selected_date_range, 
                               selected_topics, sentiment_range, show_outliers, min_comment_length)
    
    # --- Main Dashboard Tabs ---
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Overview", "☁️ Word Clouds", "📈 Trends", "🧠 Topics", "🔍 Deep Insights"])
    
    with tab1:
        overview_tab(filtered_df, df)
    
    with tab2:
        wordcloud_tab(filtered_df)
    
    with tab3:
        trends_tab(filtered_df)
    
    with tab4:
        topics_tab(filtered_df)
    
    with tab5:
        insights_tab(filtered_df, df)

# --- Filter Application Function ---
def apply_filters(df, subreddit, date_range, topics, sentiment_range, show_outliers, min_length):
    """Apply all selected filters to the dataframe."""
    filtered_df = df.copy()
    
    # Subreddit filter
    if subreddit != "All":
        filtered_df = filtered_df[filtered_df['subreddit'] == subreddit]
    
    # Date filter
    if len(date_range) == 2:
        start_date, end_date = date_range
        filtered_df = filtered_df[
            (filtered_df['created_utc'].dt.date >= start_date) & 
            (filtered_df['created_utc'].dt.date <= end_date)
        ]
    
    # Topic filter
    if 'topic_name' in filtered_df.columns and topics and "All Topics" not in topics:
        filtered_df = filtered_df[filtered_df['topic_name'].isin(topics)]
    
    # Sentiment filter
    min_sent, max_sent = sentiment_range
    filtered_df = filtered_df[
        (filtered_df['sentiment'] >= min_sent) & 
        (filtered_df['sentiment'] <= max_sent)
    ]
    
    # Outliers filter
    if not show_outliers and 'topic_id' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['topic_id'] != -1]
    
    # Comment length filter
    if min_length > 0:
        filtered_df = filtered_df[filtered_df['comment_body'].str.len() >= min_length]
    
    return filtered_df

# --- Tab Content Functions ---
def overview_tab(filtered_df, full_df):
    """Overview tab with key metrics and basic visualizations."""
    
    # Key Metrics
    st.subheader("📋 Key Metrics")
    
    if not filtered_df.empty:
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Total Comments", f"{len(filtered_df):,}")
        with col2:
            avg_sentiment = filtered_df['sentiment'].mean()
            st.metric("Avg Sentiment", f"{avg_sentiment:.3f}")
        with col3:
            topics_count = filtered_df['topic_id'].nunique()
            st.metric("Unique Topics", f"{topics_count:,}")
        with col4:
            positive_pct = (filtered_df['sentiment'] > 0.05).sum() / len(filtered_df) * 100
            st.metric("Positive %", f"{positive_pct:.1f}%")
        with col5:
            avg_length = filtered_df['comment_body'].str.len().mean()
            st.metric("Avg Length", f"{avg_length:.0f} chars")
        
        # Community Comparison
        if len(filtered_df['subreddit'].unique()) > 1:
            st.subheader("🏟️ Community Comparison")
            
            comparison_data = []
            for subreddit in filtered_df['subreddit'].unique():
                sub_data = filtered_df[filtered_df['subreddit'] == subreddit]
                comparison_data.append({
                    'Subreddit': f"r/{subreddit}",
                    'Comments': len(sub_data),
                    'Avg Sentiment': sub_data['sentiment'].mean(),
                    'Positive %': (sub_data['sentiment'] > 0.05).sum() / len(sub_data) * 100,
                    'Negative %': (sub_data['sentiment'] < -0.05).sum() / len(sub_data) * 100,
                    'Avg Length': sub_data['comment_body'].str.len().mean()
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df.round(2), use_container_width=True)
        
        # Quick Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Sentiment Distribution
            fig_sent = px.histogram(
                filtered_df,
                x='sentiment',
                color='subreddit',
                title="📊 Sentiment Distribution",
                barmode='overlay',
                opacity=0.7,
                color_discrete_map={'technology': '#0079D3', 'startups': '#FF4500'}
            )
            fig_sent.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="Neutral")
            st.plotly_chart(fig_sent, use_container_width=True)
        
        with col2:
            # Comments by Subreddit
            subreddit_counts = filtered_df['subreddit'].value_counts()
            fig_sub = px.pie(
                values=subreddit_counts.values,
                names=subreddit_counts.index,
                title="📱 Comments by Subreddit",
                color_discrete_map={'technology': '#0079D3', 'startups': '#FF4500'}
            )
            st.plotly_chart(fig_sub, use_container_width=True)
    
    else:
        st.warning("⚠️ No data available for the selected filters.")

def wordcloud_tab(filtered_df):
    """Word clouds tab."""
    st.subheader("☁️ Word Cloud Analysis")
    
    if not filtered_df.empty and 'processed_text' in filtered_df.columns:
        
        # Overall Word Cloud
        st.subheader("🌐 Overall Word Cloud")
        all_text = ' '.join(filtered_df['processed_text'].dropna())
        if all_text:
            wc_overall = generate_wordcloud(all_text, 'viridis', 150)
            if wc_overall:
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.imshow(wc_overall, interpolation='bilinear')
                ax.axis('off')
                ax.set_title('Most Common Words (All Data)', fontsize=16, fontweight='bold')
                st.pyplot(fig)
        
        # Subreddit-specific Word Clouds
        subreddits = filtered_df['subreddit'].unique()
        
        if len(subreddits) > 1:
            col1, col2 = st.columns(2)
            colors = ['Blues', 'Oranges']
            
            for idx, subreddit in enumerate(subreddits):
                sub_data = filtered_df[filtered_df['subreddit'] == subreddit]
                sub_text = ' '.join(sub_data['processed_text'].dropna())
                
                with col1 if idx == 0 else col2:
                    st.subheader(f"r/{subreddit}")
                    if sub_text:
                        wc_sub = generate_wordcloud(sub_text, colors[idx % len(colors)], 100)
                        if wc_sub:
                            fig, ax = plt.subplots(figsize=(10, 5))
                            ax.imshow(wc_sub, interpolation='bilinear')
                            ax.axis('off')
                            st.pyplot(fig)
                    else:
                        st.write("No text data available")
        
        # Sentiment-based Word Clouds
        st.subheader("😊 Sentiment-based Word Clouds")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Positive Sentiment (>0.1)**")
            positive_text = ' '.join(filtered_df[filtered_df['sentiment'] > 0.1]['processed_text'].dropna())
            if positive_text:
                wc_pos = generate_wordcloud(positive_text, 'Greens', 75)
                if wc_pos:
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.imshow(wc_pos, interpolation='bilinear')
                    ax.axis('off')
                    st.pyplot(fig)
            else:
                st.write("No positive sentiment data available")
        
        with col2:
            st.write("**Negative Sentiment (<-0.1)**")
            negative_text = ' '.join(filtered_df[filtered_df['sentiment'] < -0.1]['processed_text'].dropna())
            if negative_text:
                wc_neg = generate_wordcloud(negative_text, 'Reds', 75)
                if wc_neg:
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.imshow(wc_neg, interpolation='bilinear')
                    ax.axis('off')
                    st.pyplot(fig)
            else:
                st.write("No negative sentiment data available")
        
        # Word Frequency Analysis
        st.subheader("📝 Word Frequency Analysis")
        all_words = ' '.join(filtered_df['processed_text'].dropna()).split()
        word_freq = Counter(all_words)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Top 20 Most Common Words**")
            top_words = pd.DataFrame(word_freq.most_common(20), columns=['Word', 'Frequency'])
            fig_words = px.bar(top_words, x='Frequency', y='Word', orientation='h',
                              title="Most Frequent Words")
            fig_words.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_words, use_container_width=True)
        
        with col2:
            st.write("**Word Statistics**")
            st.write(f"Total unique words: {len(word_freq):,}")
            st.write(f"Total word count: {sum(word_freq.values()):,}")
            st.write(f"Average word frequency: {np.mean(list(word_freq.values())):.1f}")
            st.write(f"Most common word: '{word_freq.most_common(1)[0][0]}' ({word_freq.most_common(1)[0][1]:,} times)")
    
    else:
        st.warning("⚠️ No text data available for word clouds.")

def trends_tab(filtered_df):
    """Trends analysis tab."""
    st.subheader("📈 Temporal Trends Analysis")
    
    if not filtered_df.empty:
        
        # Sentiment over time
        st.subheader("🎭 Sentiment Evolution")
        
        # Daily sentiment trend
        daily_sentiment = filtered_df.groupby(['date', 'subreddit'])['sentiment'].agg(['mean', 'count']).reset_index()
        daily_sentiment.columns = ['date', 'subreddit', 'avg_sentiment', 'comment_count']
        
        fig_daily = px.line(
            daily_sentiment,
            x='date',
            y='avg_sentiment',
            color='subreddit',
            title="Daily Average Sentiment",
            markers=True,
            color_discrete_map={'technology': '#0079D3', 'startups': '#FF4500'}
        )
        fig_daily.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Neutral")
        st.plotly_chart(fig_daily, use_container_width=True)
        
        # Activity patterns
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🕐 Hourly Activity")
            hourly_activity = filtered_df.groupby(['hour', 'subreddit']).size().reset_index(name='comments')
            fig_hourly = px.line(
                hourly_activity,
                x='hour',
                y='comments',
                color='subreddit',
                title="Comments by Hour of Day",
                markers=True,
                color_discrete_map={'technology': '#0079D3', 'startups': '#FF4500'}
            )
            st.plotly_chart(fig_hourly, use_container_width=True)
        
        with col2:
            st.subheader("📅 Daily Activity")
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            daily_activity = filtered_df.groupby(['day_of_week', 'subreddit']).size().reset_index(name='comments')
            daily_activity['day_of_week'] = pd.Categorical(daily_activity['day_of_week'], categories=day_order, ordered=True)
            daily_activity = daily_activity.sort_values('day_of_week')
            
            fig_days = px.bar(
                daily_activity,
                x='day_of_week',
                y='comments',
                color='subreddit',
                title="Comments by Day of Week",
                color_discrete_map={'technology': '#0079D3', 'startups': '#FF4500'}
            )
            st.plotly_chart(fig_days, use_container_width=True)
        
        # Volume vs Sentiment correlation
        st.subheader("📊 Volume vs Sentiment Analysis")
        
        volume_sentiment = daily_sentiment.copy()
        fig_scatter = px.scatter(
            volume_sentiment,
            x='comment_count',
            y='avg_sentiment',
            color='subreddit',
            size='comment_count',
            title="Daily Comment Volume vs Average Sentiment",
            labels={'comment_count': 'Comments per Day', 'avg_sentiment': 'Average Sentiment'},
            color_discrete_map={'technology': '#0079D3', 'startups': '#FF4500'}
        )
        fig_scatter.add_hline(y=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    else:
        st.warning("⚠️ No data available for trends analysis.")

def topics_tab(filtered_df):
    """Topics analysis tab."""
    st.subheader("🧠 Topic Analysis")
    
    if not filtered_df.empty and 'topic_name' in filtered_df.columns:
        
        # Topic distribution
        topic_counts = filtered_df[filtered_df['topic_id'] != -1]['topic_name'].value_counts().head(20)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig_topics = px.bar(
                x=topic_counts.values,
                y=topic_counts.index,
                orientation='h',
                title="Top 20 Topics by Comment Count",
                labels={'x': 'Number of Comments', 'y': 'Topic'}
            )
            fig_topics.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_topics, use_container_width=True)
        
        with col2:
            st.write("**Topic Statistics**")
            total_topics = filtered_df['topic_id'].nunique() - 1
            outliers = (filtered_df['topic_id'] == -1).sum()
            
            st.metric("Total Topics", f"{total_topics}")
            st.metric("Outlier Comments", f"{outliers:,}")
            st.metric("Avg Comments/Topic", f"{len(filtered_df[filtered_df['topic_id'] != -1]) / total_topics:.0f}")
        
        # Topic sentiment analysis
        st.subheader("🎭 Sentiment by Topic")
        
        topic_sentiment = filtered_df[filtered_df['topic_id'] != -1].groupby('topic_name')['sentiment'].agg(['mean', 'count']).reset_index()
        topic_sentiment = topic_sentiment[topic_sentiment['count'] >= 10].sort_values('mean', ascending=False).head(15)
        
        fig_topic_sent = px.bar(
            topic_sentiment,
            x='mean',
            y='topic_name',
            orientation='h',
            title="Average Sentiment by Topic (Top 15, min 10 comments)",
            labels={'mean': 'Average Sentiment', 'topic_name': 'Topic'},
            color='mean',
            color_continuous_scale='RdBu'
        )
        fig_topic_sent.add_vline(x=0, line_dash="dash", line_color="red")
        fig_topic_sent.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_topic_sent, use_container_width=True)
        
        # Topic comparison between subreddits
        if len(filtered_df['subreddit'].unique()) > 1:
            st.subheader("🔄 Topic Comparison Between Subreddits")
            
            topic_comparison = filtered_df[filtered_df['topic_id'] != -1].groupby(['subreddit', 'topic_name']).size().reset_index(name='count')
            topic_comparison_pivot = topic_comparison.pivot(index='topic_name', columns='subreddit', values='count').fillna(0)
            
            # Calculate topic preferences
            for col in topic_comparison_pivot.columns:
                topic_comparison_pivot[f'{col}_pct'] = topic_comparison_pivot[col] / topic_comparison_pivot[col].sum() * 100
            
            # Show top topics for each subreddit
            for subreddit in filtered_df['subreddit'].unique():
                st.write(f"**r/{subreddit} Top Topics:**")
                if subreddit in topic_comparison_pivot.columns:
                    top_sub_topics = topic_comparison_pivot.sort_values(subreddit, ascending=False).head(5)
                    for idx, (topic, row) in enumerate(top_sub_topics.iterrows(), 1):
                        st.write(f"{idx}. {topic}: {row[subreddit]:.0f} comments ({row[f'{subreddit}_pct']:.1f}%)")
                st.write("")
    
    else:
        st.warning("⚠️ No topic data available.")

def insights_tab(filtered_df, full_df):
    """Deep insights tab."""
    st.subheader("🔍 Deep Insights & Analytics")
    
    if not filtered_df.empty:
        
        # Calculate insights
        insights = calculate_insights(filtered_df)
        
        # Community profiles
        st.subheader("🏛️ Community Profiles")
        
        for subreddit, data in insights.items():
            with st.expander(f"r/{subreddit} Profile", expanded=True):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("**📊 Basic Stats**")
                    st.write(f"Total Comments: {data['total_comments']:,}")
                    st.write(f"Avg Comment Length: {data['avg_comment_length']:.0f} chars")
                    st.write(f"Topic Diversity: {data['topic_diversity']} topics")
                
                with col2:
                    st.markdown("**😊 Sentiment Profile**")
                    st.write(f"Average: {data['avg_sentiment']:.3f}")
                    st.write(f"Median: {data['median_sentiment']:.3f}")
                    st.write(f"Std Dev: {data['sentiment_std']:.3f}")
                
                with col3:
                    st.markdown("**📈 Distribution**")
                    st.write(f"Positive: {data['positive_pct']:.1f}%")
                    st.write(f"Neutral: {data['neutral_pct']:.1f}%")
                    st.write(f"Negative: {data['negative_pct']:.1f}%")
                
                st.markdown("**🕐 Activity Patterns**")
                st.write(f"Most Active Hour: {data['most_active_hour']}:00")
                st.write(f"Most Active Day: {data['most_active_day']}")
                st.write(f"Top Topic: {data['top_topic']}")
        
        # Statistical analysis
        st.subheader("📊 Statistical Analysis")
        
        if len(insights) > 1:
            # Compare communities
            subreddit_names = list(insights.keys())
            sub1, sub2 = subreddit_names[0], subreddit_names[1]
            
            sentiment_diff = insights[sub1]['avg_sentiment'] - insights[sub2]['avg_sentiment']
            
            st.markdown('<div class="insight-box">', unsafe_allow_html=True)
            st.markdown("**🔍 Key Insights:**")
            
            # Sentiment comparison
            if abs(sentiment_diff) > 0.05:
                more_positive = sub1 if sentiment_diff > 0 else sub2
                st.write(f"• r/{more_positive} is more positive by {abs(sentiment_diff):.3f} points")
            else:
                st.write("• Both communities show similar sentiment patterns")
            
            # Activity comparison
            more_active = sub1 if insights[sub1]['total_comments'] > insights[sub2]['total_comments'] else sub2
            activity_ratio = max(insights[sub1]['total_comments'], insights[sub2]['total_comments']) / min(insights[sub1]['total_comments'], insights[sub2]['total_comments'])
            st.write(f"• r/{more_active} is {activity_ratio:.1f}x more active in this period")
            
            # Length comparison
            longer_comments = sub1 if insights[sub1]['avg_comment_length'] > insights[sub2]['avg_comment_length'] else sub2
            st.write(f"• r/{longer_comments} tends to write longer comments")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Advanced metrics
        st.subheader("🎯 Advanced Metrics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Sentiment volatility
            st.markdown("**📈 Sentiment Volatility**")
            volatility_data = []
            for subreddit in filtered_df['subreddit'].unique():
                sub_data = filtered_df[filtered_df['subreddit'] == subreddit]
                volatility = sub_data['sentiment'].std()
                volatility_data.append({'Subreddit': subreddit, 'Volatility': volatility})
            
            volatility_df = pd.DataFrame(volatility_data)
            fig_vol = px.bar(volatility_df, x='Subreddit', y='Volatility', 
                           title="Sentiment Volatility by Subreddit")
            st.plotly_chart(fig_vol, use_container_width=True)
        
        with col2:
            # Engagement quality
            st.markdown("**⭐ Engagement Quality**")
            if 'comment_score' in filtered_df.columns:
                engagement_data = []
                for subreddit in filtered_df['subreddit'].unique():
                    sub_data = filtered_df[filtered_df['subreddit'] == subreddit]
                    avg_score = sub_data['comment_score'].mean()
                    engagement_data.append({'Subreddit': subreddit, 'Avg Score': avg_score})
                
                engagement_df = pd.DataFrame(engagement_data)
                fig_eng = px.bar(engagement_df, x='Subreddit', y='Avg Score',
                               title="Average Comment Score by Subreddit")
                st.plotly_chart(fig_eng, use_container_width=True)
            else:
                st.write("Comment score data not available")
        
        # Data quality metrics
        st.subheader("🔍 Data Quality Report")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Data Coverage", f"{len(filtered_df) / len(full_df) * 100:.1f}%")
        with col2:
            missing_processed = filtered_df['processed_text'].isna().sum()
            st.metric("Missing Processed Text", f"{missing_processed:,}")
        with col3:
            outlier_ratio = (filtered_df['topic_id'] == -1).sum() / len(filtered_df) * 100
            st.metric("Topic Outliers", f"{outlier_ratio:.1f}%")
    
    else:
        st.warning("⚠️ No data available for insights analysis.")

if __name__ == "__main__":
    main()
