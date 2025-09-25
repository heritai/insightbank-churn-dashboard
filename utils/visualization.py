"""
Visualization utilities for InsightBank dashboard
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns

def create_kpi_cards(kpis):
    """Create KPI cards for the dashboard"""
    cards = []
    
    # Total Customers
    cards.append({
        'title': 'Total Customers',
        'value': f"{kpis['total_customers']:,}",
        'icon': '👥',
        'color': 'blue'
    })
    
    # Churn Rate
    cards.append({
        'title': 'Churn Rate',
        'value': f"{kpis['churn_rate']:.1f}%",
        'icon': '📉',
        'color': 'red' if kpis['churn_rate'] > 50 else 'orange' if kpis['churn_rate'] > 30 else 'green'
    })
    
    # Average Tenure
    cards.append({
        'title': 'Avg Tenure',
        'value': f"{kpis['avg_tenure']:.1f} years",
        'icon': '⏰',
        'color': 'blue'
    })
    
    # Average Income
    cards.append({
        'title': 'Avg Income',
        'value': f"${kpis['avg_income']:,.0f}",
        'icon': '💰',
        'color': 'green'
    })
    
    return cards

def create_churn_distribution_chart(df):
    """Create churn distribution chart with insights"""
    churn_counts = df['Churn'].value_counts()
    churn_rate = (churn_counts['Yes'] / len(df)) * 100
    
    fig = px.pie(
        values=churn_counts.values,
        names=churn_counts.index,
        title=f'Customer Churn Distribution (Overall Rate: {churn_rate:.1f}%)',
        color_discrete_map={'Yes': '#FF6B6B', 'No': '#4ECDC4'}
    )
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(
        width=400,
        height=400,
        showlegend=True
    )
    
    return fig, {
        'title': 'Churn Distribution Analysis',
        'description': f'This pie chart shows the overall customer churn rate of {churn_rate:.1f}%.',
        'insights': [
            f'• {churn_counts["Yes"]:,} customers ({churn_rate:.1f}%) have churned',
            f'• {churn_counts["No"]:,} customers ({100-churn_rate:.1f}%) are retained',
            f'• Industry benchmark for retail banking is typically 15-25% annually',
            f'• Current rate is {"above" if churn_rate > 25 else "within" if churn_rate > 15 else "below"} industry average'
        ],
        'recommendations': [
            '🎯 Focus on retention strategies for the at-risk segment',
            '📊 Implement early warning systems to identify churn signals',
            '💡 Develop targeted retention campaigns based on customer segments',
            '📈 Set quarterly churn reduction targets (aim for <20%)'
        ]
    }

def create_age_distribution_chart(df):
    """Create age distribution chart with insights"""
    # Calculate age group churn rates
    age_groups = pd.cut(df['Age'], bins=[0, 30, 50, 70, 100], labels=['18-30', '31-50', '51-70', '71+'])
    age_churn = df.groupby(age_groups)['Churn'].apply(lambda x: (x == 'Yes').mean() * 100)
    
    fig = px.histogram(
        df, 
        x='Age',
        color='Churn',
        title='Age Distribution by Churn Status',
        nbins=20,
        color_discrete_map={'Yes': '#FF6B6B', 'No': '#4ECDC4'}
    )
    
    fig.update_layout(
        width=600,
        height=400,
        xaxis_title='Age',
        yaxis_title='Number of Customers'
    )
    
    return fig, {
        'title': 'Age-Based Churn Analysis',
        'description': 'This histogram shows customer distribution by age and churn status, revealing generational patterns in customer retention.',
        'insights': [
            f'• Young customers (18-30): {age_churn["18-30"]:.1f}% churn rate',
            f'• Middle-aged customers (31-50): {age_churn["31-50"]:.1f}% churn rate',
            f'• Senior customers (51-70): {age_churn["51-70"]:.1f}% churn rate',
            f'• Peak churn age group: {age_churn.idxmax()} with {age_churn.max():.1f}% churn rate',
            f'• Most stable age group: {age_churn.idxmin()} with {age_churn.min():.1f}% churn rate'
        ],
        'recommendations': [
            '🎯 Develop age-specific retention strategies',
            '📱 Focus on digital engagement for younger customers',
            '🏦 Emphasize relationship banking for older customers',
            '💡 Create generational marketing campaigns',
            '📊 Monitor life-stage transitions that may trigger churn'
        ]
    }

def create_income_distribution_chart(df):
    """Create income distribution chart with insights"""
    # Calculate income group churn rates
    income_groups = pd.cut(df['Income'], bins=[0, 30000, 50000, 75000, 200000], labels=['Low', 'Medium', 'High', 'Very High'])
    income_churn = df.groupby(income_groups)['Churn'].apply(lambda x: (x == 'Yes').mean() * 100)
    
    fig = px.histogram(
        df, 
        x='Income',
        color='Churn',
        title='Income Distribution by Churn Status',
        nbins=30,
        color_discrete_map={'Yes': '#FF6B6B', 'No': '#4ECDC4'}
    )
    
    fig.update_layout(
        width=600,
        height=400,
        xaxis_title='Income ($)',
        yaxis_title='Number of Customers'
    )
    
    return fig, {
        'title': 'Income-Based Churn Analysis',
        'description': 'This histogram reveals the relationship between customer income levels and churn behavior, showing how financial capacity affects retention.',
        'insights': [
            f'• Low income customers (<$30K): {income_churn["Low"]:.1f}% churn rate',
            f'• Medium income customers ($30K-$50K): {income_churn["Medium"]:.1f}% churn rate',
            f'• High income customers ($50K-$75K): {income_churn["High"]:.1f}% churn rate',
            f'• Very high income customers (>$75K): {income_churn["Very High"]:.1f}% churn rate',
            f'• Income-churn correlation: {"Strong negative" if income_churn["Low"] - income_churn["Very High"] > 20 else "Moderate negative" if income_churn["Low"] - income_churn["Very High"] > 10 else "Weak"}'
        ],
        'recommendations': [
            '💰 Develop income-tiered product offerings',
            '🎯 Focus retention efforts on high-value customers',
            '💡 Create financial wellness programs for lower-income segments',
            '📊 Implement value-based pricing strategies',
            '🏆 Design loyalty programs that scale with income levels'
        ]
    }

def create_activity_vs_churn_chart(df):
    """Create activity score vs churn chart with insights"""
    # Calculate activity statistics
    churned_activity = df[df['Churn'] == 'Yes']['Activity_Score']
    retained_activity = df[df['Churn'] == 'No']['Activity_Score']
    
    activity_diff = retained_activity.mean() - churned_activity.mean()
    
    fig = px.box(
        df,
        x='Churn',
        y='Activity_Score',
        title='Activity Score Distribution by Churn Status',
        color='Churn',
        color_discrete_map={'Yes': '#FF6B6B', 'No': '#4ECDC4'}
    )
    
    fig.update_layout(
        width=600,
        height=400,
        xaxis_title='Churn Status',
        yaxis_title='Activity Score'
    )
    
    return fig, {
        'title': 'Activity Score vs Churn Analysis',
        'description': 'This box plot shows the distribution of customer activity scores by churn status, revealing the critical role of engagement in retention.',
        'insights': [
            f'• Retained customers average activity: {retained_activity.mean():.1f}',
            f'• Churned customers average activity: {churned_activity.mean():.1f}',
            f'• Activity difference: {activity_diff:.1f} points higher for retained customers',
            f'• Activity threshold: {churned_activity.quantile(0.75):.1f} (75% of churned customers below this)',
            f'• High-activity retention rate: {(retained_activity > 70).mean() * 100:.1f}%'
        ],
        'recommendations': [
            '📱 Implement gamification to boost customer engagement',
            '🎯 Create activity-based loyalty rewards',
            '📊 Set activity score alerts for at-risk customers',
            '💡 Develop personalized engagement strategies',
            '🔄 Launch re-engagement campaigns for low-activity customers'
        ]
    }

def create_tenure_vs_churn_chart(df):
    """Create tenure vs churn chart with insights"""
    # Calculate tenure statistics
    churned_tenure = df[df['Churn'] == 'Yes']['Tenure']
    retained_tenure = df[df['Churn'] == 'No']['Tenure']
    
    tenure_diff = retained_tenure.mean() - churned_tenure.mean()
    
    # Calculate tenure group churn rates
    tenure_groups = pd.cut(df['Tenure'], bins=[0, 1, 3, 5, 20], labels=['New (0-1yr)', 'Short (1-3yr)', 'Medium (3-5yr)', 'Long (5+yr)'])
    tenure_churn = df.groupby(tenure_groups)['Churn'].apply(lambda x: (x == 'Yes').mean() * 100)
    
    fig = px.box(
        df,
        x='Churn',
        y='Tenure',
        title='Tenure Distribution by Churn Status',
        color='Churn',
        color_discrete_map={'Yes': '#FF6B6B', 'No': '#4ECDC4'}
    )
    
    fig.update_layout(
        width=600,
        height=400,
        xaxis_title='Churn Status',
        yaxis_title='Tenure (years)'
    )
    
    return fig, {
        'title': 'Tenure vs Churn Analysis',
        'description': 'This box plot reveals how customer tenure (length of relationship) correlates with churn behavior, showing the importance of relationship building.',
        'insights': [
            f'• Retained customers average tenure: {retained_tenure.mean():.1f} years',
            f'• Churned customers average tenure: {churned_tenure.mean():.1f} years',
            f'• Tenure difference: {tenure_diff:.1f} years longer for retained customers',
            f'• New customers (0-1yr): {tenure_churn["New (0-1yr)"]:.1f}% churn rate',
            f'• Long-term customers (5+yr): {tenure_churn["Long (5+yr)"]:.1f}% churn rate',
            f'• Critical period: First {churned_tenure.quantile(0.75):.1f} years show highest churn risk'
        ],
        'recommendations': [
            '🎯 Focus on first-year customer success programs',
            '📞 Implement proactive check-ins for new customers',
            '💎 Develop long-term relationship building initiatives',
            '🔄 Create milestone-based retention campaigns',
            '📊 Monitor tenure-based churn patterns quarterly'
        ]
    }

def create_products_vs_churn_chart(df):
    """Create number of products vs churn chart with insights"""
    # Calculate product statistics
    churned_products = df[df['Churn'] == 'Yes']['Number_of_Products']
    retained_products = df[df['Churn'] == 'No']['Number_of_Products']
    
    product_diff = retained_products.mean() - churned_products.mean()
    
    # Calculate product group churn rates
    product_groups = pd.cut(df['Number_of_Products'], bins=[0, 2, 4, 6, 20], labels=['Low (1-2)', 'Medium (3-4)', 'High (5-6)', 'Very High (7+)'])
    product_churn = df.groupby(product_groups)['Churn'].apply(lambda x: (x == 'Yes').mean() * 100)
    
    fig = px.box(
        df,
        x='Churn',
        y='Number_of_Products',
        title='Number of Products by Churn Status',
        color='Churn',
        color_discrete_map={'Yes': '#FF6B6B', 'No': '#4ECDC4'}
    )
    
    fig.update_layout(
        width=600,
        height=400,
        xaxis_title='Churn Status',
        yaxis_title='Number of Products'
    )
    
    return fig, {
        'title': 'Product Portfolio vs Churn Analysis',
        'description': 'This box plot shows how the number of products a customer uses correlates with their likelihood to churn, demonstrating the value of product diversification.',
        'insights': [
            f'• Retained customers average products: {retained_products.mean():.1f}',
            f'• Churned customers average products: {churned_products.mean():.1f}',
            f'• Product difference: {product_diff:.1f} more products for retained customers',
            f'• Low product users (1-2): {product_churn["Low (1-2)"]:.1f}% churn rate',
            f'• High product users (5-6): {product_churn["High (5-6)"]:.1f}% churn rate',
            f'• Product stickiness threshold: {churned_products.quantile(0.75):.0f} products'
        ],
        'recommendations': [
            '🔄 Implement cross-selling strategies to increase product adoption',
            '🎯 Focus on product bundling to improve stickiness',
            '💡 Create product usage incentives and rewards',
            '📊 Develop product portfolio optimization campaigns',
            '🏆 Design loyalty programs based on product diversity'
        ]
    }

def create_correlation_heatmap(df):
    """Create correlation heatmap for numerical features with insights"""
    # Select numerical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    correlation_matrix = df[numerical_cols].corr()
    
    # Find strongest correlations
    corr_pairs = []
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            corr_val = correlation_matrix.iloc[i, j]
            if abs(corr_val) > 0.3:  # Only significant correlations
                corr_pairs.append((correlation_matrix.columns[i], correlation_matrix.columns[j], corr_val))
    
    corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    
    fig = px.imshow(
        correlation_matrix,
        text_auto=True,
        aspect="auto",
        title="Feature Correlation Heatmap",
        color_continuous_scale='RdBu_r'
    )
    
    fig.update_layout(
        width=600,
        height=600
    )
    
    return fig, {
        'title': 'Feature Correlation Analysis',
        'description': 'This heatmap shows the correlation between different customer attributes, revealing hidden relationships that can inform retention strategies.',
        'insights': [
            f'• Strongest positive correlation: {corr_pairs[0][0]} ↔ {corr_pairs[0][1]} ({corr_pairs[0][2]:.3f})' if corr_pairs else '• No strong correlations found',
            f'• Strongest negative correlation: {corr_pairs[-1][0]} ↔ {corr_pairs[-1][1]} ({corr_pairs[-1][2]:.3f})' if corr_pairs else '',
            f'• Total significant correlations: {len(corr_pairs)} pairs',
            f'• Income-Product correlation: {correlation_matrix.loc["Income", "Number_of_Products"]:.3f}',
            f'• Activity-Tenure correlation: {correlation_matrix.loc["Activity_Score", "Tenure"]:.3f}'
        ],
        'recommendations': [
            '🔍 Investigate highly correlated features for predictive modeling',
            '📊 Use correlation insights to identify redundant data collection',
            '💡 Develop composite metrics from highly correlated features',
            '🎯 Focus on independent variables for targeted interventions',
            '📈 Monitor correlation changes over time for trend analysis'
        ]
    }

def create_churn_by_segments_chart(df):
    """Create churn rate by different segments"""
    # Age groups
    df['Age_Group'] = pd.cut(df['Age'], bins=[0, 30, 50, 70, 100], 
                            labels=['18-30', '31-50', '51-70', '71+'])
    
    # Activity groups
    df['Activity_Group'] = pd.cut(df['Activity_Score'], bins=[0, 30, 60, 100], 
                                 labels=['Low', 'Medium', 'High'])
    
    # Calculate churn rates
    age_churn = df.groupby('Age_Group')['Churn'].apply(lambda x: (x == 'Yes').mean() * 100)
    activity_churn = df.groupby('Activity_Group')['Churn'].apply(lambda x: (x == 'Yes').mean() * 100)
    
    # Create subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Churn Rate by Age Group', 'Churn Rate by Activity Level')
    )
    
    # Age group chart
    fig.add_trace(
        go.Bar(x=age_churn.index, y=age_churn.values, name='Age Groups', marker_color='#FF6B6B'),
        row=1, col=1
    )
    
    # Activity group chart
    fig.add_trace(
        go.Bar(x=activity_churn.index, y=activity_churn.values, name='Activity Groups', marker_color='#4ECDC4'),
        row=1, col=2
    )
    
    fig.update_layout(
        height=400,
        showlegend=False,
        title_text="Churn Rate by Customer Segments"
    )
    
    fig.update_xaxes(title_text="Age Group", row=1, col=1)
    fig.update_xaxes(title_text="Activity Level", row=1, col=2)
    fig.update_yaxes(title_text="Churn Rate (%)", row=1, col=1)
    fig.update_yaxes(title_text="Churn Rate (%)", row=1, col=2)
    
    return fig

def create_feature_importance_chart(importance_df, top_n=10):
    """Create feature importance chart"""
    top_features = importance_df.head(top_n)
    
    fig = px.bar(
        top_features,
        x='Importance',
        y='Feature',
        orientation='h',
        title=f'Top {top_n} Most Important Features for Churn Prediction',
        color='Importance',
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(
        width=600,
        height=400,
        xaxis_title='Feature Importance',
        yaxis_title='Features'
    )
    
    return fig

def create_risk_distribution_chart(predictions):
    """Create risk distribution chart for churn predictions"""
    # Categorize predictions into risk levels
    risk_levels = []
    for prob in predictions:
        if prob < 0.3:
            risk_levels.append('Low Risk')
        elif prob < 0.6:
            risk_levels.append('Medium Risk')
        else:
            risk_levels.append('High Risk')
    
    risk_counts = pd.Series(risk_levels).value_counts()
    
    fig = px.pie(
        values=risk_counts.values,
        names=risk_counts.index,
        title='Customer Risk Distribution',
        color_discrete_map={
            'Low Risk': '#4ECDC4',
            'Medium Risk': '#FFE66D',
            'High Risk': '#FF6B6B'
        }
    )
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(
        width=400,
        height=400
    )
    
    return fig

def create_customer_profile_chart(customer_data):
    """Create radar chart for customer profile"""
    # Normalize values to 0-100 scale for radar chart
    normalized_data = {
        'Age': min(customer_data['Age'] / 80 * 100, 100),
        'Income': min(customer_data['Income'] / 200000 * 100, 100),
        'Tenure': min(customer_data['Tenure'] / 15 * 100, 100),
        'Products': min(customer_data['Number_of_Products'] / 8 * 100, 100),
        'Activity': customer_data['Activity_Score']
    }
    
    categories = list(normalized_data.keys())
    values = list(normalized_data.values())
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Customer Profile',
        line_color='#4ECDC4'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        showlegend=True,
        title="Customer Profile Radar Chart",
        width=500,
        height=500
    )
    
    return fig
