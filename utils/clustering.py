"""
Customer segmentation using KMeans clustering
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import plotly.express as px
import plotly.graph_objects as go

def find_optimal_clusters(df, max_clusters=8, random_state=42):
    """Find optimal number of clusters using elbow method and silhouette score"""
    feature_cols = ['Age', 'Income', 'Tenure', 'Number_of_Products', 'Activity_Score']
    X = df[feature_cols]
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Calculate metrics for different numbers of clusters
    inertias = []
    silhouette_scores = []
    K_range = range(2, max_clusters + 1)
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        cluster_labels = kmeans.fit_predict(X_scaled)
        
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X_scaled, cluster_labels))
    
    # Find optimal k (elbow method + silhouette score)
    # Use silhouette score as primary metric
    optimal_k = K_range[np.argmax(silhouette_scores)]
    
    return optimal_k, inertias, silhouette_scores, K_range

def perform_clustering(df, n_clusters=4, random_state=42):
    """Perform KMeans clustering on customer data"""
    feature_cols = ['Age', 'Income', 'Tenure', 'Number_of_Products', 'Activity_Score']
    X = df[feature_cols]
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Perform clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    # Add cluster labels to dataframe
    df_clustered = df.copy()
    df_clustered['Cluster'] = cluster_labels
    
    # Calculate cluster centers in original scale
    cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
    cluster_centers_df = pd.DataFrame(
        cluster_centers, 
        columns=feature_cols,
        index=[f'Cluster {i}' for i in range(n_clusters)]
    )
    
    return df_clustered, kmeans, scaler, cluster_centers_df

def analyze_clusters(df_clustered):
    """Analyze cluster characteristics and create profiles"""
    cluster_analysis = []
    
    for cluster_id in sorted(df_clustered['Cluster'].unique()):
        cluster_data = df_clustered[df_clustered['Cluster'] == cluster_id]
        
        # Calculate cluster statistics
        cluster_stats = {
            'Cluster': f'Cluster {cluster_id}',
            'Size': len(cluster_data),
            'Percentage': len(cluster_data) / len(df_clustered) * 100,
            'Avg_Age': cluster_data['Age'].mean(),
            'Avg_Income': cluster_data['Income'].mean(),
            'Avg_Tenure': cluster_data['Tenure'].mean(),
            'Avg_Products': cluster_data['Number_of_Products'].mean(),
            'Avg_Activity': cluster_data['Activity_Score'].mean(),
            'Churn_Rate': (cluster_data['Churn'] == 'Yes').mean() * 100,
            'Male_Percentage': (cluster_data['Gender'] == 'Male').mean() * 100
        }
        
        cluster_analysis.append(cluster_stats)
    
    cluster_df = pd.DataFrame(cluster_analysis)
    
    # Sort by churn rate (highest risk first)
    cluster_df = cluster_df.sort_values('Churn_Rate', ascending=False)
    
    return cluster_df

def create_cluster_visualization(df_clustered, x_col='Income', y_col='Activity_Score'):
    """Create 2D scatter plot of clusters"""
    fig = px.scatter(
        df_clustered, 
        x=x_col, 
        y=y_col,
        color='Cluster',
        hover_data=['CustomerID', 'Age', 'Tenure', 'Number_of_Products', 'Churn'],
        title=f'Customer Segments: {x_col} vs {y_col}',
        labels={
            x_col: x_col.replace('_', ' '),
            y_col: y_col.replace('_', ' ')
        }
    )
    
    fig.update_layout(
        width=800,
        height=600,
        showlegend=True
    )
    
    return fig

def create_cluster_comparison_chart(cluster_df):
    """Create comparison chart for cluster characteristics in 2x2 subplot layout"""
    from plotly.subplots import make_subplots
    
    # Create 2x2 subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Age Distribution', 'Income Distribution', 
                       'Tenure & Products', 'Activity & Churn Rate'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": True}, {"secondary_y": True}]]
    )
    
    # Age chart (top-left)
    fig.add_trace(
        go.Bar(
            name='Age',
            x=cluster_df['Cluster'],
            y=cluster_df['Avg_Age'],
            marker_color='#1f77b4',
            showlegend=False
        ),
        row=1, col=1
    )
    
    # Income chart (top-right)
    fig.add_trace(
        go.Bar(
            name='Income',
            x=cluster_df['Cluster'],
            y=cluster_df['Avg_Income'],
            marker_color='#ff7f0e',
            showlegend=False
        ),
        row=1, col=2
    )
    
    # Tenure chart (bottom-left, left y-axis)
    fig.add_trace(
        go.Bar(
            name='Tenure',
            x=cluster_df['Cluster'],
            y=cluster_df['Avg_Tenure'],
            marker_color='#2ca02c',
            showlegend=False
        ),
        row=2, col=1
    )
    
    # Products chart (bottom-left, right y-axis)
    fig.add_trace(
        go.Bar(
            name='Products',
            x=cluster_df['Cluster'],
            y=cluster_df['Avg_Products'],
            marker_color='#d62728',
            showlegend=False
        ),
        row=2, col=1, secondary_y=True
    )
    
    # Activity chart (bottom-right, left y-axis)
    fig.add_trace(
        go.Bar(
            name='Activity',
            x=cluster_df['Cluster'],
            y=cluster_df['Avg_Activity'],
            marker_color='#9467bd',
            showlegend=False
        ),
        row=2, col=2
    )
    
    # Churn Rate chart (bottom-right, right y-axis)
    fig.add_trace(
        go.Bar(
            name='Churn Rate',
            x=cluster_df['Cluster'],
            y=cluster_df['Churn_Rate'],
            marker_color='#8c564b',
            showlegend=False
        ),
        row=2, col=2, secondary_y=True
    )
    
    # Update layout
    fig.update_layout(
        title='Cluster Characteristics Comparison',
        height=700,
        width=1000,
        showlegend=True
    )
    
    # Update axes labels
    fig.update_xaxes(title_text="Customer Segments", row=1, col=1)
    fig.update_xaxes(title_text="Customer Segments", row=1, col=2)
    fig.update_xaxes(title_text="Customer Segments", row=2, col=1)
    fig.update_xaxes(title_text="Customer Segments", row=2, col=2)
    
    fig.update_yaxes(title_text="Age (years)", row=1, col=1)
    fig.update_yaxes(title_text="Income ($)", row=1, col=2)
    fig.update_yaxes(title_text="Tenure (years)", row=2, col=1)
    fig.update_yaxes(title_text="Products", row=2, col=1, secondary_y=True)
    fig.update_yaxes(title_text="Activity Score", row=2, col=2)
    fig.update_yaxes(title_text="Churn Rate (%)", row=2, col=2, secondary_y=True)
    
    return fig

def get_cluster_recommendations(cluster_df):
    """Generate business recommendations for each cluster"""
    recommendations = []
    
    for _, cluster in cluster_df.iterrows():
        cluster_name = cluster['Cluster']
        churn_rate = cluster['Churn_Rate']
        avg_income = cluster['Avg_Income']
        avg_activity = cluster['Avg_Activity']
        avg_tenure = cluster['Avg_Tenure']
        
        # Generate recommendations based on cluster characteristics
        if churn_rate > 60:
            risk_level = "High Risk"
            if avg_income < 30000:
                recommendation = "Focus on retention campaigns, consider loyalty programs and financial education"
            elif avg_activity < 40:
                recommendation = "Increase engagement through targeted marketing and product recommendations"
            else:
                recommendation = "Investigate specific pain points, conduct customer interviews"
        elif churn_rate > 40:
            risk_level = "Medium Risk"
            if avg_tenure < 2:
                recommendation = "Improve onboarding experience and early engagement"
            else:
                recommendation = "Monitor closely, implement proactive retention measures"
        else:
            risk_level = "Low Risk"
            recommendation = "Maintain current service levels, focus on upselling opportunities"
        
        recommendations.append({
            'Cluster': cluster_name,
            'Risk_Level': risk_level,
            'Recommendation': recommendation,
            'Priority': 'High' if churn_rate > 60 else 'Medium' if churn_rate > 40 else 'Low'
        })
    
    return pd.DataFrame(recommendations)
