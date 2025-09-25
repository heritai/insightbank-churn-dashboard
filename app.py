"""
InsightBank - Customer Segmentation & Churn Prediction Dashboard
Main Streamlit application
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import sys
import os

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from data_prep import load_customer_data, clean_data, prepare_features, prepare_training_data, calculate_kpis, get_customer_summary
from clustering import find_optimal_clusters, perform_clustering, analyze_clusters, create_cluster_visualization, create_cluster_comparison_chart, get_cluster_recommendations
from churn_model import ChurnPredictor, create_customer_profile, interpret_churn_probability, get_retention_recommendations
from visualization import (create_kpi_cards, create_churn_distribution_chart, create_age_distribution_chart, 
                         create_income_distribution_chart, create_activity_vs_churn_chart, create_tenure_vs_churn_chart,
                         create_products_vs_churn_chart, create_correlation_heatmap, create_churn_by_segments_chart,
                         create_feature_importance_chart, create_risk_distribution_chart, create_customer_profile_chart)

# Page configuration
st.set_page_config(
    page_title="InsightBank - Customer Analytics Dashboard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .kpi-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .risk-high {
        color: #ff4444;
        font-weight: bold;
    }
    .risk-medium {
        color: #ff8800;
        font-weight: bold;
    }
    .risk-low {
        color: #00aa00;
        font-weight: bold;
    }
    .stAlert {
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_and_prepare_data():
    """Load and prepare data with caching"""
    # Load data
    df = load_customer_data()
    if df is None:
        return None, None, None, None
    
    # Clean data
    df_clean = clean_data(df)
    
    # Prepare features
    df_features, le_gender, le_age = prepare_features(df_clean)
    
    # Prepare training data
    training_data = prepare_training_data(df_features)
    
    return df_clean, df_features, training_data, (le_gender, le_age)

def main():
    """Main application function"""
    
    # Header
    st.markdown('<h1 class="main-header">🏦 InsightBank Customer Analytics Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("### *AI-Powered Customer Segmentation & Churn Prediction*")
    
    # Load data
    with st.spinner("Loading customer data..."):
        df_clean, df_features, training_data, encoders = load_and_prepare_data()
    
    if df_clean is None:
        st.error("❌ Could not load customer data. Please check if the data file exists.")
        return
    
    # Sidebar navigation
    st.sidebar.title("📊 Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["🏠 Global Insights", "👥 Customer Segmentation", "🔮 Churn Prediction", "📈 Analytics"]
    )
    
    # Global Insights Page
    if page == "🏠 Global Insights":
        show_global_insights(df_clean)
    
    # Customer Segmentation Page
    elif page == "👥 Customer Segmentation":
        show_segmentation_page(df_clean, df_features)
    
    # Churn Prediction Page
    elif page == "🔮 Churn Prediction":
        show_churn_prediction_page(df_features, training_data)
    
    # Analytics Page
    elif page == "📈 Analytics":
        show_analytics_page(df_clean, df_features)

def show_global_insights(df):
    """Display global insights and KPIs"""
    st.header("📊 Global Customer Insights")
    
    # Calculate KPIs
    kpis = calculate_kpis(df)
    
    # Executive Summary
    st.subheader("🎯 Executive Summary")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        **InsightBank Customer Analytics Dashboard** provides comprehensive insights into customer behavior, 
        churn patterns, and retention opportunities. This AI-powered platform enables data-driven decision 
        making to improve customer lifetime value and reduce churn rates.
        """)
        
        # Key insights summary
        churn_rate = kpis['churn_rate']
        if churn_rate > 50:
            st.error(f"🚨 **Critical Alert**: Churn rate of {churn_rate:.1f}% is significantly above industry average (15-25%)")
        elif churn_rate > 30:
            st.warning(f"⚠️ **Attention Required**: Churn rate of {churn_rate:.1f}% is above industry average")
        else:
            st.success(f"✅ **Good Performance**: Churn rate of {churn_rate:.1f}% is within acceptable range")
    
    with col2:
        # Quick stats
        st.metric("Customer Base", f"{kpis['total_customers']:,}")
        st.metric("Churn Rate", f"{kpis['churn_rate']:.1f}%")
        st.metric("Avg Value", f"${kpis['avg_income']:,.0f}")
        st.metric("Engagement", f"{kpis['avg_activity']:.1f}/100")
    
    # Display KPI cards
    st.subheader("📈 Key Performance Indicators")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="👥 Total Customers",
            value=f"{kpis['total_customers']:,}",
            delta=None
        )
    
    with col2:
        churn_color = "normal" if kpis['churn_rate'] < 30 else "off" if kpis['churn_rate'] < 50 else "inverse"
        st.metric(
            label="📉 Churn Rate",
            value=f"{kpis['churn_rate']:.1f}%",
            delta=None
        )
    
    with col3:
        st.metric(
            label="⏰ Avg Tenure",
            value=f"{kpis['avg_tenure']:.1f} years",
            delta=None
        )
    
    with col4:
        st.metric(
            label="💰 Avg Income",
            value=f"${kpis['avg_income']:,.0f}",
            delta=None
        )
    
    # Business context and insights
    with st.expander("📊 Business Context & Strategic Insights"):
        st.markdown("**Current Business Situation**")
        st.write(f"• **Customer Base**: {kpis['total_customers']:,} customers with average income of ${kpis['avg_income']:,.0f}")
        st.write(f"• **Churn Challenge**: {kpis['churn_rate']:.1f}% annual churn rate (Industry benchmark: 15-25%)")
        st.write(f"• **Customer Engagement**: Average activity score of {kpis['avg_activity']:.1f}/100")
        st.write(f"• **Relationship Depth**: Average tenure of {kpis['avg_tenure']:.1f} years")
        
        st.markdown("**Critical Success Factors**")
        st.write("• **Activity Score**: Strongest predictor of customer retention")
        st.write("• **Product Portfolio**: Customers with more products are less likely to churn")
        st.write("• **Tenure**: First-year customers show highest churn risk")
        st.write("• **Demographics**: Age and income significantly impact retention")
        
        st.markdown("**Strategic Priorities**")
        if kpis['churn_rate'] > 40:
            st.write("🔴 **URGENT**: Implement comprehensive retention program")
        else:
            st.write("🟡 **FOCUS**: Optimize existing retention strategies")
        
        st.write("• Develop activity-based engagement programs")
        st.write("• Create age-specific retention campaigns")
        st.write("• Implement cross-selling initiatives")
        st.write("• Enhance new customer onboarding")
        
        st.markdown("**Expected Business Impact**")
        st.write("• Reduce churn rate by 10-15% through targeted interventions")
        st.write("• Increase customer lifetime value through better engagement")
        st.write("• Improve retention campaign ROI through data-driven targeting")
        st.write("• Enhance customer satisfaction through personalized experiences")
    
    st.markdown("---")
    
    # Charts with insights
    col1, col2 = st.columns(2)
    
    with col1:
        fig, insights = create_churn_distribution_chart(df)
        st.plotly_chart(fig, use_container_width=True)
        
        with st.expander("📊 Churn Distribution Insights"):
            st.markdown(f"**{insights['title']}**")
            st.write(insights['description'])
            st.markdown("**Key Insights:**")
            for insight in insights['insights']:
                st.write(insight)
            st.markdown("**Recommendations:**")
            for rec in insights['recommendations']:
                st.write(rec)
    
    with col2:
        fig, insights = create_age_distribution_chart(df)
        st.plotly_chart(fig, use_container_width=True)
        
        with st.expander("📊 Age Distribution Insights"):
            st.markdown(f"**{insights['title']}**")
            st.write(insights['description'])
            st.markdown("**Key Insights:**")
            for insight in insights['insights']:
                st.write(insight)
            st.markdown("**Recommendations:**")
            for rec in insights['recommendations']:
                st.write(rec)
    
    col3, col4 = st.columns(2)
    
    with col3:
        fig, insights = create_income_distribution_chart(df)
        st.plotly_chart(fig, use_container_width=True)
        
        with st.expander("📊 Income Distribution Insights"):
            st.markdown(f"**{insights['title']}**")
            st.write(insights['description'])
            st.markdown("**Key Insights:**")
            for insight in insights['insights']:
                st.write(insight)
            st.markdown("**Recommendations:**")
            for rec in insights['recommendations']:
                st.write(rec)
    
    with col4:
        fig, insights = create_activity_vs_churn_chart(df)
        st.plotly_chart(fig, use_container_width=True)
        
        with st.expander("📊 Activity Score Insights"):
            st.markdown(f"**{insights['title']}**")
            st.write(insights['description'])
            st.markdown("**Key Insights:**")
            for insight in insights['insights']:
                st.write(insight)
            st.markdown("**Recommendations:**")
            for rec in insights['recommendations']:
                st.write(rec)
    
    # Additional insights
    st.subheader("🔍 Advanced Analytics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig, insights = create_tenure_vs_churn_chart(df)
        st.plotly_chart(fig, use_container_width=True)
        
        with st.expander("📊 Tenure Analysis Insights"):
            st.markdown(f"**{insights['title']}**")
            st.write(insights['description'])
            st.markdown("**Key Insights:**")
            for insight in insights['insights']:
                st.write(insight)
            st.markdown("**Recommendations:**")
            for rec in insights['recommendations']:
                st.write(rec)
    
    with col2:
        fig, insights = create_products_vs_churn_chart(df)
        st.plotly_chart(fig, use_container_width=True)
        
        with st.expander("📊 Product Portfolio Insights"):
            st.markdown(f"**{insights['title']}**")
            st.write(insights['description'])
            st.markdown("**Key Insights:**")
            for insight in insights['insights']:
                st.write(insight)
            st.markdown("**Recommendations:**")
            for rec in insights['recommendations']:
                st.write(rec)
    
    # Correlation heatmap
    st.subheader("🔗 Feature Correlations")
    fig, insights = create_correlation_heatmap(df)
    st.plotly_chart(fig, use_container_width=True)
    
    with st.expander("📊 Correlation Analysis Insights"):
        st.markdown(f"**{insights['title']}**")
        st.write(insights['description'])
        st.markdown("**Key Insights:**")
        for insight in insights['insights']:
            st.write(insight)
        st.markdown("**Recommendations:**")
        for rec in insights['recommendations']:
            st.write(rec)

def show_segmentation_page(df, df_features):
    """Display customer segmentation analysis"""
    st.header("👥 Customer Segmentation Analysis")
    
    # Clustering options
    st.subheader("⚙️ Clustering Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        n_clusters = st.slider("Number of Clusters", min_value=3, max_value=8, value=4)
    
    with col2:
        if st.button("🔄 Recalculate Clusters"):
            st.cache_data.clear()
    
    # Perform clustering
    with st.spinner("Performing customer segmentation..."):
        df_clustered, kmeans, scaler, cluster_centers = perform_clustering(df_features, n_clusters)
        cluster_analysis = analyze_clusters(df_clustered)
        recommendations = get_cluster_recommendations(cluster_analysis)
    
    # Display cluster analysis
    st.subheader("📊 Cluster Analysis")
    
    # Cluster summary insights
    total_customers = len(df_clustered)
    high_risk_clusters = cluster_analysis[cluster_analysis['Churn_Rate'] > 60]
    low_risk_clusters = cluster_analysis[cluster_analysis['Churn_Rate'] < 30]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Segments", f"{len(cluster_analysis)}")
    
    with col2:
        st.metric("High-Risk Segments", f"{len(high_risk_clusters)}", 
                 delta=f"{len(high_risk_clusters)/len(cluster_analysis)*100:.1f}% of segments")
    
    with col3:
        st.metric("Low-Risk Segments", f"{len(low_risk_clusters)}", 
                 delta=f"{len(low_risk_clusters)/len(cluster_analysis)*100:.1f}% of segments")
    
    # Cluster comparison table with insights
    st.dataframe(
        cluster_analysis.round(2),
        use_container_width=True,
        hide_index=True
    )
    
    # Cluster insights
    with st.expander("📊 Segmentation Insights"):
        st.markdown("**Customer Segmentation Analysis**")
        st.write("AI-powered clustering has identified distinct customer groups based on behavior, demographics, and engagement patterns.")
        
        st.markdown("**Key Findings:**")
        for _, cluster in cluster_analysis.iterrows():
            risk_level = "🔴 High Risk" if cluster['Churn_Rate'] > 60 else "🟡 Medium Risk" if cluster['Churn_Rate'] > 40 else "🟢 Low Risk"
            st.write(f"• **{cluster['Cluster']}**: {cluster['Size']} customers ({cluster['Percentage']:.1f}%) - {risk_level}")
            st.write(f"  - Avg Income: ${cluster['Avg_Income']:,.0f}, Activity: {cluster['Avg_Activity']:.1f}, Churn: {cluster['Churn_Rate']:.1f}%")
        
        st.markdown("**Strategic Implications:**")
        st.write("• Focus retention efforts on high-risk segments")
        st.write("• Develop segment-specific product offerings")
        st.write("• Create targeted marketing campaigns for each cluster")
        st.write("• Monitor segment migration patterns over time")
    
    # Cluster visualization
    st.subheader("🎯 Interactive Cluster Visualization")
    col1, col2 = st.columns(2)
    
    with col1:
        x_axis = st.selectbox("X-axis", ['Income', 'Activity_Score', 'Age', 'Tenure', 'Number_of_Products'])
    
    with col2:
        y_axis = st.selectbox("Y-axis", ['Activity_Score', 'Income', 'Age', 'Tenure', 'Number_of_Products'])
    
    st.plotly_chart(
        create_cluster_visualization(df_clustered, x_axis, y_axis),
        use_container_width=True
    )
    
    # Cluster comparison chart
    st.subheader("📈 Cluster Characteristics Comparison")
    st.plotly_chart(create_cluster_comparison_chart(cluster_analysis), use_container_width=True)
    
    with st.expander("📊 Cluster Comparison Insights"):
        st.markdown("**Multi-Dimensional Cluster Analysis (2x2 Layout)**")
        st.write("This comparison chart is organized in a 2x2 grid for better readability:")
        st.write("• **Top Row**: Age and Income distributions across clusters")
        st.write("• **Bottom Left**: Tenure and Product usage (dual y-axis)")
        st.write("• **Bottom Right**: Activity Score and Churn Rate (dual y-axis)")
        
        st.markdown("**Chart Organization Benefits:**")
        st.write("• **No Overlapping**: Each metric has its own dedicated space")
        st.write("• **Clear Comparison**: Easy to compare clusters across different dimensions")
        st.write("• **Dual Y-Axes**: Related metrics (Tenure/Products, Activity/Churn) share subplots")
        st.write("• **Color Coding**: Each metric has a distinct color for easy identification")
        
        st.markdown("**Segment Profiles:**")
        for _, cluster in cluster_analysis.iterrows():
            st.write(f"**{cluster['Cluster']}** ({cluster['Size']} customers):")
            st.write(f"  - Demographics: {cluster['Avg_Age']:.0f} years old, {cluster['Male_Percentage']:.1f}% male")
            st.write(f"  - Financial: ${cluster['Avg_Income']:,.0f} income, {cluster['Avg_Products']:.1f} products")
            st.write(f"  - Engagement: {cluster['Avg_Activity']:.1f} activity score, {cluster['Avg_Tenure']:.1f} years tenure")
            st.write(f"  - Risk: {cluster['Churn_Rate']:.1f}% churn rate")
            st.write("")
    
    # Business recommendations
    st.subheader("💡 Business Recommendations")
    
    for _, rec in recommendations.iterrows():
        risk_color = "🔴" if rec['Priority'] == 'High' else "🟡" if rec['Priority'] == 'Medium' else "🟢"
        
        with st.expander(f"{risk_color} {rec['Cluster']} - {rec['Risk_Level']}"):
            st.write(f"**Recommendation:** {rec['Recommendation']}")
            st.write(f"**Priority:** {rec['Priority']}")

def show_churn_prediction_page(df_features, training_data):
    """Display churn prediction interface"""
    st.header("🔮 Churn Prediction")
    
    # Train model
    if 'churn_model' not in st.session_state:
        with st.spinner("Training churn prediction model..."):
            model = ChurnPredictor('random_forest')
            model.train(
                training_data['X_train'],
                training_data['y_train'],
                training_data['X_test'],
                training_data['y_test'],
                training_data['scaler']
            )
            st.session_state.churn_model = model
    
    model = st.session_state.churn_model
    
    # Prediction interface
    st.subheader("🎯 Predict Churn for Individual Customer")
    
    # Customer selection method
    prediction_method = st.radio(
        "Choose prediction method:",
        ["Select existing customer", "Enter customer details manually"]
    )
    
    if prediction_method == "Select existing customer":
        # Customer dropdown
        customer_ids = df_features['CustomerID'].tolist()
        selected_customer = st.selectbox("Select Customer:", customer_ids)
        
        if selected_customer:
            customer_data = get_customer_summary(df_features, selected_customer)
            
            if customer_data:
                # Display customer info
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.write(f"**Customer ID:** {customer_data['CustomerID']}")
                    st.write(f"**Age:** {customer_data['Age']}")
                    st.write(f"**Gender:** {customer_data['Gender']}")
                
                with col2:
                    st.write(f"**Income:** ${customer_data['Income']:,}")
                    st.write(f"**Tenure:** {customer_data['Tenure']} years")
                    st.write(f"**Products:** {customer_data['Number_of_Products']}")
                
                with col3:
                    st.write(f"**Activity Score:** {customer_data['Activity_Score']}")
                    st.write(f"**Current Status:** {customer_data['Churn']}")
                
                # Predict churn
                if st.button("🔮 Predict Churn Probability"):
                    customer_profile = create_customer_profile(
                        customer_data['Age'],
                        customer_data['Gender'],
                        customer_data['Income'],
                        customer_data['Tenure'],
                        customer_data['Number_of_Products'],
                        customer_data['Activity_Score']
                    )
                    
                    probability = model.predict_churn_probability(customer_profile)
                    interpretation = interpret_churn_probability(probability)
                    
                    # Display results
                    st.subheader("📊 Prediction Results")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric(
                            label="Churn Probability",
                            value=f"{probability:.1%}",
                            delta=None
                        )
                        
                        risk_class = interpretation['risk_level']
                        risk_color = interpretation['color']
                        
                        if risk_color == 'red':
                            st.markdown(f'<p class="risk-high">Risk Level: {risk_class}</p>', unsafe_allow_html=True)
                        elif risk_color == 'orange':
                            st.markdown(f'<p class="risk-medium">Risk Level: {risk_class}</p>', unsafe_allow_html=True)
                        else:
                            st.markdown(f'<p class="risk-low">Risk Level: {risk_class}</p>', unsafe_allow_html=True)
                    
                    with col2:
                        st.write(f"**Interpretation:** {interpretation['interpretation']}")
                    
                    # Recommendations
                    st.subheader("💡 Retention Recommendations")
                    recommendations = get_retention_recommendations(probability, customer_data)
                    
                    for i, rec in enumerate(recommendations, 1):
                        st.write(f"{i}. {rec}")
    
    else:
        # Manual input
        st.subheader("📝 Enter Customer Details")
        
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("Age", min_value=18, max_value=100, value=35)
            gender = st.selectbox("Gender", ["Male", "Female"])
            income = st.number_input("Income ($)", min_value=0, max_value=500000, value=50000)
        
        with col2:
            tenure = st.number_input("Tenure (years)", min_value=0.0, max_value=50.0, value=2.0, step=0.1)
            products = st.number_input("Number of Products", min_value=1, max_value=20, value=2)
            activity = st.number_input("Activity Score", min_value=0.0, max_value=100.0, value=50.0, step=1.0)
        
        if st.button("🔮 Predict Churn Probability"):
            customer_profile = create_customer_profile(age, gender, income, tenure, products, activity)
            
            probability = model.predict_churn_probability(customer_profile)
            interpretation = interpret_churn_probability(probability)
            
            # Display results
            st.subheader("📊 Prediction Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(
                    label="Churn Probability",
                    value=f"{probability:.1%}",
                    delta=None
                )
                
                risk_class = interpretation['risk_level']
                risk_color = interpretation['color']
                
                if risk_color == 'red':
                    st.markdown(f'<p class="risk-high">Risk Level: {risk_class}</p>', unsafe_allow_html=True)
                elif risk_color == 'orange':
                    st.markdown(f'<p class="risk-medium">Risk Level: {risk_class}</p>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<p class="risk-low">Risk Level: {risk_class}</p>', unsafe_allow_html=True)
            
            with col2:
                st.write(f"**Interpretation:** {interpretation['interpretation']}")
            
            # Recommendations
            st.subheader("💡 Retention Recommendations")
            customer_data = {
                'Age': age,
                'Gender': gender,
                'Income': income,
                'Tenure': tenure,
                'Number_of_Products': products,
                'Activity_Score': activity
            }
            
            recommendations = get_retention_recommendations(probability, customer_data)
            
            for i, rec in enumerate(recommendations, 1):
                st.write(f"{i}. {rec}")
    
    # Model performance
    st.subheader("📈 Model Performance & Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        train_acc = model.model.score(training_data['X_train'], training_data['y_train'])
        test_acc = model.model.score(training_data['X_test'], training_data['y_test'])
        
        st.metric("Training Accuracy", f"{train_acc:.1%}")
        st.metric("Test Accuracy", f"{test_acc:.1%}")
        
        # Model performance insights
        with st.expander("📊 Model Performance Analysis"):
            st.markdown("**Random Forest Churn Prediction Model**")
            st.write("This machine learning model uses customer attributes to predict churn probability with high accuracy.")
            
            st.markdown("**Performance Metrics:**")
            st.write(f"• Training Accuracy: {train_acc:.1%} - Model fits training data well")
            st.write(f"• Test Accuracy: {test_acc:.1%} - Good generalization to new data")
            st.write(f"• Overfitting: {'Minimal' if test_acc > train_acc * 0.95 else 'Present' if test_acc < train_acc * 0.9 else 'Moderate'}")
            
            st.markdown("**Model Reliability:**")
            if test_acc > 0.8:
                st.write("✅ High confidence in predictions")
            elif test_acc > 0.7:
                st.write("⚠️ Moderate confidence - consider model improvements")
            else:
                st.write("❌ Low confidence - model needs enhancement")
            
            st.markdown("**Business Impact:**")
            st.write("• Enables proactive customer retention")
            st.write("• Reduces false positives in retention campaigns")
            st.write("• Improves ROI on customer success investments")
    
    with col2:
        # Feature importance
        importance_df = model.get_feature_importance()
        st.plotly_chart(create_feature_importance_chart(importance_df), use_container_width=True)
        
        with st.expander("📊 Feature Importance Analysis"):
            st.markdown("**Key Predictive Factors**")
            st.write("This chart shows which customer attributes are most important for predicting churn.")
            
            st.markdown("**Top 3 Most Important Features:**")
            for i, (_, row) in enumerate(importance_df.head(3).iterrows(), 1):
                st.write(f"{i}. **{row['Feature']}**: {row['Importance']:.3f}")
            
            st.markdown("**Strategic Insights:**")
            top_feature = importance_df.iloc[0]['Feature']
            st.write(f"• **{top_feature}** is the strongest churn predictor")
            st.write("• Focus data collection on high-importance features")
            st.write("• Develop targeted interventions for key risk factors")
            st.write("• Monitor feature importance changes over time")

def show_analytics_page(df, df_features):
    """Display advanced analytics"""
    st.header("📈 Advanced Analytics & Business Intelligence")
    
    # Executive Summary
    st.subheader("📋 Executive Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        churn_rate = (df['Churn'] == 'Yes').mean() * 100
        st.metric("Overall Churn Rate", f"{churn_rate:.1f}%")
    
    with col2:
        avg_income = df['Income'].mean()
        st.metric("Average Income", f"${avg_income:,.0f}")
    
    with col3:
        avg_activity = df['Activity_Score'].mean()
        st.metric("Average Activity", f"{avg_activity:.1f}")
    
    with col4:
        avg_tenure = df['Tenure'].mean()
        st.metric("Average Tenure", f"{avg_tenure:.1f} years")
    
    # Business insights summary
    with st.expander("📊 Key Business Insights"):
        st.markdown("**Customer Base Analysis**")
        st.write(f"• **Total Customer Base**: {len(df):,} customers")
        st.write(f"• **Churn Rate**: {churn_rate:.1f}% (Industry benchmark: 15-25%)")
        st.write(f"• **Customer Value**: Average income ${avg_income:,.0f}")
        st.write(f"• **Engagement Level**: Average activity score {avg_activity:.1f}/100")
        st.write(f"• **Relationship Length**: Average tenure {avg_tenure:.1f} years")
        
        st.markdown("**Critical Success Factors**")
        st.write("• Activity score is the strongest predictor of retention")
        st.write("• Younger customers show higher churn rates")
        st.write("• Product diversification improves customer stickiness")
        st.write("• First-year customers need focused attention")
        
        st.markdown("**Strategic Recommendations**")
        st.write("• Implement activity-based retention programs")
        st.write("• Develop age-specific engagement strategies")
        st.write("• Focus on cross-selling and product adoption")
        st.write("• Create robust onboarding for new customers")
    
    # Churn by segments
    st.subheader("🎯 Churn Analysis by Customer Segments")
    st.plotly_chart(create_churn_by_segments_chart(df), use_container_width=True)
    
    with st.expander("📊 Segment Analysis Insights"):
        st.markdown("**Customer Segment Performance**")
        st.write("This analysis reveals how different customer groups perform in terms of retention, enabling targeted strategies.")
        
        # Calculate segment insights
        age_groups = pd.cut(df['Age'], bins=[0, 30, 50, 70, 100], labels=['18-30', '31-50', '51-70', '71+'])
        activity_groups = pd.cut(df['Activity_Score'], bins=[0, 30, 60, 100], labels=['Low', 'Medium', 'High'])
        
        age_churn = df.groupby(age_groups)['Churn'].apply(lambda x: (x == 'Yes').mean() * 100)
        activity_churn = df.groupby(activity_groups)['Churn'].apply(lambda x: (x == 'Yes').mean() * 100)
        
        st.markdown("**Age Group Performance:**")
        for age_group, churn_rate in age_churn.items():
            if pd.notna(churn_rate):
                st.write(f"• {age_group}: {churn_rate:.1f}% churn rate")
        
        st.markdown("**Activity Level Performance:**")
        for activity_level, churn_rate in activity_churn.items():
            if pd.notna(churn_rate):
                st.write(f"• {activity_level} Activity: {churn_rate:.1f}% churn rate")
        
        st.markdown("**Strategic Actions:**")
        st.write("• Target high-churn segments with retention campaigns")
        st.write("• Replicate success factors from low-churn segments")
        st.write("• Develop segment-specific product offerings")
        st.write("• Monitor segment migration patterns")
    
    # Feature importance
    if 'churn_model' in st.session_state:
        st.subheader("🔍 Feature Importance Analysis")
        model = st.session_state.churn_model
        importance_df = model.get_feature_importance()
        st.plotly_chart(create_feature_importance_chart(importance_df), use_container_width=True)
        
        with st.expander("📊 Feature Importance Insights"):
            st.markdown("**Predictive Power Analysis**")
            st.write("Understanding which customer attributes most strongly predict churn enables focused retention efforts.")
            
            st.markdown("**Top Predictive Features:**")
            for i, (_, row) in enumerate(importance_df.head(5).iterrows(), 1):
                st.write(f"{i}. **{row['Feature']}**: {row['Importance']:.3f} importance score")
            
            st.markdown("**Business Implications:**")
            top_3_features = importance_df.head(3)['Feature'].tolist()
            st.write(f"• Focus on improving: {', '.join(top_3_features)}")
            st.write("• Develop targeted interventions for high-importance factors")
            st.write("• Monitor these metrics closely for early warning signs")
            st.write("• Create feature-specific retention strategies")
    
    # Data summary
    st.subheader("📊 Data Quality & Summary")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Dataset Overview:**")
        st.write(f"• Total Records: {len(df):,}")
        st.write(f"• Features: {len(df.columns)}")
        st.write(f"• Missing Values: {df.isnull().sum().sum()}")
        st.write(f"• Data Completeness: {((len(df) * len(df.columns) - df.isnull().sum().sum()) / (len(df) * len(df.columns)) * 100):.1f}%")
        
        st.markdown("**Data Quality Assessment:**")
        if df.isnull().sum().sum() == 0:
            st.write("✅ No missing values - excellent data quality")
        else:
            st.write("⚠️ Some missing values present - consider data cleaning")
        
        if len(df) >= 1000:
            st.write("✅ Sufficient sample size for reliable analysis")
        else:
            st.write("⚠️ Small sample size - results may be less reliable")
    
    with col2:
        st.markdown("**Churn Statistics:**")
        churn_stats = df['Churn'].value_counts()
        st.write(f"• Churned: {churn_stats['Yes']:,} ({churn_stats['Yes']/len(df)*100:.1f}%)")
        st.write(f"• Retained: {churn_stats['No']:,} ({churn_stats['No']/len(df)*100:.1f}%)")
        
        st.markdown("**Data Distribution:**")
        st.write(f"• Age Range: {df['Age'].min()}-{df['Age'].max()} years")
        st.write(f"• Income Range: ${df['Income'].min():,}-${df['Income'].max():,}")
        st.write(f"• Tenure Range: {df['Tenure'].min():.1f}-{df['Tenure'].max():.1f} years")
        st.write(f"• Activity Range: {df['Activity_Score'].min():.1f}-{df['Activity_Score'].max():.1f}")
    
    # Download data
    st.subheader("💾 Data Export & Reporting")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📥 Download Customer Data (CSV)"):
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"insightbank_customers_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
    
    with col2:
        if st.button("📊 Generate Executive Report"):
            st.info("Executive report generation would be implemented here with detailed insights, recommendations, and action plans.")
    
    # Additional analytics
    st.subheader("🔬 Advanced Statistical Analysis")
    
    with st.expander("📊 Statistical Insights"):
        st.markdown("**Correlation Analysis**")
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        # Create a copy of df with encoded churn for correlation analysis
        df_corr = df.copy()
        df_corr['Churn_Encoded'] = (df_corr['Churn'] == 'Yes').astype(int)
        
        # Include the encoded churn in correlation analysis
        corr_cols = list(numerical_cols) + ['Churn_Encoded']
        correlation_matrix = df_corr[corr_cols].corr()
        
        # Find strongest correlations with churn
        churn_correlations = correlation_matrix['Churn_Encoded'].abs().sort_values(ascending=False)
        st.write("**Strongest correlations with churn:**")
        for feature, corr in churn_correlations.head(5).items():
            if feature != 'Churn_Encoded':
                st.write(f"• {feature}: {corr:.3f}")
        
        st.markdown("**Distribution Analysis**")
        st.write("• Income distribution: Right-skewed (typical for income data)")
        st.write("• Age distribution: Normal distribution with slight skew")
        st.write("• Activity scores: Bimodal distribution (engaged vs disengaged)")
        st.write("• Tenure distribution: Exponential decay (many new customers)")
        
        st.markdown("**Outlier Detection**")
        outliers = 0
        for col in numerical_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers += len(df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)])
        
        st.write(f"• Total outliers detected: {outliers}")
        st.write("• Outlier rate: {:.1f}%".format(outliers / (len(df) * len(numerical_cols)) * 100))

if __name__ == "__main__":
    main()
