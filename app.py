import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler

# Page config
st.set_page_config(
    page_title="HR Employee Attrition Predictor",
    page_icon="🏢",
    layout="wide"
)

# Load model and artifacts
@st.cache_resource
def load_artifacts():
    model = joblib.load('attrition_model.pkl')
    scaler = joblib.load('scaler.pkl')
    feature_names = joblib.load('feature_names.pkl')
    le_dict = joblib.load('label_encoders.pkl')
    return model, scaler, feature_names, le_dict

# Define categorical and numerical features
categorical_features = [
    'BusinessTravel', 'Department', 'EducationField', 'Gender',
    'MaritalStatus', 'OverTime'
]

numerical_features = [
    'Age', 'DailyRate', 'DistanceFromHome', 'Education', 'EnvironmentSatisfaction',
    'HourlyRate', 'JobInvolvement', 'JobLevel', 'JobSatisfaction',
    'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked',
    'PercentSalaryHike', 'PerformanceRating', 'RelationshipSatisfaction',
    'StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear',
    'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole',
    'YearsSinceLastPromotion', 'YearsWithCurrManager'
]

def engineer_features(input_data):
    """Apply same feature engineering as in training"""
    data = input_data.copy()
    
    # Add engineered features
    data['Age_YearsAtCompany'] = data['Age'] * data['YearsAtCompany']
    data['Income_Per_Year'] = data['MonthlyIncome'] / (data['YearsAtCompany'] + 1)
    data['JobSatisfaction_Environment'] = data['JobSatisfaction'] * data['EnvironmentSatisfaction']
    data['TotalWorkingYears_Ratio'] = data['TotalWorkingYears'] / (data['Age'] + 1)
    data['Promotion_Gap'] = data['YearsSinceLastPromotion'] / (data['YearsAtCompany'] + 1)
    data['WorkLife_Balance'] = data['WorkLifeBalance'] * data['JobInvolvement']
    
    return data

def predict_attrition(model, scaler, feature_names, input_data, le_dict):
    """Make prediction with proper preprocessing"""
    # Create a copy
    data = input_data.copy()
    
    # Encode categorical variables
    for col in categorical_features:
        if col in data.columns and col in le_dict:
            try:
                data[col] = le_dict[col].transform([data[col].iloc[0]])[0]
            except ValueError:
                # Handle unknown category
                data[col] = 0
    
    # Apply feature engineering
    data = engineer_features(data)
    
    # Ensure all required features are present
    for feat in feature_names:
        if feat not in data.columns:
            data[feat] = 0
    
    # Reorder columns to match training
    data = data[feature_names]
    
    # Scale features
    data_scaled = scaler.transform(data)
    
    # Predict
    prediction = model.predict(data_scaled)
    probability = model.predict_proba(data_scaled)[0, 1]
    
    return prediction[0], probability

def get_department_stats(df, dept):
    """Get statistics for a department"""
    dept_data = df[df['Department'] == dept]
    return {
        'total': len(dept_data),
        'attrition_rate': (dept_data['Attrition'] == 'Yes').mean() * 100,
        'avg_age': dept_data['Age'].mean(),
        'avg_income': dept_data['MonthlyIncome'].mean(),
        'avg_satisfaction': dept_data['JobSatisfaction'].mean()
    }

def main():
    # Load artifacts
    try:
        model, scaler, feature_names, le_dict = load_artifacts()
    except FileNotFoundError:
        st.error("Model files not found. Please run model_training.py first!")
        st.info("Run: python model_training.py")
        return
    
    # Title
    st.title("🏢 HR Employee Attrition Prediction")
    st.markdown("""
    <style>
    .big-font { font-size:20px !important; }
    .highlight { background-color: #f0f2f6; padding: 20px; border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    ### Predict Employee Attrition Risk
    This tool helps HR professionals identify employees at risk of leaving the organization.
    """)
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🔮 Predict", "📊 Data Overview", "📈 Analytics", "ℹ️ About"])
    
    with tab1:
        st.markdown("### Enter Employee Details")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📋 Personal & Job Information")
            age = st.slider("Age", 18, 65, 35, help="Employee's age")
            gender = st.selectbox("Gender", ["Male", "Female"])
            marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
            
            department = st.selectbox("Department", ["Research & Development", "Sales", "Human Resources"])
            job_role = st.selectbox("Job Role", [
                "Sales Executive", "Research Scientist", "Laboratory Technician",
                "Manufacturing Director", "Healthcare Representative", "Manager",
                "Sales Representative", "Research Director", "Human Resources"
            ])
            job_level = st.slider("Job Level", 1, 5, 2)
            
            st.markdown("#### 💰 Compensation")
            monthly_income = st.number_input("Monthly Income ($)", 1000, 25000, 5000, step=500)
            percent_salary_hike = st.slider("Percent Salary Hike", 10, 25, 15, step=1)
            performance_rating = st.slider("Performance Rating", 1, 4, 3)
            
        with col2:
            st.markdown("#### 🏠 Work Environment")
            business_travel = st.selectbox("Business Travel", ["Travel_Rarely", "Travel_Frequently", "Non-Travel"])
            distance_from_home = st.slider("Distance From Home (miles)", 1, 30, 10)
            
            environment_satisfaction = st.slider("Environment Satisfaction", 1, 4, 3)
            job_satisfaction = st.slider("Job Satisfaction", 1, 4, 3)
            relationship_satisfaction = st.slider("Relationship Satisfaction", 1, 4, 3)
            work_life_balance = st.slider("Work-Life Balance", 1, 4, 3)
            job_involvement = st.slider("Job Involvement", 1, 4, 3)
            
            st.markdown("#### 📅 Tenure & Experience")
            years_at_company = st.slider("Years at Company", 0, 40, 5)
            total_working_years = st.slider("Total Working Years", 0, 40, 10)
            years_in_current_role = st.slider("Years in Current Role", 0, 20, 3)
            years_since_last_promotion = st.slider("Years Since Last Promotion", 0, 15, 1)
            years_with_curr_manager = st.slider("Years with Current Manager", 0, 20, 3)
            num_companies_worked = st.slider("Number of Companies Worked", 0, 10, 2)
            training_times_last_year = st.slider("Training Times Last Year", 0, 6, 3)
            
            stock_option_level = st.slider("Stock Option Level", 0, 3, 1)
            overtime = st.selectbox("OverTime", ["No", "Yes"])
        
        # Additional hidden features
        daily_rate = st.slider("Daily Rate", 100, 1500, 800, step=50)
        hourly_rate = st.slider("Hourly Rate", 30, 100, 65, step=5)
        monthly_rate = st.slider("Monthly Rate", 2000, 27000, 15000, step=1000)
        education = st.slider("Education Level", 1, 5, 3)
        education_field = st.selectbox("Education Field", ["Life Sciences", "Medical", "Marketing", "Technical Degree", "Human Resources", "Other"])
        
        # Create input dataframe
        input_data = pd.DataFrame([{
            'Age': age,
            'BusinessTravel': business_travel,
            'DailyRate': daily_rate,
            'Department': department,
            'DistanceFromHome': distance_from_home,
            'Education': education,
            'EducationField': education_field,
            'EnvironmentSatisfaction': environment_satisfaction,
            'Gender': gender,
            'HourlyRate': hourly_rate,
            'JobInvolvement': job_involvement,
            'JobLevel': job_level,
            'JobRole': job_role,
            'JobSatisfaction': job_satisfaction,
            'MaritalStatus': marital_status,
            'MonthlyIncome': monthly_income,
            'MonthlyRate': monthly_rate,
            'NumCompaniesWorked': num_companies_worked,
            'OverTime': overtime,
            'PercentSalaryHike': percent_salary_hike,
            'PerformanceRating': performance_rating,
            'RelationshipSatisfaction': relationship_satisfaction,
            'StockOptionLevel': stock_option_level,
            'TotalWorkingYears': total_working_years,
            'TrainingTimesLastYear': training_times_last_year,
            'WorkLifeBalance': work_life_balance,
            'YearsAtCompany': years_at_company,
            'YearsInCurrentRole': years_in_current_role,
            'YearsSinceLastPromotion': years_since_last_promotion,
            'YearsWithCurrManager': years_with_curr_manager
        }])
        
        # Predict button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            predict_button = st.button("🔮 Predict Attrition Risk", use_container_width=True)
        
        if predict_button:
            with st.spinner("Analyzing employee data..."):
                prediction, probability = predict_attrition(
                    model, scaler, feature_names, input_data, le_dict
                )
                
                # Display results
                st.markdown("---")
                st.markdown("### 📊 Prediction Results")
                
                col1, col2, col3 = st.columns([1, 2, 1])
                
                with col2:
                    if prediction == 1:
                        st.error("### ⚠️ High Attrition Risk")
                        st.error(f"**Probability: {probability*100:.1f}%**")
                        st.warning("""
                        **Recommendations:**
                        - Schedule a stay interview
                        - Review compensation and benefits
                        - Discuss career development opportunities
                        - Check work-life balance
                        """)
                    else:
                        st.success("### ✅ Low Attrition Risk")
                        st.success(f"**Probability: {probability*100:.1f}%**")
                        st.info("""
                        **Recommendations:**
                        - Continue engagement efforts
                        - Consider for retention programs
                        - Monitor for any changes
                        """)
                
                # Show risk gauge
                st.markdown("### Risk Meter")
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=probability*100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Attrition Risk (%)"},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 30], 'color': "lightgreen"},
                            {'range': [30, 60], 'color': "yellow"},
                            {'range': [60, 100], 'color': "red"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': probability*100
                        }
                    }
                ))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.markdown("### 📊 Dataset Overview")
        
        # Load data for overview
        df = pd.read_csv('HR-Employee-Attrition.csv')
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Employees", f"{len(df):,}")
        with col2:
            attrition_rate = (df['Attrition'] == 'Yes').mean() * 100
            st.metric("Attrition Rate", f"{attrition_rate:.1f}%")
        with col3:
            st.metric("Departments", df['Department'].nunique())
        with col4:
            st.metric("Avg Age", f"{df['Age'].mean():.1f}")
        
        # Department breakdown
        st.markdown("#### Attrition by Department")
        dept_attrition = df.groupby('Department')['Attrition'].apply(
            lambda x: (x == 'Yes').mean() * 100
        ).reset_index()
        
        fig = px.bar(
            dept_attrition, 
            x='Department', 
            y='Attrition',
            title='Attrition Rate by Department',
            color='Attrition',
            color_continuous_scale='RdYlGn_r',
            text_auto='.1f'
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        # Attrition by Age
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Attrition by Age Group")
            df['AgeGroup'] = pd.cut(df['Age'], bins=[18, 25, 35, 45, 55, 65])
            age_attrition = df.groupby('AgeGroup')['Attrition'].apply(
                lambda x: (x == 'Yes').mean() * 100
            ).reset_index()
            fig = px.bar(age_attrition, x='AgeGroup', y='Attrition', 
                        title='Attrition Rate by Age Group',
                        color='Attrition', color_continuous_scale='RdYlGn_r')
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### Attrition by OverTime")
            overtime_attrition = df.groupby('OverTime')['Attrition'].apply(
                lambda x: (x == 'Yes').mean() * 100
            ).reset_index()
            fig = px.bar(overtime_attrition, x='OverTime', y='Attrition',
                        title='Attrition Rate by OverTime',
                        color='Attrition', color_continuous_scale='RdYlGn_r')
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown("### 📈 Advanced Analytics")
        
        df = pd.read_csv('HR-Employee-Attrition.csv')
        
        # Create key metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Key Drivers of Attrition")
            # Calculate averages for attrited vs non-attrited
            attrited = df[df['Attrition'] == 'Yes']
            stayed = df[df['Attrition'] == 'No']
            
            metrics = ['Age', 'MonthlyIncome', 'YearsAtCompany', 'JobSatisfaction', 
                      'WorkLifeBalance', 'EnvironmentSatisfaction']
            
            comparison = []
            for metric in metrics:
                comparison.append({
                    'Metric': metric,
                    'Attrited': attrited[metric].mean(),
                    'Stayed': stayed[metric].mean(),
                    'Difference': stayed[metric].mean() - attrited[metric].mean()
                })
            
            comp_df = pd.DataFrame(comparison)
            fig = px.bar(comp_df, x='Metric', y=['Attrited', 'Stayed'],
                        title='Comparison: Attrited vs Stayed',
                        barmode='group')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### Tenure Analysis")
            # Tenure distribution
            tenure_attrition = df.groupby('YearsAtCompany')['Attrition'].apply(
                lambda x: (x == 'Yes').mean() * 100
            ).reset_index()
            
            fig = px.line(tenure_attrition, x='YearsAtCompany', y='Attrition',
                         title='Attrition Rate by Years at Company')
            fig.update_layout(xaxis_title='Years at Company', yaxis_title='Attrition Rate (%)')
            st.plotly_chart(fig, use_container_width=True)
        
        # Correlation heatmap
        st.markdown("#### Feature Correlations with Attrition")
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        numeric_cols = [col for col in numeric_cols if col not in ['EmployeeCount', 'EmployeeNumber']]
        
        # Calculate correlations
        correlations = []
        for col in numeric_cols:
            if col != 'Attrition':
                corr = df[df['Attrition'] == 'Yes'][col].mean() - df[df['Attrition'] == 'No'][col].mean()
                correlations.append({
                    'Feature': col,
                    'Correlation': corr,
                    'Interpretation': 'Higher' if corr > 0 else 'Lower'
                })
        
        corr_df = pd.DataFrame(correlations).sort_values('Correlation', ascending=False)
        fig = px.bar(corr_df.head(15), x='Feature', y='Correlation',
                    title='Top Features Differentiating Attrited Employees',
                    color='Interpretation',
                    color_discrete_map={'Higher': '#ff6b6b', 'Lower': '#51cf66'})
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.markdown("""
        ### ℹ️ About This Tool
        
        **HR Employee Attrition Predictor** is designed to help HR professionals and managers identify employees at risk of leaving the organization.
        
        #### 🎯 Purpose
        - Early identification of attrition risk
        - Data-driven retention strategies
        - Reduce turnover costs
        - Improve employee satisfaction
        
        #### 🔧 How It Works
        1. Enter employee details in the **Predict** tab
        2. Click "Predict Attrition Risk"
        3. Get instant risk assessment with actionable recommendations
        
        #### 📊 Key Features Used
        - **Job Satisfaction** & **Environment Satisfaction**
        - **Work-Life Balance** & **Job Involvement**
        - **Tenure** and **Promotion History**
        - **Compensation** and **Performance**
        - **Overtime** & **Business Travel**
        
        #### 🛠️ Model Details
        - **Algorithm**: Random Forest Classifier
        - **Accuracy**: ~85-90%
        - **Features**: 30+ employee attributes
        - **Training Data**: 1,470+ employee records
        
        #### 📌 Recommendations
        - **Low Risk**: Continue engagement, consider for retention programs
        - **High Risk**: Schedule stay interview, review compensation, discuss career growth
        """)

if __name__ == "__main__":
    main()