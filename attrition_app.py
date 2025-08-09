import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from scipy.stats import chi2_contingency
import warnings
warnings.filterwarnings('ignore')

class EmployeeAttritionAnalyzer:
    def __init__(self, data_path):
        self.df = self.load_data(data_path)
        self.preprocess_data()
        self.categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns
        self.numerical_cols = self.df.select_dtypes(include=['int64', 'float64']).columns
        
    def load_data(self, path):
        """Load and validate dataset"""
        df = pd.read_csv(path)
        print(f"Dataset loaded with {df.shape[0]} records and {df.shape[1]} features")
        return df
    
    def preprocess_data(self):
        """Clean and prepare data for analysis"""
        # Convert to categorical
        cat_cols = ['Attrition', 'BusinessTravel', 'Department', 'EducationField', 
                   'Gender', 'JobRole', 'MaritalStatus', 'OverTime']
        self.df[cat_cols] = self.df[cat_cols].astype('category')
        
        # Create derived features
        self.df['IncomeToAgeRatio'] = self.df['MonthlyIncome'] / self.df['Age']
        self.df['TenureGroup'] = pd.cut(self.df['YearsAtCompany'], 
                                       bins=[0, 2, 5, 10, 20, np.inf],
                                       labels=['0-2', '3-5', '6-10', '11-20', '20+'])
        
        print("Data preprocessing completed")
    
    def plot_attrition_flow(self):
        """Alternative visualization using stacked bar chart"""
        plt.figure(figsize=(12, 6))
        dept_attrition = self.df.groupby(['Department', 'Attrition']).size().unstack()
        dept_attrition.plot(kind='bar', stacked=True, color=['#2ecc71', '#e74c3c'])
        plt.title('Attrition Flow by Department')
        plt.ylabel('Number of Employees')
        plt.xticks(rotation=45)
        plt.legend(title='Attrition Status')
        
        # Add percentage annotations
        total = dept_attrition.sum(axis=1)
        for i, (idx, row) in enumerate(dept_attrition.iterrows()):
            plt.text(i, row['Yes']/2, f"{row['Yes']}\n({row['Yes']/total[i]*100:.1f}%)", 
                    ha='center', va='center', color='white')
            plt.text(i, row['Yes'] + row['No']/2, f"{row['No']}\n({row['No']/total[i]*100:.1f}%)", 
                    ha='center', va='center', color='black')
        
        plt.tight_layout()
        plt.show()
    
    def analyze_categorical_features(self):
        """Statistical analysis of categorical features"""
        results = []
        for col in self.categorical_cols:
            if col != 'Attrition':
                contingency = pd.crosstab(self.df[col], self.df['Attrition'])
                chi2, p, _, _ = chi2_contingency(contingency)
                results.append({
                    'Feature': col,
                    'Chi-Square': chi2,
                    'P-Value': p,
                    'Significant': p < 0.05
                })
        
        return pd.DataFrame(results).sort_values('P-Value')
    
    def plot_feature_importance(self):
        """Visualize predictive importance of features"""
        # Prepare data for modeling
        model_df = self.df.copy()
        le = LabelEncoder()
        for col in self.categorical_cols:
            model_df[col] = le.fit_transform(model_df[col])
            
        # Train simple model
        from sklearn.ensemble import RandomForestClassifier
        X = model_df.drop('Attrition', axis=1)
        y = model_df['Attrition']
        
        model = RandomForestClassifier(random_state=42)
        model.fit(X, y)
        
        # Plot importance
        feat_importances = pd.Series(model.feature_importances_, index=X.columns)
        plt.figure(figsize=(12, 8))
        feat_importances.nlargest(15).plot(kind='barh', color='#3498db')
        plt.title('Top 15 Predictive Features for Attrition')
        plt.xlabel('Relative Importance')
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.show()
    
    def generate_attrition_profile(self):
        """Create composite profile of employees who left"""
        left_df = self.df[self.df['Attrition'] == 'Yes']
        
        profile = {
            'Demographics': {
                'Average Age': left_df['Age'].mean(),
                'Gender Distribution': left_df['Gender'].value_counts(normalize=True).to_dict(),
                'Marital Status': left_df['MaritalStatus'].value_counts(normalize=True).to_dict()
            },
            'Employment': {
                'Average Tenure (Years)': left_df['YearsAtCompany'].mean(),
                'Common Job Roles': left_df['JobRole'].value_counts(normalize=True).nlargest(3).to_dict(),
                'Overtime Rate': (left_df['OverTime'] == 'Yes').mean()
            },
            'Compensation': {
                'Median Income': left_df['MonthlyIncome'].median(),
                'Income vs Stayers': (left_df['MonthlyIncome'].median() / 
                                    self.df[self.df['Attrition'] == 'No']['MonthlyIncome'].median() - 1)
            }
        }
        
        return profile
    
    def print_formatted_results(self, df, title):
        """Print DataFrame with nice formatting"""
        print(f"\n{title}")
        print("-" * 50)
        print(df.to_string(index=False))
        print("-" * 50)

if __name__ == "__main__":
    analyzer = EmployeeAttritionAnalyzer("greendestination.csv")
    
    # Display basic statistics
    print("\nAttrition Rate: {:.1f}%".format(
        100 * analyzer.df['Attrition'].value_counts(normalize=True)['Yes']))
    
    # Run analyses
    analyzer.plot_attrition_flow()
    analyzer.plot_feature_importance()
    
    # Show statistical significance
    cat_results = analyzer.analyze_categorical_features()
    analyzer.print_formatted_results(cat_results, "Categorical Feature Analysis")
    
    # Generate profile
    print("\nAttrition Profile:")
    profile = analyzer.generate_attrition_profile()
    for category, stats in profile.items():
        print(f"\n{category}:")
        for k, v in stats.items():
            print(f"  {k}: {v}")