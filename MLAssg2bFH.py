# Import required libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler # For Standard Scaling
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr, chi2_contingency  # Pearson correlation, Chi-Square test
import pickle
import warnings
warnings.filterwarnings("ignore")
# *******************1 EDA*******************
# *******************1.1 Base Structure of EDA*******************
df = pd.read_csv("insuranceFH.csv")
print(df.head())
print(df.tail())
print(df.shape)
df.info()
print(df.describe())

# *******************2-Visualization of EDA[1-histplot 2-countplot 3-boxplot 4-heatmap]
# *******************2.1 Histplot for Numeric Data
cols_num = df.select_dtypes(include='number').columns
fig,axes = plt.subplots(nrows=3,ncols=3,figsize=(10,10))
axes=axes.flatten()

for i, col in enumerate(cols_num): # P:enumerates(cols_num)
    sns.histplot(data=df, x=col, kde=True, bins=50, ax=axes[i])
    axes[i].set_title(f"Distribution of {col}")
    axes[i].set_xlabel(col)
    axes[i].set_ylabel("Count")

# Remove empty subplots if any
for j in range(len(cols_num),len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.subplots_adjust(
    top = 0.90,     # More space above plots
    hspace = 0.5,   # Vertical spacing
    wspace = 0.3    # Horizontal spacing
)
plt.savefig("HistplotBMLFH.png",
            dpi=300,
            bbox_inches="tight")
plt.show()
# *******************2.2 Countplot for Categorical Data
cols_cat = ["gender","children","smoker","region"]
fig,axes = plt.subplots(nrows=2,ncols=2,figsize=(10,10))
axes=axes.flatten()

for i, col in enumerate(cols_cat): # P:enumerate(cols_num)
    sns.countplot(data=df, x=col, ax=axes[i])
    axes[i].set_title(f'Frequency of {col}')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel("Count")

plt.tight_layout()
plt.subplots_adjust(
    top = 0.90,     # More space above plots
    hspace = 0.5,   # Vertical spacing
    wspace = 0.2,    # Horizontal spacing
    bottom = 0.1
)
plt.savefig("CountplotBMLFH.png",
            dpi=300,
            bbox_inches="tight")
plt.show()

# *******************2.3 Boxplot for checking outliers
cols_num = df.select_dtypes(include='number').columns
fig,axes = plt.subplots(nrows=3, ncols=3, figsize=(10,10))
axes=axes.flatten()

for i, col in enumerate(cols_num): # P:enumerates(cols_num)
    sns.boxplot(data=df, x=col, ax=axes[i]) # P:
    axes[i].set_title(f'Boxplot of {col}')
    axes[i].set_xlabel(col)

# Remove empty subplots if any
for j in range(len(cols_num), len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.subplots_adjust(
    top = 0.90,     # More space above plots
    hspace = 0.5,   # Vertical spacing
    wspace = 0.3    # Horizontal spacing
)
plt.savefig("BoxplotBMLFH.png",
            dpi=300,
            bbox_inches="tight")
plt.show()

# *******************2.4 Heatmap for checking co-relation
sns.heatmap(df.corr(numeric_only=True), annot=True)

plt.title("Correlation Heatmap")
plt.xlabel("Features")
plt.ylabel("Features")

plt.tight_layout()
plt.subplots_adjust(
    top = 0.90,     # More space above plots
    hspace = 0.5,   # Vertical spacing
    wspace = 0.3    # Horizontal spacing
)
plt.savefig("HeatmapBMLFH.png",
            dpi=300,
            bbox_inches="tight")
plt.show()
# *******************3 Data Cleaning*******************
# Create a copy of the dataset for data cleaning
df_cleaned = df.copy()
# print(df_cleaned.head())

# print(df_cleaned.shape)
# Remove duplicate rows
df_cleaned.drop_duplicates(inplace=True)
# print(df_cleaned.shape)
print(df_cleaned.isnull().sum())
# print(df_cleaned.dtypes)

# Male/Female, male/female, MALE/FEMALE etc
# print(df_cleaned["gender"].value_counts())

# Encode gender using Label Encoding
df_cleaned["gender"] = df_cleaned["gender"].str.lower().map({"male":1,"female":0})

# Encode smoker column using Label Encoding
df_cleaned["smoker"] = df_cleaned["smoker"].str.lower().map({"yes":1,"no":0})
# print(df_cleaned.head())
# Rename encoded columns for better readability
df_cleaned.rename(columns={"gender" : "is_gender",
                           "smoker" : "is_smoker"},
                           inplace = True)
# print(df_cleaned.head())
# Encode region using One-Hot Encoding
df_cleaned = pd.get_dummies(df_cleaned,columns=["region"],drop_first=True)
# print(df_cleaned.head())
pd.set_option('display.max_columns',None)
# print(df_cleaned.head())
# Convert one-hot encoded boolean values (True/False) to integers
df_cleaned = df_cleaned.astype(int)
# print(df_cleaned.head())
# *******************4 Feature Scaling*******************
"""
CBMI                    Range
Underweight             Less than 18.5
Normal / Healthy        18.5 to 24.9
Overweight              25.0 to 29.9
Obese                   30.0 or higher
"""
# Convert BMI into categorical ranges
df_cleaned["bmi_cat"] = pd.cut(
                            df_cleaned["bmi"],
                            bins=[0,18.5,24.9,29.9,float('inf')],
                            labels=["Underweight" , "HealthyWeight" , "Overweight" , "Obesity"])
# print(df_cleaned.head())
df_cleaned = pd.get_dummies(df_cleaned,columns=["bmi_cat"],drop_first=True)
pd.set_option('display.max_columns',None)
# print(df_cleaned.head())
df_cleaned = df_cleaned.astype(int)
# print(df_cleaned.head())
# print(df_cleaned.columns)

# Standardize numerical features
cols = ["age","bmi","children"]
scaler = StandardScaler()
df_cleaned[cols] = scaler.fit_transform(df_cleaned[cols])
# print(df_cleaned.head())
print(df_cleaned.columns)

# *******************5 Feature Extraction*******************
selected_features = {'age', 'is_gender', 'bmi', 'children', 'is_smoker',
       'region_northwest', 'region_southeast', 'region_southwest',
       'bmi_cat_HealthyWeight', 'bmi_cat_Overweight',
       'bmi_cat_Obesity'}

# Store Pearson correlation values in a dictionary
correlations = {
                  feature:pearsonr(df_cleaned[feature],df_cleaned["charges"])[0]
                for feature in selected_features
}
# Convert dictionary to DataFrame
correlations_df = pd.DataFrame(list(correlations.items()),columns=["Feature","Pearson Correlation"])

# Sort features by correlation value
correlations_df.sort_values(by="Pearson Correlation",ascending=False,inplace=True)
# print(correlations_df)

# Categorical features
cat_features = ['is_gender', 'is_smoker', 'region_northwest',
                'region_southeast', 'region_southwest', 'bmi_cat_HealthyWeight' ,
                'bmi_cat_Overweight', 'bmi_cat_Obesity']

# Chi-Square Tes to Compare categories
alpha = 0.05

# Create quartile-based bins of charges for Chi-Square analysis
df_cleaned['charges_bin'] = pd.qcut(df_cleaned['charges'], q=4, labels=False)

chi2_results = {}

for col in cat_features:
    contingency = pd.crosstab(df_cleaned[col], df_cleaned['charges_bin'])
    chi2_stat, p_val, _, _ = chi2_contingency(contingency)
    decision = 'Reject Null (Keep Feature)' if p_val < alpha else 'Accept Null (Drop Feature)'
    chi2_results[col] = {
        'chi2_statistic': chi2_stat,
        'p_value': p_val,
        'Decision': decision
    }
# Convert Chi-Square results into a DataFrame
chi2_df = pd.DataFrame(chi2_results).T
chi2_df = chi2_df.sort_values(by='p_value')
print(chi2_df)
print(df_cleaned.columns.tolist())
# final_df = df_cleaned[['age', 'is_gender', 'bmi', 'children', 'is_smoker', 'charges','region_southeast','bmi_cat_Obesity']]
final_df = df_cleaned[['age', 'is_gender', 'bmi', 'children', 'is_smoker', 'charges','region_southeast','bmi_cat_Obesity','region_southwest','bmi_cat_HealthyWeight']]
# print(final_df.head())

# *******************6 Model Training*******************
print("Linear Regression using Scikit-Learn (Train-Test Split + Synthetic Inference)")

# Features and target
X = final_df.drop(columns='charges')
Y = final_df['charges']

# Split the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# Train Linear Regression model
lr = LinearRegression()
lr.fit(X_train, Y_train)
intercept = lr.intercept_
coefficient = lr.coef_

# Save Model For GUI
pickle.dump(lr, open("insurance_model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))

# Predict on the test set
Y_pred_test = lr.predict(X_test)

# Residuals
residuals = Y_test - Y_pred_test
# print(intercept)
# print(coefficient)

# Generate synthetic (unseen) Random Features values
X_synthetic = X.sample(10, random_state=42).copy()

X_synthetic["age"] += np.random.normal(0, 0.2, 10)
X_synthetic["bmi"] += np.random.normal(0, 0.2, 10)
X_synthetic["children"] += np.random.normal(0, 0.2, 10)

# Don't add noise to binary (0/1) features
# X_synthetic["is_gender"] += np.random.normal(0, 0.2, 10)
# X_synthetic["is_smoker"] += np.random.normal(0, 0.2, 10)
# X_synthetic["region_southeast"] += np.random.normal(0, 0.2, 10)
# X_synthetic["bmi_cat_Obesity"] += np.random.normal(0, 0.2, 10)

# Predict using the trained model
predictions = lr.predict(X_synthetic)

# Manual prediction using regression equation
expected_predictions = []

for _, row in X_synthetic.iterrows():
    pred = intercept + np.sum(coefficient * row.values)
    expected_predictions.append(pred)

# Create result table
result_df = X_synthetic.copy()
result_df["Predicted_Charges"] = predictions
result_df["Expected_Charges"] = expected_predictions

print("\nSynthetic Inference")
print(result_df)

print(f"\nIntercept = {intercept:.2f}")

# Store feature coefficients in a DataFrame
coef_df = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": coefficient
})

print("\nFeature Coefficients")
print(coef_df)

# Model evaluation
# print("R² score (test data)['age', 'is_gender', 'bmi', 'children', 'is_smoker', 'charges','region_southeast','bmi_cat_Obesity']:", lr.score(X_test, Y_test)) -FH
print("\nModel Evaluation")
print(f"R² Score (Test Data): {lr.score(X_test, Y_test):.4f}")
#
# ******************* 7 Visualization of data after ML *******************
#
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 5))
#  (1 )Charges Distribution Plot
sns.histplot(Y, kde=True, bins=40,  color="forestgreen",
    edgecolor="black",
    alpha=0.7,ax=axes[1])

axes[1].set_xlabel("Charges")
axes[1].set_ylabel("Count")
axes[1].set_title("Charges Distribution")

# (2) Actual vs Predicted

Y_pred = lr.predict(X_test)
# Predicted Charges
axes[0].scatter(
    Y_test,
    Y_pred_test,
    color="royalblue",
    edgecolor="black",
    alpha=0.75,
    s=55,
    label="Predicted Charges"
)
# Actual Charges
axes[0].scatter(
    Y_test,
    Y_test,
    color="forestgreen",
    marker="o",
    edgecolor="black",
    alpha=0.65,
    s=40,
    label="Actual Charges"
)

# Perfect prediction line
min_val = min(Y_test.min(), Y_pred.min())
max_val = max(Y_test.max(), Y_pred.max())

axes[0].plot(
    [min_val, max_val],
    [min_val, max_val],
    color="red",
    linestyle="--",
    linewidth=2,
    label="Perfect Prediction"
)

axes[0].set_xlabel("Actual Charges")
axes[0].set_ylabel("Predicted Charges")
axes[0].set_title("Actual vs Predicted")
axes[0].legend()

# (3) Residual Plot
residuals = Y_test - Y_pred
# Positive Residuals
positive = residuals >= 0

axes[2].scatter(
    Y_pred_test[positive],
    residuals[positive],
    color="darkorange",
    edgecolor="black",
    s=70,
    alpha=0.8,
    label="Positive Residuals"
)

# Negative Residuals
axes[2].scatter(
    Y_pred_test[~positive],
    residuals[~positive],
    color="purple",
    edgecolor="black",
    s=70,
    alpha=0.8,
    label="Negative Residuals"
)

axes[2].axhline(
    y=0,
    color="red",
    linestyle="--",
    linewidth=2,
    label="Zero Error"
)

axes[2].set_title("Residual Plot")
axes[2].set_xlabel("Predicted Charges")
axes[2].set_ylabel("Residuals (Actual - Predicted)")
axes[2].legend()


plt.tight_layout()

plt.subplots_adjust(
    left=0.06,
    right=0.98,
    wspace=0.35,
    top=0.90
)
# Save the visualization as an image
plt.savefig("MultipleLinearRegressionFH.png",
            dpi=300,
            bbox_inches="tight")
plt.show()