# Retail-Shop
ustomer Segmentation using K-Means Clustering

📌 Project Overview
This project performs Customer Segmentation using the K-Means clustering algorithm on the Mall Customers dataset.
The aim is to group customers based on their Age, Annual Income, and Spending Score, helping businesses target specific customer segments effectively.

📂 Dataset
Name: Mall_Customers.csv

Source: Kaggle Dataset

Description: The dataset contains information about customers, including:

CustomerID

Gender

Age

Annual Income (k$)

Spending Score (1-100)

⚙️ Steps Followed
1. Data Preprocessing
Filled missing numeric values with column mean.

Label encoded categorical columns (e.g., Gender).

Selected relevant features: Age, Annual Income (k$), Spending Score (1-100).

Standardized features using StandardScaler.

2. Choosing Optimal Number of Clusters
Used the Elbow Method to determine the optimal value of k.

Selected k = 5 based on SSE curve.

3. Model Training
Applied K-Means clustering with k = 5.

Assigned each customer to a cluster.

4. Visualization
Created a scatter plot showing customer segments by Annual Income and Spending Score.

📊 Results
Generated a clustered_customers.csv file containing customer data with their cluster labels.

Plotted Elbow Method graph and cluster visualization
