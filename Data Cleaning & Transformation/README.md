# Data Cleaning & Preprocessing
> Before conducting any analysis, we must clean and prepare the dataset to ensure data accuracy.

```python
# Upload Kaggle API Key to Colab

from google.colab import files
files.upload()

# Move the File to the Correct Directory

import os
!mkdir -p ~/.kaggle
!mv kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

# Download and Extract Dataset

!kaggle datasets download -d rohitsahoo/sales-forecasting
!unzip sales-forecasting.zip

# import matplotlib

import matplotlib.pyplot as plt

# Load Dataset

import pandas as pd
df = pd.read_csv("train.csv")
df.head(5)
```

>```console
> Saving kaggle.json to kaggle.json
> Dataset URL: https://www.kaggle.com/datasets/rohitsahoo/sales-forecasting
> License(s): GPL-2.0
> sales-forecasting.zip: Skipping, found more recently modified local copy (use --force to force download)
> Archive:  sales-forecasting.zip
> inflating: train.csv
>
> | Row ID | Order ID       | Order Date | Ship Date  | Ship Mode      | Customer ID | Customer Name     | Segment    | Country       | City            | State      | Postal Code > | Region | Product ID       | Category       | Sub-Category | Product Name                                             | Sales   |
> |--------|---------------|------------|------------|----------------|-------------|------------------|------------|--------------|----------------|------------|-------------|---> -----|------------------|---------------|--------------|----------------------------------------------------------|---------|
> | 0      | CA-2017-152156 | 08/11/2017 | 11/11/2017 | Second Class   | CG-12520    | Claire Gute      | Consumer   | United States | Henderson      | Kentucky   | 42420.0     | > South  | FUR-BO-10001798  | Furniture     | Bookcases    | Bush Somerset Collection Bookcase                       | 261.96  |
> | 1      | CA-2017-152156 | 08/11/2017 | 11/11/2017 | Second Class   | CG-12520    | Claire Gute      | Consumer   | United States | Henderson      | Kentucky   | 42420.0     | > South  | FUR-CH-10000454  | Furniture     | Chairs       | Hon Deluxe Fabric Upholstered Stacking Chairs,...       | 731.94  |
> | 2      | CA-2017-138688 | 12/06/2017 | 16/06/2017 | Second Class   | DV-13045    | Darrin Van Huff  | Corporate  | United States | Los Angeles    | California | 90036.0     | > West   | OFF-LA-10000240  | Office Supplies | Labels       | Self-Adhesive Address Labels for Typewriters b...       | 14.62   |
> | 3      | US-2016-108966 | 11/10/2016 | 18/10/2016 | Standard Class | SO-20335    | Sean O'Donnell   | Consumer   | United States | Fort Lauderdale | Florida    | 33311.0     > | South  | FUR-TA-10000577  | Furniture     | Tables       | Bretford CR4500 Series Slim Rectangular Table           | 957.58  |
> | 4      | US-2016-108966 | 11/10/2016 | 18/10/2016 | Standard Class | SO-20335    | Sean O'Donnell   | Consumer   | United States | Fort Lauderdale | Florida    | 33311.0     > | South  | OFF-ST-10000760  | Office Supplies | Storage      | Eldon Fold 'N Roll Cart System                         | 22.37   |
>```
> We start by loading the dataset using Pandas in Google Colab. The dataset contains sales transactions with details like order date, product category, revenue and discount.

```python
# Check missing values

df.isnull().sum()
```
>```console
> | Column Name      | Missing Values |
> |-----------------|---------------|
> | Row ID         | 0             |
> | Order ID       | 0             |
> | Order Date     | 0             |
> | Ship Date      | 0             |
> | Ship Mode      | 0             |
> | Customer ID    | 0             |
> | Customer Name  | 0             |
> | Segment        | 0             |
> | Country        | 0             |
> | City          | 0             |
> | State         | 0             |
> | Postal Code   | 11            |
> | Region        | 0             |
> | Product ID    | 0             |
> | Category      | 0             |
> | Sub-Category  | 0             |
> | Product Name  | 0             |
> | Sales         | 0             |
>```

```python
# Fill missing values with forward fill method

df.ffill(inplace=True)
```
> Rename the sales column to include currency ($)
```python
df = pd.read_csv("train.csv").rename(columns ={'Sales': 'Sales ($)'})
df['Sales ($)'] = df['Sales ($)'].round(2)
```
> Here, we identify and handle missing values using forward-fill to maintain data consistency.
```python
df['Order Date'] = pd.to_datetime(df['Order Date'], dayfirst=True)
```
> Converting Order Data to a datetime format allows for proper time-series forecasting.

> We can now extract Year, Month, and Weekday trends to observe sales patterns
```python
df['Year'] = df['Order Date'].dt.year
df['Month'] = df['Order Date'].dt.month
df['Weekday'] = df['Order Date'].dt.day_name()
```
> Extracting date features allows us to perform time-based trend analysis and identify peak sales periods.

> Confirmation that year, month and weekday attributes have been added to the dataframe

```python
df.head(5)
```
```console
| Row ID | Order ID        | Order Date | Ship Date  | Ship Mode      | Customer ID | Customer Name     | Segment    | Country        | City            | Postal Code | Region | Product ID       | Category        | Sub-Category | Product Name                                              | Sales ($) | Year | Month | Weekday  |
|--------|---------------|------------|------------|---------------|-------------|------------------|------------|---------------|----------------|-------------|--------|-----------------|---------------|-------------|------------------------------------------------------|-----------|------|-------|----------|
| 1      | CA-2017-152156 | 2017-11-08 | 11/11/2017 | Second Class  | CG-12520    | Claire Gute      | Consumer   | United States | Henderson      | 42420.0     | South  | FUR-BO-10001798  | Furniture      | Bookcases   | Bush Somerset Collection Bookcase                    | 261.96    | 2017 | 11    | Wednesday |
| 2      | CA-2017-152156 | 2017-11-08 | 11/11/2017 | Second Class  | CG-12520    | Claire Gute      | Consumer   | United States | Henderson      | 42420.0     | South  | FUR-CH-10000454  | Furniture      | Chairs      | Hon Deluxe Fabric Upholstered Stacking Chairs,...    | 731.94    | 2017 | 11    | Wednesday |
| 3      | CA-2017-138688 | 2017-06-12 | 16/06/2017 | Second Class  | DV-13045    | Darrin Van Huff  | Corporate  | United States | Los Angeles    | 90036.0     | West   | OFF-LA-10000240  | Office Supplies | Labels      | Self-Adhesive Address Labels for Typewriters b...   | 14.62     | 2017 | 6     | Monday    |
| 4      | US-2016-108966 | 2016-10-11 | 18/10/2016 | Standard Class | SO-20335    | Sean O'Donnell   | Consumer   | United States | Fort Lauderdale | 33311.0     | South  | FUR-TA-10000577  | Furniture      | Tables      | Bretford CR4500 Series Slim Rectangular Table     | 957.58    | 2016 | 10    | Tuesday   |
| 5      | US-2016-108966 | 2016-10-11 | 18/10/2016 | Standard Class | SO-20335    | Sean O'Donnell   | Consumer   | United States | Fort Lauderdale | 33311.0     | South  | OFF-ST-10000760  | Office Supplies | Storage     | Eldon Fold 'N Roll Cart System                      | 22.37     | 2016 | 10    | Tuesday   |
```
