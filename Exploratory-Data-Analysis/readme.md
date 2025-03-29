# Step 3: Exploratory Data Analysis (EDA)
  > A. Identifying Best Selling Products
```python
top_products = df.groupby('Product Name')['Sales ($)'].sum().sort_values(ascending=False).head(5)
top_products
````
>```console
> | Product Name                                                                | Sales ($)   |
> |-----------------------------------------------------------------------------|------------:|
> | Canon imageCLASS 2200 Advanced Copier                                       | 61599.83    |
> | Fellowes PB500 Electric Punch Plastic Comb Binding Machine with Manual Bind | 27453.38    |
> | Cisco TelePresence System EX90 Videoconferencing Unit                       | 22638.48    |
> | HON 5400 Series Task Chairs for Big and Tall                                | 21870.57    |
> | GBC DocuBind TL300 Electric Binding System                                  | 19823.47    |
>```
  > B. Sales by Category & Sub-Category
```python
category_sales = df.groupby(['Category')['Sales ($)'].sum().sort_values(ascending=False).head(5)
category_sales
```
>```console
> | Category        | Sales ($)   |
> |-----------------|------------:|
> | Technology      | 827455.86   |
> | Furniture       | 728658.50   |
> | Office Supplies | 705422.19   |
>```
> C. Regional Sales Analysis
```python
region_sales = df.groupby(['Region'])['Sales ($)'].sum().sort_values(ascending=False)
region_sales.head()
```
>```console
> | Region  | Sales ($)   |
> |---------|------------:|
> | West    | 710219.60   |
> | East    | 669518.79   |
> | Central | 492646.78   |
> | South   | 389151.38   |
>```
> D. Monthly Sales Trends
```python
monthly_sales = df.groupby('Month')['Sales ($)'].sum().head(12)
monthly_sales
```

>```console
> | Month | Sales ($)   |
> |-------|------------:|
> | 1     | 94291.66    |
> | 2     | 59371.12    |
> | 3     | 197573.57   |
> | 4     | 136283.01   |
> | 5     | 154086.73   |
> | 6     | 145837.47   |
> | 7     | 145535.66   |
> | 8     | 157315.84   |
> | 9     | 300103.36   |
> | 10    | 199496.28   |
> | 11    | 350161.68   |
> | 12    | 321480.17   |
>```
