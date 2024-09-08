import re
import matplotlib.pyplot as plt
import pandas
import statsmodels.api as sm
from collections import Counter
from itertools import combinations


def load_dataset():
    # Load dataset for analysis
    dataset = pandas.read_csv('analysis_data.csv')
    return dataset


def normalize_name(name):
    if isinstance(name, str):
        # Convert to lowercase, remove spaces, punctuations, symbols
        name = name.lower()
        name = re.sub(r'\s+', '', name).strip()
        name = re.sub(r'[^\w\s]', '', name)
    return name


def visualize_smish_over_time_by_day(data):
    # Convert date to datetime format
    data['DateMsgSent'] = pandas.to_datetime(data['DateMsgSent'], format='%d/%m/%Y')

    # Group data by date
    data_grouped = data.groupby(data['DateMsgSent'].dt.date).size()

    # Specify date ranges in the dataset
    # Full dataset
    # date_range = pandas.date_range(start=data['DateMsgSent'].min(), end=data['DateMsgSent'].max(), freq='D')
    # Zoomed dataset
    date_range = pandas.date_range(start='01/01/2022', end=data['DateMsgSent'].max(), freq='D')

    # Include all dates in the range, missing dates are filled with 0
    data_grouped = data_grouped.reindex(date_range, fill_value=0)

    # Plot
    plt.figure(figsize=(10, 6))
    data_grouped.plot(kind='line')
    plt.xlabel('Date')
    plt.ylabel('Number of Messages sent')
    plt.tight_layout()
    plt.show()


def visualize_brand_over_time(data):
    # Convert date to datetime format and handle missing values
    data['DateMsgSent'] = pandas.to_datetime(data['DateMsgSent'], format='%d/%m/%Y', errors='coerce')

    # Normalize brand names
    data['Brand'] = data['Brand'].apply(normalize_name)

    # Filter data - entries from 01/01/2022; exclude 'whatsapp'
    data = data[(data['DateMsgSent'] >= pandas.to_datetime('01/01/2022')) & (data['Brand'] != 'whatsapp')]

    # Group data by month and brand
    data_grouped = data.groupby([data['DateMsgSent'].dt.to_period('M'), 'Brand']).size().unstack(fill_value=0)

    # Focus on top 10 brands
    top_brands = data_grouped.sum().sort_values(ascending=False).head(10).index
    data_grouped = data_grouped[top_brands]

    # Plot
    plt.figure(figsize=(10, 6))
    data_grouped.plot(kind='line', marker='o')
    plt.xlabel('Month')
    plt.ylabel('Number of Messages Sent')
    plt.legend(title='Brand', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='x-small', ncol=1)
    plt.tight_layout()
    plt.show()


def sum_brand_over_time(data):
    # Convert date to datetime format and handle missing values
    data['DateMsgSent'] = pandas.to_datetime(data['DateMsgSent'], format='%d/%m/%Y', errors='coerce')

    # Normalize brand names
    data['Brand'] = data['Brand'].apply(normalize_name)

    # Filter data - entries from 01/01/2022
    data = data[data['DateMsgSent'] >= pandas.to_datetime('01/01/2022')]

    # Group data by year and brand, count impersonations
    data_grouped = data.groupby([data['DateMsgSent'].dt.to_period('Y'), 'Brand']).size()

    # Reset index; turn into dataframe
    summary_df = data_grouped.reset_index()
    summary_df.columns = ['Year', 'Brand', 'Impersonations']

    for index, row in summary_df.iterrows():
        print(f"{row['Year']}, {row['Brand']}, {row['Impersonations']}")


def visualize_sender_type_over_time(data):
    # Convert date to datetime format and handle missing values
    data['DateMsgSent'] = pandas.to_datetime(data['DateMsgSent'], format='%d/%m/%Y', errors='coerce')

    # Filter data - entries from 01/01/2022
    data = data[data['DateMsgSent'] >= pandas.to_datetime('01/01/2022')]

    # Group data by month and sender type
    data_grouped = data.groupby([data['DateMsgSent'].dt.to_period('M'), 'SenderType']).size().unstack(fill_value=0)

    # Plot
    plt.figure(figsize=(10, 6))
    data_grouped.plot(kind='bar', stacked=True, colormap='viridis')
    plt.xlabel('Month')
    plt.ylabel('Number of Messages Sent')
    plt.xticks(rotation=90)
    plt.legend(title='Sender Type')
    plt.tight_layout()
    plt.show()


def visualize_url_type_over_time(data):
    # Convert date to datetime format and handle missing values
    data['DateMsgSent'] = pandas.to_datetime(data['DateMsgSent'], format='%d/%m/%Y', errors='coerce')

    # Filter data - entries from 01/01/2022
    data = data[data['DateMsgSent'] >= pandas.to_datetime('01/01/2022')]

    # Group by month and URL category
    grouped = data.groupby([data['DateMsgSent'].dt.to_period('M'), 'URLCategory']).size().unstack(fill_value=0)

    # Show all categories
    all_categories = ['Deceptive top-level domain', 'Deceptive second-level domain', 'Deceptive subdomain',
                      'URL shortener', 'IP address', 'Random domain']
    # Reindex
    grouped = grouped.reindex(columns=all_categories, fill_value=0)

    # Plot
    plt.figure(figsize=(10, 6))
    grouped.plot(kind='bar', stacked=True, colormap='viridis')
    plt.xlabel('Month')
    plt.ylabel('Number of Messages Sent')
    plt.xticks(rotation=90)
    plt.legend(title='URL Category')
    plt.tight_layout()
    plt.show()


def sum_lure_principle(data):
    # Counters for principles
    distraction = 0
    social_compliance = 0
    herd = 0
    kindness = 0
    need_and_greed = 0
    time = 0

    # Sum up lure principles
    for principles in data['LurePrinciple'].dropna():
        if 'Distraction' in principles:
            distraction += 1
        if 'Social Compliance' in principles:
            social_compliance += 1
        if 'Herd' in principles:
            herd += 1
        if 'Kindness' in principles:
            kindness += 1
        if 'Need and Greed' in principles:
            need_and_greed += 1
        if 'Time' in principles:
            time += 1
    print(f"Distraction: {distraction}\nSocial Compliance: {social_compliance}\nHerd: {herd}\nKindness: {kindness}\n"
          f"Need and Greed: {need_and_greed}\nTime: {time}")


def get_lure_principle_stats(data):
    # Count how many messages
    count_msg = data['MessageText'].notna().sum()
    # Count how many have a lure principle
    count_lure = data['LurePrinciple'].notna().sum()
    # % how much lure principles are used
    percentage_lure = (count_lure / count_msg) * 100

    # Count combinations of principles
    usage_count_lure = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
    for principles in data['LurePrinciple'].dropna():
        # Split lure principles and count
        principles_list = principles.split(', ')
        count = len(principles_list)
        usage_count_lure[count] += 1

    # Counter for principle pairs
    pairs = Counter()
    for principles in data['LurePrinciple'].dropna():
        principles_list = principles.split(', ')
        if len(principles_list) > 1:
            # Generate combinations of two principles from the list of principles
            for combination in combinations(sorted(principles_list), 2):
                pairs[combination] += 1
    # 10 most common pairs
    most_common_pairs = pairs.most_common(10)

    print(f"Number of messages: {count_msg}")
    print(f"Number of messages w/ lure principles: {count_lure}")
    print(f"Percentage of messages w/ lure principles: {percentage_lure}")
    print(f"Number of combinations: {usage_count_lure}")
    print(f"Most common pairs: {most_common_pairs}")

    # Counter for all combinations of principles
    all_combinations = Counter()
    for principles in data['LurePrinciple'].dropna():
        principles_list = principles.split(', ')
        if len(principles_list) > 1:
            # Generate all combinations for 2 to size of principles_list
            for i in range(2, len(principles_list) + 1):
                for combination in combinations(sorted(principles_list), i):
                    all_combinations[combination] += 1
    # 10 most common combinations
    most_common_combinations = all_combinations.most_common(10)
    print(f"Most common combinations: {most_common_combinations}")


def sum_lure_principle_over_time(data):
    # Convert date to datetime format and handle missing values
    data['DateMsgSent'] = pandas.to_datetime(data['DateMsgSent'], format='%d/%m/%Y', errors='coerce')

    # Filter data - entries from 01/01/2022
    filtered_data = data[data['DateMsgSent'] >= pandas.to_datetime('01/01/2022')]

    # Counter for each lure principle per year
    yearly_principle_counts = {}
    for index, row in filtered_data.iterrows():
        if pandas.notna(row['LurePrinciple']):
            # Prepare year and principles
            year = row['DateMsgSent'].year
            principles = row['LurePrinciple'].split(', ')

            # Create year count if it does not exist
            if year not in yearly_principle_counts:
                yearly_principle_counts[year] = {}
            for principle in principles:
                if principle in yearly_principle_counts[year]:
                    yearly_principle_counts[year][principle] += 1
                else:
                    yearly_principle_counts[year][principle] = 1

    print(yearly_principle_counts)
    return yearly_principle_counts


def visualize_lure_principle_over_time(data):
    # Lure principle counts per year
    yearly_principle_counts = sum_lure_principle_over_time(data)

    # Convert the dictionary to dataframe
    data = []
    for year, principles in yearly_principle_counts.items():
        for principle, count in principles.items():
            data.append({'Year': year, 'Principle': principle, 'Count': count})

    dataframe = pandas.DataFrame(data)

    # Pivot dataframe
    pivot_dataframe = dataframe.pivot(index='Year', columns='Principle', values='Count').fillna(0)

    # Plot
    plt.figure(figsize=(10, 6))
    pivot_dataframe.plot(kind='bar', stacked=True, colormap='viridis')
    plt.xlabel('Year')
    plt.ylabel('Number of Messages with Lure Principles')
    plt.xticks(rotation=0)
    plt.legend(title='Lure Principle')
    plt.tight_layout()
    plt.show()


def calc_for_domain_lifetime_a(data):
    # List of IDs to exclude
    ids = [1792, 1818, 1749, 1748, 1747, 1657, 1647, 1627, 1613, 1581, 1579, 1577, 1576, 1575, 1559, 1539, 2517, 2518]

    # Filter specific IDs and URL shorteners
    data = data[~data['MessageID'].isin(ids)]
    data = data[data['URLCategory'] != 'URL shortener']

    # Convert domain lifetime to numeric or NaN
    data['DomainActiveFor'] = pandas.to_numeric(data['DomainActiveFor'], errors='coerce')

    # Calculate - count, median, average, minimum, and maximum
    count = data['DomainActiveFor'].notnull().sum()
    med_val = data['DomainActiveFor'].median()
    avg_val = data['DomainActiveFor'].mean()
    min_val = data['DomainActiveFor'].min()
    max_val = data['DomainActiveFor'].max()

    print(f'Count: {count}\nMedian: {med_val}\nAverage: {avg_val}\nMinimum: {min_val}\nMaximum: {max_val}')


def get_min_date_msg_origin(data):
    # Convert date to datetime format and handle missing values
    data['DateMsgSent'] = pandas.to_datetime(data['DateMsgSent'], format='%d/%m/%Y', errors='coerce')

    # Earliest date
    earliest_date = data['DateMsgSent'].min()
    print(f"Earliest date: {earliest_date}")


def calc_for_domain_lifetime_b(data):
    # Convert date to datetime format and handle missing values
    data['DomainCreationDate'] = pandas.to_datetime(data['DomainCreationDate'], format='%d/%m/%Y', errors='coerce')

    # List of IDs to exclude
    ids = [1792, 1818, 1749, 1748, 1747, 1657, 1647, 1627, 1613, 1581, 1579, 1577, 1576, 1575, 1559, 1539, 2517, 2518,
           1101, 1623, 1625]

    # Filter specific IDs and URL shorteners
    data = data[~data['MessageID'].isin(ids)]
    data = data[data['URLCategory'] != 'URL shortener']

    # Filter data before 1/11/2017
    data = data[data['DomainCreationDate'] >= '2017-11-01']

    # Convert domain lifetime to numeric or NaN
    data['DomainActiveFor'] = pandas.to_numeric(data['DomainActiveFor'], errors='coerce')

    # Calculate - count, median, average, minimum, and maximum
    count = data['DomainActiveFor'].notnull().sum()
    med_val = data['DomainActiveFor'].median()
    avg_val = data['DomainActiveFor'].mean()
    min_val = data['DomainActiveFor'].min()
    max_val = data['DomainActiveFor'].max()

    print(f'Count: {count}\nMedian: {med_val}\nAverage: {avg_val}\nMinimum: {min_val}\nMaximum: {max_val}')
    # Provide dataset B for further analysis
    return data


def calc_domain_lifetime_for_url_cat(data):
    # dataset B
    dataset = calc_for_domain_lifetime_b(data)

    # Group data by URL category
    grouped_data = dataset.groupby('URLCategory')

    # Calculate for each category - count, median, average, minimum, and maximum
    for category, group in grouped_data:
        count = group['DomainActiveFor'].notnull().sum()
        med_val = group['DomainActiveFor'].median()
        avg_val = group['DomainActiveFor'].mean()
        min_val = group['DomainActiveFor'].min()
        max_val = group['DomainActiveFor'].max()

        print(f"Category: {category}")
        print(f"Count: {count}\nMedian: {med_val}\nAverage: {avg_val}\nMinimum: {min_val}\nMaximum: {max_val}\n")


def calc_domain_lifetime_for_brand(data):
    # dataset B
    dataset = calc_for_domain_lifetime_b(data)

    # Normalize brand name to group them as needed
    dataset['Brand'] = dataset['Brand'].apply(normalize_name)

    # Get the ACTUAL top 10 most impersonated brands
    top_brands = dataset['Brand'].value_counts().nlargest(12).index.tolist()

    # Filter dataset to include only the most impersonated brands
    filter_dataset = dataset[dataset['Brand'].isin(top_brands)]

    # Group data by brand
    grouped_data = filter_dataset.groupby('Brand')

    # Calculate for each brand - count, median, average, minimum, and maximum
    for brand, group in grouped_data:
        count = group['DomainActiveFor'].notnull().sum()
        med_val = group['DomainActiveFor'].median()
        avg_val = group['DomainActiveFor'].mean()
        min_val = group['DomainActiveFor'].min()
        max_val = group['DomainActiveFor'].max()

        print(f"Brand: {brand}")
        print(f"Count: {count}\nMedian: {med_val}\nAverage: {avg_val}\nMinimum: {min_val}\nMaximum: {max_val}\n")


def calc_domain_lifetime_for_dom_reg(data):
    # dataset B
    dataset = calc_for_domain_lifetime_b(data)

    # Normalize domain registrar names to group them as needed
    dataset['DomainRegistrar'] = dataset['DomainRegistrar'].apply(normalize_name)

    # Get the top 10 most common registrars
    top_registrars = dataset['DomainRegistrar'].value_counts().nlargest(10).index.tolist()

    # Filter dataset to include only the most common registrars
    filter_dataset = dataset[dataset['DomainRegistrar'].isin(top_registrars)]

    # Group data by registrar
    grouped_data = filter_dataset.groupby('DomainRegistrar')

    # Calculate for each registrar - count, median, average, minimum, and maximum
    for registrar, group in grouped_data:
        count = group['DomainActiveFor'].notnull().sum()
        med_val = group['DomainActiveFor'].median()
        avg_val = group['DomainActiveFor'].mean()
        min_val = group['DomainActiveFor'].min()
        max_val = group['DomainActiveFor'].max()

        print(f"Registrar: {registrar}")
        print(f"Count: {count}\nMedian: {med_val}\nAverage: {avg_val}\nMinimum: {min_val}\nMaximum: {max_val}\n")


def regr_analysis_brand(data):
    # dataset B
    dataset = calc_for_domain_lifetime_b(data)

    # Select only relevant columns and drop missing values
    dataset = dataset[['DomainActiveFor', 'Brand']].dropna()

    # Normalize the domain registrar names to group them as needed
    dataset['Brand'] = dataset['Brand'].apply(normalize_name)

    # Focus on top 5 brands
    selected_brands = ['irs', 'usps', 'tmobile', 'netflix', 'homedepot']
    dataset = dataset[dataset['Brand'].isin(selected_brands)]

    # Encode categorical data
    dataset = pandas.get_dummies(dataset, columns=['Brand'], drop_first=True)

    # Define the response variable and predictors
    X = sm.add_constant(dataset.drop('DomainActiveFor', axis=1))
    y = pandas.to_numeric(dataset['DomainActiveFor'], errors='coerce')

    # Fit linear regression model
    model = sm.OLS(y, X.astype(float)).fit()

    print(model.summary())


def regr_analysis_registrar(data):
    # dataset B
    dataset = calc_for_domain_lifetime_b(data)

    # Select only relevant columns and drop missing values
    dataset = dataset[['DomainActiveFor', 'DomainRegistrar']].dropna()

    # Normalize the domain registrar names to group them as needed
    dataset['DomainRegistrar'] = dataset['DomainRegistrar'].apply(normalize_name)

    # Focus on top 10 registrars
    selected_registrars = ['alibabacomsingaporeecommerceprivatelimited', 'dynadotllc', 'gnamecompteltd',
                           'godaddycomllc', 'hostingconceptsbvdbaregistrareu', 'hostingeroperationsuab',
                           'internetdomainservicebscorp', 'namecheapinc', 'namesilollc',
                           'nicenicinternationalgroupcolimited']
    dataset = dataset[dataset['DomainRegistrar'].isin(selected_registrars)]

    # Encode categorical data
    dataset = pandas.get_dummies(dataset, columns=['DomainRegistrar'], drop_first=True)

    # Define the response variable and predictors
    X = sm.add_constant(dataset.drop('DomainActiveFor', axis=1))
    y = pandas.to_numeric(dataset['DomainActiveFor'], errors='coerce')

    # Fit linear regression model
    model = sm.OLS(y, X.astype(float)).fit()

    print(model.summary())


def regr_analysis_url(data):
    # dataset B
    dataset = calc_for_domain_lifetime_b(data)

    # Select only relevant columns and drop missing values
    dataset = dataset[['DomainActiveFor', 'URLCategory']].dropna()

    # Encode categorical data
    dataset = pandas.get_dummies(dataset, columns=['URLCategory'], drop_first=True)

    # Define the response variable and predictors
    X = sm.add_constant(dataset.drop('DomainActiveFor', axis=1))
    y = pandas.to_numeric(dataset['DomainActiveFor'], errors='coerce')

    # Fit linear regression model
    model = sm.OLS(y, X.astype(float)).fit()

    print(model.summary())


def virus_total_results(data):
    columns = ['Detected', 'Malicious', 'Malware', 'Phishing', 'Suspicious']

    # Get stats of each column
    for column in columns:
        # Filter out missing data
        filtered_data = data[column].dropna()
        # Total number of entries
        filled = filtered_data.count()
        # Not flagged entries
        not_flagged = (filtered_data == 0).sum()
        # Flagged entries
        flagged = (filtered_data != 0).sum()

        print(f'{column} rows: {filled}')
        print(f'{column} not flagged: {not_flagged}')
        print(f'{column} flagged: {flagged}')


if __name__ == '__main__':
    # Load dataset
    dataset_df = load_dataset()

    # Analysis
    # visualize_smish_over_time_by_day(dataset_df)
    # visualize_brand_over_time(dataset_df)
    sum_brand_over_time(dataset_df)
    visualize_sender_type_over_time(dataset_df)
    visualize_url_type_over_time(dataset_df)
    sum_lure_principle(dataset_df)
    get_lure_principle_stats(dataset_df)
    sum_lure_principle_over_time(dataset_df)
    visualize_lure_principle_over_time(dataset_df)
    calc_for_domain_lifetime_a(dataset_df)
    get_min_date_msg_origin(dataset_df)
    calc_for_domain_lifetime_b(dataset_df)
    calc_domain_lifetime_for_url_cat(dataset_df)
    calc_domain_lifetime_for_brand(dataset_df)
    calc_domain_lifetime_for_dom_reg(dataset_df)
    regr_analysis_brand(dataset_df)
    regr_analysis_registrar(dataset_df)
    regr_analysis_url(dataset_df)
    virus_total_results(dataset_df)
