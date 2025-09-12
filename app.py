import streamlit as st
import pandas as pd
import numpy as np
import uuid
import base64
import plotly.express as px

# Function to convert Excel to Parquet
def convert_excel_to_parquet(excel_path, parquet_path):
    try:
        # Read Excel file
        df = pd.read_excel(excel_path)
        df.columns = df.columns.str.strip()
        # Convert 'Date Range' column to string if it exists
        if 'Date Range' in df.columns:
            df['Date Range'] = df['Date Range'].astype(str)
        # Save to Parquet, overwrite if exists
        df.to_parquet(parquet_path, index=False)
        return True
    except FileNotFoundError:
        st.error(f"Excel file '{excel_path}' not found. Please ensure the file is in the same directory as `app.py`.")
        return False
    except Exception as e:
        st.error(f"Error converting Excel to Parquet: {str(e)}")
        return False

# Function to encode favicon image to base64
def get_base64_image(image_path):
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode()
    except FileNotFoundError:
        st.warning("Favicon file 'rideblitz_logo.jpeg' not found. Please ensure the file is in the same directory as `app.py`.")
        return ""

# Initialize session state for login and page navigation
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
    st.session_state['login_id'] = str(uuid.uuid4())
if 'page' not in st.session_state:
    st.session_state['page'] = 'login'

# Function to display logo
def display_logo():
    st.markdown("""
        <style>
        .logo-container {
            position: absolute;
            top: 10px;
            left: 10px;
            z-index: 1000;
        }
        </style>
    """, unsafe_allow_html=True)
    try:
        st.markdown('<div class="logo-container">', unsafe_allow_html=True)
        st.image('blitz logo.png', width=150)
        st.markdown('</div>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("Logo file 'blitz logo.png' not found. Please ensure the file is in the same directory as `app.py`.")

# Function to handle logout
def logout():
    st.session_state['logged_in'] = False
    st.session_state['page'] = 'login'
    st.session_state['login_id'] = str(uuid.uuid4())  # Reset login ID
    st.rerun()

# Login Page
def login_page():
    # Set page config with custom title and favicon
    favicon_base64 = get_base64_image('rideblitz_logo.jpeg')
    st.set_page_config(
        page_title="Blitz 3PL Margin Dashboard",
        page_icon=f"data:image/jpeg;base64,{favicon_base64}" if favicon_base64 else None,
        layout="wide")
    display_logo()
    st.title("Login to Blitz 3PL Margin Dashboard")
    st.markdown("Please enter your credentials to access the dashboard.")

    # Input fields for username and password
    username = st.text_input("Username", key=f"username_{st.session_state['login_id']}")
    password = st.text_input("Password", type="password", key=f"password_{st.session_state['login_id']}")

    # Login button
    if st.button("Login"):
        if username == "theo" and password == "rideblitz2019":
            st.session_state['logged_in'] = True
            st.session_state['page'] = 'home'
            st.success("Login successful! Redirecting to feature selection...")
            st.rerun()
        else:
            st.error("Invalid username or password. Please try again.")

# Home Page with Feature Selection
def home_page():
    # Set page config with custom title and favicon
    favicon_base64 = get_base64_image('rideblitz_logo.jpeg')
    st.set_page_config(
        page_title="Blitz 3PL Margin Dashboard",
        page_icon=f"data:image/jpeg;base64,{favicon_base64}" if favicon_base64 else None,
        layout="wide"
    )
    display_logo()
    st.title("Blitz 3PL Margin Dashboard")
    st.markdown("Select a feature to explore the dashboard functionalities.")

    # Logout button
    st.button("Logout", on_click=logout, key="logout_home")

    # Custom CSS for bordered feature cards
    st.markdown("""
        <style>
        .feature-card {
            border: 2px solid #4CAF50;
            border-radius: 10px;
            padding: 15px;
            margin: 10px 0;
            background-color: #f9f9f9;
        }
        .feature-card h3 {
            margin-top: 0;
            color: #333;
        }
        .feature-card p {
            color: #666;
        }
        </style>
    """, unsafe_allow_html=True)

# Create a grid layout for feature cards (3 rows of 3 columns)
    row1_col1, row1_col2, row1_col3 = st.columns(3)
    row2_col1, row2_col2, row2_col3 = st.columns(3)
    row3_col1, row3_col2, row3_col3 = st.columns(3)

    with row1_col1:
        st.markdown("""
            <div class="feature-card">
                <h3>🔍Main Margin Data</h3>
                <p>Easily identify which projects are driving profits and which projects require attention.</p>
            </div>
        """, unsafe_allow_html=True)
        if st.button("Access Main Margin Data", key="main_margin"):
            st.session_state['page'] = 'dashboard'
            st.rerun()

    with row1_col2:
        st.markdown("""
            <div class="feature-card">
                <h3>🧐Detailed Margin Data</h3>
                <p>Break down every revenue stream and cost component to understand the full financial picture.</p>
            </div>
        """, unsafe_allow_html=True)
        if st.button("Access Detailed Margin Data", key="detailed_margin"):
            st.session_state['page'] = 'detailed_dashboard'
            st.rerun()

    with row1_col3:
        st.markdown("""
            <div class="feature-card">
                <h3>🗓️Week on Week Performance</h3>
                <p>Summarize all projects performance trends like volume and financial condition with a week-on-week view.</p>
            </div>
        """, unsafe_allow_html=True)
        if st.button("Access WoW Performance", key="wow_performance"):
            st.session_state['page'] = 'wow_performance'
            st.rerun()

    with row2_col1:
        st.markdown("""
            <div class="feature-card">
                <h3>🌕Monthly Performance</h3>
                <p>Summarize all projects performance trends like volume and financial condition with a monthly view.</p>
            </div>
        """, unsafe_allow_html=True)
        if st.button("Access Monthly Performance", key="monthly_performance"):
            st.session_state['page'] = 'monthly_performance'
            st.rerun()

    with row2_col2:
        st.markdown("""
            <div class="feature-card">
                <h3>🏆Top Clients</h3>
                <p>See a weekly and monthly ranking of top clients by revenue, volume, and margin to inform strategic decisions.</p>
            </div>
        """, unsafe_allow_html=True)
        st.button("Under Development", disabled=True, key="top_clients")

    with row2_col3:
        st.markdown("""
            <div class="feature-card">
                <h3>🎯SLA Calculation</h3>
                <p>Ensure operational excellence by tracking SLA compliance for every project. Use this data to identify issues and drive quality improvements.</p>
            </div>
        """, unsafe_allow_html=True)
        if st.button("Access SLA Calculation", key="sla_calculation"):
            st.session_state['page'] = 'sla_calculation'
            st.rerun()

    with row3_col1:
        st.markdown("""
            <div class="feature-card">
                <h3>🌿Carbon Reduced</h3>
                <p>Accurately measure the cumulative distance covered by electric vehicles (EVs) in overall deliveries</p>
            </div>
        """, unsafe_allow_html=True)
        st.button("Under Development", disabled=True, key="feature_7")

    with row3_col2:
        st.markdown("""
            <div class="feature-card">
                <h3>🔜Feature 8</h3>
                <p>Under Development</p>
            </div>
        """, unsafe_allow_html=True)
        st.button("Under Development", disabled=True, key="feature_8")

    with row3_col3:
        st.markdown("""
            <div class="feature-card">
                <h3>🔜Feature 9</h3>
                <p>Under Development</p>
            </div>
        """, unsafe_allow_html=True)
        st.button("Under Development", disabled=True, key="feature_9")

# Main Margin Page
def main_margin_page():
    # Set page config with custom title and favicon
    favicon_base64 = get_base64_image('rideblitz_logo.jpeg')
    st.set_page_config(
        page_title="Blitz 3PL Margin Dashboard",
        page_icon=f"data:image/jpeg;base64,{favicon_base64}" if favicon_base64 else None,
        layout="wide")
    display_logo()
    st.title('🎯Main Margin Data')
    st.markdown('Easily identify which projects are driving profits and which projects require attention.')
    
    # Add back and logout buttons
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Back to Feature Selection"):
            st.session_state['page'] = 'home'
            st.rerun()
    with col2:
        st.button("Logout", on_click=logout, key="logout_dashboard")

    # Load data from Parquet if exists, otherwise convert from Excel
    excel_path = 'Ops Data Collection.xlsx'
    parquet_path = 'Ops Data Collection.parquet'
    try:
        # Convert Excel to Parquet (updates Parquet file automatically)
        convert_excel_to_parquet(excel_path, parquet_path)
        # Load from Parquet
        df = pd.read_parquet(parquet_path)
        df.columns = df.columns.str.strip()
        st.info("Data loaded successfully from Parquet!")
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.stop()

    # --- Pre-processing Data ---
    currency_cols = [
        'TOTAL DELIVERY REVENUE', 'Total Revenue', 'Selling Price (Regular Rate)',
        'Additional Charge (KM, KG, Etc)', 'Return/Delivery Rate', 'Lalamove Bills (Invoicing to Client)',
        'EV Reduction (3PL & KSJ)', 'EV Manpower', 'EV Revenue + Battery (Rental Client)',
        'Claim/COD/Own Risk', 'Hub, COD Fee (SBY) & Service Korlap', 'Other Revenue',
        'Attribute Fee', 'Rider Cost', 'Manpower Cost', 'OEM Cost', 'Mid-Mile/ Linehaul Cost',
        'Add. 3PL Cost', 'DM Program', 'Claim Damaged/Loss', 'Outstanding COD',
        'Claim Ownrisk', 'Attribute Cost', 'HUB Cost', 'Other Cost', 'Total Cost',
        'Delivery Volume'
    ]

    for col in currency_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            mask = df[col].isna()
            if mask.any():
                df.loc[mask, col] = df.loc[mask, col].astype(str).str.replace('Rp', '', regex=False).str.replace(' ', '', regex=False).str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # --- Filters ---
    st.sidebar.header('Filter Data')
    all_years = sorted(df['Year'].unique())
    selected_year = st.sidebar.selectbox('Year', all_years, index=0)  # No (All) option, default to first year

    df_filtered_year = df[df['Year'] == selected_year]

    all_teams = ['(All)'] + sorted(df_filtered_year['Blitz Team'].unique())
    selected_team = st.sidebar.selectbox('Blitz Team', all_teams)

    df_filtered_team = df_filtered_year.copy()
    if selected_team != '(All)':
        df_filtered_team = df_filtered_team[df_filtered_team['Blitz Team'] == selected_team]

    all_locations = ['(All)'] + sorted(df_filtered_team['Client Location'].unique())
    selected_location = st.sidebar.selectbox('Client Location', all_locations)

    df_filtered_location = df_filtered_team.copy()
    if selected_location != '(All)':
        df_filtered_location = df_filtered_location[df_filtered_location['Client Location'] == selected_location]

    all_clients = sorted(df_filtered_location['Client Name'].unique())
    selected_client = st.sidebar.selectbox('Client Name', all_clients)

    # Apply filters only if a Client Name is selected
    if not selected_client:
        st.warning('Please select at least one Client Name to display the dashboard.')
        st.stop()

    df_filtered_client = df_filtered_location[df_filtered_location['Client Name'] == selected_client]
    
    all_projects = ['(All)'] + sorted(df_filtered_client['Project'].unique())
    selected_project = st.sidebar.selectbox('Project', all_projects)

    filtered_df = df_filtered_client.copy()
    if selected_project != '(All)':
        filtered_df = filtered_df[filtered_df['Project'] == selected_project]

    if filtered_df.empty:
        st.warning('Tidak ada data yang sesuai dengan filter yang dipilih.')
        st.stop()

    # --- Create Main Pivot Table ---
    st.subheader('Clients Week-by-Week')

    # Group data by Client and Week
    df_grouped = filtered_df.groupby(['Client Name', 'Week (by Year)']).agg(
        Delivery_Volume=('Delivery Volume', 'sum'),
        Total_Delivery_Revenue=('TOTAL DELIVERY REVENUE', 'sum'),
        Total_Revenue=('Total Revenue', 'sum'),
        Total_Cost=('Total Cost', 'sum'),
        Rider_Cost=('Rider Cost', 'sum')
    ).reset_index()

    # Calculate weekly metrics
    df_grouped['Profit_Value'] = df_grouped['Total_Revenue'] - df_grouped['Total_Cost']
    df_grouped['Profit_Margins'] = (df_grouped['Profit_Value'] / df_grouped['Total_Revenue']).replace([np.inf, -np.inf], np.nan).fillna(0) * 100
    df_grouped['SRPO'] = (df_grouped['Total_Delivery_Revenue'] / df_grouped['Delivery_Volume']).replace([np.inf, -np.inf], np.nan).fillna(0)
    df_grouped['RCPO'] = (df_grouped['Rider_Cost'] / df_grouped['Delivery_Volume']).replace([np.inf, -np.inf], np.nan).fillna(0)
    df_grouped['TCOP'] = (df_grouped['Total_Cost'] / df_grouped['Delivery_Volume']).replace([np.inf, -np.inf], np.nan).fillna(0)

    # Calculate totals per Client
    df_total_client = filtered_df.groupby('Client Name').agg(
        Delivery_Volume=('Delivery Volume', 'sum'),
        Total_Delivery_Revenue=('TOTAL DELIVERY REVENUE', 'sum'),
        Total_Revenue=('Total Revenue', 'sum'),
        Total_Cost=('Total Cost', 'sum'),
        Rider_Cost=('Rider Cost', 'sum')
    ).reset_index()

    # Calculate total metrics per Client
    df_total_client['Profit_Value'] = df_total_client['Total_Revenue'] - df_total_client['Total_Cost']
    df_total_client['Profit_Margins'] = (df_total_client['Profit_Value'] / df_total_client['Total_Revenue']).replace([np.inf, -np.inf], np.nan).fillna(0) * 100
    df_total_client['SRPO'] = (df_total_client['Total_Delivery_Revenue'] / df_total_client['Delivery_Volume']).replace([np.inf, -np.inf], np.nan).fillna(0)
    df_total_client['RCPO'] = (df_total_client['Rider_Cost'] / df_total_client['Delivery_Volume']).replace([np.inf, -np.inf], np.nan).fillna(0)
    df_total_client['TCOP'] = (df_total_client['Total_Cost'] / df_total_client['Delivery_Volume']).replace([np.inf, -np.inf], np.nan).fillna(0)

    # Combine weekly and total data
    final_df = pd.DataFrame()
    for client in df_total_client['Client Name'].unique():
        client_total_row = df_total_client[df_total_client['Client Name'] == client].copy()
        client_total_row.rename(columns={'Client Name': 'Row Labels'}, inplace=True)
        client_total_row['Week (by Year)'] = 'Total'
        
        client_weekly_data = df_grouped[df_grouped['Client Name'] == client].copy()
        client_weekly_data.rename(columns={'Client Name': 'Row Labels'}, inplace=True)
        
        # Add comparison (diff) for the last weeks
        client_weekly_data.sort_values('Week (by Year)', inplace=True)

        if len(client_weekly_data) >= 3:
            last_three_weeks = client_weekly_data.tail(3).copy()
            third_last_week = last_three_weeks.iloc[0]
            second_last_week = last_three_weeks.iloc[1]
            last_week_orig = last_three_weeks.iloc[2]
            diff_data_earlier = {}
            for col in ['Delivery_Volume', 'Total_Delivery_Revenue', 'Total_Revenue', 'Total_Cost',
                        'Profit_Value', 'Profit_Margins', 'SRPO', 'RCPO', 'TCOP']:
                val_second_last = second_last_week[col]
                val_third_last = third_last_week[col]
                if val_third_last != 0 and not pd.isna(val_third_last):
                    diff_data_earlier[col] = ((val_second_last - val_third_last) / abs(val_third_last)) * 100
                else:
                    diff_data_earlier[col] = np.nan
            diff_row_earlier = pd.Series(diff_data_earlier)
            week_value_earlier = str(second_last_week['Week (by Year)']).strip()
            if week_value_earlier and week_value_earlier.replace('.', '').replace('-', '').isdigit():
                diff_row_earlier['Row Labels'] = f'Diff W{int(float(week_value_earlier))}%'
            else:
                diff_row_earlier['Row Labels'] = 'Diff WNA%'
            diff_row_earlier['Week (by Year)'] = ''
            for col in client_weekly_data.columns:
                if col not in diff_row_earlier.index:
                    diff_row_earlier[col] = np.nan
            diff_row_earlier = diff_row_earlier[client_weekly_data.columns]
            insert_idx = client_weekly_data.index[client_weekly_data['Week (by Year)'] == second_last_week['Week (by Year)']].tolist()[-1] + 1
            client_weekly_data = pd.concat([client_weekly_data.iloc[:insert_idx], pd.DataFrame([diff_row_earlier]), client_weekly_data.iloc[insert_idx:]], ignore_index=True)

        if len(client_weekly_data) >= 2:
            last_week = last_week_orig
            second_last_week = client_weekly_data[client_weekly_data['Week (by Year)'] == last_week_orig['Week (by Year)'] - 1].iloc[0] if last_week_orig['Week (by Year)'] > 1 else client_weekly_data.iloc[-2]
            diff_data = {}
            for col in ['Delivery_Volume', 'Total_Delivery_Revenue', 'Total_Revenue', 'Total_Cost',
                        'Profit_Value', 'Profit_Margins', 'SRPO', 'RCPO', 'TCOP']:
                val_last = last_week[col]
                val_second_last = second_last_week[col]
                if val_second_last != 0 and not pd.isna(val_second_last):
                    diff_data[col] = ((val_last - val_second_last) / abs(val_second_last)) * 100
                else:
                    diff_data[col] = np.nan
            diff_row = pd.Series(diff_data)
            week_value = str(last_week['Week (by Year)']).strip()
            if week_value and week_value.replace('.', '').replace('-', '').isdigit():
                diff_row['Row Labels'] = f'Diff W{int(float(week_value))}%'
            else:
                diff_row['Row Labels'] = 'Diff WNA%'
            diff_row['Week (by Year)'] = ''
            for col in client_weekly_data.columns:
                if col not in diff_row.index:
                    diff_row[col] = np.nan
            diff_row = diff_row[client_weekly_data.columns]
            insert_idx_last = client_weekly_data.index[client_weekly_data['Week (by Year)'] == last_week['Week (by Year)']].tolist()[-1] + 1
            client_weekly_data = pd.concat([client_weekly_data.iloc[:insert_idx_last], pd.DataFrame([diff_row]), client_weekly_data.iloc[insert_idx_last:]], ignore_index=True)

        combined_data = pd.concat([client_total_row, client_weekly_data], ignore_index=True)
        final_df = pd.concat([final_df, combined_data], ignore_index=True)

    # Select and rename columns for display
    final_df = final_df[[
        'Row Labels', 'Week (by Year)', 'Delivery_Volume', 'Total_Delivery_Revenue',
        'Total_Revenue', 'Total_Cost', 'Profit_Value', 'Profit_Margins',
        'SRPO', 'RCPO', 'TCOP'
    ]]
    final_df.columns = [
        'Row Labels', 'Week (by Year)', 'Sum of Delivery Volume', 'Sum of Total Sellings',
        'Sum of Total Revenue', 'Sum of Total Cost', 'Sum of Profit Value',
        'Sum of Profit Margins %', 'Sum of Selling Revenue per Order (SRPO)',
        'Sum of Rider Cost per Order (RCPO)', 'Sum of Total Cost per Order (TCOP)'
    ]

    # --- Styling DataFrame ---
    def color_rows(row):
        is_total = 'Total' in str(row['Week (by Year)'])
        is_diff = 'Diff W' in str(row['Row Labels'])
        styles = [''] * len(row)

        if is_total:
            styles = ['background-color: #e6ffe6; color: black'] * len(row)
        elif is_diff:
            styles = ['background-color: #fff2e6; color: black'] * len(row)
        
        return styles

    # Format values for display
    display_df = final_df.copy()
    for col in display_df.columns:
        if col not in ['Row Labels', 'Week (by Year)']:
            display_df[col] = display_df.apply(
                lambda row: f"{row[col]:,.2f}%" if pd.notna(row[col]) and ('Diff' in str(row['Row Labels']) or 'Profit Margins' in col) else (
                    f"Rp {row[col]:,.2f}" if pd.notna(row[col]) and ('SRPO' in col or 'RCPO' in col or 'TCOP' in col) else (
                        f"Rp {row[col]:,.0f}" if pd.notna(row[col]) and ('Total' in col or 'Selling' in col or 'Cost' in col or 'Profit' in col) else (
                            f"{row[col]:,.0f}" if pd.notna(row[col]) else ''
                        )
                    )
                ),
                axis=1
            )
            display_df[col] = display_df[col].astype(str).str.replace('nan', '', regex=False)

    # Display styled dataframe
    st.dataframe(
        display_df.style.apply(color_rows, axis=1), 
        hide_index=True
    )

    # --- Generate Chart ---
    st.subheader('Generate Chart')
    chart_columns = [
        'Sum of Delivery Volume', 'Sum of Total Sellings', 'Sum of Total Revenue',
        'Sum of Total Cost', 'Sum of Profit Value', 'Sum of Profit Margins %',
        'Sum of Selling Revenue per Order (SRPO)', 'Sum of Rider Cost per Order (RCPO)',
        'Sum of Total Cost per Order (TCOP)'
    ]
    selected_chart_col = st.selectbox('Select Variable for Chart', chart_columns)
    
    if st.button('Generate Chart'):
        # Filter data for the selected client and exclude total/diff rows
        chart_df = final_df[final_df['Row Labels'] == selected_client]
        chart_df = chart_df[~chart_df['Week (by Year)'].isin(['Total', ''])]
        
        # Ensure 'Week (by Year)' is numeric for proper sorting
        chart_df['Week (by Year)'] = pd.to_numeric(chart_df['Week (by Year)'], errors='coerce')
        
        # Create chart
        if selected_chart_col in ['Sum of Delivery Volume', 'Sum of Profit Margins %']:
            # Bar chart for Delivery Volume and Profit Margins %
            fig = px.bar(
                chart_df,
                x='Week (by Year)',
                y=selected_chart_col,
                title=f'{selected_chart_col} for {selected_client}',
                labels={'Week (by Year)': 'Week', selected_chart_col: selected_chart_col}
            )
        else:
            # Line chart for other columns
            fig = px.line(
                chart_df,
                x='Week (by Year)',
                y=selected_chart_col,
                title=f'{selected_chart_col} for {selected_client}',
                labels={'Week (by Year)': 'Week', selected_chart_col: selected_chart_col}
            )
        
        # Update layout for dynamic Y-axis and better readability
        fig.update_layout(
            xaxis_title='Week',
            yaxis_title=selected_chart_col,
            xaxis=dict(tickmode='linear', tick0=1, dtick=1),
            yaxis=dict(zeroline=True, zerolinecolor='black', zerolinewidth=1),
            showlegend=False
        )
        
        # Display the chart
        st.plotly_chart(fig, use_container_width=True)

# Detailed Dashboard Page
def detailed_dashboard_page():
    # Set page config with custom title and favicon
    favicon_base64 = get_base64_image('rideblitz_logo.jpeg')
    st.set_page_config(
        page_title="Blitz 3PL Margin Dashboard",
        page_icon=f"data:image/jpeg;base64,{favicon_base64}" if favicon_base64 else None,
        layout="wide"
    )
    display_logo()
    st.title('🧐Detailed Margin Data')
    st.markdown('Break down every revenue stream and cost component to understand the full financial picture.')
    
    # Add back and logout buttons
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Back to Feature Selection"):
            st.session_state['page'] = 'home'
            st.rerun()
    with col2:
        st.button("Logout", on_click=logout, key="logout_detailed_dashboard")

    # Load data from Parquet if exists, otherwise convert from Excel
    excel_path = 'Ops Data Collection.xlsx'
    parquet_path = 'Ops Data Collection.parquet'
    try:
        # Convert Excel to Parquet (updates Parquet file automatically)
        convert_excel_to_parquet(excel_path, parquet_path)
        # Load from Parquet
        df = pd.read_parquet(parquet_path)
        df.columns = df.columns.str.strip()
        st.info("Data loaded successfully from Parquet!")
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.stop()

    # --- Pre-processing Data ---
    currency_cols = [
        'TOTAL DELIVERY REVENUE', 'Total Revenue', 'Selling Price (Regular Rate)',
        'Additional Charge (KM, KG, Etc)', 'Return/Delivery Rate', 'Lalamove Bills (Invoicing to Client)',
        'EV Reduction (3PL & KSJ)', 'EV Manpower', 'EV Revenue + Battery (Rental Client)',
        'Claim/COD/Own Risk', 'Hub, COD Fee (SBY) & Service Korlap', 'Other Revenue',
        'Attribute Fee', 'Rider Cost', 'Manpower Cost', 'OEM Cost', 'Mid-Mile/ Linehaul Cost',
        'Add. 3PL Cost', 'DM Program', 'Claim Damaged/Loss', 'Outstanding COD',
        'Claim Ownrisk', 'Attribute Cost', 'HUB Cost', 'Other Cost', 'Total Cost',
        'Delivery Volume'
    ]

    for col in currency_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            mask = df[col].isna()
            if mask.any():
                df.loc[mask, col] = df.loc[mask, col].astype(str).str.replace('Rp', '', regex=False).str.replace(' ', '', regex=False).str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # --- Filters ---
    st.sidebar.header('Filter Data')
    all_years = sorted(df['Year'].unique())
    selected_year = st.sidebar.selectbox('Year', all_years, index=0)  # No (All) option, default to first year

    df_filtered_year = df[df['Year'] == selected_year]

    all_teams = ['(All)'] + sorted(df_filtered_year['Blitz Team'].unique())
    selected_team = st.sidebar.selectbox('Blitz Team', all_teams)

    df_filtered_team = df_filtered_year.copy()
    if selected_team != '(All)':
        df_filtered_team = df_filtered_team[df_filtered_team['Blitz Team'] == selected_team]

    all_locations = ['(All)'] + sorted(df_filtered_team['Client Location'].unique())
    selected_location = st.sidebar.selectbox('Client Location', all_locations)

    df_filtered_location = df_filtered_team.copy()
    if selected_location != '(All)':
        df_filtered_location = df_filtered_location[df_filtered_location['Client Location'] == selected_location]

    all_clients = sorted(df_filtered_location['Client Name'].unique())
    selected_client = st.sidebar.selectbox('Client Name', all_clients)

    # Apply filters only if a Client Name is selected
    if not selected_client:
        st.warning('Please select at least one Client Name to display the dashboard.')
        st.stop()

    df_filtered_client = df_filtered_location[df_filtered_location['Client Name'] == selected_client]
    
    all_projects = ['(All)'] + sorted(df_filtered_client['Project'].unique())
    selected_project = st.sidebar.selectbox('Project', all_projects)

    filtered_df = df_filtered_client.copy()
    if selected_project != '(All)':
        filtered_df = filtered_df[filtered_df['Project'] == selected_project]

    if filtered_df.empty:
        st.warning('Tidak ada data yang sesuai dengan filter yang dipilih.')
        st.stop()

    # --- Create Detailed Pivot Table ---
    st.subheader('Clients Week-by-Week (Detailed)')

    # Group data by Client and Week
    df_grouped = filtered_df.groupby(['Client Name', 'Week (by Year)']).agg(
        Delivery_Volume=('Delivery Volume', 'sum'),
        Selling_Price_Regular_Rate=('Selling Price (Regular Rate)', 'sum'),
        Additional_Charge=('Additional Charge (KM, KG, Etc)', 'sum'),
        Total_Delivery_Revenue=('TOTAL DELIVERY REVENUE', 'sum'),
        EV_Reduction=('EV Reduction (3PL & KSJ)', 'sum'),
        Claim_COD_Own_Risk=('Claim/COD/Own Risk', 'sum'),
        Hub_COD_Fee=('Hub, COD Fee (SBY) & Service Korlap', 'sum'),
        Lalamove_Bills=('Lalamove Bills (Invoicing to Client)', 'sum'),
        Attribute_Fee=('Attribute Fee', 'sum'),
        Total_Revenue=('Total Revenue', 'sum'),
        Rider_Cost=('Rider Cost', 'sum'),
        Manpower_Cost=('Manpower Cost', 'sum'),
        OEM_Cost=('OEM Cost', 'sum'),
        Mid_Mile_Cost=('Mid-Mile/ Linehaul Cost', 'sum'),
        Add_3PL_Cost=('Add. 3PL Cost', 'sum'),
        DM_Program=('DM Program', 'sum'),
        Claim_Damaged_Loss=('Claim Damaged/Loss', 'sum'),
        Outstanding_COD=('Outstanding COD', 'sum'),
        Claim_Ownrisk=('Claim Ownrisk', 'sum'),
        HUB_Cost=('HUB Cost', 'sum'),
        Other_Cost=('Other Cost', 'sum'),
        Total_Cost=('Total Cost', 'sum')
    ).reset_index()

    # Calculate additional metrics
    df_grouped['Profit_Value'] = df_grouped['Total_Revenue'] - df_grouped['Total_Cost']
    df_grouped['Profit_Margins'] = (df_grouped['Profit_Value'] / df_grouped['Total_Revenue']).replace([np.inf, -np.inf], np.nan).fillna(0) * 100
    df_grouped['SRPO'] = (df_grouped['Total_Delivery_Revenue'] / df_grouped['Delivery_Volume']).replace([np.inf, -np.inf], np.nan).fillna(0)
    df_grouped['RCPO'] = (df_grouped['Rider_Cost'] / df_grouped['Delivery_Volume']).replace([np.inf, -np.inf], np.nan).fillna(0)
    df_grouped['Rider_Cost_Percent'] = (df_grouped['Rider_Cost'] / df_grouped['Total_Delivery_Revenue']).replace([np.inf, -np.inf], np.nan).fillna(0) * 100
    df_grouped['Margin_Per_Order'] = (df_grouped['Profit_Value'] / df_grouped['Delivery_Volume']).replace([np.inf, -np.inf], np.nan).fillna(0)
    df_grouped['TCOP'] = (df_grouped['Total_Cost'] / df_grouped['Delivery_Volume']).replace([np.inf, -np.inf], np.nan).fillna(0)

    # Calculate totals per Client
    df_total_client = filtered_df.groupby('Client Name').agg(
        Delivery_Volume=('Delivery Volume', 'sum'),
        Selling_Price_Regular_Rate=('Selling Price (Regular Rate)', 'sum'),
        Additional_Charge=('Additional Charge (KM, KG, Etc)', 'sum'),
        Total_Delivery_Revenue=('TOTAL DELIVERY REVENUE', 'sum'),
        EV_Reduction=('EV Reduction (3PL & KSJ)', 'sum'),
        Claim_COD_Own_Risk=('Claim/COD/Own Risk', 'sum'),
        Hub_COD_Fee=('Hub, COD Fee (SBY) & Service Korlap', 'sum'),
        Lalamove_Bills=('Lalamove Bills (Invoicing to Client)', 'sum'),
        Attribute_Fee=('Attribute Fee', 'sum'),
        Total_Revenue=('Total Revenue', 'sum'),
        Rider_Cost=('Rider Cost', 'sum'),
        Manpower_Cost=('Manpower Cost', 'sum'),
        OEM_Cost=('OEM Cost', 'sum'),
        Mid_Mile_Cost=('Mid-Mile/ Linehaul Cost', 'sum'),
        Add_3PL_Cost=('Add. 3PL Cost', 'sum'),
        DM_Program=('DM Program', 'sum'),
        Claim_Damaged_Loss=('Claim Damaged/Loss', 'sum'),
        Outstanding_COD=('Outstanding COD', 'sum'),
        Claim_Ownrisk=('Claim Ownrisk', 'sum'),
        HUB_Cost=('HUB Cost', 'sum'),
        Other_Cost=('Other Cost', 'sum'),
        Total_Cost=('Total Cost', 'sum')
    ).reset_index()

    # Calculate total metrics per Client
    df_total_client['Profit_Value'] = df_total_client['Total_Revenue'] - df_total_client['Total_Cost']
    df_total_client['Profit_Margins'] = (df_total_client['Profit_Value'] / df_total_client['Total_Revenue']).replace([np.inf, -np.inf], np.nan).fillna(0) * 100
    df_total_client['SRPO'] = (df_total_client['Total_Delivery_Revenue'] / df_total_client['Delivery_Volume']).replace([np.inf, -np.inf], np.nan).fillna(0)
    df_total_client['RCPO'] = (df_total_client['Rider_Cost'] / df_total_client['Delivery_Volume']).replace([np.inf, -np.inf], np.nan).fillna(0)
    df_total_client['Rider_Cost_Percent'] = (df_total_client['Rider_Cost'] / df_total_client['Total_Delivery_Revenue']).replace([np.inf, -np.inf], np.nan).fillna(0) * 100
    df_total_client['Margin_Per_Order'] = (df_total_client['Profit_Value'] / df_total_client['Delivery_Volume']).replace([np.inf, -np.inf], np.nan).fillna(0)
    df_total_client['TCOP'] = (df_total_client['Total_Cost'] / df_total_client['Delivery_Volume']).replace([np.inf, -np.inf], np.nan).fillna(0)

    # Combine weekly and total data
    final_df = pd.DataFrame()
    for client in df_total_client['Client Name'].unique():
        client_total_row = df_total_client[df_total_client['Client Name'] == client].copy()
        client_total_row.rename(columns={'Client Name': 'Row Labels'}, inplace=True)
        client_total_row['Week (by Year)'] = 'Total'
        
        client_weekly_data = df_grouped[df_grouped['Client Name'] == client].copy()
        client_weekly_data.rename(columns={'Client Name': 'Row Labels'}, inplace=True)
        
        # Add comparison (diff) for the last weeks
        client_weekly_data.sort_values('Week (by Year)', inplace=True)

        if len(client_weekly_data) >= 3:
            last_three_weeks = client_weekly_data.tail(3).copy()
            third_last_week = last_three_weeks.iloc[0]
            second_last_week = last_three_weeks.iloc[1]
            last_week_orig = last_three_weeks.iloc[2]
            diff_data_earlier = {}
            for col in ['Delivery_Volume', 'Selling_Price_Regular_Rate', 'Additional_Charge', 'Total_Delivery_Revenue',
                        'EV_Reduction', 'Claim_COD_Own_Risk', 'Hub_COD_Fee', 'Lalamove_Bills', 'Attribute_Fee',
                        'Total_Revenue', 'Rider_Cost', 'Rider_Cost_Percent', 'Manpower_Cost', 'OEM_Cost',
                        'Mid_Mile_Cost', 'Add_3PL_Cost', 'DM_Program', 'Claim_Damaged_Loss', 'Outstanding_COD',
                        'Claim_Ownrisk', 'HUB_Cost', 'Other_Cost', 'Total_Cost', 'Profit_Value', 'Profit_Margins',
                        'SRPO', 'RCPO', 'Margin_Per_Order', 'TCOP']:
                val_second_last = second_last_week[col]
                val_third_last = third_last_week[col]
                if val_third_last != 0 and not pd.isna(val_third_last):
                    diff_data_earlier[col] = ((val_second_last - val_third_last) / abs(val_third_last)) * 100
                else:
                    diff_data_earlier[col] = np.nan
            diff_row_earlier = pd.Series(diff_data_earlier)
            week_value_earlier = str(second_last_week['Week (by Year)']).strip()
            if week_value_earlier and week_value_earlier.replace('.', '').replace('-', '').isdigit():
                diff_row_earlier['Row Labels'] = f'Diff W{int(float(week_value_earlier))}%'
            else:
                diff_row_earlier['Row Labels'] = 'Diff WNA%'
            diff_row_earlier['Week (by Year)'] = ''
            for col in client_weekly_data.columns:
                if col not in diff_row_earlier.index:
                    diff_row_earlier[col] = np.nan
            diff_row_earlier = diff_row_earlier[client_weekly_data.columns]
            insert_idx = client_weekly_data.index[client_weekly_data['Week (by Year)'] == second_last_week['Week (by Year)']].tolist()[-1] + 1
            client_weekly_data = pd.concat([client_weekly_data.iloc[:insert_idx], pd.DataFrame([diff_row_earlier]), client_weekly_data.iloc[insert_idx:]], ignore_index=True)

        if len(client_weekly_data) >= 2:
            last_week = last_week_orig
            second_last_week = client_weekly_data[client_weekly_data['Week (by Year)'] == last_week_orig['Week (by Year)'] - 1].iloc[0] if last_week_orig['Week (by Year)'] > 1 else client_weekly_data.iloc[-2]
            diff_data = {}
            for col in ['Delivery_Volume', 'Selling_Price_Regular_Rate', 'Additional_Charge', 'Total_Delivery_Revenue',
                        'EV_Reduction', 'Claim_COD_Own_Risk', 'Hub_COD_Fee', 'Lalamove_Bills', 'Attribute_Fee',
                        'Total_Revenue', 'Rider_Cost', 'Rider_Cost_Percent', 'Manpower_Cost', 'OEM_Cost',
                        'Mid_Mile_Cost', 'Add_3PL_Cost', 'DM_Program', 'Claim_Damaged_Loss', 'Outstanding_COD',
                        'Claim_Ownrisk', 'HUB_Cost', 'Other_Cost', 'Total_Cost', 'Profit_Value', 'Profit_Margins',
                        'SRPO', 'RCPO', 'Margin_Per_Order', 'TCOP']:
                val_last = last_week[col]
                val_second_last = second_last_week[col]
                if val_second_last != 0 and not pd.isna(val_second_last):
                    diff_data[col] = ((val_last - val_second_last) / abs(val_second_last)) * 100
                else:
                    diff_data[col] = np.nan
            diff_row = pd.Series(diff_data)
            week_value = str(last_week['Week (by Year)']).strip()
            if week_value and week_value.replace('.', '').replace('-', '').isdigit():
                diff_row['Row Labels'] = f'Diff W{int(float(week_value))}%'
            else:
                diff_row['Row Labels'] = 'Diff WNA%'
            diff_row['Week (by Year)'] = ''
            for col in client_weekly_data.columns:
                if col not in diff_row.index:
                    diff_row[col] = np.nan
            diff_row = diff_row[client_weekly_data.columns]
            insert_idx_last = client_weekly_data.index[client_weekly_data['Week (by Year)'] == last_week['Week (by Year)']].tolist()[-1] + 1
            client_weekly_data = pd.concat([client_weekly_data.iloc[:insert_idx_last], pd.DataFrame([diff_row]), client_weekly_data.iloc[insert_idx_last:]], ignore_index=True)

        combined_data = pd.concat([client_total_row, client_weekly_data], ignore_index=True)
        final_df = pd.concat([final_df, combined_data], ignore_index=True)

    # Select and rename columns for display
    final_df = final_df[[
        'Row Labels', 'Week (by Year)', 'Delivery_Volume', 'Selling_Price_Regular_Rate',
        'Additional_Charge', 'Total_Delivery_Revenue', 'EV_Reduction', 'Claim_COD_Own_Risk',
        'Hub_COD_Fee', 'Lalamove_Bills', 'Attribute_Fee', 'Total_Revenue', 'Rider_Cost',
        'Rider_Cost_Percent', 'Manpower_Cost', 'OEM_Cost', 'Mid_Mile_Cost', 'Add_3PL_Cost',
        'DM_Program', 'Claim_Damaged_Loss', 'Outstanding_COD', 'Claim_Ownrisk', 'HUB_Cost',
        'Other_Cost', 'Total_Cost', 'Profit_Value', 'Profit_Margins', 'SRPO', 'RCPO',
        'Margin_Per_Order', 'TCOP'
    ]]
    final_df.columns = [
        'Row Labels', 'Week (by Year)', 'Sum of Delivery Volume', 'Sum of Selling Price (Regular Rate)',
        'Sum of Additional Charge (KM, KG, Etc)', 'Sum of Total Sellings', 'Sum of EV Reduction (3PL & KSJ)',
        'Sum of Claim/COD/Own Risk', 'Sum of Hub, COD Fee (SBY) & Service Korlap',
        'Sum of Lalamove Bills (Invoicing to Client)', 'Sum of Attribute Fee', 'Sum of Total Revenue',
        'Sum of Rider Cost', 'Sum of Rider Cost as % of Total Sellings', 'Sum of Manpower Cost',
        'Sum of OEM Cost', 'Sum of Mid-Mile/ Linehaul Cost', 'Sum of Add. 3PL Cost',
        'Sum of DM Program', 'Sum of Claim Damaged/Loss', 'Sum of Outstanding COD',
        'Sum of Claim Ownrisk', 'Sum of HUB Cost', 'Sum of Other Cost', 'Sum of Total Cost',
        'Sum of Profit Value', 'Sum of Profit Margins %', 'Sum of Selling Revenue per Order (SRPO)',
        'Sum of Rider Cost per Order (RCPO)', 'Sum of Margin % Per Order',
        'Sum of Total Cost per Order (TCOP)'
    ]

    # --- Styling DataFrame ---
    def color_rows(row):
        is_total = 'Total' in str(row['Week (by Year)'])
        is_diff = 'Diff W' in str(row['Row Labels'])
        styles = [''] * len(row)

        if is_total:
            styles = ['background-color: #e6ffe6; color: black'] * len(row)
        elif is_diff:
            styles = ['background-color: #fff2e6; color: black'] * len(row)
        
        return styles

    # Format values for display
    display_df = final_df.copy()
    for col in display_df.columns:
        if col == 'Row Labels':
            display_df[col] = display_df[col].astype(str)
        elif col == 'Week (by Year)':
            display_df[col] = display_df[col].apply(lambda x: f"{int(float(x))}" if pd.notna(x) and str(x).replace('.', '').replace('-', '').isdigit() else x)
        elif col == 'Sum of Delivery Volume':
            display_df[col] = display_df.apply(
                lambda row: f"{row[col]:,.2f}%" if pd.notna(row[col]) and 'Diff' in str(row['Row Labels']) else (
                    f"{int(row[col]):,}" if pd.notna(row[col]) else ''
                ), axis=1
            )
        elif col in ['Sum of Rider Cost as % of Total Sellings', 'Sum of Profit Margins %']:
            display_df[col] = display_df.apply(
                lambda row: f"{row[col]:,.2f}%" if pd.notna(row[col]) else '', axis=1
            )
        elif col in ['Sum of Selling Revenue per Order (SRPO)', 'Sum of Rider Cost per Order (RCPO)', 
                     'Sum of Margin % Per Order', 'Sum of Total Cost per Order (TCOP)']:
            display_df[col] = display_df.apply(
                lambda row: f"{row[col]:,.2f}%" if pd.notna(row[col]) and 'Diff' in str(row['Row Labels']) else (
                    f"Rp {row[col]:,.2f}" if pd.notna(row[col]) else ''
                ), axis=1
            )
        else:
            display_df[col] = display_df.apply(
                lambda row: f"{row[col]:,.2f}%" if pd.notna(row[col]) and 'Diff' in str(row['Row Labels']) else (
                    f"Rp {row[col]:,.0f}" if pd.notna(row[col]) else ''
                ), axis=1
            )
        display_df[col] = display_df[col].astype(str).str.replace('nan', '', regex=False)

    # Display styled dataframe
    st.dataframe(
        display_df.style.apply(color_rows, axis=1), 
        hide_index=True
    )

# Week on Week Performance Page
def wow_performance_page():
    # Set page config with custom title and favicon
    favicon_base64 = get_base64_image('rideblitz_logo.jpeg')
    st.set_page_config(
        page_title="Blitz 3PL Margin Dashboard",
        page_icon=f"data:image/jpeg;base64,{favicon_base64}" if favicon_base64 else None,
        layout="wide")
    display_logo()
    st.title('🗓️Week on Week Performance')
    st.markdown('Summarize all projects performance trends like volume and financial condition with a week-on-week view.')
    
    # Add back and logout buttons
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Back to Feature Selection"):
            st.session_state['page'] = 'home'
            st.rerun()
    with col2:
        st.button("Logout", on_click=logout, key="logout_wow_performance")

    # Load data from Parquet if exists, otherwise convert from Excel
    excel_path = 'Ops Data Collection.xlsx'
    parquet_path = 'Ops Data Collection.parquet'
    try:
        # Convert Excel to Parquet (updates Parquet file automatically)
        convert_excel_to_parquet(excel_path, parquet_path)
        # Load from Parquet
        df = pd.read_parquet(parquet_path)
        df.columns = df.columns.str.strip()
        st.info("Data loaded successfully from Parquet!")
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.stop()

    # --- Pre-processing Data ---
    currency_cols = [
        'TOTAL DELIVERY REVENUE', 'Total Revenue', 'Selling Price (Regular Rate)',
        'Additional Charge (KM, KG, Etc)', 'Return/Delivery Rate', 'Lalamove Bills (Invoicing to Client)',
        'EV Reduction (3PL & KSJ)', 'EV Manpower', 'EV Revenue + Battery (Rental Client)',
        'Claim/COD/Own Risk', 'Hub, COD Fee (SBY) & Service Korlap', 'Other Revenue',
        'Attribute Fee', 'Rider Cost', 'Manpower Cost', 'OEM Cost', 'Mid-Mile/ Linehaul Cost',
        'Add. 3PL Cost', 'DM Program', 'Claim Damaged/Loss', 'Outstanding COD',
        'Claim Ownrisk', 'Attribute Cost', 'HUB Cost', 'Other Cost', 'Total Cost',
        'Delivery Volume'
    ]

    for col in currency_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            mask = df[col].isna()
            if mask.any():
                df.loc[mask, col] = df.loc[mask, col].astype(str).str.replace('Rp', '', regex=False).str.replace(' ', '', regex=False).str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # --- Filters ---
    st.sidebar.header('Filter Data')
    all_years = sorted(df['Year'].unique())
    selected_year = st.sidebar.selectbox('Year', all_years, index=0)  # No (All) option, default to first year

    df_filtered_year = df[df['Year'] == selected_year]

    all_teams = ['(All)'] + sorted(df_filtered_year['Blitz Team'].unique())
    selected_team = st.sidebar.selectbox('Blitz Team', all_teams)

    df_filtered_team = df_filtered_year.copy()
    if selected_team != '(All)':
        df_filtered_team = df_filtered_team[df_filtered_team['Blitz Team'] == selected_team]

    all_locations = ['(All)'] + sorted(df_filtered_team['Client Location'].unique())
    selected_location = st.sidebar.selectbox('Client Location', all_locations)

    filtered_df = df_filtered_team.copy()
    if selected_location != '(All)':
        filtered_df = filtered_df[filtered_df['Client Location'] == selected_location]

    if filtered_df.empty:
        st.warning('Tidak ada data yang sesuai dengan filter yang dipilih.')
        st.stop()

    # --- Create Pivot Table ---
    st.subheader('Week-by-Week Performance')

    # Group data by Week
    df_grouped = filtered_df.groupby('Week (by Year)').agg(
        Delivery_Volume=('Delivery Volume', 'sum'),
        Total_Delivery_Revenue=('TOTAL DELIVERY REVENUE', 'sum'),
        Total_Revenue=('Total Revenue', 'sum'),
        Total_Cost=('Total Cost', 'sum'),
        Rider_Cost=('Rider Cost', 'sum')
    ).reset_index()

    # Calculate weekly metrics
    df_grouped['Profit_Value'] = df_grouped['Total_Revenue'] - df_grouped['Total_Cost']
    df_grouped['Profit_Margins'] = (df_grouped['Profit_Value'] / df_grouped['Total_Revenue']).replace([np.inf, -np.inf], np.nan).fillna(0) * 100
    df_grouped['SRPO'] = (df_grouped['Total_Delivery_Revenue'] / df_grouped['Delivery_Volume']).replace([np.inf, -np.inf], np.nan).fillna(0)
    df_grouped['RCPO'] = (df_grouped['Rider_Cost'] / df_grouped['Delivery_Volume']).replace([np.inf, -np.inf], np.nan).fillna(0)
    df_grouped['TCOP'] = (df_grouped['Total_Cost'] / df_grouped['Delivery_Volume']).replace([np.inf, -np.inf], np.nan).fillna(0)

    # Calculate totals
    df_total = pd.DataFrame({
        'Week (by Year)': ['Total'],
        'Delivery_Volume': [df_grouped['Delivery_Volume'].sum()],
        'Total_Delivery_Revenue': [df_grouped['Total_Delivery_Revenue'].sum()],
        'Total_Revenue': [df_grouped['Total_Revenue'].sum()],
        'Total_Cost': [df_grouped['Total_Cost'].sum()],
        'Rider_Cost': [df_grouped['Rider_Cost'].sum()],
        'Profit_Value': [df_grouped['Total_Revenue'].sum() - df_grouped['Total_Cost'].sum()],
        'Profit_Margins': [((df_grouped['Total_Revenue'].sum() - df_grouped['Total_Cost'].sum()) / df_grouped['Total_Revenue'].sum()) * 100 if df_grouped['Total_Revenue'].sum() != 0 else 0],
        'SRPO': [df_grouped['Total_Delivery_Revenue'].sum() / df_grouped['Delivery_Volume'].sum() if df_grouped['Delivery_Volume'].sum() != 0 else 0],
        'RCPO': [df_grouped['Rider_Cost'].sum() / df_grouped['Delivery_Volume'].sum() if df_grouped['Delivery_Volume'].sum() != 0 else 0],
        'TCOP': [df_grouped['Total_Cost'].sum() / df_grouped['Delivery_Volume'].sum() if df_grouped['Delivery_Volume'].sum() != 0 else 0]
    })

    # Prepare weekly data with Row Labels
    df_grouped['Row Labels'] = df_grouped['Week (by Year)'].astype(str)
    df_total['Row Labels'] = 'Total'

    # Sort weekly data by week
    df_grouped.sort_values('Week (by Year)', inplace=True)

    # Add diff rows if there are at least 3 weeks
    if len(df_grouped) >= 3:
        last_three_weeks = df_grouped.tail(3).copy()
        third_last_week = last_three_weeks.iloc[0]
        second_last_week = last_three_weeks.iloc[1]
        last_week_orig = last_three_weeks.iloc[2]
        diff_data_earlier = {}
        for col in ['Delivery_Volume', 'Total_Delivery_Revenue', 'Total_Revenue', 'Total_Cost',
                    'Profit_Value', 'Profit_Margins', 'SRPO', 'RCPO', 'TCOP']:
            val_second_last = second_last_week[col]
            val_third_last = third_last_week[col]
            if val_third_last != 0 and not pd.isna(val_third_last):
                diff_data_earlier[col] = ((val_second_last - val_third_last) / abs(val_third_last)) * 100
            else:
                diff_data_earlier[col] = np.nan
        diff_row_earlier = pd.Series(diff_data_earlier)
        week_value_earlier = str(second_last_week['Week (by Year)']).strip()
        if week_value_earlier and week_value_earlier.replace('.', '').replace('-', '').isdigit():
            diff_row_earlier['Row Labels'] = f'Diff W{int(float(week_value_earlier))}%'
        else:
            diff_row_earlier['Row Labels'] = 'Diff WNA%'
        diff_row_earlier['Week (by Year)'] = ''
        for col in df_grouped.columns:
            if col not in diff_row_earlier.index:
                diff_row_earlier[col] = np.nan
        diff_row_earlier = diff_row_earlier[df_grouped.columns]
        insert_idx = df_grouped.index[df_grouped['Week (by Year)'] == second_last_week['Week (by Year)']].tolist()[-1] + 1
        df_grouped = pd.concat([df_grouped.iloc[:insert_idx], pd.DataFrame([diff_row_earlier]), df_grouped.iloc[insert_idx:]], ignore_index=True)

    # Add diff for the last two weeks if at least 2 weeks
    if len(df_grouped) >= 2:
        last_week = df_grouped[df_grouped['Week (by Year)'] == last_week_orig['Week (by Year)']].iloc[0]
        second_last_week = df_grouped[df_grouped['Week (by Year)'] == last_week_orig['Week (by Year)'] - 1].iloc[0] if last_week_orig['Week (by Year)'] > 1 else df_grouped.iloc[-2]
        diff_data = {}
        for col in ['Delivery_Volume', 'Total_Delivery_Revenue', 'Total_Revenue', 'Total_Cost',
                    'Profit_Value', 'Profit_Margins', 'SRPO', 'RCPO', 'TCOP']:
            val_last = last_week[col]
            val_second_last = second_last_week[col]
            if val_second_last != 0 and not pd.isna(val_second_last):
                diff_data[col] = ((val_last - val_second_last) / abs(val_second_last)) * 100
            else:
                diff_data[col] = np.nan
        diff_row = pd.Series(diff_data)
        week_value = str(last_week['Week (by Year)']).strip()
        if week_value and week_value.replace('.', '').replace('-', '').isdigit():
            diff_row['Row Labels'] = f'Diff W{int(float(week_value))}%'
        else:
            diff_row['Row Labels'] = 'Diff WNA%'
        diff_row['Week (by Year)'] = ''
        for col in df_grouped.columns:
            if col not in diff_row.index:
                diff_row[col] = np.nan
        diff_row = diff_row[df_grouped.columns]
        insert_idx_last = df_grouped.index[df_grouped['Week (by Year)'] == last_week['Week (by Year)']].tolist()[-1] + 1
        df_grouped = pd.concat([df_grouped.iloc[:insert_idx_last], pd.DataFrame([diff_row]), df_grouped.iloc[insert_idx_last:]], ignore_index=True)

    # Combine total and weekly data
    final_df = pd.concat([df_total, df_grouped], ignore_index=True)

    # Select and rename columns for display
    final_df = final_df[[
        'Row Labels', 'Week (by Year)', 'Delivery_Volume', 'Total_Delivery_Revenue',
        'Total_Revenue', 'Total_Cost', 'Profit_Value', 'Profit_Margins',
        'SRPO', 'RCPO', 'TCOP'
    ]]
    final_df.columns = [
        'Row Labels', 'Week (by Year)', 'Sum of Delivery Volume', 'Sum of Total Sellings',
        'Sum of Total Revenue', 'Sum of Total Cost', 'Sum of Profit Value',
        'Sum of Profit Margins %', 'Sum of Selling Revenue per Order (SRPO)',
        'Sum of Rider Cost per Order (RCPO)', 'Sum of Total Cost per Order (TCOP)'
    ]

    # --- Styling DataFrame ---
    def color_rows(row):
        is_total = 'Total' in str(row['Week (by Year)'])
        is_diff = 'Diff W' in str(row['Row Labels'])
        styles = [''] * len(row)

        if is_total:
            styles = ['background-color: #e6ffe6; color: black'] * len(row)
        elif is_diff:
            styles = ['background-color: #fff2e6; color: black'] * len(row)
        
        return styles

    # Format values for display
    display_df = final_df.copy()
    for col in display_df.columns:
        if col not in ['Row Labels', 'Week (by Year)']:
            display_df[col] = display_df.apply(
                lambda row: f"{row[col]:,.2f}%" if pd.notna(row[col]) and ('Diff' in str(row['Row Labels']) or 'Profit Margins' in col) else (
                    f"Rp {row[col]:,.2f}" if pd.notna(row[col]) and ('SRPO' in col or 'RCPO' in col or 'TCOP' in col) else (
                        f"Rp {row[col]:,.0f}" if pd.notna(row[col]) and ('Total' in col or 'Selling' in col or 'Cost' in col or 'Profit' in col) else (
                            f"{row[col]:,.0f}" if pd.notna(row[col]) else ''
                        )
                    )
                ),
                axis=1
            )
            display_df[col] = display_df[col].astype(str).str.replace('nan', '', regex=False)

    # Display styled dataframe
    st.dataframe(
        display_df.style.apply(color_rows, axis=1), 
        hide_index=True
    )

    # --- Generate Chart ---
    st.subheader('Generate Chart')
    chart_columns = [
        'Sum of Delivery Volume', 'Sum of Total Sellings', 'Sum of Total Revenue',
        'Sum of Total Cost', 'Sum of Profit Value', 'Sum of Profit Margins %',
        'Sum of Selling Revenue per Order (SRPO)', 'Sum of Rider Cost per Order (RCPO)',
        'Sum of Total Cost per Order (TCOP)'
    ]
    selected_chart_col = st.selectbox('Select Variable for Chart', chart_columns)
    
    if st.button('Generate Chart'):
        # Filter data for chart, exclude total/diff rows
        chart_df = final_df[~final_df['Week (by Year)'].isin(['Total', ''])]
        
        # Ensure 'Week (by Year)' is numeric for proper sorting
        chart_df['Week (by Year)'] = pd.to_numeric(chart_df['Week (by Year)'], errors='coerce')
        
        # Create chart
        if selected_chart_col in ['Sum of Delivery Volume', 'Sum of Profit Margins %']:
            # Bar chart for Delivery Volume and Profit Margins %
            fig = px.bar(
                chart_df,
                x='Week (by Year)',
                y=selected_chart_col,
                title=f'{selected_chart_col} Week-by-Week',
                labels={'Week (by Year)': 'Week', selected_chart_col: selected_chart_col}
            )
        else:
            # Line chart for other columns
            fig = px.line(
                chart_df,
                x='Week (by Year)',
                y=selected_chart_col,
                title=f'{selected_chart_col} Week-by-Week',
                labels={'Week (by Year)': 'Week', selected_chart_col: selected_chart_col}
            )
        
        # Update layout for dynamic Y-axis and better readability
        fig.update_layout(
            xaxis_title='Week',
            yaxis_title=selected_chart_col,
            xaxis=dict(tickmode='linear', tick0=1, dtick=1),
            yaxis=dict(zeroline=True, zerolinecolor='black', zerolinewidth=1),
            showlegend=False
        )
        
        # Display the chart
        st.plotly_chart(fig, use_container_width=True)

def monthly_performance_page():
    # Set page config with custom title and favicon
    favicon_base64 = get_base64_image('rideblitz_logo.jpeg')
    st.set_page_config(
        page_title="Blitz 3PL Margin Dashboard",
        page_icon=f"data:image/jpeg;base64,{favicon_base64}" if favicon_base64 else None,
        layout="wide")
    display_logo()
    st.title('🌕Monthly Performance')
    st.markdown('Summarize all projects performance trends like volume and financial condition with a monthly view.')
    
    # Add back and logout buttons
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Back to Feature Selection"):
            st.session_state['page'] = 'home'
            st.rerun()
    with col2:
        st.button("Logout", on_click=logout, key="logout_monthly_performance")

    # Load data from Parquet if exists, otherwise convert from Excel
    excel_path = 'Ops Data Collection.xlsx'
    parquet_path = 'Ops Data Collection.parquet'
    try:
        # Convert Excel to Parquet (updates Parquet file automatically)
        convert_excel_to_parquet(excel_path, parquet_path)
        # Load from Parquet
        df = pd.read_parquet(parquet_path)
        df.columns = df.columns.str.strip()
        st.info("Data loaded successfully from Parquet!")
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.stop()

    # --- Pre-processing Data ---
    currency_cols = [
        'TOTAL DELIVERY REVENUE', 'Total Revenue', 'Selling Price (Regular Rate)',
        'Additional Charge (KM, KG, Etc)', 'Return/Delivery Rate', 'Lalamove Bills (Invoicing to Client)',
        'EV Reduction (3PL & KSJ)', 'EV Manpower', 'EV Revenue + Battery (Rental Client)',
        'Claim/COD/Own Risk', 'Hub, COD Fee (SBY) & Service Korlap', 'Other Revenue',
        'Attribute Fee', 'Rider Cost', 'Manpower Cost', 'OEM Cost', 'Mid-Mile/ Linehaul Cost',
        'Add. 3PL Cost', 'DM Program', 'Claim Damaged/Loss', 'Outstanding COD',
        'Claim Ownrisk', 'Attribute Cost', 'HUB Cost', 'Other Cost', 'Total Cost',
        'Delivery Volume'
    ]

    for col in currency_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            mask = df[col].isna()
            if mask.any():
                df.loc[mask, col] = df.loc[mask, col].astype(str).str.replace('Rp', '', regex=False).str.replace(' ', '', regex=False).str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # --- Filters ---
    st.sidebar.header('Filter Data')
    all_years = sorted(df['Year'].unique())
    selected_year = st.sidebar.selectbox('Year', all_years, index=0)  # No (All) option, default to first year

    df_filtered_year = df[df['Year'] == selected_year]

    all_teams = ['(All)'] + sorted(df_filtered_year['Blitz Team'].unique())
    selected_team = st.sidebar.selectbox('Blitz Team', all_teams)

    df_filtered_team = df_filtered_year.copy()
    if selected_team != '(All)':
        df_filtered_team = df_filtered_team[df_filtered_team['Blitz Team'] == selected_team]

    all_locations = ['(All)'] + sorted(df_filtered_team['Client Location'].unique())
    selected_location = st.sidebar.selectbox('Client Location', all_locations)

    filtered_df = df_filtered_team.copy()
    if selected_location != '(All)':
        filtered_df = filtered_df[filtered_df['Client Location'] == selected_location]

    if filtered_df.empty:
        st.warning('Tidak ada data yang sesuai dengan filter yang dipilih.')
        st.stop()

    # --- Create Pivot Table ---
    st.subheader('Month-by-Month Performance')

    # Group data by Month
    df_grouped = filtered_df.groupby('Month').agg(
        Delivery_Volume=('Delivery Volume', 'sum'),
        Total_Delivery_Revenue=('TOTAL DELIVERY REVENUE', 'sum'),
        Total_Revenue=('Total Revenue', 'sum'),
        Total_Cost=('Total Cost', 'sum'),
        Rider_Cost=('Rider Cost', 'sum')
    ).reset_index()

    # Calculate monthly metrics
    df_grouped['Profit_Value'] = df_grouped['Total_Revenue'] - df_grouped['Total_Cost']
    df_grouped['Profit_Margins'] = (df_grouped['Profit_Value'] / df_grouped['Total_Revenue']).replace([np.inf, -np.inf], np.nan).fillna(0) * 100
    df_grouped['SRPO'] = (df_grouped['Total_Delivery_Revenue'] / df_grouped['Delivery_Volume']).replace([np.inf, -np.inf], np.nan).fillna(0)
    df_grouped['RCPO'] = (df_grouped['Rider_Cost'] / df_grouped['Delivery_Volume']).replace([np.inf, -np.inf], np.nan).fillna(0)
    df_grouped['TCOP'] = (df_grouped['Total_Cost'] / df_grouped['Delivery_Volume']).replace([np.inf, -np.inf], np.nan).fillna(0)

    # Calculate totals
    df_total = pd.DataFrame({
        'Month': ['Total'],
        'Delivery_Volume': [df_grouped['Delivery_Volume'].sum()],
        'Total_Delivery_Revenue': [df_grouped['Total_Delivery_Revenue'].sum()],
        'Total_Revenue': [df_grouped['Total_Revenue'].sum()],
        'Total_Cost': [df_grouped['Total_Cost'].sum()],
        'Rider_Cost': [df_grouped['Rider_Cost'].sum()],
        'Profit_Value': [df_grouped['Total_Revenue'].sum() - df_grouped['Total_Cost'].sum()],
        'Profit_Margins': [((df_grouped['Total_Revenue'].sum() - df_grouped['Total_Cost'].sum()) / df_grouped['Total_Revenue'].sum()) * 100 if df_grouped['Total_Revenue'].sum() != 0 else 0],
        'SRPO': [df_grouped['Total_Delivery_Revenue'].sum() / df_grouped['Delivery_Volume'].sum() if df_grouped['Delivery_Volume'].sum() != 0 else 0],
        'RCPO': [df_grouped['Rider_Cost'].sum() / df_grouped['Delivery_Volume'].sum() if df_grouped['Delivery_Volume'].sum() != 0 else 0],
        'TCOP': [df_grouped['Total_Cost'].sum() / df_grouped['Delivery_Volume'].sum() if df_grouped['Delivery_Volume'].sum() != 0 else 0]
    })

    # Prepare monthly data with Row Labels
    df_grouped['Row Labels'] = df_grouped['Month'].astype(str)
    df_total['Row Labels'] = 'Total'

    # Sort monthly data by month
    month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                   'July', 'August', 'September', 'October', 'November', 'December']
    df_grouped['Month'] = pd.Categorical(df_grouped['Month'], categories=month_order, ordered=True)
    df_grouped.sort_values('Month', inplace=True)

    # Add user-selected month comparison
    st.sidebar.header('Compare Months')
    available_months = df_grouped['Month'].tolist()
    month_1 = st.sidebar.selectbox('Select First Month', available_months, index=0)
    month_2 = st.sidebar.selectbox('Select Second Month', available_months, index=len(available_months)-1 if len(available_months) > 1 else 0)

    if month_1 != month_2 and month_1 in df_grouped['Month'].values and month_2 in df_grouped['Month'].values:
        month_1_data = df_grouped[df_grouped['Month'] == month_1].iloc[0]
        month_2_data = df_grouped[df_grouped['Month'] == month_2].iloc[0]
        
        diff_data = {}
        for col in ['Delivery_Volume', 'Total_Delivery_Revenue', 'Total_Revenue', 'Total_Cost',
                    'Profit_Value', 'Profit_Margins', 'SRPO', 'RCPO', 'TCOP']:
            val_month_1 = month_1_data[col]
            val_month_2 = month_2_data[col]
            if val_month_1 != 0 and not pd.isna(val_month_1):
                diff_data[col] = ((val_month_2 - val_month_1) / abs(val_month_1)) * 100
            else:
                diff_data[col] = np.nan
        
        diff_row = pd.Series(diff_data)
        diff_row['Row Labels'] = f'Diff {month_1} to {month_2}%'
        diff_row['Month'] = ''
        for col in df_grouped.columns:
            if col not in diff_row.index:
                diff_row[col] = np.nan
        diff_row = diff_row[df_grouped.columns]
        
        # Append diff row to grouped data
        df_grouped = pd.concat([df_grouped, pd.DataFrame([diff_row])], ignore_index=True)
    
    # Combine total and monthly data
    final_df = pd.concat([df_total, df_grouped], ignore_index=True)

    # Select and rename columns for display
    final_df = final_df[[
        'Row Labels', 'Month', 'Delivery_Volume', 'Total_Delivery_Revenue',
        'Total_Revenue', 'Total_Cost', 'Profit_Value', 'Profit_Margins',
        'SRPO', 'RCPO', 'TCOP'
    ]]
    final_df.columns = [
        'Row Labels', 'Month', 'Sum of Delivery Volume', 'Sum of Total Sellings',
        'Sum of Total Revenue', 'Sum of Total Cost', 'Sum of Profit Value',
        'Sum of Profit Margins %', 'Sum of Selling Revenue per Order (SRPO)',
        'Sum of Rider Cost per Order (RCPO)', 'Sum of Total Cost per Order (TCOP)'
    ]

    # --- Styling DataFrame ---
    def color_rows(row):
        is_total = 'Total' in str(row['Month'])
        is_diff = 'Diff' in str(row['Row Labels'])
        styles = [''] * len(row)

        if is_total:
            styles = ['background-color: #e6ffe6; color: black'] * len(row)
        elif is_diff:
            styles = ['background-color: #fff2e6; color: black'] * len(row)
        
        return styles

    # Format values for display
    display_df = final_df.copy()
    for col in display_df.columns:
        if col not in ['Row Labels', 'Month']:
            display_df[col] = display_df.apply(
                lambda row: f"{row[col]:,.2f}%" if pd.notna(row[col]) and ('Diff' in str(row['Row Labels']) or 'Profit Margins' in col) else (
                    f"Rp {row[col]:,.2f}" if pd.notna(row[col]) and ('SRPO' in col or 'RCPO' in col or 'TCOP' in col) else (
                        f"Rp {row[col]:,.0f}" if pd.notna(row[col]) and ('Total' in col or 'Selling' in col or 'Cost' in col or 'Profit' in col) else (
                            f"{row[col]:,.0f}" if pd.notna(row[col]) else ''
                        )
                    )
                ),
                axis=1
            )
            display_df[col] = display_df[col].astype(str).str.replace('nan', '', regex=False)

    # Display styled dataframe
    st.dataframe(
        display_df.style.apply(color_rows, axis=1), 
        hide_index=True
    )

    # --- Generate Chart ---
    st.subheader('Generate Chart')
    chart_columns = [
        'Sum of Delivery Volume', 'Sum of Total Sellings', 'Sum of Total Revenue',
        'Sum of Total Cost', 'Sum of Profit Value', 'Sum of Profit Margins %',
        'Sum of Selling Revenue per Order (SRPO)', 'Sum of Rider Cost per Order (RCPO)',
        'Sum of Total Cost per Order (TCOP)'
    ]
    selected_chart_col = st.selectbox('Select Variable for Chart', chart_columns)
    
    if st.button('Generate Chart'):
        # Filter data for chart, exclude total/diff rows
        chart_df = final_df[~final_df['Month'].isin(['Total', ''])]
        
        # Sort by month order for chart
        chart_df['Month'] = pd.Categorical(chart_df['Month'], categories=month_order, ordered=True)
        chart_df = chart_df.sort_values('Month')
        
        # Create chart
        if selected_chart_col in ['Sum of Delivery Volume', 'Sum of Profit Margins %']:
            # Bar chart for Delivery Volume and Profit Margins %
            fig = px.bar(
                chart_df,
                x='Month',
                y=selected_chart_col,
                title=f'{selected_chart_col} Month-by-Month',
                labels={'Month': 'Month', selected_chart_col: selected_chart_col}
            )
        else:
            # Line chart for other columns
            fig = px.line(
                chart_df,
                x='Month',
                y=selected_chart_col,
                title=f'{selected_chart_col} Month-by-Month',
                labels={'Month': 'Month', selected_chart_col: selected_chart_col}
            )
        
        # Update layout for dynamic Y-axis and better readability
        fig.update_layout(
            xaxis_title='Month',
            yaxis_title=selected_chart_col,
            xaxis=dict(tickmode='array', tickvals=month_order),
            yaxis=dict(zeroline=True, zerolinecolor='black', zerolinewidth=1),
            showlegend=False
        )
        
        # Display the chart
        st.plotly_chart(fig, use_container_width=True)

def sla_calculation_page():
    # Set page config with custom title and favicon
    favicon_base64 = get_base64_image('rideblitz_logo.jpeg')
    st.set_page_config(
        page_title="Blitz 3PL Margin Dashboard",
        page_icon=f"data:image/jpeg;base64,{favicon_base64}" if favicon_base64 else None,
        layout="wide"
    )
    display_logo()
    st.title('🎯SLA Calculation')
    st.markdown('Ensure operational excellence by tracking SLA compliance for every project. Use this data to identify issues and drive quality improvements.')
    
    # Add back and logout buttons
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Back to Feature Selection"):
            st.session_state['page'] = 'home'
            st.rerun()
    with col2:
        st.button("Logout", on_click=logout, key="logout_sla_calculation")

    # View selection
    st.header('Select View')
    view_option = st.radio("Choose a view:", ["Week (by Year)", "Month"], horizontal=True)

    # Load data from Parquet if exists, otherwise convert from Excel
    excel_path = 'Ops Data Collection.xlsx'
    parquet_path = 'Ops Data Collection.parquet'
    try:
        convert_excel_to_parquet(excel_path, parquet_path)
        df = pd.read_parquet(parquet_path)
        df.columns = df.columns.str.strip()
        st.info("Data loaded successfully from Parquet!")
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.stop()

    # Pre-processing Data
    currency_cols = [
        'TOTAL DELIVERY REVENUE', 'Total Revenue', 'Selling Price (Regular Rate)',
        'Additional Charge (KM, KG, Etc)', 'Return/Delivery Rate', 'Lalamove Bills (Invoicing to Client)',
        'EV Reduction (3PL & KSJ)', 'EV Manpower', 'EV Revenue + Battery (Rental Client)',
        'Claim/COD/Own Risk', 'Hub, COD Fee (SBY) & Service Korlap', 'Other Revenue',
        'Attribute Fee', 'Rider Cost', 'Manpower Cost', 'OEM Cost', 'Mid-Mile/ Linehaul Cost',
        'Add. 3PL Cost', 'DM Program', 'Claim Damaged/Loss', 'Outstanding COD',
        'Claim Ownrisk', 'Attribute Cost', 'HUB Cost', 'Other Cost', 'Total Cost',
        'Delivery Volume', '#Late', '#Late2'
    ]

    for col in currency_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            mask = df[col].isna()
            if mask.any():
                df.loc[mask, col] = df.loc[mask, col].astype(str).str.replace('Rp', '', regex=False).str.replace(' ', '', regex=False).str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    if view_option == "Week (by Year)":
        # Filters
        st.sidebar.header('Filter Data')
        all_years = sorted(df['Year'].unique())
        selected_year = st.sidebar.selectbox('Year', all_years, index=0)

        df_filtered_year = df[df['Year'] == selected_year]

        all_sla_types = ['(All)'] + sorted(df_filtered_year['SLA Type'].unique())
        selected_sla_type = st.sidebar.selectbox('SLA Type', all_sla_types)

        df_filtered_sla = df_filtered_year.copy()
        if selected_sla_type != '(All)':
            df_filtered_sla = df_filtered_sla[df_filtered_sla['SLA Type'] == selected_sla_type]

        all_teams = ['(All)'] + sorted(df_filtered_sla['Blitz Team'].unique())
        selected_team = st.sidebar.selectbox('Blitz Team', all_teams)

        df_filtered_team = df_filtered_sla.copy()
        if selected_team != '(All)':
            df_filtered_team = df_filtered_team[df_filtered_team['Blitz Team'] == selected_team]

        all_locations = ['(All)'] + sorted(df_filtered_team['Client Location'].unique())
        selected_location = st.sidebar.selectbox('Client Location', all_locations)

        df_filtered_location = df_filtered_team.copy()
        if selected_location != '(All)':
            df_filtered_location = df_filtered_location[df_filtered_location['Client Location'] == selected_location]

        all_clients = sorted(df_filtered_location['Client Name'].unique())
        selected_client = st.sidebar.selectbox('Client Name', all_clients)

        if not selected_client:
            st.warning('Please select at least one Client Name to display the dashboard.')
            st.stop()

        df_filtered_client = df_filtered_location[df_filtered_location['Client Name'] == selected_client]
        
        all_projects = ['(All)'] + sorted(df_filtered_client['Project'].unique())
        selected_project = st.sidebar.selectbox('Project', all_projects)

        filtered_df = df_filtered_client.copy()
        if selected_project != '(All)':
            filtered_df = filtered_df[filtered_df['Project'] == selected_project]

        if filtered_df.empty:
            st.warning('Tidak ada data yang sesuai dengan filter yang dipilih.')
            st.stop()

        # Create Pivot Table
        st.subheader('SLA Performance Week-by-Week')

        group_by_cols = ['Client Name', 'Week (by Year)']
        if selected_project != '(All)':
            group_by_cols.append('Project')

        df_grouped = filtered_df.groupby(group_by_cols).agg(
            Delivery_Volume=('Delivery Volume', 'sum'),
            Late_Order=('#Late', 'sum'),
            Late_Order2=('#Late2', 'sum')
        ).reset_index()

        # Calculate Late Order (sum of #Late and #Late2)
        df_grouped['Late_Order'] = df_grouped['Late_Order'] + df_grouped['Late_Order2']
        df_grouped.drop(columns=['Late_Order2'], inplace=True)

        # Calculate On Time Order and SLA Percentage
        df_grouped['On_Time_Order'] = df_grouped['Delivery_Volume'] - df_grouped['Late_Order']
        df_grouped['SLA_Percentage'] = (df_grouped['On_Time_Order'] / df_grouped['Delivery_Volume']).replace([np.inf, -np.inf], np.nan).fillna(0) * 100

        # Calculate totals per Client (and Project if selected)
        total_group_by = ['Client Name']
        if selected_project != '(All)':
            total_group_by.append('Project')

        df_total_client = filtered_df.groupby(total_group_by).agg(
            Delivery_Volume=('Delivery Volume', 'sum'),
            Late_Order=('#Late', 'sum'),
            Late_Order2=('#Late2', 'sum')
        ).reset_index()

        df_total_client['Late_Order'] = df_total_client['Late_Order'] + df_total_client['Late_Order2']
        df_total_client.drop(columns=['Late_Order2'], inplace=True)
        df_total_client['On_Time_Order'] = df_total_client['Delivery_Volume'] - df_total_client['Late_Order']
        df_total_client['SLA_Percentage'] = (df_total_client['On_Time_Order'] / df_total_client['Delivery_Volume']).replace([np.inf, -np.inf], np.nan).fillna(0) * 100

        # Combine weekly and total data with diff rows
        final_df = pd.DataFrame()
        for client in df_total_client['Client Name'].unique():
            if selected_project != '(All)':
                client_total_row = df_total_client[(df_total_client['Client Name'] == client) & (df_total_client['Project'] == selected_project)].copy()
                client_weekly_data = df_grouped[(df_grouped['Client Name'] == client) & (df_grouped['Project'] == selected_project)].copy()
                client_total_row['Row Labels'] = f"{client} - {selected_project}"
                client_weekly_data['Row Labels'] = f"{client} - {selected_project}"
            else:
                client_total_row = df_total_client[df_total_client['Client Name'] == client].copy()
                client_weekly_data = df_grouped[df_grouped['Client Name'] == client].copy()
                client_total_row['Row Labels'] = client
                client_weekly_data['Row Labels'] = client

            client_total_row['Week (by Year)'] = 'Total'
            
            # Sort weekly data by week
            client_weekly_data.sort_values('Week (by Year)', inplace=True)

            # Add diff rows for last three weeks
            if len(client_weekly_data) >= 3:
                last_three_weeks = client_weekly_data.tail(3).copy()
                third_last_week = last_three_weeks.iloc[0]
                second_last_week = last_three_weeks.iloc[1]
                last_week_orig = last_three_weeks.iloc[2]
                diff_data_earlier = {}
                for col in ['Delivery_Volume', 'SLA_Percentage']:
                    val_second_last = second_last_week[col]
                    val_third_last = third_last_week[col]
                    if val_third_last != 0 and not pd.isna(val_third_last):
                        diff_data_earlier[col] = ((val_second_last - val_third_last) / abs(val_third_last)) * 100
                    else:
                        diff_data_earlier[col] = np.nan
                # Explicitly set diff for Late_Order and On_Time_Order to NaN
                diff_data_earlier['Late_Order'] = np.nan
                diff_data_earlier['On_Time_Order'] = np.nan
                diff_row_earlier = pd.Series(diff_data_earlier)
                week_value_earlier = str(second_last_week['Week (by Year)']).strip()
                diff_row_earlier['Row Labels'] = f'Diff W{int(float(week_value_earlier))}%' if week_value_earlier.replace('.', '').replace('-', '').isdigit() else 'Diff WNA%'
                diff_row_earlier['Week (by Year)'] = ''
                for col in client_weekly_data.columns:
                    if col not in diff_row_earlier.index:
                        diff_row_earlier[col] = np.nan
                diff_row_earlier = diff_row_earlier[client_weekly_data.columns]
                insert_idx = client_weekly_data.index[client_weekly_data['Week (by Year)'] == second_last_week['Week (by Year)']].tolist()[-1] + 1
                client_weekly_data = pd.concat([client_weekly_data.iloc[:insert_idx], pd.DataFrame([diff_row_earlier]), client_weekly_data.iloc[insert_idx:]], ignore_index=True)

            if len(client_weekly_data) >= 2:
                last_week = last_week_orig
                second_last_week = client_weekly_data[client_weekly_data['Week (by Year)'] == last_week_orig['Week (by Year)'] - 1].iloc[0] if last_week_orig['Week (by Year)'] > 1 else client_weekly_data.iloc[-2]
                diff_data = {}
                for col in ['Delivery_Volume', 'SLA_Percentage']:
                    val_last = last_week[col]
                    val_second_last = second_last_week[col]
                    if val_second_last != 0 and not pd.isna(val_second_last):
                        diff_data[col] = ((val_last - val_second_last) / abs(val_second_last)) * 100
                    else:
                        diff_data[col] = np.nan
                # Explicitly set diff for Late_Order and On_Time_Order to NaN
                diff_data['Late_Order'] = np.nan
                diff_data['On_Time_Order'] = np.nan
                diff_row = pd.Series(diff_data)
                week_value = str(last_week['Week (by Year)']).strip()
                diff_row['Row Labels'] = f'Diff W{int(float(week_value))}%' if week_value.replace('.', '').replace('-', '').isdigit() else 'Diff WNA%'
                diff_row['Week (by Year)'] = ''
                for col in client_weekly_data.columns:
                    if col not in diff_row.index:
                        diff_row[col] = np.nan
                diff_row = diff_row[client_weekly_data.columns]
                insert_idx_last = client_weekly_data.index[client_weekly_data['Week (by Year)'] == last_week['Week (by Year)']].tolist()[-1] + 1
                client_weekly_data = pd.concat([client_weekly_data.iloc[:insert_idx_last], pd.DataFrame([diff_row]), client_weekly_data.iloc[insert_idx_last:]], ignore_index=True)

            if selected_project != '(All)':
                client_total_row = client_total_row.drop(columns=['Project'])
                client_weekly_data = client_weekly_data.drop(columns=['Project'])
            client_total_row = client_total_row.drop(columns=['Client Name'])
            client_weekly_data = client_weekly_data.drop(columns=['Client Name'])
            
            combined_data = pd.concat([client_total_row, client_weekly_data], ignore_index=True)
            final_df = pd.concat([final_df, combined_data], ignore_index=True)

        # Select and rename columns for display
        final_df = final_df[[
            'Row Labels', 'Week (by Year)', 'Delivery_Volume', 'Late_Order',
            'On_Time_Order', 'SLA_Percentage'
        ]]
        final_df.columns = [
            'Row Labels', 'Week (by Year)', 'Sum of Delivery Volume', 'Sum of Late Order',
            'Sum of On Time Order', 'SLA Percentage'
        ]

        # Styling DataFrame
        def color_rows(row):
            is_total = 'Total' in str(row['Week (by Year)'])
            is_diff = 'Diff W' in str(row['Row Labels'])
            styles = [''] * len(row)
            if is_total:
                styles = ['background-color: #e6ffe6; color: black'] * len(row)
            elif is_diff:
                styles = ['background-color: #fff2e6; color: black'] * len(row)
            return styles

        # Format values for display
        display_df = final_df.copy()
        for col in display_df.columns:
            if col == 'Row Labels':
                display_df[col] = display_df[col].astype(str)
            elif col == 'Week (by Year)':
                display_df[col] = display_df[col].apply(lambda x: f"{int(float(x))}" if pd.notna(x) and str(x).replace('.', '').replace('-', '').isdigit() else x)
            elif col == 'SLA Percentage':
                display_df[col] = display_df.apply(
                    lambda row: f"{row[col]:,.2f}%" if pd.notna(row[col]) else '', axis=1
                )
            else:
                display_df[col] = display_df.apply(
                    lambda row: f"{row[col]:,.2f}%" if pd.notna(row[col]) and 'Diff' in str(row['Row Labels']) else (
                        f"{int(row[col]):,}" if pd.notna(row[col]) else ''
                    ), axis=1
                )
            display_df[col] = display_df[col].astype(str).str.replace('nan', '', regex=False)

        # Display styled dataframe
        st.dataframe(
            display_df.style.apply(color_rows, axis=1), 
            hide_index=True
        )

        # Generate Chart
        st.subheader('Generate Chart')
        chart_columns = [
            'Sum of Delivery Volume', 'Sum of Late Order', 'Sum of On Time Order', 'SLA Percentage'
        ]
        selected_chart_col = st.selectbox('Select Variable for Chart', chart_columns)
        
        if st.button('Generate Chart'):
            chart_df = final_df[final_df['Row Labels'].str.contains(selected_client)]
            chart_df = chart_df[~chart_df['Week (by Year)'].isin(['Total', ''])]
            chart_df['Week (by Year)'] = pd.to_numeric(chart_df['Week (by Year)'], errors='coerce')
            
            if selected_chart_col == 'SLA Percentage':
                fig = px.line(
                    chart_df,
                    x='Week (by Year)',
                    y=selected_chart_col,
                    title=f'{selected_chart_col} for {chart_df["Row Labels"].iloc[0]}',
                    labels={'Week (by Year)': 'Week', selected_chart_col: selected_chart_col}
                )
            else:
                fig = px.bar(
                    chart_df,
                    x='Week (by Year)',
                    y=selected_chart_col,
                    title=f'{selected_chart_col} for {chart_df["Row Labels"].iloc[0]}',
                    labels={'Week (by Year)': 'Week', selected_chart_col: selected_chart_col}
                )
            
            fig.update_layout(
                xaxis_title='Week',
                yaxis_title=selected_chart_col,
                xaxis=dict(tickmode='linear', tick0=1, dtick=1),
                yaxis=dict(zeroline=True, zerolinecolor='black', zerolinewidth=1),
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
    else:
        # Filters
        st.sidebar.header('Filter Data')
        all_years = sorted(df['Year'].unique())
        selected_year = st.sidebar.selectbox('Year', all_years, index=0)

        df_filtered_year = df[df['Year'] == selected_year]

        all_sla_types = ['(All)'] + sorted(df_filtered_year['SLA Type'].unique())
        selected_sla_type = st.sidebar.selectbox('SLA Type', all_sla_types)

        df_filtered_sla = df_filtered_year.copy()
        if selected_sla_type != '(All)':
            df_filtered_sla = df_filtered_sla[df_filtered_sla['SLA Type'] == selected_sla_type]

        all_teams = ['(All)'] + sorted(df_filtered_sla['Blitz Team'].unique())
        selected_team = st.sidebar.selectbox('Blitz Team', all_teams)

        df_filtered_team = df_filtered_sla.copy()
        if selected_team != '(All)':
            df_filtered_team = df_filtered_team[df_filtered_team['Blitz Team'] == selected_team]

        all_locations = ['(All)'] + sorted(df_filtered_team['Client Location'].unique())
        selected_location = st.sidebar.selectbox('Client Location', all_locations)

        df_filtered_location = df_filtered_team.copy()
        if selected_location != '(All)':
            df_filtered_location = df_filtered_location[df_filtered_location['Client Location'] == selected_location]

        all_clients = sorted(df_filtered_location['Client Name'].unique())
        selected_client = st.sidebar.selectbox('Client Name', all_clients)

        if not selected_client:
            st.warning('Please select at least one Client Name to display the dashboard.')
            st.stop()

        df_filtered_client = df_filtered_location[df_filtered_location['Client Name'] == selected_client]
        
        all_projects = ['(All)'] + sorted(df_filtered_client['Project'].unique())
        selected_project = st.sidebar.selectbox('Project', all_projects)

        filtered_df = df_filtered_client.copy()
        if selected_project != '(All)':
            filtered_df = filtered_df[filtered_df['Project'] == selected_project]

        if filtered_df.empty:
            st.warning('Tidak ada data yang sesuai dengan filter yang dipilih.')
            st.stop()

        # Create Pivot Table
        st.subheader('SLA Performance Month-by-Month')

        group_by_cols = ['Client Name', 'Month']
        if selected_project != '(All)':
            group_by_cols.append('Project')

        df_grouped = filtered_df.groupby(group_by_cols).agg(
            Delivery_Volume=('Delivery Volume', 'sum'),
            Late_Order=('#Late', 'sum'),
            Late_Order2=('#Late2', 'sum')
        ).reset_index()

        # Calculate Late Order (sum of #Late and #Late2)
        df_grouped['Late_Order'] = df_grouped['Late_Order'] + df_grouped['Late_Order2']
        df_grouped.drop(columns=['Late_Order2'], inplace=True)

        # Calculate On Time Order and SLA Percentage
        df_grouped['On_Time_Order'] = df_grouped['Delivery_Volume'] - df_grouped['Late_Order']
        df_grouped['SLA_Percentage'] = (df_grouped['On_Time_Order'] / df_grouped['Delivery_Volume']).replace([np.inf, -np.inf], np.nan).fillna(0) * 100

        # Calculate totals per Client (and Project if selected)
        total_group_by = ['Client Name']
        if selected_project != '(All)':
            total_group_by.append('Project')

        df_total_client = filtered_df.groupby(total_group_by).agg(
            Delivery_Volume=('Delivery Volume', 'sum'),
            Late_Order=('#Late', 'sum'),
            Late_Order2=('#Late2', 'sum')
        ).reset_index()

        df_total_client['Late_Order'] = df_total_client['Late_Order'] + df_total_client['Late_Order2']
        df_total_client.drop(columns=['Late_Order2'], inplace=True)
        df_total_client['On_Time_Order'] = df_total_client['Delivery_Volume'] - df_total_client['Late_Order']
        df_total_client['SLA_Percentage'] = (df_total_client['On_Time_Order'] / df_total_client['Delivery_Volume']).replace([np.inf, -np.inf], np.nan).fillna(0) * 100

        # Month order for sorting
        month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                       'July', 'August', 'September', 'October', 'November', 'December']
        df_grouped['Month'] = pd.Categorical(df_grouped['Month'], categories=month_order, ordered=True)
        df_grouped.sort_values('Month', inplace=True)

        # Add user-selected month comparison
        st.sidebar.header('Compare Months')
        available_months = df_grouped['Month'].tolist()
        month_1 = st.sidebar.selectbox('Select First Month', available_months, index=0)
        month_2 = st.sidebar.selectbox('Select Second Month', available_months, index=len(available_months)-1 if len(available_months) > 1 else 0)

        # Combine weekly and total data with diff rows
        final_df = pd.DataFrame()
        for client in df_total_client['Client Name'].unique():
            if selected_project != '(All)':
                client_total_row = df_total_client[(df_total_client['Client Name'] == client) & (df_total_client['Project'] == selected_project)].copy()
                client_monthly_data = df_grouped[(df_grouped['Client Name'] == client) & (df_grouped['Project'] == selected_project)].copy()
                client_total_row['Row Labels'] = f"{client} - {selected_project}"
                client_monthly_data['Row Labels'] = f"{client} - {selected_project}"
            else:
                client_total_row = df_total_client[df_total_client['Client Name'] == client].copy()
                client_monthly_data = df_grouped[df_grouped['Client Name'] == client].copy()
                client_total_row['Row Labels'] = client
                client_monthly_data['Row Labels'] = client

            client_total_row['Month'] = 'Total'
            
            # Sort monthly data by month
            client_monthly_data.sort_values('Month', inplace=True)

            # Add user-selected comparison diff
            if month_1 != month_2 and month_1 in client_monthly_data['Month'].values and month_2 in client_monthly_data['Month'].values:
                month_1_data = client_monthly_data[client_monthly_data['Month'] == month_1].iloc[0]
                month_2_data = client_monthly_data[client_monthly_data['Month'] == month_2].iloc[0]
                diff_data_user = {}
                for col in ['Delivery_Volume', 'SLA_Percentage']:
                    val_month_1 = month_1_data[col]
                    val_month_2 = month_2_data[col]
                    if val_month_1 != 0 and not pd.isna(val_month_1):
                        diff_data_user[col] = ((val_month_2 - val_month_1) / abs(val_month_1)) * 100
                    else:
                        diff_data_user[col] = np.nan
                diff_data_user['Late_Order'] = np.nan
                diff_data_user['On_Time_Order'] = np.nan
                diff_row_user = pd.Series(diff_data_user)
                diff_row_user['Row Labels'] = f'Diff {month_1} to {month_2}%'
                diff_row_user['Month'] = ''
                for col in client_monthly_data.columns:
                    if col not in diff_row_user.index:
                        diff_row_user[col] = np.nan
                diff_row_user = diff_row_user[client_monthly_data.columns]
                client_monthly_data = pd.concat([client_monthly_data, pd.DataFrame([diff_row_user])], ignore_index=True)

            if selected_project != '(All)':
                client_total_row = client_total_row.drop(columns=['Project'])
                client_monthly_data = client_monthly_data.drop(columns=['Project'])
            client_total_row = client_total_row.drop(columns=['Client Name'])
            client_monthly_data = client_monthly_data.drop(columns=['Client Name'])
            
            combined_data = pd.concat([client_total_row, client_monthly_data], ignore_index=True)
            final_df = pd.concat([final_df, combined_data], ignore_index=True)

        # Select and rename columns for display
        final_df = final_df[[
            'Row Labels', 'Month', 'Delivery_Volume', 'Late_Order',
            'On_Time_Order', 'SLA_Percentage'
        ]]
        final_df.columns = [
            'Row Labels', 'Month', 'Sum of Delivery Volume', 'Sum of Late Order',
            'Sum of On Time Order', 'SLA Percentage'
        ]

        # Styling DataFrame
        def color_rows(row):
            is_total = 'Total' in str(row['Month'])
            is_diff = 'Diff' in str(row['Row Labels'])
            styles = [''] * len(row)
            if is_total:
                styles = ['background-color: #e6ffe6; color: black'] * len(row)
            elif is_diff:
                styles = ['background-color: #fff2e6; color: black'] * len(row)
            return styles

        # Format values for display
        display_df = final_df.copy()
        for col in display_df.columns:
            if col == 'Row Labels':
                display_df[col] = display_df[col].astype(str)
            elif col == 'Month':
                display_df[col] = display_df[col].astype(str)
            elif col == 'SLA Percentage':
                display_df[col] = display_df.apply(
                    lambda row: f"{row[col]:,.2f}%" if pd.notna(row[col]) else '', axis=1
                )
            else:
                display_df[col] = display_df.apply(
                    lambda row: f"{row[col]:,.2f}%" if pd.notna(row[col]) and 'Diff' in str(row['Row Labels']) else (
                        f"{int(row[col]):,}" if pd.notna(row[col]) else ''
                    ), axis=1
                )
            display_df[col] = display_df[col].astype(str).str.replace('nan', '', regex=False)

        # Display styled dataframe
        st.dataframe(
            display_df.style.apply(color_rows, axis=1), 
            hide_index=True
        )

        # Generate Chart
        st.subheader('Generate Chart')
        chart_columns = [
            'Sum of Delivery Volume', 'Sum of Late Order', 'Sum of On Time Order', 'SLA Percentage'
        ]
        selected_chart_col = st.selectbox('Select Variable for Chart', chart_columns)
        
        if st.button('Generate Chart'):
            chart_df = final_df[final_df['Row Labels'].str.contains(selected_client, na=False)]
            chart_df = chart_df[~chart_df['Month'].isin(['Total', ''])]
            chart_df['Month'] = pd.Categorical(chart_df['Month'], categories=month_order, ordered=True)
            chart_df = chart_df.sort_values('Month')
            
            if selected_chart_col == 'SLA Percentage':
                fig = px.line(
                    chart_df,
                    x='Month',
                    y=selected_chart_col,
                    title=f'{selected_chart_col} for {chart_df["Row Labels"].iloc[0]}',
                    labels={'Month': 'Month', selected_chart_col: selected_chart_col}
                )
            else:
                fig = px.bar(
                    chart_df,
                    x='Month',
                    y=selected_chart_col,
                    title=f'{selected_chart_col} for {chart_df["Row Labels"].iloc[0]}',
                    labels={'Month': 'Month', selected_chart_col: selected_chart_col}
                )
            
            fig.update_layout(
                xaxis_title='Month',
                yaxis_title=selected_chart_col,
                xaxis=dict(tickmode='array', tickvals=month_order),
                yaxis=dict(zeroline=True, zerolinecolor='black', zerolinewidth=1),
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
# Main logic to switch between pages
if st.session_state['page'] == 'login':
    login_page()
elif st.session_state['page'] == 'home':
    home_page()
elif st.session_state['page'] == 'dashboard':
    main_margin_page()
elif st.session_state['page'] == 'detailed_dashboard':
    detailed_dashboard_page()
elif st.session_state['page'] == 'wow_performance':
    wow_performance_page()
elif st.session_state['page'] == 'monthly_performance':
    monthly_performance_page()
elif st.session_state['page'] == 'sla_calculation':
    sla_calculation_page()