import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import re

def get_numeric_columns(df):
    return df.select_dtypes(include=[np.number]).columns.tolist()

def sanitize_column_name(column_name):
    return f"`{column_name}`"

def format_condition(condition):
    # Split the condition into column name and the rest
    match = re.match(r'(\S+)\s*(.*)', condition)
    if match:
        column, rest = match.groups()
        return f"{sanitize_column_name(column)} {rest}"
    return condition

def get_categorical_columns(df):
    return df.select_dtypes(include=['object', 'category']).columns.tolist()

def apply_advanced_filter(df, filter_conditions):
    try:
        return df.query(' and '.join(filter_conditions))
    except Exception as e:
        st.error(f"Error applying filter: {str(e)}")
        return df

def main():
    st.title("🧹 Data Cleaning App")
    
    st.sidebar.header("Data Cleaning App by: Niño Garcia")
    st.sidebar.subheader("Contact Details:")
    st.sidebar.markdown("[LinkedIn](https://www.linkedin.com/in/ninogarci/)")
    st.sidebar.markdown("[Upwork](https://www.upwork.com/freelancers/~01dd78612ac234aadd)")

    # Initialize session state
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'column_operations' not in st.session_state:
        st.session_state.column_operations = []
    if 'renamed_df' not in st.session_state:
        st.session_state.renamed_df = None

    # File upload
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        # Read the CSV file
        st.session_state.df = pd.read_csv(uploaded_file)
        
        # Use renamed_df if it exists, otherwise use the original df
        df = st.session_state.renamed_df if st.session_state.renamed_df is not None else st.session_state.df
        
        # Display basic information about the dataset
        with st.expander("📊 Dataset Information", expanded=True):
            st.write(f"Shape: {df.shape}")
            st.write(f"Columns: {', '.join(df.columns)}")
        
        # Column selection
        with st.expander("🔍 Column Selection", expanded=True):
            selected_columns = st.multiselect("Select columns to clean", df.columns, default=df.columns)
        
        if not selected_columns:
            st.warning("Please select at least one column to proceed with cleaning.")
            return
        
        # Filter the dataframe to include only selected columns
        df = df[selected_columns]
        
        # Display the first few rows of the filtered dataset
        with st.expander("👀 Data Preview", expanded=True):
            st.write(df.head())
        
        # Data type conversion
        with st.expander("🔄 Data Type Conversion", expanded=False):
            for column in df.columns:
                current_type = df[column].dtype
                col_type = st.selectbox(f"Select data type for {column} (current: {current_type})", 
                                        ["Keep current", "numeric", "datetime", "categorical"])
                if col_type == "numeric" and current_type != 'numeric':
                    df[column] = pd.to_numeric(df[column], errors='coerce')
                elif col_type == "datetime" and current_type != 'datetime64[ns]':
                    df[column] = pd.to_datetime(df[column], errors='coerce')
                elif col_type == "categorical" and current_type != 'category':
                    df[column] = df[column].astype('category')
        
        # Handle null values
        with st.expander("🧹 Handle Null Values", expanded=False):
            null_columns = df.columns[df.isnull().any()].tolist()
            
            if null_columns:
                st.write("Columns with null values:", null_columns)
                for col in null_columns:
                    st.write(f"Handling null values in column: {col}")
                    method = st.selectbox(f"Choose method for {col}", ["Ignore", "Drop", "Fill with Mean", "Fill with Median", "Fill with Mode"])
                    
                    if method == "Drop":
                        df = df.dropna(subset=[col])
                    elif method == "Fill with Mean" and pd.api.types.is_numeric_dtype(df[col]):
                        df[col].fillna(df[col].mean(), inplace=True)
                    elif method == "Fill with Median" and pd.api.types.is_numeric_dtype(df[col]):
                        df[col].fillna(df[col].median(), inplace=True)
                    elif method == "Fill with Mode":
                        df[col].fillna(df[col].mode()[0], inplace=True)
                    elif method == "Ignore":
                        pass
                    else:
                        st.warning(f"Method {method} not applicable for non-numeric column {col}. Ignoring.")
            else:
                st.write("No null values found in the dataset.")
        
        # Delete duplicate rows
        with st.expander("🗑️ Delete Duplicate Rows", expanded=False):
            num_duplicates = df.duplicated().sum()
            st.write(f"Number of duplicate rows: {num_duplicates}")
            if num_duplicates > 0:
                if st.button("Remove Duplicate Rows"):
                    df = df.drop_duplicates()
                    st.write("Duplicate rows removed.")
        
        # Outlier detection and handling
        with st.expander("📉 Outlier Detection and Handling", expanded=False):
            numeric_columns = get_numeric_columns(df)
            for col in numeric_columns:
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - (1.5 * iqr)
                upper_bound = q3 + (1.5 * iqr)
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                if not outliers.empty:
                    st.write(f"Outliers detected in {col}")
                    st.write(f"Number of outliers: {len(outliers)}")
                    st.write(f"Outlier range: < {lower_bound:.2f} or > {upper_bound:.2f}")
                    st.write("Sample of outliers:")
                    st.write(outliers[[col]].head())
                    
                    action = st.selectbox(f"Choose action for outliers in {col}", ["Keep", "Remove", "Cap"])
                    if action == "Remove":
                        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
                        st.write(f"Outliers removed from {col}")
                    elif action == "Cap":
                        df[col] = df[col].clip(lower_bound, upper_bound)
                        st.write(f"Outliers capped in {col}")
        
        # Column operations
        with st.expander("🔧 Column Operations", expanded=True):
            operation = st.selectbox("Select operation", ["Rename column", "Create new column", "Search and Replace"])
            if operation == "Rename column":
                old_name = st.selectbox("Select column to rename", df.columns)
                new_name = st.text_input("Enter new column name")
                if st.button("Rename"):
                    df = df.rename(columns={old_name: new_name})
                    st.session_state.column_operations.append(('rename', old_name, new_name))
                    st.session_state.renamed_df = df  # Update the renamed DataFrame in session state
                    st.write("Column renamed successfully")
            elif operation == "Create new column":
                new_col_name = st.text_input("Enter new column name")
                expression = st.text_input("Enter Python expression (e.g., df['A'] + df['B'])")
                if st.button("Create"):
                    try:
                        df[new_col_name] = eval(expression)
                        st.session_state.column_operations.append(('create', new_col_name, expression))
                        st.session_state.renamed_df = df  # Update the renamed DataFrame in session state
                        st.write("New column created successfully")
                    except Exception as e:
                        st.write(f"Error creating column: {str(e)}")
            elif operation == "Search and Replace":
                column = st.selectbox("Select column to search in", df.columns)
                search_term = st.text_input("Find what:")
                replace_term = st.text_input("Replace with:")
                if st.button("Apply Search and Replace"):
                    if column in df.columns:
                        # Convert the column to string type
                        df[column] = df[column].astype(str)
                        # Use str.replace() instead of str.contains()
                        df[column] = df[column].str.replace(search_term, replace_term, regex=False)
                        num_replaced = df[column].str.count(replace_term).sum()
                        st.session_state.column_operations.append(('search_replace', column, search_term, replace_term))
                        st.session_state.renamed_df = df  # Update the DataFrame in session state
                        st.write(f"Replaced {num_replaced} occurrences of '{search_term}' with '{replace_term}' in column '{column}'")
                    else:
                        st.write(f"Column '{column}' not found in the dataset.")

        # Advanced Filtering
        with st.expander("🔍 Advanced Filtering", expanded=False):
            st.write("Apply advanced filters to your data.")
            st.write("Example: column > 5, column == 'Value', column.isin(['A', 'B'])")
            
            filter_conditions = []
            
            col1, col2 = st.columns(2)
            
            with col1:
                num_conditions = st.number_input("Number of filter conditions", min_value=1, max_value=5, value=1)
            
            with col2:
                combine_method = st.selectbox("Combine conditions using", ["AND", "OR"])
            
            for i in range(int(num_conditions)):
                st.markdown("---")  # Add a horizontal line
                st.markdown(f"### Condition {i+1}")  # Use header formatting
                
                col = st.selectbox(f"Column for condition {i+1}", df.columns, key=f"col_select_{i}")
                operation = st.selectbox(f"Operation for {col}", 
                                        ["==", "!=", ">", "<", ">=", "<=", "in", "not in", "contains", "not contains"],
                                        key=f"op_select_{i}")
                
                if operation in ["in", "not in"]:
                    value = st.text_input(f"Value(s) for {col} (comma-separated for multiple values)", key=f"value_input_{i}")
                    condition = f"{sanitize_column_name(col)} {operation} {value.split(',')}"
                elif operation in ["contains", "not contains"]:
                    value = st.text_input(f"Value for {col}", key=f"value_input_{i}")
                    contains_op = "contains" if operation == "contains" else "not contains"
                    condition = f"{sanitize_column_name(col)}.str.{contains_op}('{value}')"
                else:
                    value = st.text_input(f"Value for {col}", key=f"value_input_{i}")
                    condition = f"{sanitize_column_name(col)} {operation} '{value}'"
                
                if condition:
                    filter_conditions.append(condition)
            
            if st.button("Apply Filters"):
                if filter_conditions:
                    combine_operator = ' & ' if combine_method == "AND" else ' | '
                    query_string = combine_operator.join(filter_conditions)
                    try:
                        df_filtered = df.query(query_string)
                        st.write(f"Filtered DataFrame (showing first 10 rows):")
                        st.write(df_filtered.head(10))
                        st.write(f"Number of rows after filtering: {len(df_filtered)}")
                        
                        # Option to update the main dataframe
                        if st.button("Update Main DataFrame with Filtered Data"):
                            df = df_filtered
                            st.session_state.renamed_df = df
                            st.success("Main DataFrame updated with filtered data.")
                    except Exception as e:
                        st.error(f"Error applying filter: {str(e)}")
                else:
                    st.warning("No filter conditions specified.")
                        
        # Data sampling
        with st.expander("🎲 Data Sampling", expanded=True):
            sample_size = st.slider("Select sample size", 1, len(df), min(10, len(df)))
            df_sample = df.sample(sample_size)
            st.write(f"Random sample of {sample_size} rows:")
            st.write(df_sample)

        # Data export
        with st.expander("💾 Data Export", expanded=True):
            if st.button("Download cleaned data as CSV"):
                csv = df.to_csv(index=False)
                b64 = BytesIO(csv.encode()).getvalue()
                st.download_button(
                    label="Download CSV",
                    data=b64,
                    file_name="cleaned_data.csv",
                    mime="text/csv",
                )

if __name__ == "__main__":
    main()