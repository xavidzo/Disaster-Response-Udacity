import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """loads messages and categories files from specified paths
    Args:
        messages_filepath (string): The file path for messages.csv
        categories_filepath (string): The file path for categories.csv
    Returns:
        df (pandas dataframe): df with merged messages and categories
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on='id')
    return df


def clean_data(df):
    """cleans the dataframe df by assigning category columns with meaningful names,
        values for each category are set to the last digit, and duplicates are dropped.
    Args:
        df (pandas dataframe): df from previous step

    Returns:
        df (pandas dataframe): df with cleaned data
    """
    categories = df['categories'].str.split(pat=';', expand=True)
    first_row = categories[:1]

    # Extract column names from the categories present in the first row
    category_colnames = first_row.applymap(lambda cname: cname[:-2]).iloc[0, :].tolist()
    categories.columns = category_colnames
    
    # Put only '0' or '1' as integer values inside the categories dataframe, thus removing the category prefix
    for column in categories:
        categories[column] = categories[column].astype(str).str[-1:]
        categories[column] = categories[column].astype(int)
        

    # Concatenate the original df with the new categories data
    df.drop(['categories'], axis=1, inplace=True)
    df = pd.concat([df,categories], axis=1)

    # Remove duplicate entries
    df.drop_duplicates(inplace=True)
    
    return df


def save_data(df, database_filename):
    """Saves the dataframe from previous step into a database using sqlite
    Args:
        df (pandas dataframe): df with cleaned data
        database_filename (string): the file path pointing to the database
    Returns:
        None
    """
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('disaster_response', engine, index=False)
    engine.dispose()

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()