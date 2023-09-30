import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from recipe_scrapers import scrape_me
from deep_translator import GoogleTranslator
from fuzzywuzzy import process
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict

MEASUREMENT_SYMBOLS = {'¼': '0.25', '½': '0.5', '¾': '0.75'}
FOODS_JSON_PATH = 'foods.json'
TRAINED_MODEL_PATH = 'trained_SVC_model.joblib'
VECTORIZER_PATH = 'vectorizer.joblib'


def replace_measurement_symbols(ingredients):
    return [
        ingredient.replace(symbol, replacement)
        for ingredient in ingredients
        for symbol, replacement in MEASUREMENT_SYMBOLS.items()
    ]


def translate_ingredients(df):
    ing_list = df['Ingredients'].tolist()
    ing_list_translated = GoogleTranslator(source='auto', target='en').translate_batch(ing_list)
    df['Ingredient_Translated'] = ing_list_translated
    return df


def map_food_category(df):
    with open(FOODS_JSON_PATH) as f:
        foods_json = json.load(f)

    df_foods = pd.DataFrame([(category, food) for category, foods in foods_json.items() for food in foods], columns=['Category', 'Food'])
    df['match'] = df['Ingredient_Translated'].apply(lambda x: process.extractOne(x, df_foods['Food'].tolist(), score_cutoff=90)[0] if process.extractOne(x, df_foods['Food'].tolist(), score_cutoff=90) else 'Unknown')

    mapping = df_foods.set_index('Food').to_dict()['Category']
    df['Category'] = df['match'].map(mapping)

    classifier = joblib.load(TRAINED_MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    missing_data = df.loc[df['Category'].isnull(), 'Ingredient_Translated']

    if not missing_data.empty:
        ingredients_vectorized = vectorizer.transform(missing_data.tolist())
        predictions = classifier.predict(ingredients_vectorized)
        df.loc[df['Category'].isnull(), 'Category'] = predictions

    df['Category'].fillna('Unknown', inplace=True)
    df.drop(columns=['Ingredient_Translated', 'match'], inplace=True)

    return df


def create_df(recipes):
    df_list = []

    for recipe in recipes:
        scraper = scrape_me(recipe)
        recipe_details = replace_measurement_symbols(scraper.ingredients())
        recipe_name = recipe.split('/')[-1]

        for ingredient in recipe_details:
            try:
                df_temp = pd.DataFrame(columns=['Ingredients', 'Measurement'])
                df_temp[str(recipe_name)] = recipe_name
                # Assume all the recipes have ingredients in a 2 * measurement * item format
                quantity, measurement, item = ingredient.split()[0:3]
                df_temp.loc[len(df_temp)] = [item, measurement, float(quantity)]
                df_list.append(df_temp)
            except (ValueError, IndexError):
                pass

    df = pd.concat(df_list, ignore_index=True)
    df = translate_ingredients(df)
    df = map_food_category(df)
    df.sort_values(by='Category', inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


def main():
    st.title("Recipe Scraper")
    st.image("https://images.unsplash.com/photo-1577308856961-8e9ec50d0c67", caption="What's for dinner?")
    st.write("[See this app's GitHub ReadMe file for more info](https://github.com/jrodriguez5909/RecipeScraper#top-daily-stock-losers--trading-opportunities)")
    st.write("## **App Instructions:**")
    st.write("""
    1. Populate the text box below with URLs for recipes you'd like to gather ingredient from - separate the URLs with commas e.g., https://www.hellofresh.nl/recipes/chicken-parmigiana-623c51bd7ed5c074f51bbb10, https://www.hellofresh.nl/recipes/quiche-met-broccoli-en-oude-kaas-628665b01dea7b8f5009b248
    2. Click **Grab ingredient list** to kick off the web scraping and creation of the ingredient shopping list dataset.
    3. Click **Download full csv file** link below if you'd like to download the ingredient shopping list dataset as a csv file.
    """)

    recs = st.text_area('', height=50)
    download = st.button('Grab ingredient list')

    if download:
        st.info('App is running, please wait...')
        recs = recs.split(",")
        df_download = create_df(recs)
        csv = df_download.to_csv(index=False)
        st.download_button("Download Ingredients List", csv, "ingredients_list.csv")
        st.write("## Processed Data")
        st.dataframe(df_download)

if __name__ == "__main__":
    main()
