import streamlit as st
import base64
import pandas as pd
import numpy as np
import json
import joblib

from time import sleep
from stqdm import stqdm
from recipe_scrapers import scrape_me
from stqdm import stqdm
from deep_translator import GoogleTranslator
from langdetect import detect
from fuzzywuzzy import process


def replace_measurement_symbols(ingredients):
    """
    Description:
    Converts measurement symbols to numbers that will later serve as floats

    Arguments:
    * ingredients: this is the ingredient list object
    """
    ingredients = [i.replace('¼', '0.25') for i in ingredients]
    ingredients = [i.replace('½', '0.5') for i in ingredients]
    ingredients = [i.replace('¾', '0.75') for i in ingredients]

    return ingredients


def justify(a, invalid_val=0, axis=1, side='left'):
    """
    Description:
    Justifies a 2D array i.e., this is used in create_df() below to merge all ingredient rows under each recipe column to eliminate NaN duplicates.

    Arguments:
    A : ndarray
        Input array to be justified
    axis : int
        Axis along which justification is to be made
    side : str
        Direction of justification. It could be 'left', 'right', 'up', 'down'
        It should be 'left' or 'right' for axis=1 and 'up' or 'down' for axis=0.
    """

    if invalid_val is np.nan:
        mask = pd.notnull(a)
    else:
        mask = a != invalid_val
    justified_mask = np.sort(mask, axis=axis)
    if (side == 'up') | (side == 'left'):
        justified_mask = np.flip(justified_mask, axis=axis)
    out = np.full(a.shape, invalid_val, dtype=object)
    if axis == 1:
        out[justified_mask] = a[mask]
    else:
        out.T[justified_mask.T] = a.T[mask.T]
    return out


def translate_ingredients(df):
    """
    Description:
    • Translates foods under "Ingredients" column from foreign language to English to prepare for food categorization

    Arguments:
    • df: this is the df created from scraping within the create_df function
    """
    ing_list = df['Ingredients'].to_list()
    ing_list_translated = GoogleTranslator(source=detect(', '.join(ing_list)), target='en').translate_batch(
        ing_list)  # Brings all the ingredients into a corpus to best detect the language
    df['Ingredient_Translated'] = ing_list_translated

    return df


def map_food_category(df):
    # fuzzywuzzy match
    score_cutoff = 90  # the higher this is, the more strict fuzzywuzzy is in looking for a match

    with open('foods.json') as f:
        foods_json = json.load(f)

    df_foods = pd.DataFrame(columns=['Category', 'Food'])

    for category, foods in foods_json.items():
        for food in foods:
            df_foods = df_foods.append({'Category': category, 'Food': food}, ignore_index=True)

    df['match'] = df['Ingredient_Translated'].apply(
        lambda x: process.extractOne(x, df_foods['Food'].tolist(), score_cutoff=score_cutoff)[0] if process.extractOne(
            x, df_foods['Food'].tolist(), score_cutoff=score_cutoff) else 'Unknown'
    )

    # create mapping
    mapping = df_foods.set_index('Food').to_dict()['Category']
    df['Category'] = df['match'].map(mapping)

    # fill missing values with classifier
    classifier = joblib.load('trained_SVC_model.joblib')
    missing_mask = df['Category'].isnull()
    missing_data = df[missing_mask].drop(['Category', 'match'], axis=1)

    # Vectorize the ingredients using the same bag of words representation
    vectorizer = joblib.load('vectorizer.joblib')
    ingredients_vectorized = vectorizer.transform(missing_data['Ingredient_Translated'].to_list())

    # make predictions using the classifier
    predictions = classifier.predict(ingredients_vectorized)
    df.loc[missing_mask, 'Category'] = predictions

    # fill remaining missing values with 'Unknown'
    df.loc[df['Category'].isnull(), 'Category'] = 'Unknown'
    cols = ['Ingredient_Translated', 'match']
    df = df.drop(columns=cols, axis=1)

    return df


def create_df(recipes, num_people=1):
    """
    Description:
    Creates one df with all recipes and their ingredients

    Arguments:
    * recipes: list of recipe URLs provided by user

    Comments:
    Note that ingredients with qualitative amounts e.g., "scheutje melk", "snufje zout" have been ommitted from the ingredient list
    """
    df_list = []

    for recipe in stqdm(recipes):
        scraper = scrape_me(recipe)
        recipe_details = replace_measurement_symbols(scraper.ingredients())

        recipe_name = recipe.split("https://www.hellofresh.nl/recipes/", 1)[1]
        recipe_name = recipe_name.rsplit('-', 1)[0]
        print("Processing data for " + recipe_name + " recipe.")

        for ingredient in recipe_details:
            try:
                df_temp = pd.DataFrame(columns=['Ingredients', 'Measurement'])
                df_temp[str(recipe_name)] = recipe_name

                ing_1 = ingredient.split("2 * ", 1)[1]
                ing_1 = ing_1.split(" ", 2)

                item = ing_1[2]
                measurement = ing_1[1]
                quantity = float(ing_1[0])/2 * num_people

                df_temp.loc[len(df_temp)] = [item, measurement, quantity]
                df_list.append(df_temp)
            except (ValueError, IndexError) as e:
                pass

        df = pd.concat(df_list)

    print(
        "Renaming duplicate ingredients e.g., Kruimige aardappelen, Voorgekookte halve kriel met schil -> Aardappelen")
    ingredient_dict = {
        'Aardappelen': ('Dunne frieten', 'Half kruimige aardappelen', 'Voorgekookte halve kriel met schil',
                        'Kruimige aardappelen', 'Roodschillige aardappelen', 'Opperdoezer Ronde aardappelen'),
        'Ui': ('Rode ui'),
        'Kipfilet': ('Kipfilet met tuinkruiden en knoflook'),
        'Kipworst': ('Gekruide kipworst'),
        'Kipgehakt': (
        'Gemengd gekruid gehakt', 'Kipgehakt met Mexicaanse kruiden', 'Half-om-halfgehakt met Italiaanse kruiden',
        'Kipgehakt met tuinkruiden'),
        'Kipshoarma': ('Kalkoenshoarma')
    }

    reverse_label_ing = {x: k for k, v in ingredient_dict.items() for x in (v if isinstance(v, tuple) else (v,))}
    df["Ingredients"].replace(reverse_label_ing, inplace=True)

    print("Assigning ingredient categories")

    # Read food category JSON file
    with open('foods.json') as f:
        category_dict = json.load(f)

    reverse_label_cat = {x: k for k, v in category_dict.items() for x in v}
    df["Category"] = df["Ingredients"].map(reverse_label_cat)
    col = "Category"
    first_col = df.pop(col)
    df.insert(0, col, first_col)
    df = df.sort_values(['Category', 'Ingredients'], ascending=[True, True])

    print("Merging ingredients by row across all recipe columns using justify()")
    gp_cols = ['Ingredients', 'Measurement']
    oth_cols = df.columns.difference(gp_cols)

    arr = np.vstack(df.groupby(gp_cols, sort=False, dropna=False).apply(
        lambda gp: justify(gp.to_numpy(), invalid_val=np.NaN, axis=0, side='up')))

    # Reconstruct DataFrame
    # Remove entirely NaN rows based on the non-grouping columns
    res = (pd.DataFrame(arr, columns=df.columns)
           .dropna(how='all', subset=oth_cols, axis=0))

    res = res.fillna(0)
    res['Total'] = res.drop(['Ingredients', 'Measurement'], axis=1).sum(axis=1)
    res = res[res['Total'] != 0]  # To drop rows that are being duplicated with 0 for some reason; will check later

    # Place "Total" column towards front
    col = "Total"
    first_col = res.pop(col)
    res.insert(3, col, first_col)
    res = res.reset_index(drop=True)

    print("Translating foods to prepare for food category mapping")

    res = translate_ingredients(df=res)

    print("Mapping foods to categories")

    res = map_food_category(df=res)
    res = res.sort_values(by='Category').reset_index(drop=True)

    print("Processing complete!")

    return res


def main():
    recipes = [
        'https://www.hellofresh.nl/recipes/luxe-burger-met-truffeltapenade-en-portobello-63ad875558b39f3da6083acd',
        'https://www.hellofresh.nl/recipes/chicken-parmigiana-623c51bd7ed5c074f51bbb10',
        'https://www.hellofresh.nl/recipes/quiche-met-broccoli-en-oude-kaas-628665b01dea7b8f5009b248'
    ]

    df = create_df(recipes)
    df.to_csv('all_recipes.csv')

    return df


st.write("""# Recipe Scraper""")
st.image(
    "https://images.unsplash.com/photo-1577308856961-8e9ec50d0c67?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxzZWFyY2h8Mnx8Zm9vZCUyMG9uJTIwdGFibGV8ZW58MHx8MHx8&auto=format&fit=crop&w=700&q=60",
    caption="What's for dinner?")
st.write(
    "[See this app's GitHub ReadMe file for more info](%s)" % "https://github.com/jrodriguez5909/RecipeScraper#top-daily-stock-losers--trading-opportunities")
st.write("""
## **App Instructions:**
1. Determine the amount of people (servings) you're cooking for. 
2. Populate the text box below with URLs for recipes you'd like to gather ingredient from - separate the URLs with commas e.g., https://www.hellofresh.nl/recipes/chicken-parmigiana-623c51bd7ed5c074f51bbb10, https://www.hellofresh.nl/recipes/quiche-met-broccoli-en-oude-kaas-628665b01dea7b8f5009b248
3. Click **Grab ingredient list** to kick off the web scraping and creation of the ingredient shopping list dataset.
4. Click **Download full csv file** link below if you'd like to download the ingredient shopping list dataset as a csv file.
""")

num_people = st.number_input("Number of people you're cooking for:", min_value=1, max_value=20, value=1)

recs = st.text_area("Your recipe URLs separated by commas per above instruction:", height=50)

download = st.button('Grab ingredient list')  # type="primary" giving issues for some reason

if download:
    st.info('App is running, please wait...')
    recs = recs.split(",")
    df_download = create_df(recs, num_people)
    csv = df_download.to_csv()
    b64 = base64.b64encode(csv.encode()).decode()  # some strings
    linko = f'<a href="data:file/csv;base64,{b64}" download="Ingredients_list.csv">Download full csv file</a>'
    st.markdown(linko, unsafe_allow_html=True)
    st.warning('⚠ Ingredient quantities follow serving size shown in the URL you provide so be mindful of this!')
    st.write("""
    • Ingredient shopping list below and available in csv format when clicking "Download full csv file" URL above:
    """)
    st.dataframe(df_download)
    st.balloons()