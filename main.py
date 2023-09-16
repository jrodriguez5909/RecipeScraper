import base64
import pandas as pd
import numpy as np

from recipe_scrapers import scrape_me
from stqdm import stqdm

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
    Justifies a 2D array i.e., this is used in create_df() below to merge all ingredient rows under each recipe column to eliminate NaN duplicates. 

    Parameters
    ----------
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
        mask = a!=invalid_val
    justified_mask = np.sort(mask,axis=axis)
    if (side=='up') | (side=='left'):
        justified_mask = np.flip(justified_mask,axis=axis)
    out = np.full(a.shape, invalid_val, dtype=object) 
    if axis==1:
        out[justified_mask] = a[mask]
    else:
        out.T[justified_mask.T] = a.T[mask.T]
    return out


def create_df(recipes):
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
        print("Processing data for "+ recipe_name +" recipe.")

        for ingredient in recipe_details:
            try:
                df_temp = pd.DataFrame(columns=['Ingredients', 'Measurement'])
                df_temp[str(recipe_name)] = recipe_name

                ing_1 = ingredient.split("2 * ", 1)[1]
                ing_1 = ing_1.split(" ", 2)

                item = ing_1[2]
                measurement = ing_1[1]
                quantity = float(ing_1[0]) * 2

                df_temp.loc[len(df_temp)] = [item, measurement, quantity]
                df_list.append(df_temp)
            except (ValueError, IndexError) as e:
                pass

        df = pd.concat(df_list)

    print("Renaming duplicate ingredients e.g., Kruimige aardappelen, Voorgekookte halve kriel met schil -> Aardappelen")
    ingredient_dict = {
                    'Aardappelen': ('Dunne frieten', 'Half kruimige aardappelen', 'Voorgekookte halve kriel met schil',
                                    'Kruimige aardappelen', 'Roodschillige aardappelen', 'Opperdoezer Ronde aardappelen'),
                    'Ui': ('Rode ui'),
                    'Kipfilet': ('Kipfilet met tuinkruiden en knoflook'),
                    'Kipworst': ('Gekruide kipworst'),
                    'Kipgehakt': ('Gemengd gekruid gehakt', 'Kipgehakt met Mexicaanse kruiden', 'Half-om-halfgehakt met Italiaanse kruiden',
                                  'Kipgehakt met tuinkruiden'),
                    'Kipshoarma': ('Kalkoenshoarma')
                    }

    # reverse_label_ing = {x:k for k,v in ingredient_dict.items() for x in v}
    reverse_label_ing = {x: k for k, v in ingredient_dict.items() for x in (v if isinstance(v, tuple) else (v,))}
    df["Ingredients"].replace(reverse_label_ing, inplace=True)

    print("Assigning ingredient categories")
    category_dict = {
                    'brood': ('Biologisch wit rozenbroodje', 'Bladerdeeg', 'Briochebroodje', 'Wit platbrood'),
                    'granen': ('Basmatirijst', 'Bulgur', 'Casarecce', 'Cashewstukjes',
                               'Gesneden snijbonen', 'Jasmijnrijst', 'Linzen', 'Ma√Øs in blik',
                               'Parelcouscous', 'Penne', 'Rigatoni', 'Rode kidneybonen',
                               'Spaghetti', 'Witte tortilla'),
                    'groenten': ('Aardappelen', 'Aubergine', 'Bosui', 'Broccoli', 'Ui',
                                 'Champignons', 'Citroen', 'Gele wortel', 'Gesneden rodekool',
                                 'Groene paprika', 'Groentemix van paprika, prei, gele wortel en courgette',
                                 'IJsbergsla', 'Kumato tomaat', 'Limoen', 'Little gem',
                                 'Paprika', 'Portobello', 'Prei', 'Pruimtomaat', 'Knoflookteen',
                                 'Radicchio en ijsbergsla', 'Rode cherrytomaten', 'Rode paprika', 'Rode peper',
                                 'Rode puntpaprika', 'Ui', 'Rucola', 'Rucola en veldsla', 'Rucolamelange',
                                 'Semi-gedroogde tomatenmix', 'Sjalot', 'Sperziebonen', 'Spinazie', 'Tomaat',
                                 'Turkse groene peper', 'Veldsla', 'Vers basilicum', 'Verse bieslook',
                                 'Verse bladpeterselie', 'Verse koriander', 'Verse krulpeterselie', 'Wortel', 'Zoete aardappel'),
                    'kruiden': ('A√Øoli', 'Bloem', 'Bruine suiker', 'Cranberrychutney', 'Extra vierge olijfolie',
                                'Extra vierge olijfolie met truffelaroma', 'Fles olijfolie', 'Gedroogde laos',
                                'Gedroogde oregano', 'Gemalen kaneel', 'Gemalen komijnzaad', 'Gemalen korianderzaad',
                                'Gemalen kurkuma', 'Gerookt paprikapoeder', 'Groene currykruiden', 'Groentebouillon',
                                'Groentebouillonblokje', 'Honing', 'Italiaanse kruiden', 'Kippenbouillonblokje',
                                'Kokosmelk', 'Koreaanse kruidenmix', 'Mayonaise', 'Mexicaanse kruiden', 'Midden-Oosterse kruidenmix',
                                'Mosterd', 'Nootmuskaat', 'Olijfolie', 'Panko paneermeel', 'Paprikapoeder', 'Passata',
                                'Pikante uienchutney', 'Runderbouillonblokje', 'Sambal', 'Sesamzaad', 'Siciliaanse kruidenmix',
                                'Sojasaus', 'Suiker', 'Sumak', 'Surinaamse kruiden', 'Tomatenblokjes', 'Tomatenblokjes met ui',
                                'Truffeltapenade', 'Verse gember', 'Visbouillon', 'Witte balsamicoazijn', 'Wittewijnazijn',
                                'Zonnebloemolie', 'Zwarte balsamicoazijn'),
                    'vlees': ('Gekruide runderburger', 'Half-om-half gehaktballetjes met Spaanse kruiden', 'Kipfilethaasjes', 'Kipfiletstukjes',
                              'Kipgehaktballetjes met Italiaanse kruiden', 'Kippendijreepjes', 'Kipshoarma', 'Kipworst', 'Spekblokjes',
                              'Vegetarische d√∂ner kebab', 'Vegetarische kaasschnitzel', 'Vegetarische schnitzel'),
                    'zuivel': ('Ei', 'Geraspte belegen kaas', 'Geraspte cheddar', 'Geraspte grana padano', 'Geraspte oude kaas',
                               'Geraspte pecorino', 'Karnemelk', 'Kruidenroomkaas', 'Labne', 'Melk', 'Mozzarella',
                               'Parmigiano reggiano', 'Roomboter', 'Slagroom', 'Volle yoghurt')
                    }

    reverse_label_cat = {x:k for k,v in category_dict.items() for x in v}
    df["Category"] = df["Ingredients"].map(reverse_label_cat)
    col = "Category"
    first_col = df.pop(col)
    df.insert(0, col, first_col)
    df = df.sort_values(['Category', 'Ingredients'], ascending = [True, True])

    print("Merging ingredients by row across all recipe columns using justify()")
    gp_cols = ['Ingredients', 'Measurement']
    oth_cols = df.columns.difference(gp_cols)

    arr = np.vstack(df.groupby(gp_cols, sort=False, dropna=False).apply(lambda gp: justify(gp.to_numpy(), invalid_val=np.NaN, axis=0, side='up')))

    # Reconstruct DataFrame
    # Remove entirely NaN rows based on the non-grouping columns
    res = (pd.DataFrame(arr, columns=df.columns)
             .dropna(how='all', subset=oth_cols, axis=0))

    res = res.fillna(0)
    # res['Total'] = res.drop(['Ingredients', 'Measurement'], axis=1).sum(axis=1)
    res['Total'] = res.sum(axis=1, numeric_only=True)
    res=res[res['Total'] !=0] #To drop rows that are being duplicated with 0 for some reason; will check later

    # Place "Total" column towards front
    col = "Total"
    first_col = res.pop(col)
    res.insert(3, col, first_col)
    res = res.reset_index(drop=True)

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


if __name__ == '__main__':
    main()