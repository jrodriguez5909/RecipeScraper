import streamlit as st

from main import *
from time import sleep
from stqdm import stqdm

st.write("""# Recipe Scraper""")
st.image("https://images.unsplash.com/photo-1577308856961-8e9ec50d0c67?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxzZWFyY2h8Mnx8Zm9vZCUyMG9uJTIwdGFibGV8ZW58MHx8MHx8&auto=format&fit=crop&w=700&q=60", caption="What's for dinner?")
st.write("[See this app's GitHub ReadMe file for more info](%s)" % "https://github.com/jrodriguez5909/RecipeScraper#top-daily-stock-losers--trading-opportunities")
st.write("""
## **App Instructions:**
1. Populate the text box below with URLs for recipes you'd like to gather ingredient from - separate the URLs with commas e.g., https://www.hellofresh.nl/recipes/chicken-parmigiana-623c51bd7ed5c074f51bbb10, https://www.hellofresh.nl/recipes/quiche-met-broccoli-en-oude-kaas-628665b01dea7b8f5009b248
2. Click **Grab ingredient list** to kick off the web scraping and creation of the ingredient shopping list dataset.
3. Click **Download full csv file** link below if you'd like to download the ingredient shopping list dataset as a csv file.
""")

recs = st.text_area('', height=50)

download = st.button('Grab ingredient list')

if download:
    recs = recs.split(",")
    df_download = create_df(recs)
    csv = df_download.to_csv()
    b64 = base64.b64encode(csv.encode()).decode()  # some strings
    linko = f'<a href="data:file/csv;base64,{b64}" download="Ingredients_list.csv">Download full csv file</a>'
    st.markdown(linko, unsafe_allow_html=True)
    st.warning('⚠ Ingredient quantities follow serving size shown in the URL you provide so be mindful of this!')
    st.write("""
    • Ingredient shopping list below and available in csv format when clicking "Download full csv file" URL above:
    """)
    st.dataframe(df_download)

    # TODO: give user frontend status bar when scraping is happening; this could be doable using stqdm package