<p align="center">
  <img src="https://images.unsplash.com/photo-1577308856961-8e9ec50d0c67?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxzZWFyY2h8Mnx8Zm9vZCUyMG9uJTIwdGFibGV8ZW58MHx8MHx8&auto=format&fit=crop&w=700&q=60" alt="Statoscope example" width="400">
</p>

# Top Daily Stock Losers & Trading Opportunities

> **⚠ NOTE:** I mostly use this app to scrape recipes from **[the Dutch HelloFresh website](https://www.hellofresh.nl/recipes)**, so the app may not be accounting for non-HelloFresh.nl recipes.

Planning meals is time-consuming and once you've determined what recipes you'll prepare then creating a consolidated grocery shopping list is cumbersome. **[My recipe scraping app hosted on streamlit](https://junegroceryshopping.streamlit.app/)** takes care of this shopping list preparation saving you tons of time in your weekly grocery planning. Click on the link above and follow the quick & easy steps to start saving time!

## Recipe Websites Supported
The app is powered by **[this recipe scraping project](https://github.com/hhursev/recipe-scrapers#scrapers-available-for)**, which supports tons of great recipe websites but the app does not currently consider cases beyond the **⚠ NOTE** above.

## Future Product Increments
* **ML Category Classification**: The "Category" column is currently mapped using a static dictionary i.e., have a look at the `df_create()` function in the `main.py` script. A trained classification model will map the Category column in the next release.
* **Scraping Status Bar**: The app currently does not provide an indication on how much of the total job has been processed and what's left. A status bar will be shown in the frontend.