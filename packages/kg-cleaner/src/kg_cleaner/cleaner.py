import json
import logging
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)

# Terms to filter out from entities
FILTERED_TERMS = [
    "Unknown",
    "unknown",
    "Cuisine",
    "cuisine",
    "Food",
    "food",
    "Users",
    "users",
    "unspecified",
    "dishes",
    "Dishes",
    "not specified",
    "Not specified",
    "food options",
    "meal",
    "people",
    "People",
    "User",
    "drinks",
    "drink",
]

# Relations to filter out (remove triplets with these relations)
# Format: ["relation1", "relation2", ...]
FILTERED_RELATIONS = [
    # Example: "appreciates", "likes", etc.
    # Add relations to filter here:
    "appreciates",
    "interested in",
    "enjoys",
]

# Entity-only replacement rules: Replace tail entity regardless of relation
# Format: {old_tail: new_tail}
ENTITY_REPLACEMENTS = {
    # Example: "Food allergies" -> "food allergies" (normalize case)
    # Add your entity replacements here:
    "Café": "Cafe",
    "beer": "beers",
    "craft beer": "craft beers",
    "sweet": "sweets",
    "Asian fusion food": "Asian fusion",
    "asian fusion food": "Asian fusion",
}

# Entity expansion rules: Replace one tail entity with multiple tail entities
# Format: {tail_to_replace: [list_of_replacement_tails]}
# Note: The original entity is removed and replaced with the new entities
ENTITY_EXPANSION_RULES = {
    # Example: "Cajun/Creole" will create two triplets with "Cajun" and "Creole"
    # "Cajun/Creole": ["Cajun", "Creole"],
    # Add your expansion rules here:
    "Cajun/Creole": ["Cajun cuisine", "Creole cuisine"],
    "Cajun/Creole cuisine": ["Cajun cuisine", "Creole cuisine"],
    "thin and crispy crusts": ["thin crust", "crispy crust"],
    "Fans of Indian and Pakistani cuisine": ["Fans of Indian cuisine", "Fans of Pakistani cuisine"],
    "Fans of affordable and authentic Mexican food": ["Fans of affordable Mexican cuisine", "Fans of authentic Mexican cuisine"],
    "Fans of authentic and affordable Mexican food": ["Fans of authentic Mexican cuisine", "Fans of affordable Mexican cuisine"],
    "clean and friendly environment": ["clean environment", "friendly environment"],
    "clean and spacious ambiance": ["clean ambiance", "spacious ambiance"],
    "clean and welcoming atmosphere": ["clean atmosphere", "welcoming atmosphere"],
    "clean and welcoming environment": ["clean environment", "welcoming environment"],
    "intimate and cozy": ["intimate", "cozy"],
    "cozy and intimate": ["cozy", "intimate"],
    "friendly and welcoming salon atmosphere": ["friendly atmosphere", "welcoming atmosphere"],
    "relaxed, friendly atmosphere": ["relaxed atmosphere", "friendly atmosphere"],
    "relaxed and friendly atmosphere": ["relaxed atmosphere", "friendly atmosphere"],
    "warm and welcoming atmosphere": ["warm atmosphere", "welcoming atmosphere"],
    "cozy, intimate dining experiences": ["cozy dining experiences", "intimate dining experiences"],
    "Bakery or cafe": ["bakery", "cafe"],
    "cafe or bakery": ["cafe", "bakery"],
    "Cafe or Bakery": ["Cafe", "Bakery"],
    "Fans of Greek and Mediterranean cuisine": ["Fans of Greek cuisine", "Fans of Mediterranean cuisine"],
    "Fans of Mediterranean and Greek cuisine": ["Fans of Mediterranean cuisine", "Fans of Greek cuisine"],
    "bright and clean atmosphere": ["bright atmosphere", "clean atmosphere"],
    "bright, clean atmosphere": ["bright atmosphere", "clean atmosphere"],
    "casual and friendly atmosphere": ["casual atmosphere", "friendly atmosphere"],
    "casual, friendly atmosphere": ["casual atmosphere", "friendly atmosphere"],
    "casual, laid-back atmosphere": ["casual atmosphere", "laid-back atmosphere"],
    "casual and laid-back atmosphere": ["casual atmosphere", "laid-back atmosphere"],
    "casual, laid-back": ["casual atmosphere", "laid-back atmosphere"],
    "casual and laid-back": ["casual atmosphere", "laid-back atmosphere"],
    "clean and cozy atmosphere": ["clean atmosphere", "cozy atmosphere"],
    "clean, cozy atmosphere": ["clean atmosphere", "cozy atmosphere"],
    "clean and friendly atmosphere": ["clean atmosphere", "friendly atmosphere"],
    "clean, friendly atmosphere": ["clean atmosphere", "friendly atmosphere"],
    "clean and professional environment": ["clean environment", "professional environment"],
    "clean, professional environment": ["clean environment", "professional environment"],
    "cozy and friendly atmosphere": ["cozy atmosphere", "friendly atmosphere"],
    "cozy, friendly atmosphere": ["cozy atmosphere", "friendly atmosphere"],
    "cozy and welcoming atmosphere": ["cozy atmosphere", "welcoming atmosphere"],
    "cozy, welcoming atmosphere": ["cozy atmosphere", "welcoming atmosphere"],
    "cozy and intimate atmosphere": ["cozy atmosphere", "intimate atmosphere"],
    "cozy, intimate setting": ["cozy setting", "intimate setting"],
    "cozy and intimate setting": ["cozy setting", "intimate setting"],
    "fun and lively atmosphere": ["fun atmosphere", "lively atmosphere"],
    "fast and friendly service": ["fast service", "friendly service"],
    "fast, friendly service": ["fast service", "friendly service"],
    "warm and friendly atmosphere": ["warm atmosphere", "friendly atmosphere"],
    "cozy and inviting atmosphere": ["cozy atmosphere", "inviting atmosphere"],
    "clean and inviting atmosphere": ["clean atmosphere", "inviting atmosphere"],
    "clean, inviting atmosphere": ["clean atmosphere", "inviting atmosphere"],
    "cozy and relaxed atmosphere": ["cozy atmosphere", "relaxed atmosphere"],
    "cozy and romantic atmosphere": ["cozy atmosphere", "romantic atmosphere"],
    "cozy and welcoming environment": ["cozy environment", "welcoming environment"],
    "casual and fun atmosphere": ["casual atmosphere", "fun atmosphere"],
    "casual, fun atmosphere": ["casual atmosphere", "fun atmosphere"],
    "Fans of Cajun/Creole cuisine": ["Fans of Cajun cuisine", "Fans of Creole cuisine"],
    "Cajun/Creole food": ["Cajun cuisine", "Creole cuisine"],
    "Cajun/Creole dishes": ["Cajun cuisine", "Creole cuisine"],
    "Fans of Cajun/Creole food": ["Fans of Cajun cuisine", "Fans of Creole cuisine"],
    "food and drink options": ["food options", "drink options"],
    "variety of food and drink options": ["variety of food options", "variety of drink options"],
    "food and drinks": ["food", "drinks"],
    "friendly and attentive service": ["friendly service", "attentive service"],
    "quick and efficient service": ["quick service", "efficient service"],
    "quick and friendly service": ["quick service", "friendly service"],
    "quick, friendly service": ["quick service", "friendly service"],
    "attentive and friendly service": ["attentive service", "friendly service"],
    "friendly and efficient service": ["friendly service", "efficient service"],
    "friendly and helpful service": ["friendly service", "helpful service"],
    "fun and vibrant atmosphere": ["fun atmosphere", "vibrant atmosphere"],
    "lively and friendly atmosphere": ["lively atmosphere", "friendly atmosphere"],
    "vibrant and welcoming atmosphere": ["vibrant atmosphere", "welcoming atmosphere"],
    "warm and inviting atmosphere": ["warm atmosphere", "inviting atmosphere"],
    "fun and friendly atmosphere": ["fun atmosphere", "friendly atmosphere"],
    "comfortable and friendly atmosphere": ["comfortable atmosphere", "friendly atmosphere"],
    "casual and welcoming atmosphere": ["casual atmosphere", "welcoming atmosphere"],
    "cozy, lively atmosphere": ["cozy atmosphere", "lively atmosphere"],
    "cozy, homey atmosphere": ["cozy atmosphere", "homey atmosphere"],
    "cozy, intimate ambiance": ["cozy ambiance", "intimate ambiance"],
    "cozy, homey ambiance": ["cozy ambiance", "homey ambiance"],
    "small, cozy setting": ["small setting", "cozy setting"],
    "small, casual setting": ["small setting", "casual setting"],
    "cozy, casual setting": ["cozy setting", "casual setting"],
    "modern and clean setting": ["modern setting", "clean setting"],
    "cozy, hip atmosphere": ["cozy atmosphere", "hip atmosphere"],
    "cozy, bustling atmosphere": ["cozy atmosphere", "bustling atmosphere"],
    "cozy, local atmosphere": ["cozy atmosphere", "local atmosphere"],
    "cozy, family-friendly atmosphere": ["cozy atmosphere", "family-friendly atmosphere"],
    "cozy, intimate settings": ["cozy settings", "intimate settings"],
    "cozy, rustic setting": ["cozy setting", "rustic setting"],
    "clean, friendly environment": ["clean environment", "friendly environment"],
    "welcoming and friendly environment": ["welcoming environment", "friendly environment"],
    "Persian/Iranian cuisine": ["Persian cuisine", "Iranian cuisine"],
    "Himalayan/Nepalese cuisine": ["Himalayan cuisine", "Nepalese cuisine"],
    "Fans of Tex-Mex and Mexican cuisine": ["Fans of Tex-Mex cuisine", "Fans of Mexican cuisine"],
    "Fans of flavorful and authentic Thai cuisine": ["Fans of flavorful Thai cuisine", "Fans of authentic Thai cuisine"],
    "diverse food and drink options": ["diverse food options", "diverse drink options"],
    "wide variety of food and drink options": ["wide variety of food options", "wide variety of drink options"],
    "high-quality food and drinks": ["high-quality food", "high-quality drinks"],
    "food and beverage options": ["food options", "beverage options"],
    "food and drink specials": ["food specials", "drink specials"],
    "breakfast and brunch options": ["breakfast options", "brunch options"],
    "breakfast and brunch": ["breakfast", "brunch"],
    "grab-and-go setting": ["grab-and-go", "setting"],
    "lively bar and grill setting": ["lively bar setting", "lively grill setting"],
    "cool and hip atmosphere": ["cool atmosphere", "hip atmosphere"],
    "delicious vegan and vegetarian dishes": ["delicious vegan dishes", "delicious vegetarian dishes"],
    "delicious vegan and vegetarian food options": ["delicious vegan food options", "delicious vegetarian food options"],
    "diverse cocktail and wine selections": ["diverse cocktail selections", "diverse wine selections"],
    "unique and artisanal ice cream flavors": ["unique ice cream flavors", "artisanal ice cream flavors"],
    "diverse food & drink menus": ["diverse food menus", "diverse drink menus"],
    "diverse and flavorful dishes": ["diverse dishes", "flavorful dishes"],
    "pizza with creative and fresh toppings": ["pizza with creative toppings", "pizza with fresh toppings"],
    "Fans of Mexican and Latin American cuisine": ["Fans of Mexican cuisine", "Fans of Latin American cuisine"],
    "Fans of flavorful and diverse Thai cuisine": ["Fans of flavorful Thai cuisine", "Fans of diverse Thai cuisine"],
    "calm and relaxed atmosphere": ["calm atmosphere", "relaxed atmosphere"],
    "friendly and casual dining": ["friendly dining", "casual dining"],
    "chic and comfortable environment": ["chic environment", "comfortable environment"],
    "comfortable and inviting ambiance": ["comfortable ambiance", "inviting ambiance"],
    "cozy and friendly": ["cozy", "friendly"],
    "cozy and inviting": ["cozy", "inviting"],
    "cozy and intimate dining atmosphere": ["cozy dining atmosphere", "intimate dining atmosphere"],
    "cozy and welcoming": ["cozy", "welcoming"],
    "friendly and welcoming": ["friendly", "welcoming"],
    "fun and nostalgic ambiance": ["fun ambiance", "nostalgic ambiance"],
    "unique and funky vibe": ["unique vibe", "funky vibe"],
    "trendy and sleek ambiance": ["trendy ambiance", "sleek ambiance"],
    "upscale and cozy": ["upscale", "cozy"],
    "diverse wine and cocktail lists": ["diverse wine lists", "diverse cocktail lists"],
    "diverse wine and cocktail selections": ["diverse wine selections", "diverse cocktail selections"],
    "excellent beer and wine selection": ["excellent beer selection", "excellent wine selection"],
    "flavorful and diverse taco options": ["flavorful taco options", "diverse taco options"],
    "flavorful and well-spiced dishes": ["flavorful dishes", "well-spiced dishes"],
    "fresh and flavorful sushi": ["fresh sushi", "flavorful sushi"],
    "fresh and tasty sushi": ["fresh sushi", "tasty sushi"],
    "slow and disorganized service": ["slow service", "disorganized service"],
    "trendy and fun dining experiences": ["trendy dining experiences", "fun dining experiences"],
    "unique and adventurous ice cream flavors": ["unique ice cream flavors", "adventurous ice cream flavors"],
    "all ages and groups": ["all ages", "groups"],
    "Customers who enjoy breakfast and brunch options": ["Customers who enjoy breakfast options", "Customers who enjoy brunch options"],
    "Lovers of breakfast and brunch options": ["Lovers of breakfast options", "Lovers of brunch options"],
    "Fans of flavorful and affordable Mexican food": ["Fans of flavorful Mexican food", "Fans of affordable Mexican food"],
    "Fans of fresh and unique sushi": ["Fans of fresh sushi", "Fans of unique sushi"],
    "Fans of cozy, friendly bars": ["Fans of cozy bars", "Fans of friendly bars"],
    "those seeking healthy and unique dining experiences": ["those seeking healthy dining experiences", "those seeking unique dining experiences"],
    "beautiful and romantic setting": ["beautiful setting", "romantic setting"],
    "casual and classy ambiance": ["casual ambiance", "classy ambiance"],
    "clean and friendly": ["clean", "friendly"],
    "clean, inviting environment": ["clean environment", "inviting environment"],
    "clean, welcoming environment": ["clean environment", "welcoming environment"],
    "comfortable and welcoming": ["comfortable", "welcoming"],
    "welcoming and friendly": ["welcoming", "friendly"],
    "cozy, bustling dining experience": ["cozy dining experience", "bustling dining experience"],
    "friendly and fun atmosphere": ["friendly atmosphere", "fun atmosphere"],
    "authentic and flavorful Thai cuisine": ["authentic Thai cuisine", "flavorful Thai cuisine"],
    "upscale, cozy": ["upscale", "cozy"],
    "delicious and authentic Thai food": ["delicious Thai food", "authentic Thai food"],
    "fresh and authentic Thai food": ["fresh Thai food", "authentic Thai food"],
    "trendy and sophisticated ambiance": ["trendy ambiance", "sophisticated ambiance"],
    "homemade chips and salsa": ["homemade chips", "homemade salsa"],
    "fresh and locally sourced ingredients": ["fresh ingredients", "locally sourced ingredients"],
    "Fans of affordable and casual dining": ["Fans of affordable dining", "Fans of casual dining"],
    "Fans of cheap and satisfying Mexican food": ["Fans of cheap Mexican food", "Fans of satisfying Mexican food"],
    "Fans of breakfast/brunch options": ["Fans of breakfast options", "Fans of brunch options"],
    "Fans of delicious breakfast and brunch options": ["Fans of delicious breakfast options", "Fans of delicious brunch options"],
    "bustling, casual setting": ["bustling setting", "casual setting"],
    "casual and modern vibe": ["casual vibe", "modern vibe"],
    "clean, welcoming atmosphere": ["clean atmosphere", "welcoming atmosphere"],
    "cool and relaxed vibe": ["cool vibe", "relaxed vibe"],
    "cozy and unique": ["cozy", "unique"],
    "cute and clean environment": ["cute environment", "clean environment"],
    "laid-back and funky vibe": ["laid-back vibe", "funky vibe"],
    "Flavorful and delicious food": ["Flavorful food", "delicious food"],
    "affordable and quality sushi": ["affordable sushi", "quality sushi"],
    "rich and creamy ice cream": ["rich ice cream", "creamy ice cream"],
    "unique and artsy vibes": ["unique vibes", "artsy vibes"],
    "flavorful and affordable Thai cuisine": ["flavorful Thai cuisine", "affordable Thai cuisine"],
    "unique and flavorful combinations": ["unique combinations", "flavorful combinations"],
    "fresh and delicious sushi": ["fresh sushi", "delicious sushi"],
    "fresh and flavorful meals": ["fresh meals", "flavorful meals"],
    "friendly, skilled stylist": ["friendly stylist", "skilled stylist"],
    "fun, casual dining experience": ["fun dining experience", "casual dining experience"],
    "rich moist cakes": ["rich cakes", "moist cakes"],
    "fish and chips": ["fish", "chips"],
    "quality food and products": ["quality food", "quality products"],
    "delicious and varied food options": ["delicious food options", "varied food options"],
    "fresh and well-seasoned dishes": ["fresh dishes", "well-seasoned dishes"],
    "unique and delicious dishes": ["unique dishes", "delicious dishes"],
    "unique and tasty dishes": ["unique dishes", "tasty dishes"],
    "Fans of Cajun and Creole cuisine": ["Fans of Cajun cuisine", "Fans of Creole cuisine"],
    "Fans of Caribbean and Jamaican cuisine": ["Fans of Caribbean cuisine", "Fans of Jamaican cuisine"],
    "Fans of Vietnamese and Chinese cuisine": ["Fans of Vietnamese cuisine", "Fans of Chinese cuisine"],
    "Fans of breakfast and brunch": ["Fans of breakfast", "Fans of brunch"],
    "Fans of breakfast and brunch meals": ["Fans of breakfast meals", "Fans of brunch meals"],
    "Fans of casual and welcoming neighborhood bars": ["Fans of casual neighborhood bars", "Fans of welcoming neighborhood bars"],
    "relaxed and friendly": ["relaxed", "friendly"],
    "clean and welcoming dining atmosphere": ["clean dining atmosphere", "welcoming dining atmosphere"],
    "cool and modern atmosphere": ["cool atmosphere", "modern atmosphere"],
    "modern and warm atmosphere": ["modern atmosphere", "warm atmosphere"],
    "cozy and chic atmosphere": ["cozy atmosphere", "chic atmosphere"],
    "cozy and elegant": ["cozy", "elegant"],
    "cozy, casual ambiance": ["cozy ambiance", "casual ambiance"],
    "cute and cozy environment": ["cute environment", "cozy environment"],
    "cozy and modern": ["cozy", "modern"],
    "modern and cozy setting": ["modern setting", "cozy setting"],
    "fancy and casual dining experiences": ["fancy dining experiences", "casual dining experiences"],
    "festive, lively atmosphere": ["festive atmosphere", "lively atmosphere"],
    "fun and pleasant atmosphere": ["fun atmosphere", "pleasant atmosphere"],
    "relaxed and enjoyable outdoor experience": ["relaxed outdoor experience", "enjoyable outdoor experience"],
    "vibrant and lively setting": ["vibrant setting", "lively setting"],
    "welcoming, elegant ambiance": ["welcoming ambiance", "elegant ambiance"],
    "card and board games": ["card games", "board games"],
    "friendly and approachable dining experience": ["friendly dining experience", "approachable dining experience"],
    "friendly and knowledgeable stylists": ["friendly stylists", "knowledgeable stylists"],
    "high-quality, affordable, and trendy clothing": ["high-quality clothing", "affordable clothing", "trendy clothing"],
    "tasty, reasonably priced meals": ["tasty meals", "reasonably priced meals"],
    "reliable and thorough service": ["reliable service", "thorough service"],
    "spacious indoor and outdoor seating": ["spacious indoor seating", "spacious outdoor seating"],
    "high-quality, unique flavors": ["high-quality flavors", "unique flavors"],
    "unique and quirky gifts": ["unique gifts", "quirky gifts"],
    "creative and traditional menu items": ["creative menu items", "traditional menu items"],
    "fresh and delicious food options": ["fresh food options", "delicious food options"],
    "frozen yogurt with flavors and toppings": ["frozen yogurt with flavors", "frozen yogurt with toppings"],
    "unique and reasonably priced menu items": ["unique menu items", "reasonably priced menu items"],
    "unique and creative tea drinks": ["unique tea drinks", "creative tea drinks"],
    "unique and innovative tea drinks": ["unique tea drinks", "innovative tea drinks"],
    "fresh, organic, and locally sourced ingredients": ["fresh ingredients", "organic ingredients", "locally sourced ingredients"],
    "Fans of flavorful and crispy chicken wings": ["Fans of flavorful chicken wings", "Fans of crispy chicken wings"],
    "Fans of fresh, customizable burgers": ["Fans of fresh burgers", "Fans of customizable burgers"],
    "Vegan and vegetarian individuals": ["Vegan individuals", "vegetarian individuals"],
    "clean and vibrant dining environment": ["clean dining environment", "vibrant dining environment"],
    "clean and welcoming": ["clean", "welcoming"],
    "cool and welcoming": ["cool", "welcoming"],
    "warm, welcoming": ["warm", "welcoming"],
    "cozy and traditional bakery setting": ["cozy bakery setting", "traditional bakery setting"],
    "cozy and welcoming cafe atmosphere": ["cozy cafe atmosphere", "welcoming cafe atmosphere"],
    "cozy mom and pop restaurant vibe": ["cozy restaurant vibe", "mom and pop restaurant vibe"],
    "peaceful and tranquil": ["peaceful", "tranquil"],
    "small, casual restaurant setting": ["small restaurant setting", "casual restaurant setting"],
    "creative and unique donut flavors": ["creative donut flavors", "unique donut flavors"],
    "delicious, well-prepared dishes": ["delicious dishes", "well-prepared dishes"],
    "fresh and generous sushi": ["fresh sushi", "generous sushi"],
    "fresh and tasty dishes": ["fresh dishes", "tasty dishes"],
    "unique and flavorful": ["unique", "flavorful"],
    "unique and flavorful creations": ["unique creations", "flavorful creations"],
    "variety of vegetarian and vegan choices": ["variety of vegetarian choices", "variety of vegan choices"],
    "Craft Beer & Wine Bar": ["Craft Beer Bar", "Wine Bar"],
    "Nail & Spa": ["Nail", "Spa"],
    "unique and creative flavors": ["unique flavors", "creative flavors"],
    "Fun and affordable dining": ["Fun dining", "affordable dining"],
    "Fans of unique, flavorful burgers": ["Fans of unique burgers", "Fans of flavorful burgers"],
    "Vintage and antique enthusiasts": ["Vintage enthusiasts", "antique enthusiasts"],
    "Steak and seafood restaurant": ["Steak restaurant", "seafood restaurant"],
    "steak and seafood": ["steak", "seafood"],
    "locals and tourists": ["locals", "tourists"],
    "clean, spacious, and friendly atmosphere": ["clean atmosphere", "spacious atmosphere", "friendly atmosphere"],
    "quaint and charming dining atmospheres": ["quaint dining atmospheres", "charming dining atmospheres"],
    "affordable and well-prepared sushi": ["affordable sushi", "well-prepared sushi"],
    "new, clean theaters": ["new theaters", "clean theaters"],
    "cozy and cute bakery setting": ["cozy bakery setting", "cute bakery setting"],
    "cute and cozy bakery experience": ["cute bakery experience", "cozy bakery experience"],
    "diverse beer and wine selection": ["diverse beer selection", "diverse wine selection"],
    "good variety and quality": ["good variety", "good quality"],
    "unique and diverse pizza choices": ["unique pizza choices", "diverse pizza choices"],
    "intimate, romantic setting": ["intimate setting", "romantic setting"],
    "moist and delicious cakes": ["moist cakes", "delicious cakes"],
    "quality and taste of the food": ["quality of the food", "taste of the food"],
    "unique and delicious food options": ["unique food options", "delicious food options"],
    "wide variety of delicious flavors": ["wide variety of flavors", "delicious flavors"],
    "wide selection of comfort food options": ["wide selection of comfort food", "comfort food options"],
    "personalized, professional nail care": ["personalized nail care", "professional nail care"],
    "options for different dietary needs": ["options for dietary needs", "different dietary needs"],
    "breakfast & brunch": ["breakfast", "brunch"],
    "breakfast/brunch options": ["breakfast options", "brunch options"],
    "nutritious breakfast and brunch options": ["nutritious breakfast options", "nutritious brunch options"],
    "variety of tasty sauces": ["variety of sauces", "tasty sauces"],
    "vegan/vegetarian options": ["vegan options", "vegetarian options"],
    "Fans of fresh and flavorful Vietnamese cuisine": ["Fans of fresh Vietnamese cuisine", "Fans of flavorful Vietnamese cuisine"],
    "Fans of cozy and intimate bars": ["Fans of cozy bars", "Fans of intimate bars"],
    "Breakfast and brunch spot": ["Breakfast spot", "brunch spot"],
    "parents and kids": ["parents", "kids"],
    "vibrant and clean atmosphere": ["vibrant atmosphere", "clean atmosphere"],
    "cleaner and modern atmosphere": ["cleaner atmosphere", "modern atmosphere"],
    "comfortable and clean ambiance": ["comfortable ambiance", "clean ambiance"],
    "cozy and comfortable cafe": ["cozy cafe", "comfortable cafe"],
    "cozy, sunny patio setting": ["cozy patio setting", "sunny patio setting"],
    "fun and creative atmosphere": ["fun atmosphere", "creative atmosphere"],
    "sophisticated and elegant atmosphere": ["sophisticated atmosphere", "elegant atmosphere"],
    "friendly and lively atmosphere": ["friendly atmosphere", "lively atmosphere"],
    "lively and relaxed": ["lively", "relaxed"],
    "intimate and quiet dining atmospheres": ["intimate dining atmospheres", "quiet dining atmospheres"],
    "quick and casual dining experiences": ["quick dining experiences", "casual dining experiences"],
    "quick and relaxed dining experience": ["quick dining experience", "relaxed dining experience"],
    "relaxed and unassuming dining atmosphere": ["relaxed dining atmosphere", "unassuming dining atmosphere"],
    "relaxed, quiet ambiance": ["relaxed ambiance", "quiet ambiance"],
    "spacious and open restaurant settings": ["spacious restaurant settings", "open restaurant settings"],
    "vegan and vegetarian": ["vegan", "vegetarian"],
    "thin, crispy crust": ["thin crust", "crispy crust"],
    "thin, crispy crusts": ["thin crusts", "crispy crusts"],
    "thin, crunchy crusts": ["thin crusts", "crunchy crusts"],
    "fast and efficient service": ["fast service", "efficient service"],
    "fresh, flavorful cuisine": ["fresh cuisine", "flavorful cuisine"],
    "personable and friendly service": ["personable service", "friendly service"],
    "quick, efficient service": ["quick service", "efficient service"],
    "intimate, romantic ambiance": ["intimate ambiance", "romantic ambiance"],
    "varied food and drink options": ["varied food options", "varied drink options"],
    "variety in food and drink choices": ["variety in food choices", "variety in drink choices"],
    "wide variety of food and beverage options": ["wide variety of food options", "wide variety of beverage options"],
    "wide variety of food and drink choices": ["wide variety of food choices", "wide variety of drink choices"],
    "unique and affordable furniture": ["unique furniture", "affordable furniture"],
    "fish & chips": ["fish", "chips"],
    "slightly pricey but reasonable prices": ["slightly pricey prices", "reasonable prices"],
    "delicious grilled cheese sandwiches": ["delicious grilled cheese", "grilled cheese sandwiches"],
    "innovative and fresh dishes": ["innovative dishes", "fresh dishes"],
    "hot hearty soups": ["hot soups", "hearty soups"],
    "home cooked meals": ["home cooked", "meals"],
    "home-cooked style food": ["home-cooked style", "food"],
    "local and organic ingredients": ["local ingredients", "organic ingredients"],
    "fresh, natural ingredients": ["fresh ingredients", "natural ingredients"],
    "Fans of trendy and unique dessert experiences": ["Fans of trendy dessert experiences", "Fans of unique dessert experiences"],
    "Fans of unique and indulgent dessert experiences": ["Fans of unique dessert experiences", "Fans of indulgent dessert experiences"],
    "Those who appreciate attentive and friendly staff": ["Those who appreciate attentive staff", "Those who appreciate friendly staff"],
    "hipster and alternative": ["hipster", "alternative"],
    "diverse range of tastes and preferences": ["diverse range of tastes", "diverse range of preferences"],
    "casual and welcoming setting": ["casual setting", "welcoming setting"],
    "chic and modern setting": ["chic setting", "modern setting"],
    "classy, fun atmosphere": ["classy atmosphere", "fun atmosphere"],
    "clean, upscale environment": ["clean environment", "upscale environment"],
    "cozy and artistic": ["cozy", "artistic"],
    "cozy and quaint setting": ["cozy setting", "quaint setting"],
    "intimate, cozy ambiance": ["intimate ambiance", "cozy ambiance"],
    "cozy, dark pub ambiance": ["cozy pub ambiance", "dark pub ambiance"],
    "friendly and nostalgic atmosphere": ["friendly atmosphere", "nostalgic atmosphere"],
    "fun and welcoming": ["fun", "welcoming"],
    "fun and social atmosphere": ["fun atmosphere", "social atmosphere"],
    "fun and social experience": ["fun experience", "social experience"],
    "laid-back and friendly dining atmosphere": ["laid-back dining atmosphere", "friendly dining atmosphere"],
    "modern and welcoming environment": ["modern environment", "welcoming environment"],
    "modern, clean, and welcoming": ["modern", "clean", "welcoming"],
    "delicious breakfast and brunch options": ["delicious breakfast options", "delicious brunch options"],
    "fresh, delicious breakfast/brunch options": ["fresh breakfast/brunch options", "delicious breakfast/brunch options"],
    "flavorful and meaty chicken wings": ["flavorful chicken wings", "meaty chicken wings"],
    "fresh and unique bagels": ["fresh bagels", "unique bagels"],
    "great food and service": ["great food", "great service"],
    "hippie/hipster vibe": ["hippie vibe", "hipster vibe"],
    "trendy and lively restaurant": ["trendy restaurant", "lively restaurant"],
    "unique and innovative brews": ["unique brews", "innovative brews"],
    "wide selection of local and craft beers": ["wide selection of local beers", "wide selection of craft beers"],
    "Cafe Bakery": ["Cafe", "Bakery"],
    "creative Asian fusion dishes": ["creative Asian fusion", "Asian fusion dishes"],
    "Himalayan/Nepalese food": ["Himalayan food", "Nepalese food"],
    "hearty Italian meals": ["hearty meals", "Italian meals"],
    "tasty Latin American food": ["tasty food", "Latin American food"],
    "tasty breakfast foods": ["tasty foods", "breakfast foods"],
    "food with limited vegetarian options": ["food", "limited vegetarian options"],
    "vegetarian/vegan options": ["vegetarian options", "vegan options"],
    "Fans of Cajun/Creole flavors": ["Fans of Cajun flavors", "Fans of Creole flavors"],
    "Fans of flavorful Cajun/Creole cuisine": ["Fans of flavorful Cajun cuisine", "Fans of flavorful Creole cuisine"],
    "Fans of breakfast, brunch, lunch": ["Fans of breakfast", "Fans of brunch", "Fans of lunch"],
    "Fans of quick, hearty, and authentic Vietnamese cuisine": ["Fans of quick Vietnamese cuisine", "Fans of hearty Vietnamese cuisine", "Fans of authentic Vietnamese cuisine"],
    "Fans of authentic and delicious Cajun/Creole cuisine": ["Fans of authentic Cajun/Creole cuisine", "Fans of delicious Cajun/Creole cuisine"],
    "bar and grill": ["bar", "grill"],
    "casual, vibrant bar atmosphere": ["casual bar atmosphere", "vibrant bar atmosphere"],
    "clean, classy atmosphere": ["clean atmosphere", "classy atmosphere"],
    "comfortable and warm atmosphere": ["comfortable atmosphere", "warm atmosphere"],
    "warm and comfortable atmosphere": ["warm atmosphere", "comfortable atmosphere"],
    "friendly and communal atmosphere": ["friendly atmosphere", "communal atmosphere"],
    "cozy and vibrant cafe setting": ["cozy setting", "vibrant setting"],
    "stylish, cozy atmosphere": ["stylish atmosphere", "cozy atmosphere"],
    "cozy, small diner-style setting": ["cozy diner-style setting", "small diner-style setting"],
    "cozy, homey setting": ["cozy setting", "homey setting"],
    "intimate, cozy and warm settings": ["intimate settings", "cozy settings", "warm settings"],
    "vibrant and bustling atmosphere": ["vibrant atmosphere", "bustling atmosphere"],
    "lively sports bar/pub atmosphere": ["lively sports bar atmosphere", "lively pub atmosphere"],
    "quiet, relaxing atmosphere": ["quiet atmosphere", "relaxing atmosphere"],
    "cozy cafe and bar setting": ["cozy setting"],
    "unique and diverse selections": ["unique selections", "diverse selections"],
    "flavorful and juicy chicken": ["flavorful chicken", "juicy chicken"],
    "healthy and flavorful seafood": ["healthy seafood", "flavorful seafood"],
    "unique and delicious breakfast/brunch options": ["unique breakfast/brunch options", "delicious breakfast/brunch options"],
    "tasty, seasoned fries": ["tasty fries", "seasoned fries"],
    "variety of food and drinks": ["variety of food", "variety of drinks"],
    "wide variety of services and products": ["wide variety of services", "wide variety of products"],
    "Wine Shop & Bar": ["Wine Shop", "Bar"],
    "creative and unique breakfast options": ["creative breakfast options", "unique breakfast options"],
    "vegetarian/vegan-friendly dishes": ["vegetarian-friendly dishes", "vegan-friendly dishes"],
    "diverse lineup of musical acts": ["diverse lineup", "musical acts"],
    "delicious and authentic dishes": ["delicious dishes", "authentic dishes"],
    "small local businesses": ["small businesses", "local businesses"],
    "Individuals seeking relaxation and tranquility": ["Individuals seeking relaxation", "Individuals seeking tranquility"],
    "Parents with young kids": ["Parents", "young kids"],
    "different dietary preferences": ["different preferences", "dietary preferences"],
    "bar/restaurant": ["bar", "restaurant"],
    "clean and beautifully decorated restaurants": ["clean restaurants", "beautifully decorated restaurants"],
    "calm and serene environment": ["calm environment", "serene environment"],
    "casual and nice ambience": ["casual ambience", "nice ambience"],
    "casual, intimate setting": ["casual setting", "intimate setting"],
    "intimate and relaxed settings": ["intimate settings", "relaxed settings"],
    "chill and pleasant atmosphere": ["chill atmosphere", "pleasant atmosphere"],
    "relaxed and chill atmosphere": ["relaxed atmosphere", "chill atmosphere"],
    "cool, hip atmosphere": ["cool atmosphere", "hip atmosphere"],
    "cozy and quiet atmosphere": ["cozy atmosphere", "quiet atmosphere"],
    "cozy, quiet atmosphere": ["cozy atmosphere", "quiet atmosphere"],
    "cute and welcoming atmosphere": ["cute atmosphere", "welcoming atmosphere"],
    "cute, welcoming atmosphere": ["cute atmosphere", "welcoming atmosphere"],
    "friendly and low-key atmosphere": ["friendly atmosphere", "low-key atmosphere"],
    "friendly, low-key atmosphere": ["friendly atmosphere", "low-key atmosphere"],
    "fun and laid-back atmosphere": ["fun atmosphere", "laid-back atmosphere"],
    "fun, laid-back atmosphere": ["fun atmosphere", "laid-back atmosphere"],
    "authentic and flavorful dishes": ["authentic dishes", "flavorful dishes"],
    "authentic, flavorful dishes": ["authentic dishes", "flavorful dishes"],
    "cozy and intimate dining experience": ["cozy dining experience", "intimate dining experience"],
    "cozy, intimate dining experience": ["cozy dining experience", "intimate dining experience"],
    "cozy and quaint cafes": ["cozy cafes", "quaint cafes"],
    "cozy, quaint cafes": ["cozy cafes", "quaint cafes"],
    "fresh and delicious dishes": ["fresh dishes", "delicious dishes"],
    "fresh, delicious dishes": ["fresh dishes", "delicious dishes"],
    "fresh and flavorful dishes": ["fresh dishes", "flavorful dishes"],
    "fresh, flavorful dishes": ["fresh dishes", "flavorful dishes"],
    "quiet and spacious environment": ["quiet environment", "spacious environment"],
    "quiet, spacious environment": ["quiet environment", "spacious environment"],
    "cozy and chic ambiance": ["cozy ambiance", "chic ambiance"],
    "cozy, chic ambiance": ["cozy ambiance", "chic ambiance"],
    "cozy and laid-back environment": ["cozy environment", "laid-back environment"],
    "cozy, laid-back environment": ["cozy environment", "laid-back environment"],
    "laid-back and cozy atmosphere": ["laid-back atmosphere", "cozy atmosphere"],
    "laid-back, cozy atmosphere": ["laid-back atmosphere", "cozy atmosphere"],
    "lively and fun atmosphere": ["lively atmosphere", "fun atmosphere"],
    "lively, fun atmosphere": ["lively atmosphere", "fun atmosphere"],
    "lively and spacious setting": ["lively setting", "spacious setting"],
    "spacious and lively setting": ["spacious setting", "lively setting"],
    "sleek and modern ambiance": ["sleek ambiance", "modern ambiance"],
    "sleek, modern ambiance": ["sleek ambiance", "modern ambiance"],
    "trendy and chic atmosphere": ["trendy atmosphere", "chic atmosphere"],
    "trendy, chic atmosphere": ["trendy atmosphere", "chic atmosphere"],
    "fresh and flavorful food": ["fresh food", "flavorful food"],
    "fresh, flavorful food": ["fresh food", "flavorful food"],
    "fresh and healthy options": ["fresh options", "healthy options"],
    "fresh, healthy options": ["fresh options", "healthy options"],
    "fresh and seasonal ingredients": ["fresh ingredients", "seasonal ingredients"],
    "fresh, seasonal ingredients": ["fresh ingredients", "seasonal ingredients"],
    "creative and unique dishes": ["creative dishes", "unique dishes"],
    "unique and creative dishes": ["unique dishes", "creative dishes"],
    "local and fresh ingredients": ["local ingredients", "fresh ingredients"],
    "local, fresh ingredients": ["local ingredients", "fresh ingredients"],
    "clean, modern dining environments": ["clean dining environments", "modern dining environments"],
    "modern, clean dining environments": ["modern dining environments", "clean dining environments"],
    "clean, modern environment": ["clean environment", "modern environment"],
    "modern, clean environment": ["modern environment", "clean environment"],
    "eclectic and vibrant atmosphere": ["eclectic atmosphere", "vibrant atmosphere"],
    "vibrant and eclectic atmosphere": ["vibrant atmosphere", "eclectic atmosphere"],
    "friendly local atmosphere": ["friendly atmosphere", "local atmosphere"],
    "friendly, local atmosphere": ["friendly atmosphere", "local atmosphere"],
    "trendy and energetic atmosphere": ["trendy atmosphere", "energetic atmosphere"],
    "trendy, energetic atmosphere": ["trendy atmosphere", "energetic atmosphere"],
    "cozy and upscale atmosphere": ["cozy atmosphere", "upscale atmosphere"],
    "cozy, upscale atmosphere": ["cozy atmosphere", "upscale atmosphere"],
    "delicious and unique dishes": ["delicious dishes", "unique dishes"],
    "delicious, unique dishes": ["delicious dishes", "unique dishes"],
    "modern and spacious setting": ["modern setting", "spacious setting"],
    "modern, spacious setting": ["modern setting", "spacious setting"],
    "friendly and knowledgeable staff": ["friendly staff", "knowledgeable staff"],
    "friendly, knowledgeable staff": ["friendly staff", "knowledgeable staff"],
    "fresh local ingredients": ["fresh ingredients", "local ingredients"],
    "fresh, local ingredients": ["fresh ingredients", "local ingredients"],
    "clean and modern atmosphere": ["clean atmosphere", "modern atmosphere"],
    "modern and clean atmosphere": ["modern atmosphere", "clean atmosphere"],
    "cozy local setting": ["cozy setting", "local setting"],
    "cozy, local setting": ["cozy setting", "local setting"],
    "friendly and personable service": ["friendly service", "personable service"],
    "friendly, personable service": ["friendly service", "personable service"],
    "diverse cocktail and wine selections": ["diverse cocktail selections", "diverse wine selections"],
    "diverse wine and cocktail selections": ["diverse wine selections", "diverse cocktail selections"],
    "bustling, lively atmosphere": ["bustling atmosphere", "lively atmosphere"],
    "lively and bustling atmosphere": ["lively atmosphere", "bustling atmosphere"],
    "cozy neighborhood setting": ["cozy setting", "neighborhood setting"],
    "cozy, neighborhood setting": ["cozy setting", "neighborhood setting"],
    "charming, cozy atmosphere": ["charming atmosphere", "cozy atmosphere"],
    "cozy and charming atmosphere": ["cozy atmosphere", "charming atmosphere"],
    "clean and friendly dining environment": ["clean dining environment", "friendly dining environment"],
    "cozy and homey atmosphere": ["cozy atmosphere", "homey atmosphere"],
    "cozy family-owned atmosphere": ["cozy atmosphere", "family-owned atmosphere"],
    "cozy, family-owned atmosphere": ["cozy atmosphere", "family-owned atmosphere"],
    "cozy romantic atmosphere": ["cozy atmosphere", "romantic atmosphere"],
    "cozy, romantic atmosphere": ["cozy atmosphere", "romantic atmosphere"],
    "friendly neighborhood setting": ["friendly setting", "neighborhood setting"],
    "friendly, neighborhood setting": ["friendly setting", "neighborhood setting"],
    "lively, loud atmosphere": ["lively atmosphere", "loud atmosphere"],
    "loud and lively atmosphere": ["loud atmosphere", "lively atmosphere"],
    "relaxed neighborhood vibe": ["relaxed vibe", "neighborhood vibe"],
    "relaxed, neighborhood vibe": ["relaxed vibe", "neighborhood vibe"],
    "trendy, laid-back atmosphere": ["trendy atmosphere", "laid-back atmosphere"],
    "cozy and quaint atmosphere": ["cozy atmosphere", "quaint atmosphere"],
    "quaint and cozy atmosphere": ["quaint atmosphere", "cozy atmosphere"],
    "casual hipster atmosphere": ["casual atmosphere", "hipster atmosphere"],
    "casual, hipster atmosphere": ["casual atmosphere", "hipster atmosphere"],
    "clean and welcoming environments": ["clean environment", "welcoming environment"],
    "cozy family-run atmosphere": ["cozy atmosphere", "family-run atmosphere"],
    "cozy, family-run atmosphere": ["cozy atmosphere", "family-run atmosphere"],
    "fun vibrant atmosphere": ["fun atmosphere", "vibrant atmosphere"],
    "fun, vibrant atmosphere": ["fun atmosphere", "vibrant atmosphere"],
    "fresh and organic produce": ["fresh produce", "organic produce"],
    "fresh organic produce": ["fresh produce", "organic produce"],
    "lively but casual atmosphere": ["lively atmosphere", "casual atmosphere"],
    "cozy old school atmosphere": ["cozy atmosphere", "old-school atmosphere"],
    "cozy, old-school atmosphere": ["cozy atmosphere", "old-school atmosphere"],
    "cozy, laid-back atmosphere": ["cozy atmosphere", "laid-back atmosphere"],
    "laid-back, cozy atmosphere": ["laid-back atmosphere", "cozy atmosphere"],
    "creative and high-quality dishes": ["creative dishes", "high-quality dishes"],
    "creative and quality dishes": ["creative dishes", "quality dishes"],
    "comfortable and friendly environment": ["comfortable environment", "friendly environment"],
    "friendly and comfortable environment": ["friendly environment", "comfortable environment"],
    "trendy, cozy atmosphere": ["trendy atmosphere", "cozy atmosphere"],
    "vibrant and fun atmosphere": ["vibrant atmosphere", "fun atmosphere"],
    "fresh and tasty food": ["fresh food", "tasty food"],
    "tasty, fresh food": ["tasty food", "fresh food"],
    "fresh quality food": ["fresh food", "quality food"],
    "fresh, quality food": ["fresh food", "quality food"],
    "cozy homey atmosphere": ["cozy atmosphere", "homey atmosphere"],
    "cozy, homely atmosphere": ["cozy atmosphere", "homely atmosphere"],
    "laid-back but upscale atmosphere": ["laid-back atmosphere", "upscale atmosphere"],
    "laid-back upscale atmosphere": ["laid-back atmosphere", "upscale atmosphere"],
    "laid-back, local atmosphere": ["laid-back atmosphere", "local atmosphere"],
    "local and laid-back atmosphere": ["local atmosphere", "laid-back atmosphere"],
    "fresh and organic ingredients": ["fresh ingredients", "organic ingredients"],
    "fresh organic ingredients": ["fresh ingredients", "organic ingredients"],
    "cozy and vibrant atmosphere": ["cozy atmosphere", "vibrant atmosphere"],
    "modern and chic decor": ["modern decor", "chic decor"],
    "modern chic decor": ["modern decor", "chic decor"],
    "cozy, relaxed atmosphere": ["cozy atmosphere", "relaxed atmosphere"],
    "cozy, relaxing atmosphere": ["cozy atmosphere", "relaxing atmosphere"],
    "cozy, welcoming environment": ["cozy environment", "welcoming environment"],
    "welcoming, cozy atmosphere": ["welcoming atmosphere", "cozy atmosphere"],
    "laid-back, retro atmosphere": ["laid-back atmosphere", "retro atmosphere"],
    "retro, laid-back atmosphere": ["retro atmosphere", "laid-back atmosphere"],
    "small and cozy setting": ["small setting", "cozy setting"],
    "small cozy setting": ["small setting", "cozy setting"],
    "cozy, small restaurant": ["cozy restaurant", "small restaurant"],
    "small, cozy restaurant": ["small restaurant", "cozy restaurant"],
    "fresh, hot food": ["fresh food", "hot food"],
    "hot and fresh food": ["hot food", "fresh food"],
    "modern and cozy ambiance": ["modern ambiance", "cozy ambiance"],
    "modern cozy ambiance": ["modern ambiance", "cozy ambiance"],
    "spacious and inviting atmosphere": ["spacious atmosphere", "inviting atmosphere"],
    "attentive and friendly staff": ["attentive staff", "friendly staff"],
    "friendly and attentive staff": ["friendly staff", "attentive staff"],
    "bright, welcoming atmosphere": ["bright atmosphere", "welcoming atmosphere"],
    "cozy and elegant atmosphere": ["cozy atmosphere", "elegant atmosphere"],
    "trendy and hip atmosphere": ["trendy atmosphere", "hip atmosphere"],
    "friendly and helpful staff": ["friendly staff", "helpful staff"],
    "helpful and friendly staff": ["helpful staff", "friendly staff"],
    "colorful and cozy environment": ["colorful environment", "cozy environment"],
    "cozy, colorful atmosphere": ["cozy atmosphere", "colorful atmosphere"],
    "cozy, small-town charm": ["cozy charm", "small-town charm"],
    "quiet, cozy atmosphere": ["quiet atmosphere", "cozy atmosphere"],
    "great service and ambiance": ["great service", "great ambiance"],
    "clean and well-decorated restaurant": ["clean restaurant", "well-decorated restaurant"],
    "cozy and cute restaurant environment": ["cozy restaurant environment", "cute restaurant environment"],
    "attentive and knowledgeable bartenders": ["attentive bartenders", "knowledgeable bartenders"],
    "friendly and welcoming dining experience": ["friendly dining experience", "welcoming dining experience"],
    "sleek and contemporary ambiance": ["sleek ambiance", "contemporary ambiance"],
    "hot and cold beverages": ["hot beverages", "cold beverages"],
    "hot and cold drinks": ["hot drinks", "cold drinks"],
    "friendly, casual dining experience": ["friendly dining experience", "casual dining experience"],
    "fun, festive atmosphere": ["fun atmosphere", "festive atmosphere"],
    "elegant and formal dining experiences": ["elegant dining experiences", "formal dining experiences"],
    "fast & friendly service": ["fast service", "friendly service"],
    "fun and lively events": ["fun events", "lively events"],
    "luxurious and intimate settings": ["luxurious settings", "intimate settings"],
    "rustic, charming ambiance": ["rustic ambiance", "charming ambiance"],
    "wide selection of beauty and spa products": ["wide selection of beauty products", "wide selection of spa products"],
    "unique and affordable jewelry": ["unique jewelry", "affordable jewelry"],
    "creative, flavorful dishes": ["creative dishes", "flavorful dishes"],
    "creative and flavorful dishes": ["creative dishes", "flavorful dishes"],
    "trendy and lively atmosphere": ["trendy atmosphere", "lively atmosphere"],
    "casual and modern atmospheres": ["casual atmospheres", "modern atmospheres"],
    "modern, casual atmosphere": ["modern atmosphere", "casual atmosphere"],
    "casual, family-friendly atmosphere": ["casual atmosphere", "family-friendly atmosphere"],
    "clean and cozy environment": ["clean environment", "cozy environment"],
    "clean and nice atmosphere": ["clean atmosphere", "nice atmosphere"],
    "clean and pleasant atmosphere": ["clean atmosphere", "pleasant atmosphere"],
    "modern and cleaner ambiance": ["modern ambiance", "cleaner ambiance"],
    "modern, clean ambiance": ["modern ambiance", "clean ambiance"],
    "bright and friendly atmosphere": ["bright atmosphere", "friendly atmosphere"],
    "casual and relaxed dining experience": ["casual dining experience", "relaxed dining experience"],
    "casual, comfortable dining experience": ["casual dining experience", "comfortable dining experience"],
    "cozy and friendly environment": ["cozy environment", "friendly environment"],
    "friendly and cozy atmosphere": ["friendly atmosphere", "cozy atmosphere"],
    "cozy and welcoming ambiance": ["cozy ambiance", "welcoming ambiance"],
    "hip and cozy atmosphere": ["hip atmosphere", "cozy atmosphere"],
    "fun and engaging environment": ["fun environment", "engaging environment"],
    "fun and enjoyable environment": ["fun environment", "enjoyable environment"],
    "small and large parties": ["small parties", "large parties"],
    "beautiful and cozy atmosphere": ["beautiful atmosphere", "cozy atmosphere"],
    "bright and inviting atmosphere": ["bright atmosphere", "inviting atmosphere"],
    "casual and hip atmosphere": ["casual atmosphere", "hip atmosphere"],
    "hip and casual atmosphere": ["hip atmosphere", "casual atmosphere"],
    "clean and hip atmosphere": ["clean atmosphere", "hip atmosphere"],
    "hip, clean atmosphere": ["hip atmosphere", "clean atmosphere"],
    "cozy and comfortable atmosphere": ["cozy atmosphere", "comfortable atmosphere"],
    "friendly and entertaining dining experience": ["friendly dining experience", "entertaining dining experience"],
    "fun and friendly dining experience": ["fun dining experience", "friendly dining experience"],
    "welcoming and enjoyable dining experience": ["welcoming dining experience", "enjoyable dining experience"],
    "lively, vibrant dining atmosphere": ["lively dining atmosphere", "vibrant dining atmosphere"],
    "beautiful and clean atmosphere": ["beautiful atmosphere", "clean atmosphere"],
    "bright and inviting ambiance": ["bright ambiance", "inviting ambiance"],
    "classy, cozy atmosphere": ["classy atmosphere", "cozy atmosphere"],
    "clean and friendly shopping environment": ["clean shopping environment", "friendly shopping environment"],
    "clean and friendly shopping experience": ["clean shopping experience", "friendly shopping experience"],
    "cozy and romantic ambiance": ["cozy ambiance", "romantic ambiance"],
    "delicious and satisfying dining experience": ["delicious dining experience", "satisfying dining experience"],
    "friendly and efficient dining experience": ["friendly dining experience", "efficient dining experience"],
    "friendly, inviting atmosphere": ["friendly atmosphere", "inviting atmosphere"],
    "relaxed but lively atmosphere": ["relaxed atmosphere", "lively atmosphere"],
    "clean and pleasant dining experience": ["clean dining experience", "pleasant dining experience"],
    "clean, comfortable dining experience": ["clean dining experience", "comfortable dining experience"],
    "cozy and comfortable dining atmosphere": ["cozy dining atmosphere", "comfortable dining atmosphere"],
    "fun and energetic atmosphere": ["fun atmosphere", "energetic atmosphere"],
    "fun and lively nightlife": ["fun nightlife", "lively nightlife"],
    "friendly and cozy cafe setting": ["friendly cafe setting", "cozy cafe setting"],
    "clean and friendly atmospheres": ["clean atmospheres", "friendly atmospheres"],
    "sophisticated but comfortable dining experience": ["sophisticated dining experience", "comfortable dining experience"],
    "cozy, modern atmosphere": ["cozy atmosphere", "modern atmosphere"],
    "cozy, friendly vibe": ["cozy vibe", "friendly vibe"],
    "friendly and lively pub atmosphere": ["friendly pub atmosphere", "lively pub atmosphere"],
    "imaginative and creative cocktails": ["imaginative cocktails", "creative cocktails"],
    "casual and friendly dining environment": ["casual dining environment", "friendly dining environment"],
    "clean and comfortable dining atmosphere": ["clean dining atmosphere", "comfortable dining atmosphere"],
    "clean, comfortable dining experience": ["clean dining experience", "comfortable dining experience"],
    "friendly and accommodating dining experience": ["friendly dining experience", "accommodating dining experience"],
    "spacious and welcoming atmosphere": ["spacious atmosphere", "welcoming atmosphere"],
    "welcoming and spacious environment": ["welcoming environment", "spacious environment"],
    "diverse and unique menu": ["diverse menu", "unique menu"],
    "cheap and delicious Mexican food": ["cheap Mexican food", "delicious Mexican food"],
    "cheap and satisfying Mexican food": ["cheap Mexican food", "satisfying Mexican food"],
    "casual, cozy outdoor setting": ["casual outdoor setting", "cozy outdoor setting"],
    "contemporary and clean dining environment": ["contemporary dining environment", "clean dining environment"],
    "warm and inviting setting": ["warm setting", "inviting setting"],
    "welcoming and friendly atmosphere": ["welcoming atmosphere", "friendly atmosphere"],
    "delicious and flavorful offerings": ["delicious offerings", "flavorful offerings"],
    "lively and entertaining dining experience": ["lively dining experience", "entertaining dining experience"],
    "unique and adventurous ice cream flavors": ["unique ice cream flavors", "adventurous ice cream flavors"],
    "unique and interesting ice cream flavors": ["unique ice cream flavors", "interesting ice cream flavors"],
    "unique and innovative dishes": ["unique dishes", "innovative dishes"],
    "unique and flavorful dishes": ["unique dishes", "flavorful dishes"],
    "cool, hipster ambiance": ["cool ambiance", "hipster ambiance"],
    "cozy and intimate dining experiences": ["cozy dining experiences", "intimate dining experiences"],
    "engaging and knowledgeable tour guides": ["engaging tour guides", "knowledgeable tour guides"],
    "fun and interactive dining experience": ["fun dining experience", "interactive dining experience"],
    "flavorful and spicy dishes": ["flavorful dishes", "spicy dishes"],
    "spicy and flavorful dishes": ["spicy dishes", "flavorful dishes"],
    "delicious and unique menu items": ["delicious menu items", "unique menu items"],
    "unique and delicious menu offerings": ["unique menu offerings", "delicious menu offerings"],
    "beautiful and lively dining atmosphere": ["beautiful dining atmosphere", "lively dining atmosphere"],
    "casual chic atmosphere": ["casual atmosphere", "chic atmosphere"],
    "clean and cozy setting": ["clean setting", "cozy setting"],
    "cozy and friendly setting": ["cozy setting", "friendly setting"],
    "cute and simple ambiance": ["cute ambiance", "simple ambiance"],
    "cozy, intimate dining spots": ["cozy dining spots", "intimate dining spots"],
    "moist and flavorful cupcakes": ["moist cupcakes", "flavorful cupcakes"],
    "comfortable and aesthetically pleasing atmospheres": ["comfortable atmospheres", "aesthetically pleasing atmospheres"],
    "cheerful and lively dining atmosphere": ["cheerful dining atmosphere", "lively dining atmosphere"],
    "clean and friendly dining experience": ["clean dining experience", "friendly dining experience"],
    "cozy and friendly dining experience": ["cozy dining experience", "friendly dining experience"],
    "cozy, minimalist setting": ["cozy setting", "minimalist setting"],
    "fun and lively ambiance": ["fun ambiance", "lively ambiance"],
    "spacious and classy ambiance": ["spacious ambiance", "classy ambiance"],
    "trendy and Instagram-worthy setting": ["trendy setting", "Instagram-worthy setting"],
    "moist, flavorful cakes": ["moist cakes", "flavorful cakes"],
    "variety of unique and delicious flavors": ["variety of unique flavors", "variety of delicious flavors"],
    "Fans of affordable and tasty Mexican food": ["Fans of affordable Mexican food", "Fans of tasty Mexican food"],
    "comfortable and intimate setting": ["comfortable setting", "intimate setting"],
    "cozy and friendly cafe environment": ["cozy cafe environment", "friendly cafe environment"],
    "cozy, casual dining spots": ["cozy dining spots", "casual dining spots"],
    "cozy, charming ambiance": ["cozy ambiance", "charming ambiance"],
    "comfortable and relaxed atmosphere": ["comfortable atmosphere", "relaxed atmosphere"],
    "knowledgeable and friendly bartenders": ["knowledgeable bartenders", "friendly bartenders"],
    "vegetarian and vegan Indian cuisine": ["vegetarian Indian cuisine", "vegan Indian cuisine"],
    "Fans of cheap and delicious Mexican food": ["Fans of cheap Mexican food", "Fans of delicious Mexican food"],
    "Fans of authentic, high-quality sandwiches": ["Fans of authentic sandwiches", "Fans of high-quality sandwiches"],
    "Fans of flavorful and diverse Thai cuisine": ["Fans of flavorful Thai cuisine", "Fans of diverse Thai cuisine"],
    "fun and casual dining experience": ["fun dining experience", "casual dining experience"],
    "friendly and charming atmosphere": ["friendly atmosphere", "charming atmosphere"],
    "clean and friendly environments": ["clean environments", "friendly environments"],
    "clean and welcoming setting": ["clean setting", "welcoming setting"],
    "cute and cozy hangout spot": ["cute hangout spot", "cozy hangout spot"],
    "friendly laid-back atmosphere": ["friendly atmosphere", "laid-back atmosphere"],
    "friendly, welcoming atmosphere": ["friendly atmosphere", "welcoming atmosphere"],
    "delightful and satisfying dining experience": ["delightful dining experience", "satisfying dining experience"],
    "clean and spacious facilities": ["clean facilities", "spacious facilities"],
    "unique and creative ice cream flavors": ["unique ice cream flavors", "creative ice cream flavors"],
    "unique and flavorful offerings": ["unique offerings", "flavorful offerings"],
    "knowledgeable and helpful staff": ["knowledgeable staff", "helpful staff"],
    "artistic and unique dishes": ["artistic dishes", "unique dishes"],
    "delicious and unique food options": ["delicious food options", "unique food options"],
    "Fans of flavorful and affordable Mexican food": ["Fans of flavorful Mexican food", "Fans of affordable Mexican food"],
    "Fans of fresh and flavorful Thai cuisine": ["Fans of fresh Thai cuisine", "Fans of flavorful Thai cuisine"],
    "Fans of fresh and unique sushi": ["Fans of fresh sushi", "Fans of unique sushi"],
    "cozy and serene atmosphere": ["cozy atmosphere", "serene atmosphere"],
    "casual and elegant dining": ["casual dining", "elegant dining"],
    "clean and friendly ambiance": ["clean ambiance", "friendly ambiance"],
    "cozy, casual cafe atmosphere": ["cozy cafe atmosphere", "casual cafe atmosphere"],
    "cozy, inviting atmosphere": ["cozy atmosphere", "inviting atmosphere"],
    "unique and creative food combinations": ["unique food combinations", "creative food combinations"],
    "unique and diverse beer selections": ["unique beer selections", "diverse beer selections"],
    "unique and quirky dining experience": ["unique dining experience", "quirky dining experience"],
    "unique and delicious brunch options": ["unique brunch options", "delicious brunch options"],
    "people who enjoy a friendly and lively atmosphere": ["people who enjoy a friendly atmosphere", "people who enjoy a lively atmosphere"],
    "casual, relaxed ambiance": ["casual ambiance", "relaxed ambiance"],
    "casual, lively setting": ["casual setting", "lively setting"],
    "relaxed, upscale setting": ["relaxed setting", "upscale setting"],
    "roomy, clean dining spaces": ["roomy dining spaces", "clean dining spaces"],
    "relaxed hipster ambiance": ["relaxed ambiance", "hipster ambiance"],
    "relaxed, cozy environment": ["relaxed environment", "cozy environment"],
    "trendy and casual dining atmosphere": ["trendy dining atmosphere", "casual dining atmosphere"],
    "Flavorful and delicious food": ["Flavorful food", "delicious food"],
    "affordable and generous breakfast options": ["affordable breakfast options", "generous breakfast options"],
    "social and interactive dining experience": ["social dining experience", "interactive dining experience"],
    "serene, tranquil atmosphere": ["serene atmosphere", "tranquil atmosphere"],
    "delicious and diverse menu offerings": ["delicious menu offerings", "diverse menu offerings"],
    "casual, laid-back dining experiences": ["casual dining experiences", "laid-back dining experiences"],
    "cozy and elegant ambiance": ["cozy ambiance", "elegant ambiance"],
    "lively and pleasant atmosphere": ["lively atmosphere", "pleasant atmosphere"],
    "well-decorated and friendly environment": ["well-decorated environment", "friendly environment"],
    "casual, family-friendly dining experience": ["casual dining experience", "family-friendly dining experience"],
    "cozy and cheerful atmosphere": ["cozy atmosphere", "cheerful atmosphere"],
    "spacious and clean dining experience": ["spacious dining experience", "clean dining experience"],
    "elegant and diverse menus": ["elegant menus", "diverse menus"],
    "unique and diverse menu offerings": ["unique menu offerings", "diverse menu offerings"],
    "flavorful and creative Mexican cuisine": ["flavorful Mexican cuisine", "creative Mexican cuisine"],
    "friendly owners and staff": ["friendly owners", "friendly staff"],
    "spacious indoor and outdoor seating": ["spacious indoor seating", "spacious outdoor seating"],
    "upscale and casual dining experiences": ["upscale dining experiences", "casual dining experiences"],
    "creative and original dishes": ["creative dishes", "original dishes"],
    "delicious, flavorful burgers": ["delicious burgers", "flavorful burgers"],
    "vegan and traditional pizzas": ["vegan pizzas", "traditional pizzas"],
    "dining with family and friends": ["dining with family", "dining with friends"],
    "classy and clean dining environment": ["classy dining environment", "clean dining environment"],
    "clean and safe environment": ["clean environment", "safe environment"],
    "cozy and artsy atmosphere": ["cozy atmosphere", "artsy atmosphere"],
    "cozy and warm dining atmosphere": ["cozy dining atmosphere", "warm dining atmosphere"],
    "friendly and efficient customer service": ["friendly customer service", "efficient customer service"],
    "friendly and intimate setting": ["friendly setting", "intimate setting"],
    "sleek, minimalist ambiance": ["sleek ambiance", "minimalist ambiance"],
    "upscale and classy settings": ["upscale settings", "classy settings"],
    # From lines 1-100
    "$10 burger and beer special": ["$10 burger special", "$10 beer special"],
    "1950s/1960s atmosphere": ["1950s atmosphere", "1960s atmosphere"],
    # From lines 101-200
    "Adults looking for unique and nostalgic toys": ["Adults looking for unique toys", "Adults looking for nostalgic toys"],
    "African American heritage and culture": ["African American heritage", "African American culture"],
    "Air-Heating And Air Conditioning": ["Air-Heating", "Air Conditioning"],
    "American Cuisine & BBQ": ["American Cuisine", "BBQ"],
    "American and Mediterranean flavors": ["American flavors", "Mediterranean flavors"],
    "American art from the 18th and 19th century": ["American art from the 18th century", "American art from the 19th century"],
    # From lines 201-300
    "Animal Emergency and Specialty Center": ["Animal Emergency Center", "Animal Specialty Center"],
    "Art and culture": ["Art", "culture"],
    "Art and culture enthusiasts": ["Art enthusiasts", "culture enthusiasts"],
    "Arts & entertainment enthusiasts": ["Arts enthusiasts", "entertainment enthusiasts"],
    "Arts and crafts enthusiasts": ["Arts enthusiasts", "crafts enthusiasts"],
    "Arts and crafts store": ["Arts store", "crafts store"],
    "Arts and entertainment enthusiasts": ["Arts enthusiasts", "entertainment enthusiasts"],
    "Arts and music enthusiasts": ["Arts enthusiasts", "music enthusiasts"],
    "Arts and music venue": ["Arts venue", "music venue"],
    # From lines 301-400
    "BBQ and catering": ["BBQ", "catering"],
    # From lines 401-500
    "Bagel & Deli": ["Bagel", "Deli"],
    "Bakery & Cafe": ["Bakery", "Cafe"],
    "Bakery & Store": ["Bakery", "Store"],
    "Bakery and Cafe": ["Bakery", "Cafe"],
    "Bakery and Cuisine": ["Bakery", "Cuisine"],
    "Bangers and Mash": ["Bangers", "Mash"],
    "Banh Mi Bar & Bistro": ["Banh Mi Bar", "Bistro"],
    "Bar & Chicken": ["Bar", "Chicken"],
    "Bar & Deli": ["Bar", "Deli"],
    "Bar & Kitchen": ["Bar", "Kitchen"],
    "Bar & Lounge": ["Bar", "Lounge"],
    "Bar & Restaurant": ["Bar", "Restaurant"],
    "Bar & Urban Kitchen": ["Bar", "Urban Kitchen"],
    "Bar + Lounge": ["Bar", "Lounge"],
    "Bar / Pub": ["Bar", "Pub"],
    "Bar and Kitchen": ["Bar", "Kitchen"],
    "Bar and cafe": ["Bar", "cafe"],
    "Bar and music venue": ["Bar", "music venue"],
    "Bar and pub": ["Bar", "pub"],
    # From lines 501-600
    "Bar-Cafe": ["Bar", "Cafe"],
    "Bar/Club": ["Bar", "Club"],
    "Bar/Grill": ["Bar", "Grill"],
    "Bavarian and German cuisine": ["Bavarian cuisine", "German cuisine"],
    "Beer and wine connoisseurs": ["Beer connoisseurs", "wine connoisseurs"],
    # From lines 601-700
    "Board Game Bar & Cafe": ["Board Game Bar", "Cafe"],
    "Bodega Kitchen & Wine": ["Bodega Kitchen", "Wine"],
    "Bottle & Bistro": ["Bottle", "Bistro"],
    "Breakfast & Lunch": ["Breakfast", "Lunch"],
    "Breakfast & Lunch restaurant": ["Breakfast restaurant", "Lunch restaurant"],
    "Breakfast & brunch restaurant": ["Breakfast restaurant", "brunch restaurant"],
    "Breakfast and Burgers": ["Breakfast", "Burgers"],
    "Breakfast and brunch lovers": ["Breakfast lovers", "brunch lovers"],
    "Breakfast and brunch restaurant": ["Breakfast restaurant", "brunch restaurant"],
    "Breakfast, Lunch and Tacos": ["Breakfast", "Lunch", "Tacos"],
    "Breakfast/Brunch restaurant": ["Breakfast restaurant", "Brunch restaurant"],
    "Brewery & Distillery": ["Brewery", "Distillery"],
    "Brewery & Restaurant": ["Brewery", "Restaurant"],
    "Brewery/Wine shop": ["Brewery shop", "Wine shop"],
    "Brie & bacon crepe": ["Brie crepe", "bacon crepe"],
    "British pub & restaurant": ["British pub", "restaurant"],
    # From lines 701-800
    "Buffet & Grille": ["Buffet", "Grille"],
    "Burger and fries enthusiasts": ["Burger enthusiasts", "fries enthusiasts"],
    "Burgers & Booze": ["Burgers", "Booze"],
    "Cafe & Bakery": ["Cafe", "Bakery"],
    "Cafe & Bar": ["Cafe", "Bar"],
    "Cafe & Deli": ["Cafe", "Deli"],
    "Cafe & Gallery": ["Cafe", "Gallery"],
    "Cafe & Lounge": ["Cafe", "Lounge"],
    "Cafe & Market": ["Cafe", "Market"],
    "Cafe & Pizzeria": ["Cafe", "Pizzeria"],
    "Cafe or Restaurant": ["Cafe", "Restaurant"],
    "Cajun shrimp and grits": ["Cajun shrimp", "grits"],
    # From lines 801-900
    "Cantina & Grill": ["Cantina", "Grill"],
    "Casino & Hotel": ["Casino", "Hotel"],
    "Casino & Restaurant": ["Casino", "Restaurant"],
    "Ceviche Tapas Bar And Restaurant": ["Ceviche Tapas Bar", "Restaurant"],
    # From lines 901-1000
    "Cheese and wine enthusiasts": ["Cheese enthusiasts", "wine enthusiasts"],
    # From lines 1001-1100
    "Christmas decorations and lighting": ["Christmas decorations", "lighting"],
    "Cider and mead enthusiasts": ["Cider enthusiasts", "mead enthusiasts"],
    "Clients seeking luxury and quality grooming services": ["Clients seeking luxury grooming services", "Clients seeking quality grooming services"],
    "Clothing and household goods": ["Clothing", "household goods"],
    "Coffee & Bakehouse": ["Coffee", "Bakehouse"],
    "Coffee & Cocktails": ["Coffee", "Cocktails"],
    "Coffee & tea lovers": ["Coffee lovers", "tea lovers"],
    "Coffee and Foods": ["Coffee", "Foods"],
    "Coffee and tea enthusiasts": ["Coffee enthusiasts", "tea enthusiasts"],
    # From lines 1101-1200
    "Costume & Dancewear store": ["Costume store", "Dancewear store"],
    "Country Store & Farms": ["Country Store", "Farms"],
    "Craft Beer & Burgers": ["Craft Beer", "Burgers"],
    "Craft Wine & Beer": ["Craft Wine", "Beer"],
    # From lines 1201-1300
    "Customers seeking a mature, sophisticated bar experience": ["Customers seeking a mature bar experience", "Customers seeking a sophisticated bar experience"],
    "Customers who appreciate easy checkout processes and a large selection of Apple products": ["Customers who appreciate easy checkout processes", "Customers who appreciate a large selection of Apple products"],
    "Customers who enjoy trying new breakfast and brunch spots": ["Customers who enjoy trying new breakfast spots", "Customers who enjoy trying new brunch spots"],
    "Customers who value cleanliness, personalized service, and attention to detail": ["Customers who value cleanliness", "Customers who value personalized service", "Customers who value attention to detail"],
    # From lines 1301-1400
    "Diner & Pub": ["Diner", "Pub"],
    "Dining & Catering": ["Dining", "Catering"],
    # From lines 1401-1500
    "Doughnuts & Dragons": ["Doughnuts", "Dragons"],
    "Eiteljorg Museum of American Indians & Western Art": ["Eiteljorg Museum of American Indians", "Western Art"],
    "Electric & HVAC services": ["Electric services", "HVAC services"],
    # From lines 1501-1600
    "Ethiopian culture and history": ["Ethiopian culture", "Ethiopian history"],
    "European-inspired breakfast and brunch options": ["European-inspired breakfast options", "European-inspired brunch options"],
    "Exceptional food and ambiance": ["Exceptional food", "Exceptional ambiance"],
    "Exploration and Innovation": ["Exploration", "Innovation"],
    "Families interested in a wide variety of games, educational toys, and interactive train sets": ["Families interested in a wide variety of games", "Families interested in educational toys", "Families interested in interactive train sets"],
    # From lines 1601-1700
    "Fans of Dominican and Latin American cuisine": ["Fans of Dominican cuisine", "Fans of Latin American cuisine"],
    # From lines 1701-1800
    "Fans of Italian culture and cooking": ["Fans of Italian culture", "Fans of Italian cooking"],
    "Fans of Korean and Vietnamese cuisine": ["Fans of Korean cuisine", "Fans of Vietnamese cuisine"],
    "Fans of Mediterranean, Italian, and BYO restaurants": ["Fans of Mediterranean restaurants", "Fans of Italian restaurants", "Fans of BYO restaurants"],
    "Fans of Mexican, Honduran, and Central American cuisine": ["Fans of Mexican cuisine", "Fans of Honduran cuisine", "Fans of Central American cuisine"],
    # From lines 1801-1900
    "Fans of Pakistani and Indian cuisine": ["Fans of Pakistani cuisine", "Fans of Indian cuisine"],
    "Fans of Southern, Italian and Cajun/Creole food": ["Fans of Southern food", "Fans of Italian food", "Fans of Cajun/Creole food"],
    "Fans of Taiwanese and Asian fusion cuisine": ["Fans of Taiwanese cuisine", "Fans of Asian fusion cuisine"],
    "Fans of Tex-Mex and American cuisine": ["Fans of Tex-Mex cuisine", "Fans of American cuisine"],
    "Fans of Thai and Lao cuisine": ["Fans of Thai cuisine", "Fans of Lao cuisine"],
    "Fans of Turkish and Mediterranean cuisine": ["Fans of Turkish cuisine", "Fans of Mediterranean cuisine"],
    "Fans of Western movies and TV shows": ["Fans of Western movies", "Fans of TV shows"],
    "Fans of affordable and generous breakfast options": ["Fans of affordable breakfast options", "Fans of generous breakfast options"],
    # From lines 1901-2000
    "Fans of authentic and innovative ramen": ["Fans of authentic ramen", "Fans of innovative ramen"],
    "Fans of authentic, affordable, and nostalgic dining experiences": ["Fans of authentic dining experiences", "Fans of affordable dining experiences", "Fans of nostalgic dining experiences"],
    "Fans of big, juicy burgers": ["Fans of big burgers", "Fans of juicy burgers"],
    # From lines 2001-2100
    "Fans of bold and savory flavors": ["Fans of bold flavors", "Fans of savory flavors"],
    "Fans of breakfast & brunch": ["Fans of breakfast", "Fans of brunch"],
    "Fans of breakfast and Mexican cuisine": ["Fans of breakfast", "Fans of Mexican cuisine"],
    "Fans of casual and fun bars": ["Fans of casual bars", "Fans of fun bars"],
    "Fans of charming, cozy atmospheres": ["Fans of charming atmospheres", "Fans of cozy atmospheres"],
    "Fans of cheap and delicious Salvadoran food": ["Fans of cheap Salvadoran food", "Fans of delicious Salvadoran food"],
    "Fans of chic and artsy ambiance": ["Fans of chic ambiance", "Fans of artsy ambiance"],
    # From lines 2101-2200
    "Fans of cozy, dive bars": ["Fans of cozy bars", "Fans of dive bars"],
    "Fans of cozy, historic settings": ["Fans of cozy settings", "Fans of historic settings"],
    "Fans of cozy, intimate settings": ["Fans of cozy settings", "Fans of intimate settings"],
    "Fans of cozy, trendy brunch spots": ["Fans of cozy brunch spots", "Fans of trendy brunch spots"],
    "Fans of cozy, unique cafes": ["Fans of cozy cafes", "Fans of unique cafes"],
    "Fans of creamy, rich ice cream": ["Fans of creamy ice cream", "Fans of rich ice cream"],
    "Fans of creative and unconventional donuts": ["Fans of creative donuts", "Fans of unconventional donuts"],
    "Fans of crispy, flavorful pizzas": ["Fans of crispy pizzas", "Fans of flavorful pizzas"],
    # From lines 2201-2300
    "Fans of delicious and affordable sandwich options": ["Fans of delicious sandwich options", "Fans of affordable sandwich options"],
    "Fans of delicious and natural frozen treats": ["Fans of delicious frozen treats", "Fans of natural frozen treats"],
    "Fans of diverse and delicious menu items": ["Fans of diverse menu items", "Fans of delicious menu items"],
    "Fans of diverse and flavorful Mexican-themed pizzas": ["Fans of diverse Mexican-themed pizzas", "Fans of flavorful Mexican-themed pizzas"],
    "Fans of diverse and flavorful street food": ["Fans of diverse street food", "Fans of flavorful street food"],
    "Fans of diverse and unique hot dog options": ["Fans of diverse hot dog options", "Fans of unique hot dog options"],
    "Fans of diverse, delicious, and eco-conscious food": ["Fans of diverse food", "Fans of delicious food", "Fans of eco-conscious food"],
    # From lines 2301-2400
    "Fans of fast and friendly service": ["Fans of fast service", "Fans of friendly service"],
    "Fans of flavorful Caribbean and Puerto Rican cuisine": ["Fans of flavorful Caribbean cuisine", "Fans of flavorful Puerto Rican cuisine"],
    "Fans of flavorful Mexican and Tex-Mex cuisine": ["Fans of flavorful Mexican cuisine", "Fans of flavorful Tex-Mex cuisine"],
    "Fans of flavorful and customizable comfort food": ["Fans of flavorful comfort food", "Fans of customizable comfort food"],
    "Fans of flavorful and modern Thai cuisine": ["Fans of flavorful Thai cuisine", "Fans of modern Thai cuisine"],
    "Fans of fresh and authentic Japanese sushi": ["Fans of fresh Japanese sushi", "Fans of authentic Japanese sushi"],
    "Fans of fresh and creative bagels": ["Fans of fresh bagels", "Fans of creative bagels"],
    "Fans of fresh and customizable Mexican seafood": ["Fans of fresh Mexican seafood", "Fans of customizable Mexican seafood"],
    "Fans of fresh and customizable food": ["Fans of fresh food", "Fans of customizable food"],
    "Fans of fresh and delicious sushi": ["Fans of fresh sushi", "Fans of delicious sushi"],
    "Fans of fresh and flavorful seafood": ["Fans of fresh seafood", "Fans of flavorful seafood"],
    "Fans of fresh and flavorful thin-crust pizza": ["Fans of fresh thin-crust pizza", "Fans of flavorful thin-crust pizza"],
    "Fans of fresh and gourmet deli sandwiches": ["Fans of fresh deli sandwiches", "Fans of gourmet deli sandwiches"],
    "Fans of fresh and healthy Asian fusion food": ["Fans of fresh Asian fusion food", "Fans of healthy Asian fusion food"],
    "Fans of fresh and healthy options": ["Fans of fresh options", "Fans of healthy options"],
    "Fans of fresh and tasty deli sandwiches": ["Fans of fresh deli sandwiches", "Fans of tasty deli sandwiches"],
    "Fans of fresh and tasty deli-style sandwiches": ["Fans of fresh deli-style sandwiches", "Fans of tasty deli-style sandwiches"],
    "Fans of fresh and tasty food": ["Fans of fresh food", "Fans of tasty food"],
    # From lines 2401-2500
    "Fans of fresh, flavorful deli sandwiches": ["Fans of fresh deli sandwiches", "Fans of flavorful deli sandwiches"],
    "Fans of fresh, flavorful pizza": ["Fans of fresh pizza", "Fans of flavorful pizza"],
    "Fans of fun and new dishes": ["Fans of fun dishes", "Fans of new dishes"],
    "Fans of fun and photogenic spots": ["Fans of fun spots", "Fans of photogenic spots"],
    "Fans of funky, fun eateries": ["Fans of funky eateries", "Fans of fun eateries"],
    "Fans of gel and regular manis": ["Fans of gel manis", "Fans of regular manis"],
    "Fans of healthy breakfast and brunch": ["Fans of healthy breakfast", "Fans of healthy brunch"],
    "Fans of hearty breakfast and brunch": ["Fans of hearty breakfast", "Fans of hearty brunch"],
    "Fans of hearty breakfast and brunch options": ["Fans of hearty breakfast options", "Fans of hearty brunch options"],
    "Fans of hearty, large portioned sandwiches": ["Fans of hearty sandwiches", "Fans of large portioned sandwiches"],
    # From lines 2501-2600
    "Fans of homemade chips and salsa": ["Fans of homemade chips", "Fans of homemade salsa"],
    "Fans of indulgent and unique comfort food": ["Fans of indulgent comfort food", "Fans of unique comfort food"],
    "Fans of innovative and delicious toast creations": ["Fans of innovative toast creations", "Fans of delicious toast creations"],
    "Fans of intimate breakfast/brunch spots": ["Fans of intimate breakfast spots", "Fans of intimate brunch spots"],
    "Fans of intimate, cozy ambiance": ["Fans of intimate ambiance", "Fans of cozy ambiance"],
    "Fans of live jazz and blues music": ["Fans of live jazz music", "Fans of live blues music"],
    # From lines 2601-2700
    "Fans of pizza and wings": ["Fans of pizza", "Fans of wings"],
    # From lines 2701-2800
    "Fans of quirky and unique dive bars": ["Fans of quirky dive bars", "Fans of unique dive bars"],
    "Fans of retro, laid-back atmospheres": ["Fans of retro atmospheres", "Fans of laid-back atmospheres"],
    "Fans of rich, decadent cakes": ["Fans of rich cakes", "Fans of decadent cakes"],
    "Fans of simple and delicious dinners": ["Fans of simple dinners", "Fans of delicious dinners"],
    "Fans of southern and traditional American BBQ": ["Fans of southern BBQ", "Fans of traditional American BBQ"],
    "Fans of spicy and flavorful Thai food": ["Fans of spicy Thai food", "Fans of flavorful Thai food"],
    "Fans of steak and seafood": ["Fans of steak", "Fans of seafood"],
    # From lines 2801-2900
    "Fans of sweet and savory breakfast options": ["Fans of sweet breakfast options", "Fans of savory breakfast options"],
    "Fans of sweet and savory options": ["Fans of sweet options", "Fans of savory options"],
    "Fans of sweet and savory pastries": ["Fans of sweet pastries", "Fans of savory pastries"],
    "Fans of tasty, healthy, quality food": ["Fans of tasty food", "Fans of healthy food", "Fans of quality food"],
    "Fans of tender, flavorsome BBQ": ["Fans of tender BBQ", "Fans of flavorsome BBQ"],
    "Fans of traditional delis and authentic kosher cuisine": ["Fans of traditional delis", "Fans of authentic kosher cuisine"],
    "Fans of trendy and casual eateries": ["Fans of trendy eateries", "Fans of casual eateries"],
    "Fans of trendy bar and restaurant scenes": ["Fans of trendy bar scenes", "Fans of trendy restaurant scenes"],
    "Fans of trendy, upscale Asian fusion cuisine": ["Fans of trendy Asian fusion cuisine", "Fans of upscale Asian fusion cuisine"],
    "Fans of unique and creative coffee shops": ["Fans of unique coffee shops", "Fans of creative coffee shops"],
    "Fans of unique and creative toasted subs": ["Fans of unique toasted subs", "Fans of creative toasted subs"],
    "Fans of unique and customizable desserts": ["Fans of unique desserts", "Fans of customizable desserts"],
    "Fans of unique and flavorful Japanese-Vietnamese fusion cuisine": ["Fans of unique Japanese-Vietnamese fusion cuisine", "Fans of flavorful Japanese-Vietnamese fusion cuisine"],
    "Fans of unique and flavorful bubble teas": ["Fans of unique bubble teas", "Fans of flavorful bubble teas"],
    "Fans of unique and flavorful healthy food": ["Fans of unique healthy food", "Fans of flavorful healthy food"],
    "Fans of unique and funky cocktail bars": ["Fans of unique cocktail bars", "Fans of funky cocktail bars"],
    "Fans of unique and historic confectionaries": ["Fans of unique confectionaries", "Fans of historic confectionaries"],
    "Fans of unique and intimate bar experiences": ["Fans of unique bar experiences", "Fans of intimate bar experiences"],
    # From lines 2901-3000
    "Fans of unique drinks and food": ["Fans of unique drinks", "Fans of unique food"],
    "Fans of unique, themed nightlife experiences": ["Fans of unique nightlife experiences", "Fans of themed nightlife experiences"],
    "Fans of upscale Tex-Mex and Mexican cuisine": ["Fans of upscale Tex-Mex cuisine", "Fans of upscale Mexican cuisine"],
    "Fashion and home decor store": ["Fashion store", "home decor store"],
    # From lines 3001-3100
    "Fig and Pig flatbread": ["Fig flatbread", "Pig flatbread"],
    "Fish & chips enthusiasts": ["Fish enthusiasts", "chips enthusiasts"],
    "Fish and Chip Shop": ["Fish Shop", "Chip Shop"],
    "Food & Brewery": ["Food", "Brewery"],
    "Food & Coffee Bar": ["Food", "Coffee Bar"],
    "Food & Drink": ["Food", "Drink"],
    "Food + Booze": ["Food", "Booze"],
    "Food + Brew": ["Food", "Brew"],
    "Food + Drink": ["Food", "Drink"],
    "Food and Juice Bar": ["Food", "Juice Bar"],
    "Food and Wine": ["Food", "Wine"],
    "Food and beverage venue": ["Food venue", "beverage venue"],
    "Food and drink establishment": ["Food establishment", "drink establishment"],
    # From lines 3101-3200
    "Fragrance and beauty store": ["Fragrance store", "beauty store"],
    "French/European drinks": ["French drinks", "European drinks"],
    "Fresh and authentic food enthusiasts": ["Fresh food enthusiasts", "authentic food enthusiasts"],
    "Friends and family": ["Friends", "family"],
    "Frozen Yogurt & Smoothies": ["Frozen Yogurt", "Smoothies"],
    "Fun and unique experience": ["Fun experience", "unique experience"],
    # From lines 3201-3300
    "Gastropub & Pizzeria": ["Gastropub", "Pizzeria"],
    "Gelato & Dessert cafe": ["Gelato", "Dessert cafe"],
    "German/European dishes": ["German dishes", "European dishes"],
    "Gourmet Burgers and Brews": ["Gourmet Burgers", "Brews"],
    # From lines 3301-3400
    "Grill & Cafe": ["Grill", "Cafe"],
    "Grocery & Deli": ["Grocery", "Deli"],
    "Grocery & Restaurant": ["Grocery", "Restaurant"],
    "Groups looking for a great atmosphere and outstanding food for events": ["Groups looking for a great atmosphere for events", "Groups looking for outstanding food for events"],
    "Groups looking for accommodating and friendly service": ["Groups looking for accommodating service", "Groups looking for friendly service"],
    "Groups seeking easy reservations and large party accommodations": ["Groups seeking easy reservations", "Groups seeking large party accommodations"],
    # From lines 3401-3500
    "Hawaiian and island-style dining": ["Hawaiian dining", "island-style dining"],
    "Health & Nutrition": ["Health", "Nutrition"],
    "Heating and Air Conditioning services": ["Heating services", "Air Conditioning services"],
    "Heating and cooling units": ["Heating units", "cooling units"],
    "Hibachi Steak House & Sushi Bar": ["Hibachi Steak House", "Sushi Bar"],
    # From lines 3501-3600
    "High-quality food and service enthusiasts": ["High-quality food enthusiasts", "High-quality service enthusiasts"],
    "Hot and Spicy Chicken": ["Hot Chicken", "Spicy Chicken"],
    "Hotcakes Emporium Pancake House & Restaurant": ["Hotcakes Emporium Pancake House", "Restaurant"],
    "Hotel & Casino": ["Hotel", "Casino"],
    # From lines 3601-3700
    "Ian's Tire and Auto Repair": ["Ian's Tire", "Auto Repair"],
    "Ice Cream & Coffee shop": ["Ice Cream", "Coffee shop"],
    "Indian and Pakistani restaurant": ["Indian restaurant", "Pakistani restaurant"],
    "Individuals and families": ["Individuals", "families"],
    "Individuals craving greasy, cheesy goodness": ["Individuals craving greasy goodness", "Individuals craving cheesy goodness"],
    "Individuals interested in animals and wildlife": ["Individuals interested in animals", "Individuals interested in wildlife"],
    "Individuals interested in heritage and culture": ["Individuals interested in heritage", "Individuals interested in culture"],
    "Individuals interested in historical sites and education": ["Individuals interested in historical sites", "Individuals interested in education"],
    "Individuals interested in trying new and tasty food options": ["Individuals interested in trying new food options", "Individuals interested in trying tasty food options"],
    "Individuals interested in wide variety of food and drink": ["Individuals interested in wide variety of food", "Individuals interested in wide variety of drink"],
    "Individuals looking for a clean and friendly atmosphere": ["Individuals looking for a clean atmosphere", "Individuals looking for a friendly atmosphere"],
    "Individuals looking for a cool and modern dining experience": ["Individuals looking for a cool dining experience", "Individuals looking for a modern dining experience"],
    # From lines 3701-3800
    "Individuals looking for a cozy, artistic atmosphere": ["Individuals looking for a cozy atmosphere", "Individuals looking for an artistic atmosphere"],
    "Individuals looking for a high-quality and flavorful dishes": ["Individuals looking for a high-quality dishes", "Individuals looking for a flavorful dishes"],
    "Individuals looking for a unique and fun group activity": ["Individuals looking for a unique group activity", "Individuals looking for a fun group activity"],
    "Individuals seeking a friendly and talented salon experience": ["Individuals seeking a friendly salon experience", "Individuals seeking a talented salon experience"],
    "Individuals seeking a unique and intimate bar experience": ["Individuals seeking a unique bar experience", "Individuals seeking an intimate bar experience"],
    "Individuals seeking affordable and efficient haircuts": ["Individuals seeking affordable haircuts", "Individuals seeking efficient haircuts"],
    "Individuals seeking convenient and clean movie-going experience": ["Individuals seeking convenient movie-going experience", "Individuals seeking clean movie-going experience"],
    "Individuals seeking friendly and knowledgeable butchers": ["Individuals seeking friendly butchers", "Individuals seeking knowledgeable butchers"],
    "Individuals seeking fun and active day out": ["Individuals seeking fun day out", "Individuals seeking active day out"],
    "Individuals seeking local and unique food items": ["Individuals seeking local food items", "Individuals seeking unique food items"],
    "Individuals seeking personalized and attentive service": ["Individuals seeking personalized service", "Individuals seeking attentive service"],
    "Individuals seeking personalized and trendy haircuts": ["Individuals seeking personalized haircuts", "Individuals seeking trendy haircuts"],
    "Individuals seeking piercings and tattoos": ["Individuals seeking piercings", "Individuals seeking tattoos"],
    # From lines 3801-3900
    "Individuals seeking tasty breakfast and brunch options": ["Individuals seeking tasty breakfast options", "Individuals seeking tasty brunch options"],
    "Individuals who appreciate beautiful views and public art": ["Individuals who appreciate beautiful views", "Individuals who appreciate public art"],
    "Individuals who appreciate chic and trendy atmospheres": ["Individuals who appreciate chic atmospheres", "Individuals who appreciate trendy atmospheres"],
    "Individuals who appreciate creative and unique treats": ["Individuals who appreciate creative treats", "Individuals who appreciate unique treats"],
    "Individuals who appreciate flavorful and authentic dishes": ["Individuals who appreciate flavorful dishes", "Individuals who appreciate authentic dishes"],
    "Individuals who appreciate fresh and flavorful Mediterranean food": ["Individuals who appreciate fresh Mediterranean food", "Individuals who appreciate flavorful Mediterranean food"],
    "Individuals who appreciate high-quality seafood and steak": ["Individuals who appreciate high-quality seafood", "Individuals who appreciate high-quality steak"],
    "Individuals who appreciate quality and local products": ["Individuals who appreciate quality products", "Individuals who appreciate local products"],
    "Individuals who appreciate unique and locally-inspired fashion": ["Individuals who appreciate unique fashion", "Individuals who appreciate locally-inspired fashion"],
    "Individuals who enjoy Mexican cuisine and Southern cuisine": ["Individuals who enjoy Mexican cuisine", "Individuals who enjoy Southern cuisine"],
    "Individuals who enjoy Vietnamese cuisine, vegetarian options, and bubble tea": ["Individuals who enjoy Vietnamese cuisine", "Individuals who enjoy vegetarian options", "Individuals who enjoy bubble tea"],
    "Individuals who enjoy a casual and fun environment": ["Individuals who enjoy a casual environment", "Individuals who enjoy a fun environment"],
    "Individuals who enjoy a spacious and renovated setting": ["Individuals who enjoy a spacious setting", "Individuals who enjoy a renovated setting"],
    "Individuals who enjoy affordable and quick Indian/Pakistani cuisine": ["Individuals who enjoy affordable Indian/Pakistani cuisine", "Individuals who enjoy quick Indian/Pakistani cuisine"],
    "Individuals who enjoy art and creativity": ["Individuals who enjoy art", "Individuals who enjoy creativity"],
    # From lines 3901-4000
    "Individuals who enjoy bowling and arcade games": ["Individuals who enjoy bowling", "Individuals who enjoy arcade games"],
    "Individuals who enjoy customizable and quick-service dining experiences": ["Individuals who enjoy customizable dining experiences", "Individuals who enjoy quick-service dining experiences"],
    "Individuals who enjoy fast and customizable pizzas": ["Individuals who enjoy fast pizzas", "Individuals who enjoy customizable pizzas"],
    "Individuals who enjoy fresh and flavorful Mediterranean food": ["Individuals who enjoy fresh Mediterranean food", "Individuals who enjoy flavorful Mediterranean food"],
    "Individuals who enjoy fresh, local produce": ["Individuals who enjoy fresh produce", "Individuals who enjoy local produce"],
    "Individuals who enjoy gluten-free, vegetarian, and vegan comfort food": ["Individuals who enjoy gluten-free comfort food", "Individuals who enjoy vegetarian comfort food", "Individuals who enjoy vegan comfort food"],
    "Individuals who enjoy live music and outdoor seating": ["Individuals who enjoy live music", "Individuals who enjoy outdoor seating"],
    "Individuals who enjoy trendy and creative food": ["Individuals who enjoy trendy food", "Individuals who enjoy creative food"],
    "Individuals who enjoy trendy, creative cocktails": ["Individuals who enjoy trendy cocktails", "Individuals who enjoy creative cocktails"],
    "Individuals who enjoy trendy, upscale bars": ["Individuals who enjoy trendy bars", "Individuals who enjoy upscale bars"],
    "Individuals who enjoy trying new and adventurous dishes": ["Individuals who enjoy trying new dishes", "Individuals who enjoy trying adventurous dishes"],
    "Individuals who enjoy unique and trendy ice cream flavors": ["Individuals who enjoy unique ice cream flavors", "Individuals who enjoy trendy ice cream flavors"],
    "Individuals who enjoy unique, dive bar experiences": ["Individuals who enjoy unique bar experiences", "Individuals who enjoy dive bar experiences"],
    # From lines 4001-4100
    "Individuals who value clean, bright establishments": ["Individuals who value clean establishments", "Individuals who value bright establishments"],
    "Individuals with gluten or dairy intolerances": ["Individuals with gluten intolerances", "Individuals with dairy intolerances"],
    # From lines 4101-4200
    "Japanese Sushi and Hibachi restaurant": ["Japanese Sushi restaurant", "Hibachi restaurant"],
    "Japanese and Korean cuisine": ["Japanese cuisine", "Korean cuisine"],
    "Jazz & Heritage Festival": ["Jazz Festival", "Heritage Festival"],
    "Jazz & blues": ["Jazz", "blues"],
    "Jazz & blues enthusiasts": ["Jazz enthusiasts", "blues enthusiasts"],
    # From lines 4201-4300
    "Juice and smoothie bar": ["Juice bar", "smoothie bar"],
    "K-POT Korean BBQ & Hot Pot": ["K-POT Korean BBQ", "Hot Pot"],
    "Kitchen + Rooftop Bar": ["Kitchen", "Rooftop Bar"],
    "Korean and Chinese flavors": ["Korean flavors", "Chinese flavors"],
    "Kosher Cajun NY Deli & Grocery": ["Kosher Cajun NY Deli", "Grocery"],
    "Kosher Meat & Deli": ["Kosher Meat", "Deli"],
    "Kouzina Cafe Gyros and Subs": ["Kouzina Cafe Gyros", "Subs"],
    # From lines 4301-4400
    "Landmark & historical buildings enthusiasts": ["Landmark enthusiasts", "historical buildings enthusiasts"],
    # From lines 4401-4500
    "Lovers of cozy, intimate dining experiences": ["Lovers of cozy dining experiences", "Lovers of intimate dining experiences"],
    # From lines 4501-4600
    "MaGerk's Pub & Grill": ["Pub", "Grill"],
    "Market & Cafe": ["Market", "Cafe"],
    "Market & Delicatessen": ["Market", "Delicatessen"],
    "Market & Farm": ["Market", "Farm"],
    "Market & Kitchen": ["Market", "Kitchen"],
    "Massage & Wellness": ["Massage", "Wellness"],
    "Massage & Wellness Center": ["Massage Center", "Wellness Center"],
    "Meat and Deli": ["Meat", "Deli"],
    # From lines 4601-4700
    "Mediterranean/Middle Eastern cuisine": ["Mediterranean cuisine", "Middle Eastern cuisine"],
    "Mexican and Greek ambiance": ["Mexican ambiance", "Greek ambiance"],
    "Mexican and Japanese fusion": ["Mexican fusion", "Japanese fusion"],
    "Mexican restaurant and bar": ["Mexican restaurant", "bar"],
    # From lines 4701-4800
    "Middle Eastern/Moroccan dishes": ["Middle Eastern dishes", "Moroccan dishes"],
    "Midwestern and coastal vibes": ["Midwestern vibes", "coastal vibes"],
    "Modern Indian Food & Spirits": ["Modern Indian Food", "Spirits"],
    "Museum of American Indians & Western Art": ["Museum of American Indians", "Western Art"],
    # From lines 4801-4900
    "Music and arts enthusiasts": ["Music enthusiasts", "arts enthusiasts"],
    "Music and event enthusiasts": ["Music enthusiasts", "event enthusiasts"],
    "National Park and Preserve": ["National Park", "Preserve"],
    "Nature and animal lovers": ["Nature lovers", "animal lovers"],
    # From lines 4901-5000
    "New Orleans' history and heritage": ["New Orleans' history", "New Orleans' heritage"],
    # From lines 5001-5100
    "Outdoor & Bike Shop": ["Outdoor", "Bike Shop"],
    "Parks and Recreation area": ["Parks area", "Recreation area"],
    "Pasta and Market": ["Pasta", "Market"],
    # From lines 5101-5200
    "Persian/Iranian food": ["Persian food", "Iranian food"],
    "Pizza & Deli": ["Pizza", "Deli"],
    "Pizza & Grill": ["Pizza", "Grill"],
    "Pizza & Martini Bar": ["Pizza", "Martini Bar"],
    "Pizza & Pasta": ["Pizza", "Pasta"],
    # From lines 5201-5300
    "Plumbing and Heating service": ["Plumbing service", "Heating service"],
    "Produce and Juice Bar": ["Produce", "Juice Bar"],
    "Provisions and Spirits": ["Provisions", "Spirits"],
    "Pub & Deli": ["Pub", "Deli"],
    "Pub & Grill": ["Pub", "Grill"],
    "Pub & Grub": ["Pub", "Grub"],
    "Pub & Kitchen": ["Pub", "Kitchen"],
    "Pub & Restaurant": ["Pub", "Restaurant"],
    "Pub + Restaurant": ["Pub", "Restaurant"],
    "Pub and Grill": ["Pub", "Grill"],
    "Quick and satisfying meal": ["Quick meal", "satisfying meal"],
    # From lines 5301-5400
    "Red beans and rice enthusiasts": ["Red beans enthusiasts", "rice enthusiasts"],
    "Resort & Convention Center": ["Resort", "Convention Center"],
    "Resort & Spa": ["Resort", "Spa"],
    "Restaurant & Bar": ["Restaurant", "Bar"],
    "Restaurant & Brewhouse": ["Restaurant", "Brewhouse"],
    "Restaurant & Deli": ["Restaurant", "Deli"],
    "Restaurant & Grill": ["Restaurant", "Grill"],
    "Restaurant & Grocery": ["Restaurant", "Grocery"],
    "Restaurant & Inn": ["Restaurant", "Inn"],
    "Restaurant & Lounge": ["Restaurant", "Lounge"],
    "Restaurant and Brewery": ["Restaurant", "Brewery"],
    "Rotisserie and Bar": ["Rotisserie", "Bar"],
    # From lines 5401-5500
    "Salon & Spa": ["Salon", "Spa"],
    "Salon and Spa": ["Salon", "Spa"],
    "Sandwich and soup lovers": ["Sandwich lovers", "soup lovers"],
    "Satisfying and affordable dining experience": ["Satisfying dining experience", "affordable dining experience"],
    "Seafood & Steaks": ["Seafood", "Steaks"],
    # From lines 5501-5600
    "Shopping & Dining destination": ["Shopping destination", "Dining destination"],
    "Shopping and dining complex": ["Shopping complex", "dining complex"],
    "Small Plates & Noodle Bar": ["Small Plates", "Noodle Bar"],
    "Snack and Sweets shop": ["Snack shop", "Sweets shop"],
    # From lines 5601-5700
    "South/Central American cuisine": ["South American cuisine", "Central American cuisine"],
    "Southern and American menu": ["Southern menu", "American menu"],
    "Southern and American traditional cuisine": ["Southern traditional cuisine", "American traditional cuisine"],
    "Spa & Nails": ["Spa", "Nails"],
    "Spanish wine bar and restaurant": ["Spanish wine bar", "restaurant"],
    "Specialty food and cheese shop": ["Specialty food shop", "cheese shop"],
    "Spice & Tea Shoppe": ["Spice Shoppe", "Tea Shoppe"],
    # From lines 5701-5800
    "Sports Bar & Grill": ["Sports Bar", "Grill"],
    "Sports Bar & Patio": ["Sports Bar", "Patio"],
    "Sports bar and casual dining restaurant": ["Sports bar", "casual dining restaurant"],
    "Sports bar and grill": ["Sports bar", "grill"],
    "Steak & Seafood restaurant": ["Steak restaurant", "Seafood restaurant"],
    "Steak and seafood enthusiasts": ["Steak enthusiasts", "seafood enthusiasts"],
    "Steakhouse & Bar": ["Steakhouse", "Bar"],
    "Steakhouse & Italian Grill": ["Steakhouse", "Italian Grill"],
    # From lines 5801-5900
    "Tailor and Alterations service": ["Tailor service", "Alterations service"],
    "Taiwanese and Asian fusion cuisine": ["Taiwanese cuisine", "Asian fusion cuisine"],
    "Tap & Grill": ["Tap", "Grill"],
    "Taproom & Kitchen": ["Taproom", "Kitchen"],
    "Tattoo and piercing shop": ["Tattoo shop", "piercing shop"],
    "Tea and bakery": ["Tea", "bakery"],
    # From lines 5901-6000
    "Thai Cuisine & Sushi": ["Thai Cuisine", "Sushi"],
    "Thai and Asian fusion cuisine": ["Thai cuisine", "Asian fusion cuisine"],
    "The Hangar Restaurant & Flight Lounge": ["The Hangar Restaurant", "Flight Lounge"],
    # From lines 6001-6100
    "Those interested in socializing over drinks and food": ["Those interested in socializing over drinks", "Those interested in socializing over food"],
    # From lines 6101-6200
    "Unique and flavorful Mexican cuisine": ["Unique Mexican cuisine", "flavorful Mexican cuisine"],
    "Unique and flavorful breakfast": ["Unique breakfast", "flavorful breakfast"],
    "Urban and trendy individuals": ["Urban individuals", "trendy individuals"],
    "Vegan and organic consumers": ["Vegan consumers", "organic consumers"],
    "Vegan/vegetarian food enthusiasts": ["Vegan food enthusiasts", "vegetarian food enthusiasts"],
    "Vegan/vegetarian restaurant": ["Vegan restaurant", "vegetarian restaurant"],
    "Vegans and omnivores": ["Vegans", "omnivores"],
    # From lines 6201-6300
    "Vibrant and upbeat atmosphere": ["Vibrant atmosphere", "upbeat atmosphere"],
    "Vietnamese and Thai fusion cuisine": ["Vietnamese fusion cuisine", "Thai fusion cuisine"],
    "Wafflerie & Cafe": ["Wafflerie", "Cafe"],
    "Waffles & Ice Cream shop": ["Waffles shop", "Ice Cream shop"],
    # From lines 6301-6400
    "Waterside Pub & Patio": ["Waterside Pub", "Patio"],
    "Welcoming for adults and kids": ["Welcoming for adults", "Welcoming for kids"],
    "Whiskey/bourbon/scotch enthusiasts": ["Whiskey enthusiasts", "bourbon enthusiasts", "scotch enthusiasts"],
    "Wine & Beer": ["Wine", "Beer"],
    "Wine & Beer store": ["Wine store", "Beer store"],
    "Wine & Spirits": ["Wine", "Spirits"],
    "Wine & Spirits store": ["Wine store", "Spirits store"],
    "Wine & Whisky Bar": ["Wine", "Whisky Bar"],
    "Wine & Wood": ["Wine", "Wood"],
    "Wine Dive & Ripple Kitchen": ["Wine Dive", "Ripple Kitchen"],
    "Wine Market & Table": ["Wine Market", "Table"],
    "Wine and Burger Bar": ["Wine", "Burger Bar"],
    "Wine and cheese restaurant": ["Wine restaurant", "cheese restaurant"],
    "Wine and cheese shop": ["Wine shop", "cheese shop"],
    "Wine and spirit shop": ["Wine shop", "spirit shop"],
    "Wine, beer, and spirits store": ["Wine store", "beer store", "spirits store"],
    # From lines 6401-6500
    "Wood-Fired Grille & Bar": ["Wood-Fired Grille", "Bar"],
    "Wood-Fired Steaks & Seafood": ["Wood-Fired Steaks", "Seafood"],
    "Work or hang out": ["Work", "hang out"],
    "a clean and organized restaurant setting": ["a clean restaurant setting", "an organized restaurant setting"],
    "a cozy and historical setting": ["a cozy setting", "a historical setting"],
    "a mix of cheaper and pricier options": ["a mix of cheaper options", "a mix of pricier options"],
    # From lines 6501-6600
    "a variety of natural and organic options": ["a variety of natural options", "a variety of organic options"],
    # From lines 6601-6700
    "addictive and flavorful": ["addictive", "flavorful"],
    "affordable and authentic": ["affordable", "authentic"],
    "affordable and delicious": ["affordable", "delicious"],
    "affordable and efficient": ["affordable", "efficient"],
    # From lines 6701-6800
    "affordable and filling bar food": ["affordable bar food", "filling bar food"],
    "affordable and flavorful sushi": ["affordable sushi", "flavorful sushi"],
    "affordable and friendly atmosphere": ["affordable atmosphere", "friendly atmosphere"],
    "affordable and generous portion sandwiches": ["affordable portion sandwiches", "generous portion sandwiches"],
    "affordable and plentiful options": ["affordable options", "plentiful options"],
    "affordable and tasty chicken wings": ["affordable chicken wings", "tasty chicken wings"],
    "affordable food and drink options": ["affordable food options", "affordable drink options"],
    # From lines 6801-6900
    "affordable, efficient, and friendly nail salon experience": ["affordable nail salon experience", "efficient nail salon experience", "friendly nail salon experience"],
    "affordable, filling breakfast options": ["affordable breakfast options", "filling breakfast options"],
    "affordable, filling menu options": ["affordable menu options", "filling menu options"],
    "affordable, flavorful ramen": ["affordable ramen", "flavorful ramen"],
    "affordable, good quality pizzas": ["affordable pizzas", "good quality pizzas"],
    "affordable, high-quality sandwiches": ["affordable sandwiches", "high-quality sandwiches"],
    "affordable, quality burgers": ["affordable burgers", "quality burgers"],
    "affordable, substantial meals": ["affordable meals", "substantial meals"],
    # From lines 6901-7000
    "ambiance and service": ["ambiance", "service"],
    "ample indoor and patio seating": ["ample indoor seating", "ample patio seating"],
    # From lines 7001-7100
    "anyone looking for a fun and active indoor experience": ["anyone looking for a fun indoor experience", "anyone looking for an active indoor experience"],
    "apple and pumpkin picking": ["apple picking", "pumpkin picking"],
    "appreciators of unique and unconventional art": ["appreciators of unique art", "appreciators of unconventional art"],
    # From lines 7101-7200
    "artist-designed clothing and accessories": ["artist-designed clothing", "artist-designed accessories"],
    "arts & crafts supplies": ["arts supplies", "crafts supplies"],
    "arts & entertainment": ["arts", "entertainment"],
    "arts and crafts": ["arts", "crafts"],
    "arts and entertainment": ["arts", "entertainment"],
    # From lines 7201-7300
    "attention to detail and quality ingredients": ["attention to detail", "quality ingredients"],
    "attentive and accommodating": ["attentive", "accommodating"],
    "attentive and accommodating service": ["attentive service", "accommodating service"],
    "attentive and friendly": ["attentive", "friendly"],
    "attentive and knowledgeable staff": ["attentive staff", "knowledgeable staff"],
    "attentive and personalized service": ["attentive service", "personalized service"],
    "attentive and thorough medical professionals": ["attentive medical professionals", "thorough medical professionals"],
    "attentive, skilled hair stylists": ["attentive hair stylists", "skilled hair stylists"],
    # From lines 7301-7400
    "authentic and affordable dishes": ["authentic dishes", "affordable dishes"],
    "authentic and creative": ["authentic", "creative"],
    "authentic and delicious Cajun/Creole cuisine": ["authentic Cajun/Creole cuisine", "delicious Cajun/Creole cuisine"],
    "authentic and delicious boudin": ["authentic boudin", "delicious boudin"],
    "authentic and flavorful": ["authentic", "flavorful"],
    "authentic and fresh": ["authentic", "fresh"],
    "authentic and high-quality offerings": ["authentic offerings", "high-quality offerings"],
    "authentic and quality": ["authentic", "quality"],
    "authentic and tasty": ["authentic", "tasty"],
    "authentic and traditional dishes": ["authentic dishes", "traditional dishes"],
    # From lines 7401-7500
    "authentic, flavorful Chinese food": ["authentic Chinese food", "flavorful Chinese food"],
    "authentic, flavorful pho": ["authentic pho", "flavorful pho"],
    "authentic, high-quality BBQ": ["authentic BBQ", "high-quality BBQ"],
    "authentic, homemade Italian pastries": ["authentic Italian pastries", "homemade Italian pastries"],
    "authentic, succulent, and flavorful": ["authentic", "succulent", "flavorful"],
    "authentic, traditional Italian": ["authentic Italian", "traditional Italian"],
    "bacon mac & cheese": ["bacon mac", "cheese"],
    "bacon mac and cheese": ["bacon mac", "cheese"],
    # From lines 7501-7600
    "bakery and prepared foods": ["bakery", "prepared foods"],
    "bar & grill": ["bar", "grill"],
    "bar and dining": ["bar", "dining"],
    "bar and grill atmosphere": ["bar atmosphere", "grill atmosphere"],
    "bar and nightlife atmospheres": ["bar atmospheres", "nightlife atmospheres"],
    "bar and restaurant": ["bar", "restaurant"],
    "bar or brewpub": ["bar", "brewpub"],
    "bar or lounge": ["bar", "lounge"],
    "bar or pub": ["bar", "pub"],
    # From lines 7601-7700
    "bar/lounge": ["bar", "lounge"],
    "bar/restaurant setting": ["bar setting", "restaurant setting"],
    # From lines 7701-7800
    "beautiful and cultural setting": ["beautiful setting", "cultural setting"],
    "beautiful, intimate dining space": ["beautiful dining space", "intimate dining space"],
    "beautiful, remote beaches": ["beautiful beaches", "remote beaches"],
    "beautiful, uncrowded park": ["beautiful park", "uncrowded park"],
    # From lines 7801-7900
    "beauty and skincare products": ["beauty products", "skincare products"],
    "beauty and spa services": ["beauty services", "spa services"],
    "beauty and wellness treatments": ["beauty treatments", "wellness treatments"],
    "beef and chicken patties": ["beef patties", "chicken patties"],
    "beef and lamb dish": ["beef dish", "lamb dish"],
    # From lines 7901-8000
    "big and juicy burgers": ["big burgers", "juicy burgers"],
    "big, airy restaurant": ["big restaurant", "airy restaurant"],
    "biscuits and gravy": ["biscuits", "gravy"],
    # From lines 8001-8100
    "bold and authentic options": ["bold options", "authentic options"],
    "bold and flavorful dishes": ["bold dishes", "flavorful dishes"],
    "bold and rich flavors": ["bold flavors", "rich flavors"],
    "bold and unique clothing": ["bold clothing", "unique clothing"],
    "books on architecture and arts": ["books on architecture", "books on arts"],
    # From lines 8101-8200
    "brands & designers": ["brands", "designers"],
    "breakfast & brunch options": ["breakfast options", "brunch options"],
    "breakfast and lunch options": ["breakfast options", "lunch options"],
    "breakfast and lunch spot": ["breakfast spot", "lunch spot"],
    # From lines 8201-8300
    "bride and bridal parties": ["bride", "bridal parties"],
    "bright and friendly": ["bright", "friendly"],
    "bright and inviting": ["bright", "inviting"],
    "bright and open space": ["bright space", "open space"],
    "bright and sunny ambiance": ["bright ambiance", "sunny ambiance"],
    "bright and welcoming": ["bright", "welcoming"],
    "bright, airy ambiance": ["bright ambiance", "airy ambiance"],
    "bright, contemporary atmosphere": ["bright atmosphere", "contemporary atmosphere"],
    "bright, cute cafe setting": ["bright cafe setting", "cute cafe setting"],
    "bright, slightly cramped atmosphere": ["bright atmosphere", "slightly cramped atmosphere"],
    # From lines 8301-8400
    "burritos with french fries": ["burritos", "french fries"],
    # From lines 8401-8500
    "bustling and charming setting": ["bustling setting", "charming setting"],
    "bustling and diverse dining atmosphere": ["bustling dining atmosphere", "diverse dining atmosphere"],
    "bustling and lively": ["bustling", "lively"],
    "bustling but welcoming atmosphere": ["bustling atmosphere", "welcoming atmosphere"],
    "bustling, noisy setting": ["bustling setting", "noisy setting"],
    "bustling, popular setting": ["bustling setting", "popular setting"],
    "bustling, traditional setting": ["bustling setting", "traditional setting"],
    "bustling, upscale setting": ["bustling setting", "upscale setting"],
    "busy and crowded environment": ["busy environment", "crowded environment"],
    "busy and popular setting": ["busy setting", "popular setting"],
    "busy and touristy locations": ["busy locations", "touristy locations"],
    "busy and vibrant atmospheres": ["busy atmospheres", "vibrant atmospheres"],
    "busy, popular atmosphere": ["busy atmosphere", "popular atmosphere"],
    "butcher shop and cafe": ["butcher shop", "cafe"],
    # From lines 8501-8600
    "cafe and grocery store experience": ["cafe experience", "grocery store experience"],
    "calm and casual atmosphere": ["calm atmosphere", "casual atmosphere"],
    # From lines 8601-8700
    "cash and check payment only": ["cash payment only", "check payment only"],
    "cash or check": ["cash", "check"],
    "casual and active entertainment": ["casual entertainment", "active entertainment"],
    "casual and authentic setting": ["casual setting", "authentic setting"],
    "casual and bustling": ["casual", "bustling"],
    "casual and chill": ["casual", "chill"],
    "casual and clean atmosphere": ["casual atmosphere", "clean atmosphere"],
    "casual and eclectic atmosphere": ["casual atmosphere", "eclectic atmosphere"],
    "casual and friendly": ["casual", "friendly"],
    "casual and fun": ["casual", "fun"],
    "casual and grungy": ["casual", "grungy"],
    "casual and lively": ["casual", "lively"],
    "casual and nostalgic atmosphere": ["casual atmosphere", "nostalgic atmosphere"],
    "casual and retro ambiance": ["casual ambiance", "retro ambiance"],
    "casual and spacious environment": ["casual environment", "spacious environment"],
    "casual and spacious setting": ["casual setting", "spacious setting"],
    "casual and stylish gastropub setting": ["casual gastropub setting", "stylish gastropub setting"],
    "casual and welcoming neighborhood bars": ["casual neighborhood bars", "welcoming neighborhood bars"],
    # From lines 8701-8800
    "casual dining with friends and family": ["casual dining with friends", "casual dining with family"],
    # From lines 8801-8900
    "casual, bustling": ["casual", "bustling"],
    "casual, comfortable atmosphere": ["casual atmosphere", "comfortable atmosphere"],
    "casual, comfortable setting": ["casual setting", "comfortable setting"],
    "casual, communal setting": ["casual setting", "communal setting"],
    "casual, dive bar setting": ["casual setting", "dive bar setting"],
    "casual, friendly neighborhood atmosphere": ["casual neighborhood atmosphere", "friendly neighborhood atmosphere"],
    "casual, laid-back setting": ["casual setting", "laid-back setting"],
    "casual, local pub": ["casual pub", "local pub"],
    "casual, locally owned restaurant setting": ["casual restaurant setting", "locally owned restaurant setting"],
    "casual, modern setting": ["casual setting", "modern setting"],
    "casual, urban vibes": ["casual vibes", "urban vibes"],
    # From lines 8901-9000
    "chaotic and bustling shopping experience": ["chaotic shopping experience", "bustling shopping experience"],
    "character and charm": ["character", "charm"],
    "charming and eclectic atmosphere": ["charming atmosphere", "eclectic atmosphere"],
    "charming and inviting": ["charming", "inviting"],
    # From lines 9001-9100
    "charming, busy cafe": ["charming cafe", "busy cafe"],
    "charming, cozy environment": ["charming environment", "cozy environment"],
    "charming, cozy setting": ["charming setting", "cozy setting"],
    "charming, feminine decor": ["charming decor", "feminine decor"],
    "charming, storybook ambiance": ["charming ambiance", "storybook ambiance"],
    "charming, vintage setting": ["charming setting", "vintage setting"],
    "charming, welcoming atmosphere": ["charming atmosphere", "welcoming atmosphere"],
    "cheese and meat boards": ["cheese boards", "meat boards"],
    "chic and artsy ambiance": ["chic ambiance", "artsy ambiance"],
    "chic and cozy": ["chic", "cozy"],
    "chic and small bars": ["chic bars", "small bars"],
    "chic and trendy": ["chic", "trendy"],
    "chic and trendy dining experience": ["chic dining experience", "trendy dining experience"],
    # From lines 9101-9200
    "chic, laid back atmosphere": ["chic atmosphere", "laid back atmosphere"],
    "chicken & waffles": ["chicken", "waffles"],
    "chicken and shroom burgers": ["chicken burgers", "shroom burgers"],
    "chicken and waffles": ["chicken", "waffles"],
    "chicken and waffles enthusiasts": ["chicken enthusiasts", "waffles enthusiasts"],
    "chill and trendy ambiance": ["chill ambiance", "trendy ambiance"],
    # From lines 9201-9300
    "chill, fun atmosphere": ["chill atmosphere", "fun atmosphere"],
    "chill, laid-back": ["chill", "laid-back"],
    "chill, late-night atmosphere": ["chill atmosphere", "late-night atmosphere"],
    "chill, upscale": ["chill", "upscale"],
    "chips and queso": ["chips", "queso"],
    "chips and salsa": ["chips", "salsa"],
    "chorizo mac and cheese": ["chorizo mac", "cheese"],
    "city's history and music culture": ["city's history", "music culture"],
    # From lines 9301-9400
    "classic and modern": ["classic", "modern"],
    # From lines 9401-9500
    "classic, family-operated deli experience": ["classic deli experience", "family-operated deli experience"],
    "classic, good quality New Orleans dishes": ["classic New Orleans dishes", "good quality New Orleans dishes"],
    "classic, nostalgic": ["classic", "nostalgic"],
    "classic, old-school barber shop experience": ["classic barber shop experience", "old-school barber shop experience"],
    "classic, old-school fine dining ambiance": ["classic fine dining ambiance", "old-school fine dining ambiance"],
    "classic, old-world Italian dishes": ["classic Italian dishes", "old-world Italian dishes"],
    "classy and divey": ["classy", "divey"],
    "classy and friendly cafe setting": ["classy cafe setting", "friendly cafe setting"],
    "classy and intimate setting": ["classy setting", "intimate setting"],
    "classy and trendy": ["classy", "trendy"],
    "classy, chic atmosphere": ["classy atmosphere", "chic atmosphere"],
    "classy, intimate atmosphere": ["classy atmosphere", "intimate atmosphere"],
    "classy, relaxed environment": ["classy environment", "relaxed environment"],
    "clean and accommodating environment": ["clean environment", "accommodating environment"],
    "clean and attentive": ["clean", "attentive"],
    "clean and beautiful": ["clean", "beautiful"],
    "clean and calm environment": ["clean environment", "calm environment"],
    "clean and casual setting": ["clean setting", "casual setting"],
    "clean and colorful": ["clean", "colorful"],
    "clean and comfortable ambience": ["clean ambience", "comfortable ambience"],
    "clean and comfortable rooms": ["clean rooms", "comfortable rooms"],
    "clean and convenient locations": ["clean locations", "convenient locations"],
    "clean and cozy": ["clean", "cozy"],
    "clean and efficient service": ["clean service", "efficient service"],
    "clean and family-friendly atmosphere": ["clean atmosphere", "family-friendly atmosphere"],
    "clean and friendly cafe": ["clean cafe", "friendly cafe"],
    "clean and friendly dining environments": ["clean dining environments", "friendly dining environments"],
    "clean and friendly nail salons": ["clean nail salons", "friendly nail salons"],
    "clean and friendly restaurant setting": ["clean restaurant setting", "friendly restaurant setting"],
    "clean and friendly theater environment": ["clean theater environment", "friendly theater environment"],
    "clean and hospitable environment": ["clean environment", "hospitable environment"],
    "clean and inviting space": ["clean space", "inviting space"],
    "clean and modern cafe environment": ["clean cafe environment", "modern cafe environment"],
    "clean and modern environment": ["clean environment", "modern environment"],
    "clean and modern fitness facility": ["clean fitness facility", "modern fitness facility"],
    "clean and modern mall environment": ["clean mall environment", "modern mall environment"],
    "clean and modern setting": ["clean setting", "modern setting"],
    "clean and modern shopping environments": ["clean shopping environments", "modern shopping environments"],
    "clean and nicely decorated environment": ["clean environment", "nicely decorated environment"],
    "clean and open space": ["clean space", "open space"],
    "clean and organized cinema": ["clean cinema", "organized cinema"],
    "clean and organized environment": ["clean environment", "organized environment"],
    "clean and organized spa environment": ["clean spa environment", "organized spa environment"],
    "clean and organized store": ["clean store", "organized store"],
    "clean and organized store environment": ["clean store environment", "organized store environment"],
    "clean and organized stores": ["clean stores", "organized stores"],
    "clean and peaceful viewing environment": ["clean viewing environment", "peaceful viewing environment"],
    "clean and pretty store ambiance": ["clean store ambiance", "pretty store ambiance"],
    "clean and relaxing environment": ["clean environment", "relaxing environment"],
    "clean and renovated theaters": ["clean theaters", "renovated theaters"],
    "clean and respectful moviegoing experience": ["clean moviegoing experience", "respectful moviegoing experience"],
    "clean and spacious": ["clean", "spacious"],
    "clean and spacious casinos": ["clean casinos", "spacious casinos"],
    "clean and spacious dining areas": ["clean dining areas", "spacious dining areas"],
    "clean and spotless": ["clean", "spotless"],
    "clean and stylish surroundings": ["clean surroundings", "stylish surroundings"],
    "clean and tastefully decorated restaurant spaces": ["clean restaurant spaces", "tastefully decorated restaurant spaces"],
    "clean and tasty food": ["clean food", "tasty food"],
    "clean and tranquil": ["clean", "tranquil"],
    # From lines 9501-9600
    "clean and tranquil environment": ["clean environment", "tranquil environment"],
    "clean and tranquil nail salon": ["clean nail salon", "tranquil nail salon"],
    "clean and trendy setting": ["clean setting", "trendy setting"],
    "clean and updated": ["clean", "updated"],
    "clean and upscale salon setting": ["clean salon setting", "upscale salon setting"],
    "clean and vibrant restaurant setting": ["clean restaurant setting", "vibrant restaurant setting"],
    "clean and welcoming cafe setting": ["clean cafe setting", "welcoming cafe setting"],
    "clean and welcoming eatery": ["clean eatery", "welcoming eatery"],
    "clean and well-kept": ["clean", "well-kept"],
    "clean and well-lit store layout": ["clean store layout", "well-lit store layout"],
    "clean and well-maintained facilities": ["clean facilities", "well-maintained facilities"],
    "clean and well-maintained nail salon": ["clean nail salon", "well-maintained nail salon"],
    "clean and well-maintained theater": ["clean theater", "well-maintained theater"],
    "clean and well-organized meat department": ["clean meat department", "well-organized meat department"],
    "clean, bright establishment": ["clean establishment", "bright establishment"],
    "clean, bright store": ["clean store", "bright store"],
    "clean, caring environment": ["clean environment", "caring environment"],
    "clean, classy setting": ["clean setting", "classy setting"],
    "clean, comfortable environment": ["clean environment", "comfortable environment"],
    "clean, cute environment": ["clean environment", "cute environment"],
    "clean, friendly service": ["clean service", "friendly service"],
    "clean, healthy food": ["clean food", "healthy food"],
    "clean, intimate environment": ["clean environment", "intimate environment"],
    "clean, kid-friendly environment": ["clean environment", "kid-friendly environment"],
    "clean, professional, appointment-based": ["clean", "professional", "appointment-based"],
    "clean, relaxed beach experiences": ["clean beach experiences", "relaxed beach experiences"],
    "clean, spacious rooms": ["clean rooms", "spacious rooms"],
    "clean, spacious venue": ["clean venue", "spacious venue"],
    "clean, welcoming, and challenging environment": ["clean environment", "welcoming environment", "challenging environment"],
    # From lines 9601-9700
    "close to entertainment and dining": ["close to entertainment", "close to dining"],
    "clubs & groups": ["clubs", "groups"],
    "coffee & tea": ["coffee", "tea"],
    # From lines 9701-9800
    "colorful and hip hop cool decor": ["colorful decor", "hip hop cool decor"],
    "colorful and informative": ["colorful", "informative"],
    "colorful and inviting atmosphere": ["colorful atmosphere", "inviting atmosphere"],
    "comfortable and busy": ["comfortable", "busy"],
    "comfortable and laid-back atmosphere": ["comfortable atmosphere", "laid-back atmosphere"],
    "comfortable and luxurious": ["comfortable", "luxurious"],
    "comfortable and safe": ["comfortable", "safe"],
    "comfortable and spacious setting": ["comfortable setting", "spacious setting"],
    "comfortable and spacious venue": ["comfortable venue", "spacious venue"],
    "comfortable and stylish atmosphere": ["comfortable atmosphere", "stylish atmosphere"],
    "comfortable and welcoming settings": ["comfortable settings", "welcoming settings"],
    # From lines 9801-9900
    "comfortable, cozy dining": ["comfortable dining", "cozy dining"],
    "comfortable, versatile atmosphere": ["comfortable atmosphere", "versatile atmosphere"],
    "comforting and delicious": ["comforting", "delicious"],
    "community fitness and social event": ["community fitness event", "social event"],
    # From lines 9901-10000
    "compassionate and attentive care": ["compassionate care", "attentive care"],
    "compassionate and thorough care": ["compassionate care", "thorough care"],
    "compassionate and understanding vet care": ["compassionate vet care", "understanding vet care"],
    "complimentary chips and salsa": ["complimentary chips", "complimentary salsa"],
    # From lines 10001-10100
    "consistent and fresh": ["consistent", "fresh"],
    "consistent and popular dining experience": ["consistent dining experience", "popular dining experience"],
    "consistent and tasty diner food": ["consistent diner food", "tasty diner food"],
    "consistent quality and service": ["consistent quality", "consistent service"],
    "consistent, satisfying dining experiences": ["consistent dining experiences", "satisfying dining experiences"],
    "contemporary and clean restaurant": ["contemporary restaurant", "clean restaurant"],
    "contemporary and country inn vibes": ["contemporary vibes", "country inn vibes"],
    "contemporary, eclectic setting": ["contemporary setting", "eclectic setting"],
    "convenience in location and parking": ["convenience in location", "convenience in parking"],
    "convenience store with gas stations": ["convenience store", "gas stations"],
    # From lines 10101-10200
    "convenience with gas stations": ["convenience", "gas stations"],
    "convenient and tasty meal option": ["convenient meal option", "tasty meal option"],
    # From lines 10201-10300
    "cool and creative": ["cool", "creative"],
    "cool and funky": ["cool", "funky"],
    "cool and modern space": ["cool space", "modern space"],
    "cool and newly opened spot": ["cool spot", "newly opened spot"],
    "cool, chill setting": ["cool setting", "chill setting"],
    "cool, laid-back atmosphere": ["cool atmosphere", "laid-back atmosphere"],
    "cool, minimalist setting": ["cool setting", "minimalist setting"],
    "cool, revamped location": ["cool location", "revamped location"],
    "cool/hip": ["cool", "hip"],
    # From lines 10301-10400
    "cosy and artistic environment": ["cosy environment", "artistic environment"],
    "country & western experience": ["country experience", "western experience"],
    "cozy and European-inspired atmosphere": ["cozy atmosphere", "European-inspired atmosphere"],
    "cozy and clean dining environment": ["cozy dining environment", "clean dining environment"],
    "cozy and dark atmosphere": ["cozy atmosphere", "dark atmosphere"],
    "cozy and friendly mountain cabin setting": ["cozy mountain cabin setting", "friendly mountain cabin setting"],
    "cozy and friendly service": ["cozy service", "friendly service"],
    "cozy and fun spot": ["cozy spot", "fun spot"],
    "cozy and hidden gem taproom": ["cozy taproom", "hidden gem taproom"],
    "cozy and historical setting": ["cozy setting", "historical setting"],
    "cozy and homey": ["cozy", "homey"],
    "cozy and intimate bars": ["cozy bars", "intimate bars"],
    "cozy and intimate restaurant setting": ["cozy restaurant setting", "intimate restaurant setting"],
    "cozy and lively": ["cozy", "lively"],
    "cozy and nostalgic dining ambiance": ["cozy dining ambiance", "nostalgic dining ambiance"],
    "cozy and peaceful environments": ["cozy environments", "peaceful environments"],
    "cozy and quiet": ["cozy", "quiet"],
    "cozy and quirky atmosphere": ["cozy atmosphere", "quirky atmosphere"],
    "cozy and romantic": ["cozy", "romantic"],
    "cozy and romantic setting": ["cozy setting", "romantic setting"],
    "cozy and rustic dining environments": ["cozy dining environments", "rustic dining environments"],
    "cozy and spacious": ["cozy", "spacious"],
    "cozy and spacious coffee shop": ["cozy coffee shop", "spacious coffee shop"],
    "cozy and spacious settings": ["cozy settings", "spacious settings"],
    "cozy and stylish cafe setting": ["cozy cafe setting", "stylish cafe setting"],
    "cozy and unique converted house setting": ["cozy converted house setting", "unique converted house setting"],
    "cozy and warm atmosphere": ["cozy atmosphere", "warm atmosphere"],
    # From lines 10401-10500
    "cozy indoor/outdoor patio": ["cozy indoor patio", "cozy outdoor patio"],
    # From lines 10501-10600
    "cozy, artistic atmosphere": ["cozy atmosphere", "artistic atmosphere"],
    "cozy, authentic and homely": ["cozy", "authentic", "homely"],
    "cozy, busy shopping environment": ["cozy shopping environment", "busy shopping environment"],
    "cozy, calming spa atmosphere": ["cozy spa atmosphere", "calming spa atmosphere"],
    "cozy, casual atmosphere": ["cozy atmosphere", "casual atmosphere"],
    "cozy, casual coffeehouse": ["cozy coffeehouse", "casual coffeehouse"],
    "cozy, casual dining": ["cozy dining", "casual dining"],
    "cozy, casual dining setting": ["cozy dining setting", "casual dining setting"],
    "cozy, casual spot": ["cozy spot", "casual spot"],
    "cozy, charming": ["cozy", "charming"],
    "cozy, clean cafe setting": ["cozy cafe setting", "clean cafe setting"],
    "cozy, clean restaurant setting": ["cozy restaurant setting", "clean restaurant setting"],
    "cozy, college-town setting": ["cozy setting", "college-town setting"],
    "cozy, colorful ambiance": ["cozy ambiance", "colorful ambiance"],
    "cozy, communal setting": ["cozy setting", "communal setting"],
    "cozy, crowded cafe": ["cozy cafe", "crowded cafe"],
    "cozy, crowded setting": ["cozy setting", "crowded setting"],
    "cozy, cute atmosphere": ["cozy atmosphere", "cute atmosphere"],
    "cozy, dark, and aromatic setting": ["cozy setting", "dark setting", "aromatic setting"],
    "cozy, dark-ish atmosphere": ["cozy atmosphere", "dark-ish atmosphere"],
    "cozy, dimly lit dining experience": ["cozy dining experience", "dimly lit dining experience"],
    "cozy, down-home dining experience": ["cozy dining experience", "down-home dining experience"],
    "cozy, eclectic setting": ["cozy setting", "eclectic setting"],
    "cozy, familiar": ["cozy", "familiar"],
    "cozy, family-friendly": ["cozy", "family-friendly"],
    "cozy, family-like atmosphere": ["cozy atmosphere", "family-like atmosphere"],
    # From lines 10601-10700
    "cozy, family-run setting": ["cozy setting", "family-run setting"],
    "cozy, family-run vibe": ["cozy vibe", "family-run vibe"],
    "cozy, folkie-hipster vibes": ["cozy vibes", "folkie-hipster vibes"],
    "cozy, hidden gem setting": ["cozy setting", "hidden gem setting"],
    "cozy, hip ambiance": ["cozy ambiance", "hip ambiance"],
    "cozy, hipster ambiance": ["cozy ambiance", "hipster ambiance"],
    "cozy, hipster atmosphere": ["cozy atmosphere", "hipster atmosphere"],
    "cozy, historic settings": ["cozy settings", "historic settings"],
    "cozy, hole-in-the-wall atmosphere": ["cozy atmosphere", "hole-in-the-wall atmosphere"],
    "cozy, holiday-themed ambiance": ["cozy ambiance", "holiday-themed ambiance"],
    "cozy, home-like setting": ["cozy setting", "home-like setting"],
    "cozy, hometown atmosphere": ["cozy atmosphere", "hometown atmosphere"],
    "cozy, homey": ["cozy", "homey"],
    "cozy, homey atmospheres": ["cozy atmospheres", "homey atmospheres"],
    "cozy, intimate atmosphere": ["cozy atmosphere", "intimate atmosphere"],
    "cozy, intimate bar setting": ["cozy bar setting", "intimate bar setting"],
    "cozy, inviting": ["cozy", "inviting"],
    "cozy, local eateries": ["cozy eateries", "local eateries"],
    "cozy, local eatery": ["cozy eatery", "local eatery"],
    "cozy, local vibe": ["cozy vibe", "local vibe"],
    "cozy, low-frills atmosphere": ["cozy atmosphere", "low-frills atmosphere"],
    "cozy, minimalist": ["cozy", "minimalist"],
    "cozy, minimalist bars": ["cozy bars", "minimalist bars"],
    "cozy, modern setting": ["cozy setting", "modern setting"],
    "cozy, mom's kitchen-inspired setting": ["cozy setting", "mom's kitchen-inspired setting"],
    "cozy, mom-and-pop atmosphere": ["cozy atmosphere", "mom-and-pop atmosphere"],
    "cozy, mom-and-pop restaurant ambiance": ["cozy restaurant ambiance", "mom-and-pop restaurant ambiance"],
    "cozy, neighborhood BYOB charm": ["cozy charm", "neighborhood BYOB charm"],
    "cozy, neighborhood cafe": ["cozy cafe", "neighborhood cafe"],
    "cozy, neighborhood coffee shop": ["cozy coffee shop", "neighborhood coffee shop"],
    "cozy, neighborhood vibe": ["cozy vibe", "neighborhood vibe"],
    "cozy, no-frills ambiance": ["cozy ambiance", "no-frills ambiance"],
    "cozy, non-chain atmosphere": ["cozy atmosphere", "non-chain atmosphere"],
    "cozy, non-touristy atmosphere": ["cozy atmosphere", "non-touristy atmosphere"],
    "cozy, nostalgic atmosphere": ["cozy atmosphere", "nostalgic atmosphere"],
    "cozy, old New Orleans ambiance": ["cozy ambiance", "old New Orleans ambiance"],
    "cozy, old-school": ["cozy", "old-school"],
    "cozy, old-school deli atmosphere": ["cozy deli atmosphere", "old-school deli atmosphere"],
    "cozy, old-town ambiance": ["cozy ambiance", "old-town ambiance"],
    "cozy, quaint ambiance": ["cozy ambiance", "quaint ambiance"],
    "cozy, quiet, and welcoming ambiance": ["cozy ambiance", "quiet ambiance", "welcoming ambiance"],
    "cozy, quirky atmosphere": ["cozy atmosphere", "quirky atmosphere"],
    "cozy, relaxed dining experience": ["cozy dining experience", "relaxed dining experience"],
    "cozy, romantic": ["cozy", "romantic"],
    "cozy, romantic setting": ["cozy setting", "romantic setting"],
    "cozy, slightly cluttered atmosphere": ["cozy atmosphere", "slightly cluttered atmosphere"],
    "cozy, small": ["cozy", "small"],
    "cozy, small cafe": ["cozy cafe", "small cafe"],
    "cozy, small mom-and-pop restaurant": ["cozy mom-and-pop restaurant", "small mom-and-pop restaurant"],
    "cozy, small, and unique dining experience": ["cozy dining experience", "small dining experience", "unique dining experience"],
    "cozy, southern-style setting": ["cozy setting", "southern-style setting"],
    "cozy, tiny coffee shop": ["cozy coffee shop", "tiny coffee shop"],
    "cozy, traditional atmosphere": ["cozy atmosphere", "traditional atmosphere"],
    "cozy, trendy environment": ["cozy environment", "trendy environment"],
    "cozy, unpretentious atmosphere": ["cozy atmosphere", "unpretentious atmosphere"],
    "cozy, upscale but not pretentious setting": ["cozy setting", "upscale but not pretentious setting"],
    "cozy, well-decorated ambiance": ["cozy ambiance", "well-decorated ambiance"],
    "cozy, well-established market setting": ["cozy market setting", "well-established market setting"],
    "crabmeat and shrimp frittata": ["crabmeat frittata", "shrimp frittata"],
    # From lines 10701-10800
    "creamy and indulgent appetizers": ["creamy appetizers", "indulgent appetizers"],
    "creamy and tasty rolled ice cream": ["creamy rolled ice cream", "tasty rolled ice cream"],
    "creative and artfully presented dishes": ["creative dishes", "artfully presented dishes"],
    "creative and challenging workouts": ["creative workouts", "challenging workouts"],
    "creative and cool themed ambiance": ["creative themed ambiance", "cool themed ambiance"],
    "creative and delicious": ["creative", "delicious"],
    "creative and delicious baked goods": ["creative baked goods", "delicious baked goods"],
    "creative and delicious comfort sandwiches": ["creative comfort sandwiches", "delicious comfort sandwiches"],
    "creative and delicious food": ["creative food", "delicious food"],
    "creative and diverse breakfast options": ["creative breakfast options", "diverse breakfast options"],
    "creative and elevated comfort food": ["creative comfort food", "elevated comfort food"],
    "creative and entertaining activity": ["creative activity", "entertaining activity"],
    "creative and ever-changing menu options": ["creative menu options", "ever-changing menu options"],
    "creative and flavorful": ["creative", "flavorful"],
    "creative and flavorful food": ["creative food", "flavorful food"],
    "creative and fresh": ["creative", "fresh"],
    "creative and high-quality": ["creative", "high-quality"],
    "creative and high-quality cuisine": ["creative cuisine", "high-quality cuisine"],
    "creative and inspired Mexican dining experience": ["creative Mexican dining experience", "inspired Mexican dining experience"],
    "creative and lofty salon vibe": ["creative salon vibe", "lofty salon vibe"],
    "creative and thoughtfully prepared": ["creative", "thoughtfully prepared"],
    "creative and transformative hairstyling": ["creative hairstyling", "transformative hairstyling"],
    "creative and unconventional donuts": ["creative donuts", "unconventional donuts"],
    "creative and unique menu items": ["creative menu items", "unique menu items"],
    "creative and unique menu options": ["creative menu options", "unique menu options"],
    "creative but classic cuisine": ["creative cuisine", "classic cuisine"],
    # From lines 10801-10900
    "creative, fun, flavorful twists on Southern cuisine": ["creative twists on Southern cuisine", "fun twists on Southern cuisine", "flavorful twists on Southern cuisine"],
    # From lines 10901-11000
    "crowded and lively atmosphere": ["crowded atmosphere", "lively atmosphere"],
    "crowded but well-managed location": ["crowded location", "well-managed location"],
    "crowded or pretentious gym environments": ["crowded gym environments", "pretentious gym environments"],
    # From lines 11001-11100
    "customizable and fresh food": ["customizable food", "fresh food"],
    # From lines 11101-11200
    "cute and Instagram-worthy settings": ["cute settings", "Instagram-worthy settings"],
    "cute and beautifully decorated shop": ["cute shop", "beautifully decorated shop"],
    "cute and charming bakery settings": ["cute bakery settings", "charming bakery settings"],
    "cute and chic shopping experience": ["cute shopping experience", "chic shopping experience"],
    "cute and eco-friendly packaging": ["cute packaging", "eco-friendly packaging"],
    "cute and friendly cafe environment": ["cute cafe environment", "friendly cafe environment"],
    "cute and quaint environment": ["cute environment", "quaint environment"],
    "cute and relaxed atmosphere": ["cute atmosphere", "relaxed atmosphere"],
    "cute and renovated spaces": ["cute spaces", "renovated spaces"],
    "cute and thoughtful gift options": ["cute gift options", "thoughtful gift options"],
    # From lines 11201-11300
    "cute, classy fashion": ["cute fashion", "classy fashion"],
    "cute, comfortable cafes": ["cute cafes", "comfortable cafes"],
    "dark, romantic": ["dark", "romantic"],
    "decadent and flavorful": ["decadent", "flavorful"],
    "decadent and unique ice cream flavors": ["decadent ice cream flavors", "unique ice cream flavors"],
    "decent and consistent": ["decent", "consistent"],
    # From lines 11401-11500
    "delicious and affordable": ["delicious", "affordable"],
    "delicious and affordable dishes": ["delicious dishes", "affordable dishes"],
    "delicious and diverse options": ["delicious options", "diverse options"],
    "delicious and filling dining experience": ["delicious dining experience", "filling dining experience"],
    "delicious and filling menu options": ["delicious menu options", "filling menu options"],
    "delicious and flavorful": ["delicious", "flavorful"],
    "delicious and fresh breakfast items": ["delicious breakfast items", "fresh breakfast items"],
    "delicious and healthy food": ["delicious food", "healthy food"],
    "delicious and natural": ["delicious", "natural"],
    "delicious and thoughtfully prepared": ["delicious", "thoughtfully prepared"],
    "delicious and unique": ["delicious", "unique"],
    "delicious breakfast and brunch": ["delicious breakfast", "delicious brunch"],
    "delicious breakfast/brunch options": ["delicious breakfast options", "delicious brunch options"],
    "delicious cakes and desserts": ["delicious cakes", "delicious desserts"],
    "delicious food and drink specials": ["delicious food specials", "delicious drink specials"],
    "delicious food and drinks": ["delicious food", "delicious drinks"],
    # From lines 11601-11700
    "delicious, diverse menu options": ["delicious menu options", "diverse menu options"],
    "delicious, fresh food": ["delicious food", "fresh food"],
    "delicious, unique, and locally made specialty food items": ["delicious specialty food items", "unique specialty food items", "locally made specialty food items"],
    "dense and creamy foam": ["dense foam", "creamy foam"],
    # From lines 11701-11800
    "detailed and attentive staff": ["detailed staff", "attentive staff"],
    "dine-in and takeout": ["dine-in", "takeout"],
    "dim, romantic atmosphere": ["dim atmosphere", "romantic atmosphere"],
    # From lines 11901-12000
    "diverse and affordable beer selection": ["diverse beer selection", "affordable beer selection"],
    "diverse and ambitious programming": ["diverse programming", "ambitious programming"],
    "diverse and beautifully set restaurant": ["diverse restaurant", "beautifully set restaurant"],
    "diverse and creative menu": ["diverse menu", "creative menu"],
    "diverse and delicious menu items": ["diverse menu items", "delicious menu items"],
    "diverse and family-friendly atmosphere": ["diverse atmosphere", "family-friendly atmosphere"],
    "diverse and flavorful": ["diverse", "flavorful"],
    "diverse and flavorful offerings": ["diverse offerings", "flavorful offerings"],
    "diverse and flavorful options": ["diverse options", "flavorful options"],
    "diverse and flavorful pizzas": ["diverse pizzas", "flavorful pizzas"],
    "diverse and flavorful sausages": ["diverse sausages", "flavorful sausages"],
    "diverse and flavorful street food": ["diverse street food", "flavorful street food"],
    "diverse and high-quality menu": ["diverse menu", "high-quality menu"],
    "diverse and inclusive community": ["diverse community", "inclusive community"],
    "diverse and inclusive employee community": ["diverse employee community", "inclusive employee community"],
    "diverse and international foods": ["diverse foods", "international foods"],
    "diverse and reasonably priced": ["diverse", "reasonably priced"],
    "diverse and unique hot dog options": ["diverse hot dog options", "unique hot dog options"],
    "diverse beer and wine menu": ["diverse beer menu", "diverse wine menu"],
    # From lines 12001-12500
    "diverse selection of CDs and vinyl": ["diverse selection of CDs", "diverse selection of vinyl"],
    "diverse selection of craft beers, wines, and spirits": ["diverse selection of craft beers", "diverse selection of wines", "diverse selection of spirits"],
    "diverse selection of new and used books": ["diverse selection of new books", "diverse selection of used books"],
    "diverse selection of unique and flavorful beers": ["diverse selection of unique beers", "diverse selection of flavorful beers"],
    "diverse wine and cheese selection": ["diverse wine selection", "diverse cheese selection"],
    "diverse wine and cocktail selection": ["diverse wine selection", "diverse cocktail selection"],
    "diverse, delicious, and eco-conscious food offerings": ["diverse food offerings", "delicious food offerings", "eco-conscious food offerings"],
    "diverse, flavorful dishes": ["diverse dishes", "flavorful dishes"],
    "diverse, flavorful tacos": ["diverse tacos", "flavorful tacos"],
    "diverse, flavorful tapas": ["diverse tapas", "flavorful tapas"],
    "diverse, flavorful, and healthy food options": ["diverse food options", "flavorful food options", "healthy food options"],
    "diverse, high-quality dishes": ["diverse dishes", "high-quality dishes"],
    "diverse, multi-ethnic ambiance": ["diverse ambiance", "multi-ethnic ambiance"],
    "divey and minimalist setting": ["divey setting", "minimalist setting"],
    "donuts and gyros with a twist": ["donuts with a twist", "gyros with a twist"],
    "drinks and snacks": ["drinks", "snacks"],
    "eclectic and artistic restaurant designs": ["eclectic restaurant designs", "artistic restaurant designs"],
    "eclectic and changing menus": ["eclectic menus", "changing menus"],
    "eclectic food and drinks": ["eclectic food", "eclectic drinks"],
    "eclectic, artsy atmospheres": ["eclectic atmospheres", "artsy atmospheres"],
    "eclectic, dive bar": ["eclectic", "dive bar"],
    "eclectic, dive bar atmosphere": ["eclectic atmosphere", "dive bar atmosphere"],
    # From lines 12501-13500
    "efficient and friendly service": ["efficient service", "friendly service"],
    "efficient and friendly staff": ["efficient staff", "friendly staff"],
    "efficient and professional care": ["efficient care", "professional care"],
    "elegant and charming setting": ["elegant setting", "charming setting"],
    "entertainment and educational experience": ["entertainment experience", "educational experience"],
    "enthusiastic and friendly service": ["enthusiastic service", "friendly service"],
    "enthusiastic and helpful instructors": ["enthusiastic instructors", "helpful instructors"],
    "ethical and eco-friendly choices": ["ethical choices", "eco-friendly choices"],
    "everyday and fine dining dishes": ["everyday dishes", "fine dining dishes"],
    "excellent food and drinks": ["excellent food", "excellent drinks"],
    "exceptional coffee and hot chocolate": ["exceptional coffee", "exceptional hot chocolate"],
    "exciting and anticipatory": ["exciting", "anticipatory"],
    "extensive beer and wine selection": ["extensive beer selection", "extensive wine selection"],
    "extensive margarita and tequila": ["extensive margarita", "extensive tequila"],
    "extensive wine and cheese selections": ["extensive wine selections", "extensive cheese selections"],
    "extensive wine and cocktail menus": ["extensive wine menus", "extensive cocktail menus"],
    "extensive wine and spirits selection": ["extensive wine selection", "extensive spirits selection"],
    "extra care and attention": ["extra care", "extra attention"],
    "fair trade/organic beans": ["fair trade beans", "organic beans"],
    "families and groups": ["families", "groups"],
    "family and friends": ["family", "friends"],
    "fancy and clean atmosphere": ["fancy atmosphere", "clean atmosphere"],
    "fancy, flavorful dishes": ["fancy dishes", "flavorful dishes"],
    "fans of customizable and healthy dining options": ["fans of customizable dining options", "fans of healthy dining options"],
    "fast and affordable dining experience": ["fast dining experience", "affordable dining experience"],
    "fast and convenient service": ["fast service", "convenient service"],
    "fast and courteous": ["fast", "courteous"],
    "fast and friendly": ["fast", "friendly"],
    "fast, fresh, and affordable food": ["fast food", "fresh food", "affordable food"],
    "fast, fresh, and tasty food options": ["fast food options", "fresh food options", "tasty food options"],
    "fast, polite service": ["fast service", "polite service"],
    "fast, quality food": ["fast food", "quality food"],
    # From lines 13501-16000
    "festive and social atmosphere": ["festive atmosphere", "social atmosphere"],
    "fish and seafood": ["fish", "seafood"],
    "flavorful and affordable": ["flavorful", "affordable"],
    "flavorful and affordable food": ["flavorful food", "affordable food"],
    "flavorful and authentic": ["flavorful", "authentic"],
    "flavorful and authentic dishes": ["flavorful dishes", "authentic dishes"],
    "flavorful and authentic food": ["flavorful food", "authentic food"],
    "flavorful and bold": ["flavorful", "bold"],
    "flavorful and colorful": ["flavorful", "colorful"],
    "flavorful and creamy": ["flavorful", "creamy"],
    "flavorful and creative menu items": ["flavorful menu items", "creative menu items"],
    "flavorful and crispy": ["flavorful", "crispy"],
    "flavorful and crispy chicken dishes": ["flavorful chicken dishes", "crispy chicken dishes"],
    "flavorful and diverse menus": ["flavorful menus", "diverse menus"],
    "flavorful and filling options": ["flavorful options", "filling options"],
    "flavorful and fresh": ["flavorful", "fresh"],
    "flavorful and fresh dishes": ["flavorful dishes", "fresh dishes"],
    "flavorful and fresh vegan dishes": ["flavorful vegan dishes", "fresh vegan dishes"],
    "flavorful and generous portions": ["flavorful portions", "generous portions"],
    "flavorful and healthy": ["flavorful", "healthy"],
    "flavorful and modern": ["flavorful", "modern"],
    "flavorful and non-greasy BBQ": ["flavorful BBQ", "non-greasy BBQ"],
    "flavorful and satisfying meal options": ["flavorful meal options", "satisfying meal options"],
    "flavorful and strong": ["flavorful", "strong"],
    "flavorful and unique hot dogs": ["flavorful hot dogs", "unique hot dogs"],
    "flavorful and unusual": ["flavorful", "unusual"],
    "flavorful and well-balanced dishes": ["flavorful dishes", "well-balanced dishes"],
    "flavorful and well-seasoned": ["flavorful", "well-seasoned"],
    "flavorful and well-spiced": ["flavorful", "well-spiced"],
    "flavorful salsas and chips": ["flavorful salsas", "chips"],
    "flavorful, fresh": ["flavorful", "fresh"],
    "flavorful, healthy options": ["flavorful options", "healthy options"],
    "flavorful, unique dishes": ["flavorful dishes", "unique dishes"],
    "flavorful, well-cooked chicken wings": ["flavorful chicken wings", "well-cooked chicken wings"],
    "food and drink": ["food", "drink"],
    "food and drink enthusiasts": ["food enthusiasts", "drink enthusiasts"],
    "food and drink experience": ["food experience", "drink experience"],
    "food and drink selections": ["food selections", "drink selections"],
    "food and snack options": ["food options", "snack options"],
    "free chips and salsa": ["free chips", "free salsa"],
    "fruit and vegetable market": ["fruit market", "vegetable market"],
    "fruits and vegetables": ["fruits", "vegetables"],
    "fun and adventurous experience": ["fun experience", "adventurous experience"],
    "fun and bright space": ["fun space", "bright space"],
    "fun and busy environment": ["fun environment", "busy environment"],
    "fun and casual": ["fun", "casual"],
    "fun and casual atmosphere": ["fun atmosphere", "casual atmosphere"],
    "fun and challenging": ["fun", "challenging"],
    "fun and community-focused environment": ["fun environment", "community-focused environment"],
    "fun and cool nightlife experience": ["fun nightlife experience", "cool nightlife experience"],
    "fun and cozy atmosphere": ["fun atmosphere", "cozy atmosphere"],
    "fun and creative items": ["fun items", "creative items"],
    "fun and customizable dessert experience": ["fun dessert experience", "customizable dessert experience"],
    "fun and educational activities": ["fun activities", "educational activities"],
    "fun and energetic": ["fun", "energetic"],
    "fun and engaging": ["fun", "engaging"],
    "fun and excitement": ["fun", "excitement"],
    "fun and experimental dining experience": ["fun dining experience", "experimental dining experience"],
    "fun and festive": ["fun", "festive"],
    "fun and festive atmosphere": ["fun atmosphere", "festive atmosphere"],
    "fun and free local experience": ["fun local experience", "free local experience"],
    "fun and interactive": ["fun", "interactive"],
    "fun and interactive setting": ["fun setting", "interactive setting"],
    "fun and inviting": ["fun", "inviting"],
    "fun and lively": ["fun", "lively"],
    "fun and lively ice cream experience": ["fun ice cream experience", "lively ice cream experience"],
    "fun and memorable experience": ["fun experience", "memorable experience"],
    "fun and modern atmosphere": ["fun atmosphere", "modern atmosphere"],
    "fun and new dishes": ["fun dishes", "new dishes"],
    "fun and professional experience": ["fun experience", "professional experience"],
    "fun and relaxed": ["fun", "relaxed"],
    "fun and relaxed setting": ["fun setting", "relaxed setting"],
    "fun and relaxing": ["fun", "relaxing"],
    "fun and simple atmosphere": ["fun atmosphere", "simple atmosphere"],
    "fun and spooky atmosphere": ["fun atmosphere", "spooky atmosphere"],
    "fun and tasty experience": ["fun experience", "tasty experience"],
    "fun and unique ice cream shop setting": ["fun ice cream shop setting", "unique ice cream shop setting"],
    "fun and vibrant": ["fun", "vibrant"],
    "fun and vibrant setting": ["fun setting", "vibrant setting"],
    "fun and well-organized events": ["fun events", "well-organized events"],
    "fun, clean atmosphere": ["fun atmosphere", "clean atmosphere"],
    "fun, creative experience": ["fun experience", "creative experience"],
    "fun, interactive experience": ["fun experience", "interactive experience"],
    "fun, intimate atmosphere": ["fun atmosphere", "intimate atmosphere"],
    "fun, laid-back environment": ["fun environment", "laid-back environment"],
    "fun, lively atmosphere": ["fun atmosphere", "lively atmosphere"],
    "fun, nostalgic atmosphere": ["fun atmosphere", "nostalgic atmosphere"],
    "fun, open-air environment": ["fun environment", "open-air environment"],
    "fun, quirky atmosphere": ["fun atmosphere", "quirky atmosphere"],
    "fun, upbeat": ["fun", "upbeat"],
    "fried chicken and waffles": ["fried chicken", "waffles"],
    "friendly and accommodating environment": ["friendly environment", "accommodating environment"],
    "friendly and accommodating service": ["friendly service", "accommodating service"],
    "friendly and accommodating staff": ["friendly staff", "accommodating staff"],
    "friendly and attentive": ["friendly", "attentive"],
    "friendly and attentive environment": ["friendly environment", "attentive environment"],
    "friendly and caring service": ["friendly service", "caring service"],
    "friendly and considerate staff": ["friendly staff", "considerate staff"],
    "friendly and convenient coffee shop": ["friendly coffee shop", "convenient coffee shop"],
    "friendly and conversational staff": ["friendly staff", "conversational staff"],
    "friendly and courteous staff": ["friendly staff", "courteous staff"],
    "friendly and cozy setting": ["friendly setting", "cozy setting"],
    "friendly and efficient manner": ["friendly manner", "efficient manner"],
    "friendly and energetic food truck experience": ["friendly food truck experience", "energetic food truck experience"],
    "friendly and engaging": ["friendly", "engaging"],
    "friendly and excellent customer service": ["friendly customer service", "excellent customer service"],
    "friendly and helpful": ["friendly", "helpful"],
    "friendly and hip servers": ["friendly servers", "hip servers"],
    "friendly and hospitable staff": ["friendly staff", "hospitable staff"],
    "friendly and integrated": ["friendly", "integrated"],
    "friendly and interactive sushi chefs": ["friendly sushi chefs", "interactive sushi chefs"],
    "friendly and inviting": ["friendly", "inviting"],
    "friendly and knowledgeable wait staff": ["friendly wait staff", "knowledgeable wait staff"],
    "friendly and passionate owners": ["friendly owners", "passionate owners"],
    "friendly and patient service": ["friendly service", "patient service"],
    "friendly and personalized service": ["friendly service", "personalized service"],
    "friendly and positive customer service": ["friendly customer service", "positive customer service"],
    "friendly and professional eyecare services": ["friendly eyecare services", "professional eyecare services"],
    "friendly and professional service": ["friendly service", "professional service"],
    "friendly and quick service": ["friendly service", "quick service"],
    "friendly and sunny atmosphere": ["friendly atmosphere", "sunny atmosphere"],
    "friendly and talented artists": ["friendly artists", "talented artists"],
    "friendly and thorough staff": ["friendly staff", "thorough staff"],
    "friendly and vibrant atmosphere": ["friendly atmosphere", "vibrant atmosphere"],
    "friendly and welcoming customer service": ["friendly customer service", "welcoming customer service"],
    "friendly and welcoming neighborhood spot": ["friendly neighborhood spot", "welcoming neighborhood spot"],
    "friendly, accommodating service": ["friendly service", "accommodating service"],
    "friendly, attentive service": ["friendly service", "attentive service"],
    "friendly, busy atmosphere": ["friendly atmosphere", "busy atmosphere"],
    "friendly, clean environment": ["friendly environment", "clean environment"],
    "friendly, clean, and quick-service setting": ["friendly setting", "clean setting", "quick-service setting"],
    "friendly, country-style diner setting": ["friendly diner setting", "country-style diner setting"],
    "friendly, efficient, and thoughtful dental experience": ["friendly dental experience", "efficient dental experience", "thoughtful dental experience"],
    "friendly, family-like dining experience": ["friendly dining experience", "family-like dining experience"],
    "friendly, family-oriented atmosphere": ["friendly atmosphere", "family-oriented atmosphere"],
    "friendly, family-owned atmosphere": ["friendly atmosphere", "family-owned atmosphere"],
    "friendly, family-run atmosphere": ["friendly atmosphere", "family-run atmosphere"],
    "friendly, fast service": ["friendly service", "fast service"],
    "friendly, full-service diner setting": ["friendly diner setting", "full-service diner setting"],
    "friendly, laid-back vibe": ["friendly vibe", "laid-back vibe"],
    "friendly, local dining atmosphere": ["friendly dining atmosphere", "local dining atmosphere"],
    "friendly, local vibe": ["friendly vibe", "local vibe"],
    "friendly, mom-and-pop deli": ["friendly deli", "mom-and-pop deli"],
    "friendly, neighborly atmosphere": ["friendly atmosphere", "neighborly atmosphere"],
    "friendly, non-touristy setting": ["friendly setting", "non-touristy setting"],
    "friendly, professional service": ["friendly service", "professional service"],
    "friendly, professional setting": ["friendly setting", "professional setting"],
    "friendly, quaint ambience": ["friendly ambience", "quaint ambience"],
    "friendly, well-lit salon environment": ["friendly salon environment", "well-lit salon environment"],
    "games like darts and corn hole": ["games like darts", "games like corn hole"],
    "games like pool and arcade": ["games like pool", "games like arcade"],
    "generous and flavorful sandwiches": ["generous sandwiches", "flavorful sandwiches"],
    "generous portions of meat and cheese": ["generous portions of meat", "generous portions of cheese"],
    "gentle and attentive staff": ["gentle staff", "attentive staff"],
    "gift cards or vouchers": ["gift cards", "vouchers"],
    "good beer and wine selection": ["good beer selection", "good wine selection"],
    "good cigar and drink selections": ["good cigar selections", "good drink selections"],
    "good food and drink specials": ["good food specials", "good drink specials"],
    "good food and drinks": ["good food", "good drinks"],
    "good selection of drinks and games": ["good selection of drinks", "good selection of games"],
    "good selection of wine and beer": ["good selection of wine", "good selection of beer"],
    "good selection of wines and cocktails": ["good selection of wines", "good selection of cocktails"],
    "good value for food and drinks": ["good value for food", "good value for drinks"],
    "great beer and liquor selection": ["great beer selection", "great liquor selection"],
    "great food and drink experience": ["great food experience", "great drink experience"],
    "great food and drink selection": ["great food selection", "great drink selection"],
    "great whiskey and cider selection": ["great whiskey selection", "great cider selection"],
    "great wine and cocktail selection": ["great wine selection", "great cocktail selection"],
    "ham and cheese po-boys": ["ham po-boys", "cheese po-boys"],
    "hearty and affordable options": ["hearty options", "affordable options"],
    "hearty and authentic": ["hearty", "authentic"],
    "hearty and generous portions": ["hearty portions", "generous portions"],
    # From lines 16001-21000
    "helpful and attentive staff": ["helpful staff", "attentive staff"],
    "helpful and knowledgeable": ["helpful", "knowledgeable"],
    "helpful and patient staff": ["helpful staff", "patient staff"],
    "herb and sour flavors": ["herb flavors", "sour flavors"],
    "hibachi and regular dining": ["hibachi dining", "regular dining"],
    "hot and cold tubs": ["hot tubs", "cold tubs"],
    "hot and crispy chicken wings": ["hot chicken wings", "crispy chicken wings"],
    "hot and fresh dishes": ["hot dishes", "fresh dishes"],
    "hot and sour soup": ["hot soup", "sour soup"],
    "hot and tasty": ["hot", "tasty"],
    "high praise for drinks, ambiance, and service": ["high praise for drinks", "high praise for ambiance", "high praise for service"],
    "high-quality breakfast and brunch": ["high-quality breakfast", "high-quality brunch"],
    "high-quality breakfast and lunch options": ["high-quality breakfast options", "high-quality lunch options"],
    "high-quality steak and seafood": ["high-quality steak", "high-quality seafood"],
    "high-quality, beautifully presented dishes": ["high-quality dishes", "beautifully presented dishes"],
    "high-quality, flavorful dishes": ["high-quality dishes", "flavorful dishes"],
    "high-quality, fresh products": ["high-quality products", "fresh products"],
    "high-quality, locally sourced ingredients": ["high-quality ingredients", "locally sourced ingredients"],
    "hip and casual": ["hip", "casual"],
    "hip and cool": ["hip", "cool"],
    "hip and cozy": ["hip", "cozy"],
    "hip and cozy interior": ["hip interior", "cozy interior"],
    "hip and eclectic style": ["hip style", "eclectic style"],
    "hip and elegant setting": ["hip setting", "elegant setting"],
    "hip and friendly": ["hip", "friendly"],
    "hip and functional location": ["hip location", "functional location"],
    "hip and stylish individuals": ["hip individuals", "stylish individuals"],
    "hip and stylish restaurants": ["hip restaurants", "stylish restaurants"],
    "hip, laid-back atmosphere": ["hip atmosphere", "laid-back atmosphere"],
    "hipster and friendly crowds": ["hipster crowds", "friendly crowds"],
    "historic and charming": ["historic", "charming"],
    "historical and elegant setting": ["historical setting", "elegant setting"],
    "history and character": ["history", "character"],
    "home & garden": ["home", "garden"],
    "home & garden supplies": ["home supplies", "garden supplies"],
    "home and garden products": ["home products", "garden products"],
    "impressive beer and wine variety": ["impressive beer variety", "impressive wine variety"],
    "indoor and outdoor dining": ["indoor dining", "outdoor dining"],
    "indoor and outdoor drinking": ["indoor drinking", "outdoor drinking"],
    "indoor/outdoor seating": ["indoor seating", "outdoor seating"],
    "indoor/outdoor venues": ["indoor venues", "outdoor venues"],
    "indulgent, buttery dishes": ["indulgent dishes", "buttery dishes"],
    "industrial and classy decor": ["industrial decor", "classy decor"],
    "inexpensive and tasty pizza": ["inexpensive pizza", "tasty pizza"],
    "informative and detailed exhibits": ["informative exhibits", "detailed exhibits"],
    "informative and interactive": ["informative", "interactive"],
    "informative and well-organized guides": ["informative guides", "well-organized guides"],
    "interactive and hospitable service": ["interactive service", "hospitable service"],
    "intimate and consistent sushi bars": ["intimate sushi bars", "consistent sushi bars"],
    "intimate and exclusive": ["intimate", "exclusive"],
    "intimate and historic theaters": ["intimate theaters", "historic theaters"],
    "intimate and laid-back bars": ["intimate bars", "laid-back bars"],
    "intimate and quaint": ["intimate", "quaint"],
    "intimate, cozy setting": ["intimate setting", "cozy setting"],
    "intimate, dive bar setting": ["intimate setting", "dive bar setting"],
    "intimate, lively dive bar experience": ["intimate dive bar experience", "lively dive bar experience"],
    "jazz and blues music": ["jazz music", "blues music"],
    "juicy and savory burgers": ["juicy burgers", "savory burgers"],
    "knowledgeable and attentive service": ["knowledgeable service", "attentive service"],
    "knowledgeable and engaging staff": ["knowledgeable staff", "engaging staff"],
    "knowledgeable and friendly": ["knowledgeable", "friendly"],
    "knowledgeable and friendly bartending staff": ["knowledgeable bartending staff", "friendly bartending staff"],
    "knowledgeable and friendly staff": ["knowledgeable staff", "friendly staff"],
    "knowledgeable and passionate experience": ["knowledgeable experience", "passionate experience"],
    "laid-back and friendly atmosphere": ["laid-back atmosphere", "friendly atmosphere"],
    "laid-back and relaxing": ["laid-back", "relaxing"],
    "laid-back and romantic": ["laid-back", "romantic"],
    "laid-back, beachy dining experience": ["laid-back dining experience", "beachy dining experience"],
    "laid-back, casual atmosphere": ["laid-back atmosphere", "casual atmosphere"],
    "laid-back, casual dining": ["laid-back dining", "casual dining"],
    "laid-back, eclectic atmosphere": ["laid-back atmosphere", "eclectic atmosphere"],
    "laid-back, grunge": ["laid-back", "grunge"],
    "laid-back, quirky atmosphere": ["laid-back atmosphere", "quirky atmosphere"],
    "large and clean establishment": ["large establishment", "clean establishment"],
    "large and clean shopping environment": ["large shopping environment", "clean shopping environment"],
    "large and diverse menus": ["large menus", "diverse menus"],
    "large and relaxed": ["large", "relaxed"],
    "large beer and cocktail selection": ["large beer selection", "large cocktail selection"],
    "large whiskey and beer selections": ["large whiskey selections", "large beer selections"],
    "large, appealing dining environment": ["large dining environment", "appealing dining environment"],
    "large, flavorful": ["large", "flavorful"],
    "large, fresh burgers": ["large burgers", "fresh burgers"],
    "large, new setting": ["large setting", "new setting"],
    "large, stuffed donuts": ["large donuts", "stuffed donuts"],
    "light and healthy options": ["light options", "healthy options"],
    "light, flavorful dessert experience": ["light dessert experience", "flavorful dessert experience"],
    "light, refreshing, and healthy food options": ["light food options", "refreshing food options", "healthy food options"],
    "light, satisfying meals": ["light meals", "satisfying meals"],
    "live jazz & blues music": ["live jazz music", "live blues music"],
    "live music and performances": ["live music", "live performances"],
    "lively and bustling": ["lively", "bustling"],
    "lively and busy": ["lively", "busy"],
    "lively and busy restaurant": ["lively restaurant", "busy restaurant"],
    "lively and crowded nightlife scene": ["lively nightlife scene", "crowded nightlife scene"],
    "lively and dark bar atmosphere": ["lively bar atmosphere", "dark bar atmosphere"],
    "lively and diverse atmosphere": ["lively atmosphere", "diverse atmosphere"],
    "lively and elegant": ["lively", "elegant"],
    "lively and elegant setting": ["lively setting", "elegant setting"],
    "lively and festive atmosphere": ["lively atmosphere", "festive atmosphere"],
    "lively and popular setting": ["lively setting", "popular setting"],
    "lively and romantic": ["lively", "romantic"],
    "lively, cozy atmosphere": ["lively atmosphere", "cozy atmosphere"],
    "lively, crowded atmosphere": ["lively atmosphere", "crowded atmosphere"],
    "lively, family-friendly atmosphere": ["lively atmosphere", "family-friendly atmosphere"],
    "lively, fun, and historic diner experiences": ["lively diner experiences", "fun diner experiences", "historic diner experiences"],
    "lively, kitschy atmosphere": ["lively atmosphere", "kitschy atmosphere"],
    "lively, modern atmosphere": ["lively atmosphere", "modern atmosphere"],
    "lively, packed atmosphere": ["lively atmosphere", "packed atmosphere"],
    "lively, somewhat loud atmosphere": ["lively atmosphere", "somewhat loud atmosphere"],
    "lively, well-lit atmosphere": ["lively atmosphere", "well-lit atmosphere"],
    "lobster mac & cheese": ["lobster mac", "cheese"],
    "lobster mac and cheese": ["lobster mac", "cheese"],
    "local and fresh beer selection": ["local beer selection", "fresh beer selection"],
    "local and regional farms": ["local farms", "regional farms"],
    "local and seasonal ingredients": ["local ingredients", "seasonal ingredients"],
    "local and upscale atmosphere": ["local atmosphere", "upscale atmosphere"],
    "local and vibrant environment": ["local environment", "vibrant environment"],
    "local, authentic dining experiences": ["local dining experiences", "authentic dining experiences"],
    "local, chill environment": ["local environment", "chill environment"],
    "local, family-owned": ["local", "family-owned"],
    "local, fast-casual setting": ["local setting", "fast-casual setting"],
    "local, friendly atmosphere": ["local atmosphere", "friendly atmosphere"],
    "local, funky cafes": ["local cafes", "funky cafes"],
    "local, non-pretentious atmosphere": ["local atmosphere", "non-pretentious atmosphere"],
    "local, prompt delivery service": ["local delivery service", "prompt delivery service"],
    "locally sourced and sustainable food options": ["locally sourced food options", "sustainable food options"],
    "locally sourced, organic products": ["locally sourced products", "organic products"],
    "locals and visitors": ["locals", "visitors"],
    "loud and lively": ["loud", "lively"],
    "loud and lively atmospheres": ["loud atmospheres", "lively atmospheres"],
    "low-key and relaxed": ["low-key", "relaxed"],
    "low-key, cozy atmospheres": ["low-key atmospheres", "cozy atmospheres"],
    "low-key, family-friendly atmosphere": ["low-key atmosphere", "family-friendly atmosphere"],
    "low-key, old-school bars": ["low-key bars", "old-school bars"],
    "luxurious and relaxing": ["luxurious", "relaxing"],
    "luxurious and welcoming environment": ["luxurious environment", "welcoming environment"],
    "luxurious, fine dining experiences": ["luxurious dining experiences", "fine dining experiences"],
    "mac & cheese": ["mac", "cheese"],
    "mac and cheese": ["mac", "cheese"],
    "mac and cheese dishes": ["mac dishes", "cheese dishes"],
    "mac and cheese skillet": ["mac skillet", "cheese skillet"],
    "mac and cheese with brisket": ["mac with brisket", "cheese with brisket"],
    "mac n cheese": ["mac", "cheese"],
    "mac n' cheese": ["mac", "cheese"],
    "mac-n-cheese": ["mac", "cheese"],
    "macaroni and cheese": ["macaroni", "cheese"],
    "meat & three": ["meat", "three"],
    "meat & three dishes": ["meat dishes", "three dishes"],
    "meat & three meals": ["meat meals", "three meals"],
    "meat and seafood dishes": ["meat dishes", "seafood dishes"],
    "mellow and relaxed": ["mellow", "relaxed"],
    "messy, casual dining": ["messy dining", "casual dining"],
    "minimalist and welcoming": ["minimalist", "welcoming"],
    "minimalist romantic atmosphere": ["minimalist atmosphere", "romantic atmosphere"],
    "mix of locals and tourists": ["mix of locals", "mix of tourists"],
    "mix of music genres": ["mix of music", "mix of genres"],
    "mix of national and local shops": ["mix of national shops", "mix of local shops"],
    "mix of stores and restaurants": ["mix of stores", "mix of restaurants"],
    "mix of traditional dishes and street food": ["mix of traditional dishes", "mix of street food"],
    "mix of upscale and dive bar vibes": ["mix of upscale vibes", "mix of dive bar vibes"],
    "modern and artistic spaces": ["modern spaces", "artistic spaces"],
    "modern and classic styles": ["modern styles", "classic styles"],
    "modern and clean": ["modern", "clean"],
    "modern and clean ambiance": ["modern ambiance", "clean ambiance"],
    "modern and clean environment": ["modern environment", "clean environment"],
    "modern and clean restaurant interiors": ["modern restaurant interiors", "clean restaurant interiors"],
    "modern and clean restaurant setting": ["modern restaurant setting", "clean restaurant setting"],
    "modern and clean restaurants": ["modern restaurants", "clean restaurants"],
    "modern and hip atmosphere": ["modern atmosphere", "hip atmosphere"],
    "modern and interactive ordering environment": ["modern ordering environment", "interactive ordering environment"],
    "modern and manga-inspired atmosphere": ["modern atmosphere", "manga-inspired atmosphere"],
    "modern and minimalist setting": ["modern setting", "minimalist setting"],
    "modern and polished bakery setting": ["modern bakery setting", "polished bakery setting"],
    "modern and relaxed hangout spot": ["modern hangout spot", "relaxed hangout spot"],
    "modern and romantic setting": ["modern setting", "romantic setting"],
    "modern and simple interiors": ["modern interiors", "simple interiors"],
    "modern and spacious cinema": ["modern cinema", "spacious cinema"],
    "modern and spacious dessert establishment": ["modern dessert establishment", "spacious dessert establishment"],
    "modern and spacious hotel": ["modern hotel", "spacious hotel"],
    "modern and stylish setting": ["modern setting", "stylish setting"],
    "modern and upscale": ["modern", "upscale"],
    "modern and upscale setting": ["modern setting", "upscale setting"],
    "modern and welcoming office vibe": ["modern office vibe", "welcoming office vibe"],
    "modern and welcoming setting": ["modern setting", "welcoming setting"],
    "modern and well-equipped": ["modern", "well-equipped"],
    "modern, chic": ["modern", "chic"],
    "modern, chic ambiance": ["modern ambiance", "chic ambiance"],
    "modern, chic dining experiences": ["modern dining experiences", "chic dining experiences"],
    "modern, classy bars": ["modern bars", "classy bars"],
    "modern, classy hotel": ["modern hotel", "classy hotel"],
    "modern, clean": ["modern", "clean"],
    "modern, clean dining experience": ["modern dining experience", "clean dining experience"],
    "modern, clean rooms": ["modern rooms", "clean rooms"],
    "modern, clean setting": ["modern setting", "clean setting"],
    "modern, clean, and cozy salons": ["modern salons", "clean salons", "cozy salons"],
    "modern, fast-casual setting": ["modern setting", "fast-casual setting"],
    "modern, fast-food dining experience": ["modern dining experience", "fast-food dining experience"],
    "modern, inviting space": ["modern space", "inviting space"],
    "modern, lively setting": ["modern setting", "lively setting"],
    "modern, lively spot": ["modern spot", "lively spot"],
    "modern, minimalist ambiance": ["modern ambiance", "minimalist ambiance"],
    "modern, spacious dining experience": ["modern dining experience", "spacious dining experience"],
    "modern, specialty supermarket atmosphere": ["modern supermarket atmosphere", "specialty supermarket atmosphere"],
    "modern, stylish cafe setting": ["modern cafe setting", "stylish cafe setting"],
    "modern, swanky atmosphere": ["modern atmosphere", "swanky atmosphere"],
    "modern, upscale dishes": ["modern dishes", "upscale dishes"],
    "modern, upscale twist": ["modern twist", "upscale twist"],
    "moist and flavorful": ["moist", "flavorful"],
    "moist and flavorful cakes": ["moist cakes", "flavorful cakes"],
    "moist and rich cakes": ["moist cakes", "rich cakes"],
    "natural and unprocessed food": ["natural food", "unprocessed food"],
    "natural/vegetarian menu": ["natural menu", "vegetarian menu"],
    "patient and understanding staff": ["patient staff", "understanding staff"],
    "peaceful and quiet environment": ["peaceful environment", "quiet environment"],
    "peaceful and relaxing environment": ["peaceful environment", "relaxing environment"],
    "peaceful and relaxing experience": ["peaceful experience", "relaxing experience"],
    "peanut butter and jelly sandwiches": ["peanut butter sandwiches", "jelly sandwiches"],
    # From lines 21001-23500
    "people who enjoy festive lights and activities": ["people who enjoy festive lights", "people who enjoy activities"],
    "people who enjoy fresh produce and treats": ["people who enjoy fresh produce", "people who enjoy treats"],
    "people who enjoy fun atmospheres and unique experiences": ["people who enjoy fun atmospheres", "people who enjoy unique experiences"],
    "people who enjoy modern and clean restaurants": ["people who enjoy modern restaurants", "people who enjoy clean restaurants"],
    "people who enjoy seafood and steak options": ["people who enjoy seafood options", "people who enjoy steak options"],
    "people who enjoy sharing and trying a variety of dishes": ["people who enjoy sharing", "people who enjoy trying a variety of dishes"],
    "people who prioritize great service and hospitality": ["people who prioritize great service", "people who prioritize hospitality"],
    "people who prioritize quality tanning and hair removal services": ["people who prioritize quality tanning services", "people who prioritize hair removal services"],
    "personalized and attentive service": ["personalized service", "attentive service"],
    "pizza and drinks": ["pizza", "drinks"],
    "pizza pasta": ["pizza", "pasta"],
    "pizza with red sauce and cheese": ["pizza with red sauce", "pizza with cheese"],
    "popular among locals and visitors": ["popular among locals", "popular among visitors"],
    "premium food and drinks": ["premium food", "premium drinks"],
    "professional and attentive": ["professional", "attentive"],
    "professional and friendly stylists": ["professional stylists", "friendly stylists"],
    "professional and skillful": ["professional", "skillful"],
    "professional, attentive and gentle service": ["professional service", "attentive service", "gentle service"],
    "prompt and friendly service": ["prompt service", "friendly service"],
    "prompt and polite staff": ["prompt staff", "polite staff"],
    "prompt, professional service": ["prompt service", "professional service"],
    "quaint and comfortable": ["quaint", "comfortable"],
    "quaint and cozy dining experience": ["quaint dining experience", "cozy dining experience"],
    "quaint and friendly": ["quaint", "friendly"],
    "quaint and homey": ["quaint", "homey"],
    "quaint and magical setting": ["quaint setting", "magical setting"],
    "quaint and welcoming atmosphere": ["quaint atmosphere", "welcoming atmosphere"],
    "quaint, BYOB setting": ["quaint setting", "BYOB setting"],
    "quaint, flavorful dining experience": ["quaint dining experience", "flavorful dining experience"],
    "quaint, local cafe vibe": ["quaint cafe vibe", "local cafe vibe"],
    "quaint, local-feeling setting": ["quaint setting", "local-feeling setting"],
    "quaint, relaxing, and friendly environment": ["quaint environment", "relaxing environment", "friendly environment"],
    "quality cheese and meat boards": ["quality cheese boards", "quality meat boards"],
    "quality chips and salsa": ["quality chips", "quality salsa"],
    "quality coffee/tea": ["quality coffee", "quality tea"],
    "quality food and beer selections": ["quality food selections", "quality beer selections"],
    "quality food and drinks": ["quality food", "quality drinks"],
    "quality wine and beer tastings": ["quality wine tastings", "quality beer tastings"],
    "quick and attentive service": ["quick service", "attentive service"],
    "quick and convenient": ["quick", "convenient"],
    "quick and courteous service": ["quick service", "courteous service"],
    "quick and delicious breakfast on-the-go": ["quick breakfast on-the-go", "delicious breakfast on-the-go"],
    "quick and delicious food options": ["quick food options", "delicious food options"],
    "quick and delicious meals": ["quick meals", "delicious meals"],
    "quick and delicious sushi takeout": ["quick sushi takeout", "delicious sushi takeout"],
    "quick and easy meals": ["quick meals", "easy meals"],
    "quick and efficient": ["quick", "efficient"],
    "quick and efficient medical care": ["quick medical care", "efficient medical care"],
    "quick and efficient tire services": ["quick tire services", "efficient tire services"],
    "quick and friendly staff": ["quick staff", "friendly staff"],
    "quick and quality lunch": ["quick lunch", "quality lunch"],
    "quick and quality services": ["quick services", "quality services"],
    "quick and satisfying meals": ["quick meals", "satisfying meals"],
    "quick and tasty meals": ["quick meals", "tasty meals"],
    "quick, affordable breakfast spot": ["quick breakfast spot", "affordable breakfast spot"],
    "quick, casual dining experience": ["quick dining experience", "casual dining experience"],
    "quick, easy, and affordable experience": ["quick experience", "easy experience", "affordable experience"],
    "quick, quality meal": ["quick meal", "quality meal"],
    "quiet and cozy": ["quiet", "cozy"],
    "quiet and creative atmosphere": ["quiet atmosphere", "creative atmosphere"],
    "quiet, low-key": ["quiet", "low-key"],
    "quirky and fresh dishes": ["quirky dishes", "fresh dishes"],
    "quirky and unique dive bars": ["quirky dive bars", "unique dive bars"],
    "quirky, cozy setting": ["quirky setting", "cozy setting"],
    "quirky, intimate atmosphere": ["quirky atmosphere", "intimate atmosphere"],
    "raucous and irreverent celebrations": ["raucous celebrations", "irreverent celebrations"],
    "red beans & rice": ["red beans", "rice"],
    "red beans and rice": ["red beans", "rice"],
    "relaxed and enjoyable time with friends": ["relaxed time with friends", "enjoyable time with friends"],
    "relaxed and laid-back": ["relaxed", "laid-back"],
    "relaxed and zen atmosphere": ["relaxed atmosphere", "zen atmosphere"],
    "relaxed, casual setting": ["relaxed setting", "casual setting"],
    "relaxing and casual dining": ["relaxing dining", "casual dining"],
    "relaxing and friendly spa setting": ["relaxing spa setting", "friendly spa setting"],
    "relaxing and heartwarming experience": ["relaxing experience", "heartwarming experience"],
    "relaxing and pampering experience": ["relaxing experience", "pampering experience"],
    "relaxing and welcoming setting": ["relaxing setting", "welcoming setting"],
    "restaurant and bar": ["restaurant", "bar"],
    "restaurant and market experience": ["restaurant experience", "market experience"],
    "rich and creamy frostings": ["rich frostings", "creamy frostings"],
    "rich and delicious plates": ["rich plates", "delicious plates"],
    "rich and flavorful food": ["rich food", "flavorful food"],
    "rich and savory broth": ["rich broth", "savory broth"],
    "rich, creamy, and generous portions": ["rich portions", "creamy portions", "generous portions"],
    "rich, decadent cakes": ["rich cakes", "decadent cakes"],
    "romantic and casual": ["romantic", "casual"],
    "romantic and swanky": ["romantic", "swanky"],
    "romantic, cozy dining experiences": ["romantic dining experiences", "cozy dining experiences"],
    "romantic, historic ambiance": ["romantic ambiance", "historic ambiance"],
    "rustic and intimate setting": ["rustic setting", "intimate setting"],
    "rustic and welcoming": ["rustic", "welcoming"],
    "rustic, calming environment": ["rustic environment", "calming environment"],
    "rustic, open setting": ["rustic setting", "open setting"],
    "salt and pepper chicken wings": ["salt chicken wings", "pepper chicken wings"],
    "salt and pepper soft-shell crab": ["salt soft-shell crab", "pepper soft-shell crab"],
    "seafood & meat dishes": ["seafood dishes", "meat dishes"],
    "seafood and sushi selections": ["seafood selections", "sushi selections"],
    # From lines 23501-28434
    "shrimp & grits": ["shrimp", "grits"],
    "shrimp & pork noodle soup": ["shrimp noodle soup", "pork noodle soup"],
    "shrimp and grits": ["shrimp", "grits"],
    "simple, authentic": ["simple", "authentic"],
    "simple, comforting food": ["simple food", "comforting food"],
    "simple, high-quality ingredients": ["simple ingredients", "high-quality ingredients"],
    "simple, modest setting": ["simple setting", "modest setting"],
    "skilled and entertaining bartenders": ["skilled bartenders", "entertaining bartenders"],
    "sleek and modern": ["sleek", "modern"],
    "sleek, modern salon setting": ["sleek salon setting", "modern salon setting"],
    "sleek, modern setting": ["sleek setting", "modern setting"],
    "sophisticated and comfortable lounge": ["sophisticated lounge", "comfortable lounge"],
    "sophisticated and knowledgeable bartenders": ["sophisticated bartenders", "knowledgeable bartenders"],
    "sophisticated, vintage": ["sophisticated", "vintage"],
    "spacious and airy salon environment": ["spacious salon environment", "airy salon environment"],
    "spacious and comfortable dining settings": ["spacious dining settings", "comfortable dining settings"],
    "spacious and inviting": ["spacious", "inviting"],
    "spacious and inviting ambiance": ["spacious ambiance", "inviting ambiance"],
    "spacious and newer locations": ["spacious locations", "newer locations"],
    "spacious and quiet dining experience": ["spacious dining experience", "quiet dining experience"],
    "spacious and visually appealing bar": ["spacious bar", "visually appealing bar"],
    "spacious and well-decorated taproom": ["spacious taproom", "well-decorated taproom"],
    "spacious and well-designed setting": ["spacious setting", "well-designed setting"],
    "spacious, bright, and chill vibes": ["spacious vibes", "bright vibes", "chill vibes"],
    "spacious, clean environment": ["spacious environment", "clean environment"],
    "spacious, clean setting": ["spacious setting", "clean setting"],
    "spacious, colorful ambiance": ["spacious ambiance", "colorful ambiance"],
    "spacious, community-oriented atmosphere": ["spacious atmosphere", "community-oriented atmosphere"],
    "spacious, friendly environments": ["spacious environments", "friendly environments"],
    "spacious, group-friendly atmosphere": ["spacious atmosphere", "group-friendly atmosphere"],
    "spacious, multi-level setting": ["spacious setting", "multi-level setting"],
    "spacious, off-leash areas": ["spacious areas", "off-leash areas"],
    "spacious, trendy setting": ["spacious setting", "trendy setting"],
    "spacious, upscale": ["spacious", "upscale"],
    "spacious, well-lit coffee spots": ["spacious coffee spots", "well-lit coffee spots"],
    "spacious, well-maintained setting": ["spacious setting", "well-maintained setting"],
    "spicy and flavorful": ["spicy", "flavorful"],
    "spicy and flavorful broth": ["spicy broth", "flavorful broth"],
    "sweet & salty ham": ["sweet ham", "salty ham"],
    "sweet & tangy sauce": ["sweet sauce", "tangy sauce"],
    "sweet and delicious BBQ": ["sweet BBQ", "delicious BBQ"],
    "sweet and dry wines": ["sweet wines", "dry wines"],
    "sweet and salty flavors": ["sweet flavors", "salty flavors"],
    "sweet and savory cravings": ["sweet cravings", "savory cravings"],
    "sweet and savory flavors": ["sweet flavors", "savory flavors"],
    "sweet and spicy chicken": ["sweet chicken", "spicy chicken"],
    "tasty breakfast and brunch options": ["tasty breakfast options", "tasty brunch options"],
    "tasty, greasy burgers": ["tasty burgers", "greasy burgers"],
    "tasty, healthy, quality food": ["tasty food", "healthy food", "quality food"],
    "tender and flavorful": ["tender", "flavorful"],
    "tender and flavorful steaks": ["tender steaks", "flavorful steaks"],
    "top-notch food and drinks": ["top-notch food", "top-notch drinks"],
    "traditional and comforting dishes": ["traditional dishes", "comforting dishes"],
    "traditional and modern vibes": ["traditional vibes", "modern vibes"],
    "trendy and Instagrammable": ["trendy", "Instagrammable"],
    "trendy and affordable clothing": ["trendy clothing", "affordable clothing"],
    "trendy and artsy": ["trendy", "artsy"],
    "trendy and beautifully decorated spaces": ["trendy spaces", "beautifully decorated spaces"],
    "trendy and busy": ["trendy", "busy"],
    "trendy and clean ambience": ["trendy ambience", "clean ambience"],
    "trendy and clean food hall environment": ["trendy food hall environment", "clean food hall environment"],
    "trendy and comfortable cafe": ["trendy cafe", "comfortable cafe"],
    "trendy and cozy spot": ["trendy spot", "cozy spot"],
    "trendy and eclectic atmosphere": ["trendy atmosphere", "eclectic atmosphere"],
    "trendy and hip": ["trendy", "hip"],
    "trendy and intimate": ["trendy", "intimate"],
    "trendy and modern": ["trendy", "modern"],
    "trendy and modern spaces": ["trendy spaces", "modern spaces"],
    "trendy and spacious": ["trendy", "spacious"],
    "trendy and spacious venues": ["trendy venues", "spacious venues"],
    "trendy and unique coffee shops": ["trendy coffee shops", "unique coffee shops"],
    "trendy and unique healthy food options": ["trendy healthy food options", "unique healthy food options"],
    "trendy and upbeat": ["trendy", "upbeat"],
    "trendy and vibrant": ["trendy", "vibrant"],
    "trendy and welcoming atmospheres": ["trendy atmospheres", "welcoming atmospheres"],
    "trendy, Instagram-worthy atmosphere": ["trendy atmosphere", "Instagram-worthy atmosphere"],
    "trendy, Instagram-worthy spaces": ["trendy spaces", "Instagram-worthy spaces"],
    "trendy, affordable cafe": ["trendy cafe", "affordable cafe"],
    "trendy, comfortable rooms": ["trendy rooms", "comfortable rooms"],
    "trendy, fast-casual": ["trendy", "fast-casual"],
    "trendy, fun, and friendly dining experiences": ["trendy dining experiences", "fun dining experiences", "friendly dining experiences"],
    "trendy, industrial-chic decor": ["trendy decor", "industrial-chic decor"],
    "trendy, lively": ["trendy", "lively"],
    "trendy, minimalist coffee shop": ["trendy coffee shop", "minimalist coffee shop"],
    "trendy, modern": ["trendy", "modern"],
    "trendy, photogenic cafes": ["trendy cafes", "photogenic cafes"],
    "trendy, retro atmosphere": ["trendy atmosphere", "retro atmosphere"],
    "trendy, retro-themed": ["trendy", "retro-themed"],
    "trendy, romantic atmosphere": ["trendy atmosphere", "romantic atmosphere"],
    "trendy, unconventional, and oversized burritos": ["trendy burritos", "unconventional burritos", "oversized burritos"],
    "trendy, up-and-coming area": ["trendy area", "up-and-coming area"],
    "trendy, upscale bar": ["trendy bar", "upscale bar"],
    "trendy, upscale dining": ["trendy dining", "upscale dining"],
    "trendy, upscale environment": ["trendy environment", "upscale environment"],
    "trendy, vibrant atmosphere": ["trendy atmosphere", "vibrant atmosphere"],
    "unique and aesthetic atmosphere": ["unique atmosphere", "aesthetic atmosphere"],
    "unique and affordable clothing": ["unique clothing", "affordable clothing"],
    "unique and affordable records": ["unique records", "affordable records"],
    "unique and bold seasoning": ["unique seasoning", "bold seasoning"],
    "unique and clever drink names": ["unique drink names", "clever drink names"],
    "unique and complex beer styles": ["unique beer styles", "complex beer styles"],
    "unique and cozy setting": ["unique setting", "cozy setting"],
    "unique and creative menu offerings": ["unique menu offerings", "creative menu offerings"],
    "unique and creative pizza": ["unique pizza", "creative pizza"],
    "unique and creative takes": ["unique takes", "creative takes"],
    "unique and creative toasted subs": ["unique toasted subs", "creative toasted subs"],
    "unique and customizable desserts": ["unique desserts", "customizable desserts"],
    "unique and daring menu items": ["unique menu items", "daring menu items"],
    "unique and delicious": ["unique", "delicious"],
    "unique and delicious cake varieties": ["unique cake varieties", "delicious cake varieties"],
    "unique and delicious drinks": ["unique drinks", "delicious drinks"],
    "unique and delicious flavors": ["unique flavors", "delicious flavors"],
    "unique and delicious pizza combinations": ["unique pizza combinations", "delicious pizza combinations"],
    "unique and eclectic men's and women's clothing": ["unique men's clothing", "eclectic men's clothing", "unique women's clothing", "eclectic women's clothing"],
    "unique and engaging atmosphere": ["unique atmosphere", "engaging atmosphere"],
    "unique and entertaining dessert experiences": ["unique dessert experiences", "entertaining dessert experiences"],
    "unique and exotic drink selections": ["unique drink selections", "exotic drink selections"],
    "unique and exotic toppings": ["unique toppings", "exotic toppings"],
    "unique and experimental dishes": ["unique dishes", "experimental dishes"],
    "unique and experimental flavors": ["unique flavors", "experimental flavors"],
    "unique and flavor-packed brews": ["unique brews", "flavor-packed brews"],
    "unique and flavorful chicken dishes": ["unique chicken dishes", "flavorful chicken dishes"],
    "unique and flavorful donuts": ["unique donuts", "flavorful donuts"],
    "unique and flavorful empanadas": ["unique empanadas", "flavorful empanadas"],
    "unique and flavorful food options": ["unique food options", "flavorful food options"],
    "unique and flavorful meals": ["unique meals", "flavorful meals"],
    "unique and flavorful menu offerings": ["unique menu offerings", "flavorful menu offerings"],
    "unique and flavorful options": ["unique options", "flavorful options"],
    "unique and flavorful popsicles": ["unique popsicles", "flavorful popsicles"],
    "unique and flavorful rolls": ["unique rolls", "flavorful rolls"],
    "unique and frequently changing menu options": ["unique menu options", "frequently changing menu options"],
    "unique and fresh fruits": ["unique fruits", "fresh fruits"],
    "unique and historic": ["unique", "historic"],
    "unique and immersive setting": ["unique setting", "immersive setting"],
    "unique and intimate bar experiences": ["unique bar experiences", "intimate bar experiences"],
    "unique and inventive coffee and food options": ["unique coffee options", "inventive coffee options", "unique food options", "inventive food options"],
    "unique and inviting": ["unique", "inviting"],
    "unique and locally-inspired fashion": ["unique fashion", "locally-inspired fashion"],
    "unique and nostalgic setting": ["unique setting", "nostalgic setting"],
    "unique and personalized cake options": ["unique cake options", "personalized cake options"],
    "unique and refreshing": ["unique", "refreshing"],
    "unique and rich gelato flavors": ["unique gelato flavors", "rich gelato flavors"],
    "unique and spooky atmosphere": ["unique atmosphere", "spooky atmosphere"],
    "unique and trendy eyewear": ["unique eyewear", "trendy eyewear"],
    "unique, character-filled": ["unique", "character-filled"],
    "unique, cozy, and spacious": ["unique", "cozy", "spacious"],
    "unique, creative dishes": ["unique dishes", "creative dishes"],
    "unique, fresh offerings": ["unique offerings", "fresh offerings"],
    "unique, fresh, and changing menus": ["unique menus", "fresh menus", "changing menus"],
    "unique, handmade gifts": ["unique gifts", "handmade gifts"],
    "unique, intricately flavored dishes": ["unique dishes", "intricately flavored dishes"],
    "unique, local atmosphere": ["unique atmosphere", "local atmosphere"],
    "unique, speakeasy-style bars": ["unique bars", "speakeasy-style bars"],
    "unlimited chips and dip": ["unlimited chips", "unlimited dip"],
    "unlimited soup and salad": ["unlimited soup", "unlimited salad"],
    "unlimited soup and salad options": ["unlimited soup options", "unlimited salad options"],
    "upscale and casual eateries": ["upscale eateries", "casual eateries"],
    "upscale and romantic setting": ["upscale setting", "romantic setting"],
    "upscale but welcoming atmosphere": ["upscale atmosphere", "welcoming atmosphere"],
    "upscale, classy atmosphere": ["upscale atmosphere", "classy atmosphere"],
    "upscale, historic building": ["upscale building", "historic building"],
    "variety of beer and wine selections": ["variety of beer selections", "variety of wine selections"],
    "variety of food and beverage options": ["variety of food options", "variety of beverage options"],
    "variety of meat and seafood options": ["variety of meat options", "variety of seafood options"],
    "variety of meats and cheeses": ["variety of meats", "variety of cheeses"],
    "variety of wine and spirits": ["variety of wine", "variety of spirits"],
    "vegan and plant-based cuisine": ["vegan cuisine", "plant-based cuisine"],
    "vegetarian and vegan breakfast and brunch options": ["vegetarian breakfast options", "vegan breakfast options", "vegetarian brunch options", "vegan brunch options"],
    "vegetarian/vegan cuisine": ["vegetarian cuisine", "vegan cuisine"],
    "vegetarian/vegan food": ["vegetarian food", "vegan food"],
    "vegetarian/vegan-friendly meals": ["vegetarian-friendly meals", "vegan-friendly meals"],
    "vibrant and artistic": ["vibrant", "artistic"],
    "vibrant and artistic setting": ["vibrant setting", "artistic setting"],
    "vibrant and busy setting": ["vibrant setting", "busy setting"],
    "vibrant and colorful": ["vibrant", "colorful"],
    "vibrant and colorful hangout spot": ["vibrant hangout spot", "colorful hangout spot"],
    "vibrant and crowded environment": ["vibrant environment", "crowded environment"],
    "vibrant and cute": ["vibrant", "cute"],
    "vibrant and eclectic store atmosphere": ["vibrant store atmosphere", "eclectic store atmosphere"],
    "vibrant and enthusiastic fan atmosphere": ["vibrant fan atmosphere", "enthusiastic fan atmosphere"],
    "vibrant and fun ambiance": ["vibrant ambiance", "fun ambiance"],
    "vibrant and fun dining experience": ["vibrant dining experience", "fun dining experience"],
    "vibrant and historic setting": ["vibrant setting", "historic setting"],
    "vibrant and music-themed environment": ["vibrant environment", "music-themed environment"],
    "vibrant and welcoming setting": ["vibrant setting", "welcoming setting"],
    "vibrant, artistic atmosphere": ["vibrant atmosphere", "artistic atmosphere"],
    "vibrant, energetic atmosphere": ["vibrant atmosphere", "energetic atmosphere"],
    "vibrant, riverfront setting": ["vibrant setting", "riverfront setting"],
    "vibrant, tiki lounge atmosphere": ["vibrant atmosphere", "tiki lounge atmosphere"],
    "warm and attentive service": ["warm service", "attentive service"],
    "warm and cozy atmosphere": ["warm atmosphere", "cozy atmosphere"],
    "warm and delicious treats": ["warm treats", "delicious treats"],
    "warm and efficient service": ["warm service", "efficient service"],
    "warm and friendly": ["warm", "friendly"],
    "warm and friendly service": ["warm service", "friendly service"],
    "warm and friendly staff": ["warm staff", "friendly staff"],
    "warm and genuine atmosphere": ["warm atmosphere", "genuine atmosphere"],
    "warm and hip atmosphere": ["warm atmosphere", "hip atmosphere"],
    "warm and homely": ["warm", "homely"],
    "warm and inclusive dining experience": ["warm dining experience", "inclusive dining experience"],
    "warm and inviting": ["warm", "inviting"],
    "warm and inviting deli setting": ["warm deli setting", "inviting deli setting"],
    "warm and inviting environment": ["warm environment", "inviting environment"],
    "warm and inviting staff": ["warm staff", "inviting staff"],
    "warm and romantic": ["warm", "romantic"],
    "warm and soft naan": ["warm naan", "soft naan"],
    "warm and welcoming": ["warm", "welcoming"],
    "warm and welcoming dining experiences": ["warm dining experiences", "welcoming dining experiences"],
    "warm and welcoming owners": ["warm owners", "welcoming owners"],
    "warm and welcoming pub atmosphere": ["warm pub atmosphere", "welcoming pub atmosphere"],
    "warm, attentive service": ["warm service", "attentive service"],
    "warm, cozy setting": ["warm setting", "cozy setting"],
    "warm, freshly baked cookies": ["warm cookies", "freshly baked cookies"],
    "warm, hospitable atmosphere": ["warm atmosphere", "hospitable atmosphere"],
    "warm, inviting vibe": ["warm vibe", "inviting vibe"],
    "warm, laid-back": ["warm", "laid-back"],
    "warm, welcoming atmosphere": ["warm atmosphere", "welcoming atmosphere"],
    "welcoming and comfortable salon atmosphere": ["welcoming salon atmosphere", "comfortable salon atmosphere"],
    "welcoming and diverse atmosphere": ["welcoming atmosphere", "diverse atmosphere"],
    "welcoming and friendly neighborhood pub": ["welcoming neighborhood pub", "friendly neighborhood pub"],
    "welcoming and fun atmosphere": ["welcoming atmosphere", "fun atmosphere"],
    "welcoming and relaxed environment": ["welcoming environment", "relaxed environment"],
    "welcoming and tranquil environment": ["welcoming environment", "tranquil environment"],
    "welcoming, energetic environment": ["welcoming environment", "energetic environment"],
    "welcoming, family-friendly": ["welcoming", "family-friendly"],
    "welcoming, homey atmosphere": ["welcoming atmosphere", "homey atmosphere"],
    "welcoming, non-intimidating environment": ["welcoming environment", "non-intimidating environment"],
    "welcoming, organized space": ["welcoming space", "organized space"],
    "well-battered and fried options": ["well-battered options", "fried options"],
    "well-cooked flavorful dishes": ["well-cooked dishes", "flavorful dishes"],
    "well-crafted and unique drinks": ["well-crafted drinks", "unique drinks"],
    "well-designed and clean restrooms": ["well-designed restrooms", "clean restrooms"],
    "well-lit and safe environment": ["well-lit environment", "safe environment"],
    "well-lit, spacious environment": ["well-lit environment", "spacious environment"],
    "well-presented and flavorful food": ["well-presented food", "flavorful food"],
    "well-run and clean": ["well-run", "clean"],
    "well-spiced and flavorful": ["well-spiced", "flavorful"],
    "western and historical": ["western", "historical"],
    "wine and cocktail selection": ["wine selection", "cocktail selection"],
    "wide variety of meats and cheeses": ["wide variety of meats", "wide variety of cheeses"],
    "with friends and family": ["with friends", "with family"],
}

# Entity addition rules: Add multiple tail entities while keeping the original entity
# Format: {tail_to_expand: [list_of_additional_tails]}
# Note: The original entity is kept, and additional triplets are created with the new entities
ENTITY_ADDITION_RULES = {
    # Example: "Italian cuisine" will create triplets with both "Italian cuisine" and "Italian"
    # "Italian cuisine": ["Italian"],
    # Add your addition rules here:
    # Moved from ENTITY_EXPANSION_RULES (entries containing "yet"):
    "affordable yet classy dining": ["affordable dining", "classy dining"],
    "affordable yet quality eats": ["affordable eats", "quality eats"],
    "bustling yet cozy setting": ["bustling setting", "cozy setting"],
    "bustling yet laid-back atmosphere": ["bustling atmosphere", "laid-back atmosphere"],
    "busy yet comfortable atmosphere": ["busy atmosphere", "comfortable atmosphere"],
    "casual yet charming atmosphere": ["casual atmosphere", "charming atmosphere"],
    "casual yet classy": ["casual", "classy"],
    "casual yet contemporary dining atmosphere": ["casual dining atmosphere", "contemporary dining atmosphere"],
    "casual yet elegant": ["casual", "elegant"],
    "casual yet elegant dining atmosphere": ["casual dining atmosphere", "elegant dining atmosphere"],
    "casual yet elegant dining experience": ["casual dining experience", "elegant dining experience"],
    "casual yet festive": ["casual", "festive"],
    "casual yet inviting atmosphere": ["casual atmosphere", "inviting atmosphere"],
    "casual yet lively atmosphere": ["casual atmosphere", "lively atmosphere"],
    "casual yet nice environment": ["casual environment", "nice environment"],
    "casual yet refined atmosphere": ["casual atmosphere", "refined atmosphere"],
    "casual yet refined dining atmosphere": ["casual dining atmosphere", "refined dining atmosphere"],
    "casual yet refined dining experience": ["casual dining experience", "refined dining experience"],
    "casual yet slightly upscale ambiance": ["casual ambiance", "slightly upscale ambiance"],
    "casual yet sophisticated": ["casual", "sophisticated"],
    "casual yet sophisticated atmosphere": ["casual atmosphere", "sophisticated atmosphere"],
    "casual yet tasteful setting": ["casual setting", "tasteful setting"],
    "casual yet upscale": ["casual", "upscale"],
    "casual yet upscale setting": ["casual setting", "upscale setting"],
    "casual yet welcoming ambiance": ["casual ambiance", "welcoming ambiance"],
    "chill yet excellent vibe": ["chill vibe", "excellent vibe"],
    "classy yet comfortable": ["classy", "comfortable"],
    "classy yet relaxed atmosphere": ["classy atmosphere", "relaxed atmosphere"],
    "classy yet unassuming setting": ["classy setting", "unassuming setting"],
    "classy yet welcoming atmosphere": ["classy atmosphere", "welcoming atmosphere"],
    "cluttered yet endearing": ["cluttered", "endearing"],
    "contemporary yet comfortable dining atmosphere": ["contemporary dining atmosphere", "comfortable dining atmosphere"],
    "cozy yet classy": ["cozy", "classy"],
    "cozy yet hip atmosphere": ["cozy atmosphere", "hip atmosphere"],
    "cozy yet lively atmosphere": ["cozy atmosphere", "lively atmosphere"],
    "cozy yet popular": ["cozy", "popular"],
    "cozy yet sophisticated setting": ["cozy setting", "sophisticated setting"],
    "cozy yet trendy atmosphere": ["cozy atmosphere", "trendy atmosphere"],
    "cozy yet vibrant atmosphere": ["cozy atmosphere", "vibrant atmosphere"],
    "cozy, yet elegant atmosphere": ["cozy atmosphere", "elegant atmosphere"],
    "crispy yet soft waffles": ["crispy waffles", "soft waffles"],
    "fancy yet affordable dining": ["fancy dining", "affordable dining"],
    "laid-back yet elevated ambiance": ["laid-back ambiance", "elevated ambiance"],
    "laid-back yet sophisticated atmosphere": ["laid-back atmosphere", "sophisticated atmosphere"],
    "lively yet comfortable atmosphere": ["lively atmosphere", "comfortable atmosphere"],
    "lively yet cozy atmosphere": ["lively atmosphere", "cozy atmosphere"],
    "lively yet relaxing atmosphere": ["lively atmosphere", "relaxing atmosphere"],
    "lively yet semi-private dining environment": ["lively dining environment", "semi-private dining environment"],
    "lively yet upscale": ["lively", "upscale"],
    "messy yet tasty BBQ food": ["messy BBQ food", "tasty BBQ food"],
    "modern yet comfortable atmosphere": ["modern atmosphere", "comfortable atmosphere"],
    "modern yet historical blend": ["modern blend", "historical blend"],
    "modern yet warm ambiance": ["modern ambiance", "warm ambiance"],
    "relaxed yet refined atmosphere": ["relaxed atmosphere", "refined atmosphere"],
    "rustic yet charming environment": ["rustic environment", "charming environment"],
    "simple yet delicious menu": ["simple menu", "delicious menu"],
    "simple yet flavorful pizza": ["simple pizza", "flavorful pizza"],
    "simple yet nice ambiance": ["simple ambiance", "nice ambiance"],
    "simple yet special dishes": ["simple dishes", "special dishes"],
    "simple yet well-executed dishes": ["simple dishes", "well-executed dishes"],
    "sophisticated yet cozy ambiance": ["sophisticated ambiance", "cozy ambiance"],
    "trendy yet classy": ["trendy", "classy"],
    "trendy yet comfortable dining experience": ["trendy dining experience", "comfortable dining experience"],
    "trendy yet laid-back atmosphere": ["trendy atmosphere", "laid-back atmosphere"],
    "people who enjoy high-end yet affordable American cuisine": ["people who enjoy high-end American cuisine", "people who enjoy affordable American cuisine"],
    "people who appreciate simple yet flavorful pizza": ["people who appreciate simple pizza", "people who appreciate flavorful pizza"],
    "people who enjoy a comfortable yet aesthetically-pleasing dining experience": ["people who enjoy a comfortable dining experience", "people who appreciate aesthetically-pleasing dining experience"],
    "people who enjoy casual yet slightly more upscale dining experiences": ["people who enjoy casual dining experiences", "people who enjoy slightly more upscale dining experiences"],
    "people who appreciate a relaxed yet refined atmosphere": ["people who appreciate relaxed atmosphere", "people who appreciate refined atmosphere"],
}

# Relation replacement rules
# Format: {old_relation: new_relation}
RELATION_REPLACEMENTS = {
    # Example: "has ambience": "has atmosphere",
    "has price range": "price range",
}

# Replacement rules for (relation, tail) combinations
# Format: [(relation_pattern, tail_pattern, new_relation, new_tail), ...]
# Use None to keep the original value
RELATION_TAIL_REPLACEMENTS = [
    # Example: ("serves", "Italian cuisine", "serves", "Italian")
    # Example: ("has feature", "big portions", "has characteristic", "big portions")
    # Add your rules here:
    # ("old_relation", "old_tail", "new_relation", "new_tail"),
    ("serves", "Italian cuisine", "serves", "Italian"),
    ("serves", "Italian food", "serves", "Italian"),
    ("serves", "Polish cuisine", "serves", "Polish"),
    ("serves", "Polish food", "serves", "Polish"),
    ("appeals to", "Fans of Polish food", "appeals to", "Fans of Polish"),
    ("appeals to", "Fans of Polish cuisine", "appeals to", "Fans of Polish"),
    ("appeals to", "Fans of Italian food", "appeals to", "Fans of Italian"),
    ("appeals to", "Fans of Italian cuisine", "appeals to", "Fans of Italian"),
    ("serves", "Cajun food", "serves", "Cajun cuisine"),
    ("serves", "Creole food", "serves", "Creole cuisine"),
    ("serves", "French cuisine", "serves", "French"),
    ("serves", "French food", "serves", "French"),
    ("serves", "Japanese food", "serves", "Japanese cuisine"),
    ("serves", "Japanese dishes", "serves", "Japanese cuisine"),
    ("serves", "seafood dishes", "serves", "seafood"),
    ("has atmosphere", "cozy atmosphere", "has atmosphere", "cozy"),
    ("has feature", "cozy atmosphere", "has atmosphere", "cozy"),
    ("serves", "American", "serves", "American cuisine"),
    ("serves", "American food", "serves", "American cuisine"),
    ("serves", "Mexican cuisine", "serves", "Mexican"),
    ("serves", "Mexican food", "serves", "Mexican"),
    ("serves", "Thai dishes", "serves", "Thai cuisine"),
    ("serves", "Thai food", "serves", "Thai cioisine"),
    ("cators to", "food allergies", "accommodates", "food allergies"),
    ("serves", "Asian fusion cuisine", "serves", "Asian fusion"),
    ("offers", "Asian fusion cuisine", "serves", "Asian fusion"),
    ("price range", "reasonable prices", "price range", "reasonable"),
    ("price range", "reasonable price", "price range", "reasonable"),
    ("has feature", "reasonable prices", "price range", "reasonable"),
    ("price range", "affordable prices", "price range", "affordable"),
    ("price range", "affordable price", "price range", "affordable"),
    ("has feature", "affordable prices", "price range", "affordable"),
    ("has feature", "affordable price", "price range", "affordable"),
    ("serves", "Greek food", "serves", "Greek cuisine"),
    ("serves", "restaurant", "is a", "restaurant"),
    ("category", "restaurant", "is a", "restaurant"),
    ("category", "cafe", "is a", "cafe"),
    ("category", "bar", "is a", "bar"),
    ("category", "pub", "is a", "pub"),
    ("has feature", "family-friendly atmosphere", "has atmosphere", "family-friendly"),
    (
        "has atmosphere",
        "family-friendly atmosphere",
        "has atmosphere",
        "family-friendly",
    ),
    ("category", "brewery", "is a", "brewery"),
    ("category", "beer", "serves", "beer"),
    ("category", "pizza", "serves", "pizza"),
    ("offers", "comfort food", "serves", "comfort food"),
    ("has feature", "generous portions", "offers", "generous portions"),
    ("serves", "generous portions", "offers", "generous portions"),
    ("offers", "desserts", "serves", "desserts"),
    ("offers", "affordable prices", "price range", "affordable"),
    ("price range", "lower prices", "price range", "lower"),
    ("has feature", "lower prices", "price range", "lower"),
    ("offers", "beer", "serves", "beers"),
    ("serves", "beer", "serves", "beers"),
    ("offers", "beers", "serves", "beers"),
    ("offers", "Filipino dishes", "serves", "Filipino cuisine"),
    ("offers", "Filipino cuisine", "serves", "Filipino cuisine"),
    ("serves", "Filipino dishes", "serves", "Filipino cuisine"),
    ("serves", "Filipino food", "serves", "Filipino cuisine"),
    ("offers", "Filipino food", "serves", "Filipino cuisine"),
    ("offers", "Chinese food", "serves", "Chinese cuisine"),
    ("serves", "Chinese food", "serves", "Chinese cuisine"),
    ("offers", "Chinese cuisine", "serves", "Chinese cuisine"),
    ("serves", "Chinese", "serves", "Chinese cuisine"),
    ("serves", "Japanese", "serves", "Japanese cuisine"),
    ("offers", "creole cuisine", "serves", "creole cuisine"),
    ("offers", "cajun cuisine", "serves", "cajun cuisine"),
    ("features", "live music", "has feature", "live music"),
    ("category", "bakery", "is a", "bakery"),
    ("has atmosphere", "vibrant atmosphere", "has atmosphere", "vibrant"),
    ("has staff", "friendly staff", "has feature", "friendly staff"),
    ("has feature", "fun atmosphere", "has atmosphere", "fun atmosphere"),
    ("serves", "British food", "serves", "British cuisine"),
    ("has feature", "friendly atmosphere", "has atmosphere", "friendly atmosphere"),
    ("serves", "hight-quality food", "has feature", "high-quality food"),
    ("has atmosphere", "cozy setting", "has feature", "cozy setting"),
    ("offers", "friendly atmosphere", "has atmosphere", "friendly atmosphere"),
    ("category", "breakfast", "serves", "breakfast"),
    ("category", "brunch", "serves", "brunch"),
    ("has feature", "romantic atmosphere", "has atmosphere", "romantic atmosphere"),
    ("uses", "seasonal ingredients", "has feature", "seasonal ingredients"),
    ("uses", "local ingredients", "has feature", "local ingredients"),
    ("has atmosphere", "clean dining spaces", "has feature", "clean dining spaces"),
    ("category", "seafood", "serves", "seafood"),
    ("category", "spa", "is a", "spa"),
    ("has atmosphere", "roomy dining spaces", "has feature", "roomy dining spaces"),
]

# Tail-based relation unification rules
# When the same tail appears with multiple relations in one iid,
# if the specified (relation, tail) combination exists, keep only that one.
# Format: {tail: preferred_relation}
# Example: {"big portions": "serves"} means:
#   If tail "big portions" appears with multiple relations (e.g., "has feature" and "serves"),
#   keep only the triplet with relation "serves" and remove others.
TAIL_RELATION_UNIFICATION_RULES = {
    # Add your tail-based unification rules here:
    "coffee": "serves",
    "cakes": "serves",
    "soups": "serves",
    "cozy": "has atmosphere",
    "restaurant": "is a",
    "cafe": "is a",
}


class KGCleaner:
    """Knowledge Graph cleaner for normalizing and deduplicating triplets."""

    def __init__(self):
        self.stats = {
            "total_triplets": 0,
            "normalized_count": 0,
            "duplicate_count": 0,
            "conflict_count": 0,
            "filtered_count": 0,
            "replaced_count": 0,
            "filtered_relations_count": 0,
            "entity_replaced_count": 0,
            "relation_replaced_count": 0,
            "expanded_count": 0,
            "added_count": 0,
            "unified_count": 0,
            "accent_normalized_count": 0,
            "people_normalized_count": 0,
        }

    def _should_normalize(self, value: str) -> bool:
        """Check if a value has case variations that should be normalized."""
        return value != value.lower() and value.lower() not in ["", "_"]

    def _collect_global_variations(
        self, input_data: List[Dict[str, Any]]
    ) -> Tuple[set, set]:
        """
        Collect global case variations across all data.
        Returns sets of relations and tails that should be normalized.
        """
        # Collect all unique relations and tails globally
        relations = defaultdict(set)
        tails = defaultdict(set)

        for item in input_data:
            for h, r, t in item["triplets"]:
                relations[r.lower()].add(r)
                tails[t.lower()].add(t)

        # Identify which should be normalized (have multiple case variations)
        relations_to_normalize = {
            orig
            for lower, variations in relations.items()
            if len(variations) > 1
            for orig in variations
        }
        tails_to_normalize = {
            orig
            for lower, variations in tails.items()
            if len(variations) > 1
            for orig in variations
        }

        logger.info(
            f"Found {len(relations_to_normalize)} relations and {len(tails_to_normalize)} tails to normalize globally"
        )

        return relations_to_normalize, tails_to_normalize

    def _normalize_relation_and_tail(
        self,
        triplets: List[List[str]],
        relations_to_normalize: set,
        tails_to_normalize: set,
    ) -> List[List[str]]:
        """
        Normalize relation and tail entities by converting to lowercase
        based on global case variations.
        """
        # Apply normalization
        normalized_triplets = []
        for h, r, t in triplets:
            normalized_r = r.lower() if r in relations_to_normalize else r
            normalized_t = t.lower() if t in tails_to_normalize else t

            if normalized_r != r or normalized_t != t:
                self.stats["normalized_count"] += 1

            normalized_triplets.append([h, normalized_r, normalized_t])

        return normalized_triplets

    def _expand_entities(self, triplets: List[List[str]]) -> List[List[str]]:
        """Expand tail entities based on expansion rules (one entity -> multiple entities)."""
        if not ENTITY_EXPANSION_RULES:
            return triplets

        expanded = []

        for h, r, t in triplets:
            # Check if tail matches any expansion rule
            if t in ENTITY_EXPANSION_RULES:
                # Create multiple triplets with expanded entities
                for new_t in ENTITY_EXPANSION_RULES[t]:
                    expanded.append([h, r, new_t])
                    self.stats["expanded_count"] += 1
                logger.debug(
                    f"Expanded: [{h}, {r}, {t}] → {len(ENTITY_EXPANSION_RULES[t])} triplets"
                )
            else:
                expanded.append([h, r, t])

        if self.stats["expanded_count"] > 0:
            logger.info(
                f"Expanded {self.stats['expanded_count']} tail entities into multiple triplets"
            )
        return expanded

    def _add_entities(self, triplets: List[List[str]]) -> List[List[str]]:
        """Add additional tail entities while keeping the original entity."""
        if not ENTITY_ADDITION_RULES:
            return triplets

        # Create a set of existing triplets to check for duplicates
        existing_triplets = {(h, r, t) for h, r, t in triplets}
        added = []

        for h, r, t in triplets:
            # Keep the original triplet
            added.append([h, r, t])
            
            # Check if tail matches any addition rule
            if t in ENTITY_ADDITION_RULES:
                # Create additional triplets with new entities while keeping the original
                added_count_for_this = 0
                for new_t in ENTITY_ADDITION_RULES[t]:
                    # Skip if the triplet already exists
                    if (h, r, new_t) not in existing_triplets:
                        added.append([h, r, new_t])
                        existing_triplets.add((h, r, new_t))  # Update set to avoid duplicates in same batch
                        self.stats["added_count"] += 1
                        added_count_for_this += 1
                
                if added_count_for_this > 0:
                    logger.debug(
                        f"Added entities: [{h}, {r}, {t}] → kept original + {added_count_for_this} additional triplets"
                    )

        if self.stats["added_count"] > 0:
            logger.info(
                f"Added {self.stats['added_count']} additional tail entities while keeping originals"
            )
        return added

    def _normalize_accents(self, text: str) -> str:
        """Remove accent marks from text (e.g., é → e, à → a)."""
        # Normalize to NFD (decomposed form) and remove combining characters
        nfd = unicodedata.normalize("NFD", text)
        return "".join(
            char for char in nfd if unicodedata.category(char) != "Mn"
        )

    def _normalize_tail_accents(self, triplets: List[List[str]]) -> List[List[str]]:
        """Normalize accent marks in tail entities (e.g., é → e)."""
        normalized = []

        for h, r, t in triplets:
            normalized_t = self._normalize_accents(t)
            if normalized_t != t:
                normalized.append([h, r, normalized_t])
                self.stats["accent_normalized_count"] += 1
                logger.debug(f"Normalized accents: [{h}, {r}, {t}] → [{h}, {r}, {normalized_t}]")
            else:
                normalized.append([h, r, t])

        if self.stats["accent_normalized_count"] > 0:
            logger.info(f"Normalized accents in {self.stats['accent_normalized_count']} tail entities")
        return normalized

    def _convert_ing_to_base_verb(self, verb_ing: str) -> str:
        """Convert -ing verb to base form (e.g., 'seeking' -> 'seek')."""
        # Common verbs and their -ing forms
        ing_to_base = {
            "seeking": "seek",
            "looking": "look",
            "enjoying": "enjoy",
            "appreciating": "appreciate",
            "wanting": "want",
            "needing": "need",
            "preferring": "prefer",
            "loving": "love",
            "liking": "like",
            "trying": "try",
            "hoping": "hope",
            "expecting": "expect",
            "finding": "find",
            "getting": "get",
            "taking": "take",
            "making": "make",
            "having": "have",
            "doing": "do",
            "going": "go",
            "coming": "come",
            "seeing": "see",
            "knowing": "know",
            "thinking": "think",
            "feeling": "feel",
            "being": "be",
            "missing": "miss",
        }
        
        # Check if we have a direct mapping
        verb_lower = verb_ing.lower()
        if verb_lower in ing_to_base:
            return ing_to_base[verb_lower]
        
        # General rule: remove -ing
        # For verbs ending in -ing, try to convert to base form
        if verb_ing.endswith("ing"):
            base = verb_ing[:-3]  # Remove "ing"
            # Handle common patterns:
            # - removing 'e' before -ing (e.g., "making" -> "make")
            if verb_ing.endswith("eing"):  # e.g., "being"
                base = verb_ing[:-4] + "e"
            return base
        
        return verb_ing

    def _normalize_people_terms(self, triplets: List[List[str]]) -> List[List[str]]:
        """Normalize tail entities: 'Users'/'People'/'users' to 'people', 'those ~ing' to 'people ~ing', 'those who ~' to 'people who ~'."""
        normalized = []
        # Terms to look for (case-sensitive as specified)
        people_terms = ["Users", "People", "users"]

        for h, r, t in triplets:
            normalized_t = t
            should_normalize = False
            
            # Check if tail contains any of the people terms -> convert to "people"
            if any(term in t for term in people_terms):
                normalized_t = "people"
                should_normalize = True
            # Check if tail starts with "those" or "Those"
            elif t.startswith("those ") or t.startswith("Those "):
                # Pattern 1: "those who X" -> "people who X"
                if re.match(r"^(those|Those) who ", t, re.IGNORECASE):
                    normalized_t = re.sub(r"^(those|Those) who ", "people who ", t, flags=re.IGNORECASE)
                    should_normalize = True
                # Pattern 2: "those ~ing X" -> "people ~ing X" (ing形をそのまま)
                elif re.match(r"^(those|Those) \w+ing", t, re.IGNORECASE):
                    # "those/Those" を "people" に置き換え（ing形はそのまま）
                    normalized_t = re.sub(r"^(those|Those) ", "people ", t, flags=re.IGNORECASE)
                    should_normalize = True
                # それ以外の "those" で始まるものは置き換えない
            
            if should_normalize:
                normalized.append([h, r, normalized_t])
                self.stats["people_normalized_count"] += 1
                logger.debug(f"Normalized people term: [{h}, {r}, {t}] → [{h}, {r}, {normalized_t}]")
            else:
                normalized.append([h, r, t])

        if self.stats["people_normalized_count"] > 0:
            logger.info(f"Normalized {self.stats['people_normalized_count']} tail entities: 'Users'/'People'/'users' to 'people', 'those ~ing' to 'people ~ing', 'those who ~' to 'people who ~'")
        return normalized

    def _replace_entities(self, triplets: List[List[str]]) -> List[List[str]]:
        """Replace tail entities based on entity replacement rules."""
        if not ENTITY_REPLACEMENTS:
            return triplets

        replaced = []

        for h, r, t in triplets:
            # Check if tail matches any entity replacement rule
            if t in ENTITY_REPLACEMENTS:
                new_t = ENTITY_REPLACEMENTS[t]
                replaced.append([h, r, new_t])
                self.stats["entity_replaced_count"] += 1
                logger.debug(f"Replaced entity: [{h}, {r}, {t}] → [{h}, {r}, {new_t}]")
            else:
                replaced.append([h, r, t])

        if self.stats["entity_replaced_count"] > 0:
            logger.info(f"Replaced {self.stats['entity_replaced_count']} tail entities")
        return replaced

    def _replace_relations(self, triplets: List[List[str]]) -> List[List[str]]:
        """Replace relations based on relation replacement rules."""
        if not RELATION_REPLACEMENTS:
            return triplets

        replaced = []

        for h, r, t in triplets:
            if r in RELATION_REPLACEMENTS:
                new_r = RELATION_REPLACEMENTS[r]
                replaced.append([h, new_r, t])
                self.stats["relation_replaced_count"] += 1
                logger.debug(
                    f"Replaced relation: [{h}, {r}, {t}] → [{h}, {new_r}, {t}]"
                )
            else:
                replaced.append([h, r, t])

        if self.stats["relation_replaced_count"] > 0:
            logger.info(f"Replaced {self.stats['relation_replaced_count']} relations")
        return replaced

    def _replace_relation_tail_combinations(
        self, triplets: List[List[str]]
    ) -> List[List[str]]:
        """Replace specific (relation, tail) combinations based on rules."""
        if not RELATION_TAIL_REPLACEMENTS:
            return triplets

        processed = []

        for h, r, t in triplets:
            replaced = False
            # Check each replacement rule
            for old_r, old_t, new_r, new_t in RELATION_TAIL_REPLACEMENTS:
                if r == old_r and t == old_t:
                    # Apply replacement (None means keep original)
                    new_r = new_r if new_r is not None else r
                    new_t = new_t if new_t is not None else t
                    self.stats["replaced_count"] += 1
                    logger.debug(f"Replaced: [{h}, {r}, {t}] → [{h}, {new_r}, {new_t}]")
                    processed.append([h, new_r, new_t])
                    replaced = True
                    break

            if not replaced:
                processed.append([h, r, t])

        if self.stats["replaced_count"] > 0:
            logger.info(
                f"Replaced {self.stats['replaced_count']} relation-tail combinations"
            )
        return processed

    def _filter_triplets(self, triplets: List[List[str]]) -> List[List[str]]:
        """Remove triplets with filtered terms in head, relation, or tail."""
        filtered = []

        for h, r, t in triplets:
            # Check if any field exactly matches filtered terms
            if h in FILTERED_TERMS or r in FILTERED_TERMS or t in FILTERED_TERMS:
                self.stats["filtered_count"] += 1
                logger.debug(f"Filtered triplet: [{h}, {r}, {t}]")
                continue

            filtered.append([h, r, t])

        logger.info(
            f"Removed {self.stats['filtered_count']} triplets with filtered terms"
        )
        return filtered

    def _filter_relations(self, triplets: List[List[str]]) -> List[List[str]]:
        """Remove triplets with specific relations."""
        if not FILTERED_RELATIONS:
            return triplets

        filtered = []

        for h, r, t in triplets:
            if r in FILTERED_RELATIONS:
                self.stats["filtered_relations_count"] += 1
                logger.debug(f"Filtered relation: [{h}, {r}, {t}]")
                continue

            filtered.append([h, r, t])

        if self.stats["filtered_relations_count"] > 0:
            logger.info(
                f"Removed {self.stats['filtered_relations_count']} triplets with filtered relations"
            )
        return filtered

    def _deduplicate_triplets(self, triplets: List[List[str]]) -> List[List[str]]:
        """Remove exact duplicate triplets (h, r, t)."""
        seen = set()
        deduplicated = []

        for h, r, t in triplets:
            triplet_tuple = (h, r, t)
            if triplet_tuple not in seen:
                seen.add(triplet_tuple)
                deduplicated.append([h, r, t])
            else:
                self.stats["duplicate_count"] += 1

        logger.info(f"Removed {self.stats['duplicate_count']} duplicate triplets")
        return deduplicated

    def _unify_tail_relations(self, triplets: List[List[str]]) -> List[List[str]]:
        """
        Unify relations for tails that appear with multiple relations.
        If a tail matches the unification rules and the preferred relation exists,
        remove all other relations for that tail.

        Example:
            If TAIL_RELATION_UNIFICATION_RULES = {"big portions": "serves"}
            and triplets contain:
                ["Restaurant A", "has feature", "big portions"]
                ["Restaurant A", "serves", "big portions"]
            Then remove "has feature" and keep only "serves".
        """
        if not TAIL_RELATION_UNIFICATION_RULES:
            return triplets

        # Group triplets by tail to find tails with multiple relations
        tail_to_triplets = defaultdict(list)
        for h, r, t in triplets:
            tail_to_triplets[t].append([h, r, t])

        unified = []

        for tail, tail_triplets in tail_to_triplets.items():
            # If only one triplet for this tail, keep as is
            if len(tail_triplets) == 1:
                unified.extend(tail_triplets)
                continue

            # Multiple triplets with this tail - check for unification rule
            if tail in TAIL_RELATION_UNIFICATION_RULES:
                preferred_relation = TAIL_RELATION_UNIFICATION_RULES[tail]

                # Check if the preferred relation exists
                relations = {triplet[1] for triplet in tail_triplets}
                if preferred_relation in relations:
                    # Keep only triplets with the preferred relation
                    for triplet in tail_triplets:
                        if triplet[1] == preferred_relation:
                            unified.append(triplet)
                        else:
                            self.stats["unified_count"] += 1
                            logger.debug(
                                f"Unified: [{triplet[0]}, {triplet[1]}, {triplet[2]}] -> prefer relation '{preferred_relation}'"
                            )
                else:
                    # Preferred relation doesn't exist, keep all
                    unified.extend(tail_triplets)
            else:
                # No unification rule for this tail, keep all
                unified.extend(tail_triplets)

        if self.stats["unified_count"] > 0:
            logger.info(
                f"Unified {self.stats['unified_count']} relations based on tail unification rules"
            )

        return unified

    def _find_conflicts(
        self, triplets: List[List[str]]
    ) -> Dict[Tuple[str, str], List[str]]:
        """
        Find (h, t) pairs that have multiple relations.
        Returns dict: (head, tail) -> [list of relations]
        """
        ht_to_relations = defaultdict(set)

        for h, r, t in triplets:
            ht_to_relations[(h, t)].add(r)

        # Filter only conflicts (multiple relations for same h, t pair)
        conflicts = {
            ht: sorted(relations)
            for ht, relations in ht_to_relations.items()
            if len(relations) > 1
        }

        # Don't update stats here - will be updated after all iids are processed
        if conflicts:
            logger.debug(
                f"Found {len(conflicts)} (h, t) pairs with multiple relations in this iid"
            )

        return conflicts

    def clean_kg_data(
        self,
        input_data: List[Dict[str, Any]],
        normalize: bool = True,
        deduplicate: bool = True,
        find_conflicts: bool = True,
        filter_terms: bool = True,
        filter_relations: bool = True,
        expand_entities: bool = True,
        add_entities: bool = True,
        replace_entities: bool = True,
        replace_relations: bool = True,
        replace_combinations: bool = True,
        unify_relations: bool = True,
        normalize_accents: bool = False,
        normalize_people: bool = False,
    ) -> Tuple[List[Dict[str, Any]], Dict[Tuple[str, str], List[str]]]:
        """
        Clean knowledge graph data.

        Args:
            input_data: List of dicts with 'iid' and 'triplets' keys
            normalize: Apply case normalization to relations and tails
            deduplicate: Remove exact duplicate triplets
            find_conflicts: Find (h, t) pairs with multiple relations
            filter_terms: Remove triplets containing filtered terms
            filter_relations: Remove triplets with specific relations
            expand_entities: Expand tail entities into multiple triplets (replaces original)
            add_entities: Add additional tail entities while keeping the original
            replace_entities: Replace tail entities regardless of relation
            replace_relations: Replace relations regardless of head or tail
            replace_combinations: Replace specific (relation, tail) combinations
            unify_relations: Unify relations for tails with multiple relations
            normalize_accents: Normalize accent marks in tail entities (é → e)
            normalize_people: Normalize tail entities containing 'Users', 'People', or 'users' to 'people'

        Returns:
            Tuple of (cleaned_data, conflicts_dict)
        """
        self.stats["total_triplets"] = sum(len(item["triplets"]) for item in input_data)
        logger.info(f"Processing {self.stats['total_triplets']} total triplets")

        # Step 0: Collect global case variations if normalize is enabled
        relations_to_normalize = set()
        tails_to_normalize = set()
        if normalize:
            relations_to_normalize, tails_to_normalize = (
                self._collect_global_variations(input_data)
            )

        cleaned_data = []
        all_conflicts = {}

        for item in input_data:
            iid = item["iid"]
            triplets = item["triplets"]

            # Step 0: Filter terms (optional)
            if filter_terms:
                triplets = self._filter_triplets(triplets)

            # Step 1: Filter relations (optional)
            if filter_relations:
                triplets = self._filter_relations(triplets)

            # Step 2: Expand entities (optional)
            if expand_entities:
                triplets = self._expand_entities(triplets)

            # Step 2.5: Add entities (optional)
            if add_entities:
                triplets = self._add_entities(triplets)

            # Step 3: Replace entities (optional)
            if replace_entities:
                triplets = self._replace_entities(triplets)

            # Step 3.5: Normalize accents in tail entities (optional)
            if normalize_accents:
                triplets = self._normalize_tail_accents(triplets)

            # Step 3.6: Normalize people terms in tail entities (optional)
            if normalize_people:
                triplets = self._normalize_people_terms(triplets)

            # Step 4: Replace relations (optional)
            if replace_relations:
                triplets = self._replace_relations(triplets)

            # Step 5: Replace combinations (optional)
            if replace_combinations:
                triplets = self._replace_relation_tail_combinations(triplets)

            # Step 6: Normalize relations and tails (optional)
            if normalize:
                triplets = self._normalize_relation_and_tail(
                    triplets, relations_to_normalize, tails_to_normalize
                )

            # Step 7: Deduplicate (optional)
            if deduplicate:
                triplets = self._deduplicate_triplets(triplets)

            # Step 8: Unify relations (optional)
            if unify_relations:
                triplets = self._unify_tail_relations(triplets)

            # Step 9: Find conflicts (optional)
            if find_conflicts:
                conflicts = self._find_conflicts(triplets)
                if conflicts:
                    all_conflicts.update(conflicts)

            cleaned_data.append({"iid": iid, "triplets": triplets})

        # Update conflict count after all iids are processed
        if find_conflicts:
            self.stats["conflict_count"] = len(all_conflicts)
            logger.info(
                f"Found {len(all_conflicts)} total (h, t) pairs with multiple relations"
            )

        logger.info(f"Cleaning statistics: {self.stats}")
        return cleaned_data, all_conflicts

    def clean_file(
        self,
        input_path: Path,
        output_path: Path,
        conflicts_path: Path,
        normalize: bool = True,
        deduplicate: bool = True,
        find_conflicts: bool = True,
        filter_terms: bool = True,
        filter_relations: bool = True,
        expand_entities: bool = True,
        add_entities: bool = True,
        replace_entities: bool = True,
        replace_relations: bool = True,
        replace_combinations: bool = True,
        unify_relations: bool = True,
        normalize_accents: bool = False,
        normalize_people: bool = False,
    ):
        """
        Clean a KG file and save results.

        Args:
            input_path: Path to input JSON file
            output_path: Path to save cleaned data
            conflicts_path: Path to save conflicts
            normalize: Apply case normalization to relations and tails
            deduplicate: Remove exact duplicate triplets
            find_conflicts: Find (h, t) pairs with multiple relations
            filter_terms: Remove triplets containing filtered terms
            filter_relations: Remove triplets with specific relations
            expand_entities: Expand tail entities into multiple triplets (replaces original)
            add_entities: Add additional tail entities while keeping the original
            replace_entities: Replace tail entities regardless of relation
            replace_relations: Replace relations regardless of head or tail
            replace_combinations: Replace specific (relation, tail) combinations
            unify_relations: Unify relations for tails with multiple relations
            normalize_accents: Normalize accent marks in tail entities (é → e)
            normalize_people: Normalize tail entities containing 'Users', 'People', or 'users' to 'people'
        """
        logger.info(f"Loading data from {input_path}")
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        cleaned_data, conflicts = self.clean_kg_data(
            data,
            normalize=normalize,
            deduplicate=deduplicate,
            find_conflicts=find_conflicts,
            filter_terms=filter_terms,
            filter_relations=filter_relations,
            expand_entities=expand_entities,
            add_entities=add_entities,
            replace_entities=replace_entities,
            replace_relations=replace_relations,
            replace_combinations=replace_combinations,
            unify_relations=unify_relations,
            normalize_accents=normalize_accents,
            normalize_people=normalize_people,
        )

        # Save cleaned data
        logger.info(f"Saving cleaned data to {output_path}")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(cleaned_data, f, indent=2, ensure_ascii=False)

        # Save conflicts in the same format as input canon_kg.json
        if find_conflicts and conflicts:
            # Use head entity as iid
            conflicts_data = []
            for (h, t), relations in sorted(conflicts.items()):
                triplets = [[h, r, t] for r in relations]
                conflicts_data.append(
                    {
                        "iid": h,
                        "triplets": triplets,
                    }
                )

            logger.info(
                f"Saving {len(conflicts_data)} conflict groups to {conflicts_path}"
            )
            with open(conflicts_path, "w", encoding="utf-8") as f:
                json.dump(conflicts_data, f, indent=2, ensure_ascii=False)
        else:
            logger.info("Conflict detection disabled or no conflicts found")

        logger.info("Cleaning complete!")
        logger.info(f"  Total triplets: {self.stats['total_triplets']}")
        if filter_terms:
            logger.info(f"  Filtered: {self.stats['filtered_count']}")
        if filter_relations:
            logger.info(
                f"  Filtered relations: {self.stats['filtered_relations_count']}"
            )
        if expand_entities:
            logger.info(f"  Entities expanded: {self.stats['expanded_count']}")
        if add_entities:
            logger.info(f"  Entities added: {self.stats['added_count']}")
        if replace_entities:
            logger.info(f"  Entities replaced: {self.stats['entity_replaced_count']}")
        if replace_relations:
            logger.info(
                f"  Relations replaced: {self.stats['relation_replaced_count']}"
            )
        if replace_combinations:
            logger.info(f"  Combinations replaced: {self.stats['replaced_count']}")
        if normalize:
            logger.info(f"  Normalized: {self.stats['normalized_count']}")
        if deduplicate:
            logger.info(f"  Duplicates removed: {self.stats['duplicate_count']}")
        if unify_relations:
            logger.info(f"  Relations unified: {self.stats['unified_count']}")
        if normalize_accents:
            logger.info(f"  Accents normalized: {self.stats['accent_normalized_count']}")
        if normalize_people:
            logger.info(f"  People terms normalized: {self.stats['people_normalized_count']}")
        if find_conflicts:
            logger.info(f"  Conflicts found: {self.stats['conflict_count']}")
