import json
import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import streamlit as st
import re

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
stopwords = {"a", "the", "with", "and", "of", "on", "in", "is", "are", "to", "hot"}

USDA_API_KEY = "nLVGxo3dSmBXw7D4ctcrKwUC6nNgqJSg3D1PWQF7"
USDA_API_URL = "https://api.nal.usda.gov/fdc/v1/foods/search"

def fetch_nutritional_info(food_item):
    params = {
        "api_key": USDA_API_KEY,
        "query": food_item,
        "pageSize": 1,
    }
    response = requests.get(USDA_API_URL, params=params)
    if response.status_code == 200:
        data = response.json()
        if data.get("foods"):
            nutrients = data["foods"][0]["foodNutrients"]
            # Filter for key nutrients
            key_nutrients = {
                "Energy": "Calories",
                "Protein": "Protein",
                "Total lipid (fat)": "Fat",
                "Carbohydrate, by difference": "Carbohydrates",
                "Sugars, total including NLEA": "Sugar",
                "Fiber, total dietary": "Fiber",
            }
            result = {
                key_nutrients[nutrient["nutrientName"]]: nutrient["value"]
                for nutrient in nutrients
                if nutrient["nutrientName"] in key_nutrients
            }
            return result
    return None

def recognize_food_and_nutrition(img):
    img_input = Image.open(img)

    inputs = processor(images=img_input, return_tensors="pt")
    output = model.generate(**inputs)
    caption = processor.decode(output[0], skip_special_tokens=True)

    clean_caption = re.sub(r"[^\w\s]", "", caption.lower())
    recognized_food = [word for word in clean_caption.split() if word not in stopwords]

    results = {}
    for food_item in recognized_food:
        nutrient_info = fetch_nutritional_info(food_item)
        if nutrient_info:
            results[food_item] = nutrient_info

    result = f"Caption: {caption}\n\n"
    if results:
        result += "Recognized food items with key nutrients:\n"
        for food, nutrients in results.items():
            result += f"\n{food.capitalize()}:\n"
            for nutrient, value in nutrients.items():
                result += f"  - {nutrient}: {value}\n"
    else:
        result += "No recognizable food items found or no nutrient data available."

    return result

# Streamlit App
st.title("Food Recognition and Nutritional Analysis")
st.write("Upload an image of food, and the app will provide its nutritional profile.")

uploaded_image = st.file_uploader("Upload a food image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
    st.write("Processing...")

    with st.spinner("Analyzing the image..."):
        result = recognize_food_and_nutrition(uploaded_image)

    st.success("Done!")
    st.text(result)
