import streamlit as st
import pandas as pd
import dill

# 💡 Streamlit config must be FIRST
st.set_page_config(layout="wide")

# Define model path
MODEL_PATH = "adash2.dill"

# 🔁 Load model only once
@st.cache_resource
def load_model():
    """Load the pre-trained food prediction model from dill file"""
    try:
        with open(MODEL_PATH, 'rb') as f:
            return dill.load(f)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

predictor = load_model()

# Page Title
st.title("🍽️ Food Quantity & Price Estimator")

# Sidebar Inputs
with st.sidebar:
    st.header("🛠️ Event Details")
    event_time = st.selectbox("Event Time", ["Morning", "Afternoon", "Evening", "Night"])
    meal_type = st.selectbox("Meal Type", ["Breakfast", "Lunch", "Hi-Tea", "Dinner"])
    event_type = st.selectbox("Event Type", ["Wedding", "Birthday Party", "Corporate Event", "Casual Gathering"])
    guest_count = st.number_input("Number of Guests", min_value=1, value=50)

    st.markdown("---")
    st.header("🍱 Menu Items")
    items_text = st.text_area("Enter menu items (one per line)", height=200)
    menu_items = [item.strip() for item in items_text.strip().split("\n") if item.strip()]

# Main Output
if predictor is None:
    st.error("Failed to load the prediction model. Please check that the model file exists.")
elif menu_items:
    with st.spinner("Calculating quantities and pricing..."):
        predictions = predictor.predict(event_time, meal_type, event_type, guest_count, menu_items)

        results = []
        total_weight = 0.0
        total_volume = 0.0
        total_pieces = 0.0
        total_price_sum = 0.0
        total_per_person_price_sum = 0.0

        for item, qty_data in predictions.items():
            total_qty = qty_data['total']
            per_person_qty_str = qty_data['per_person']
            per_person_qty_val = predictor.extract_quantity_value(per_person_qty_str)
            unit = predictor.extract_unit(per_person_qty_str)

            # Fix for .category error
            std_item = predictor.standardize_item_name(item)
            mapped_item = getattr(predictor, 'item_name_mapping', {}).get(std_item, item)
            item_meta = predictor.item_metadata.get(mapped_item)
            category = item_meta.category if item_meta and hasattr(item_meta, "category") else predictor.guess_item_category(item)

            total_qty_val = predictor.extract_quantity_value(total_qty)
            try:
                total_price, base_price_per_unit, per_person_price = predictor.calculate_price(
                    total_qty_val, category, guest_count, item, unit=qty_data['converted_unit']
                )
            except Exception as e:
                st.warning(f"Error calculating price for {item}: {e}")
                total_price = 0.0
                base_price_per_unit = 0.0
                per_person_price = 0.0
            # Totals accumulation
            if unit == "pcs":
                total_pieces += per_person_qty_val
            elif unit in ["ml", "l"]:
                total_volume += per_person_qty_val if unit == "ml" else per_person_qty_val * 1000
            else:
                total_weight += per_person_qty_val if unit == "g" else per_person_qty_val * 1000

            total_price_sum += total_price
            total_per_person_price_sum += per_person_price
            current_unit = unit 
            results.append({
                "Item": item,
                "Category": category,
                "Per Person Weight": per_person_qty_str,
                "Quantity": total_qty,
                "Base Price": f"₹{base_price_per_unit:.2f}", 
                "Per Person Price (₹)": f"₹{per_person_price:.2f}",
                "Total Price (₹)": f"₹{total_price:.2f}",
            })

        # Add Totals Row
        results.append({
            "Item": "**Total**",
            "Category":"",
            "Quantity": "",
            "Per Person Weight": f"{total_weight:.2f}g | {total_volume:.2f}ml | {total_pieces:.2f}pcs",
            "Base Price": "",
            "Per Person Price (₹)": f"₹{total_per_person_price_sum:.2f}",
            "Total Price (₹)": f"₹{total_price_sum:.2f}"
        })

        df = pd.DataFrame(results)

        st.markdown("### 📊 Estimated Quantities and Prices")
        st.dataframe(df, use_container_width=True, hide_index=True)
else:
    st.info("Please enter at least one menu item to continue.")
