import streamlit as st
import pandas as pd
import dill
import os
import time

# Set page configuration
st.set_page_config(
    page_title="Food Quantity Prediction System",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define paths
MODEL_DIR = "models"
MODEL_PATH = f"{MODEL_DIR}/adash2.dill"

# Create model directory if it doesn't exist
os.makedirs(MODEL_DIR, exist_ok=True)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem !important;
        color: #FF6B6B;
    }
    .subheader {
        font-size: 1.5rem !important;
        color: #4ECDC4;
    }
    .card {
        background-color: #f9f9f9;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .footer {
        margin-top: 50px;
        text-align: center;
        color: #999;
    }
</style>
""", unsafe_allow_html=True)


# Function to load the model using dill
@st.cache_resource
def load_model():
    """Load the food prediction model from dill file"""
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        st.warning("⚠️ Model file not found locally.")
        st.error("Please make sure you have saved your model to the models directory.")
        return None

    try:
        with st.spinner("Loading prediction model... This may take a moment."):
            with open(MODEL_PATH, 'rb') as f:
                predictor = dill.load(f)
        return predictor
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


# Main function to create the app UI
def main():
    # Display header
    st.markdown('<h1 class="main-header">🍽️ Food Quantity Prediction System</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
    <p>This system helps you estimate the right amount of food to prepare for your event and provides cost estimates for planning.</p>
    <p>Enter your event details and menu items to get predictions for quantities and costs.</p>
    </div>
    """, unsafe_allow_html=True)

    # Load the model
    predictor = load_model()

    if predictor is None:
        st.error("Could not load the prediction model. Please check the error messages above.")
        st.stop()

    # Create two columns for the form
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<h3 class="subheader">Event Details</h3>', unsafe_allow_html=True)
        event_time = st.selectbox(
            "Event Time",
            options=["Morning", "Afternoon", "Evening", "Night"]
        )

        meal_type = st.selectbox(
            "Meal Type",
            options=["Breakfast", "Lunch", "Dinner", "Hi-Tea"]
        )

        event_type = st.selectbox(
            "Event Type",
            options=["Wedding", "Birthday Party", "Corporate Event", "Family Gathering"]
        )

        guest_count = st.number_input(
            "Number of Guests",
            min_value=1,
            max_value=1000,
            value=30
        )

    with col2:
        st.markdown('<h3 class="subheader">Menu Items</h3>', unsafe_allow_html=True)
        st.write("Enter each menu item on a new line")

        # Example menu items for reference
        example_items = """Chicken Biryani
Veg Manchurian
Paneer Butter Masala
Gulab Jamun
Raitha
Steamed Rice"""

        # Make the text area taller
        menu_items_text = st.text_area(
            "Menu Items",
            height=250,
            placeholder=example_items,
            help="Enter one item per line. Example: Veg Biryani"
        )

        # Process menu items
        menu_items = [item.strip() for item in menu_items_text.splitlines() if item.strip()]

        # Show number of items entered
        st.write(f"**{len(menu_items)}** items entered")

    # Prediction button with loading animation
    predict_button = st.button("Calculate Quantities and Prices", disabled=len(menu_items) == 0)

    if predict_button:
        if len(menu_items) == 0:
            st.warning("Please enter at least one menu item.")
        else:
            with st.spinner("Calculating quantities and prices... This may take a moment."):
                # Add slight delay for better UX
                time.sleep(0.5)

                try:
                    # Make predictions
                    predictions = predictor.predict(event_time, meal_type, event_type, guest_count, menu_items)

                    # Process results
                    results = []
                    total_event_cost = 0

                    for item, qty_str in predictions.items():
                        total_qty_val = predictor.extract_quantity_value(qty_str)
                        unit = predictor.extract_unit(qty_str)

                        # Get item metadata
                        std_item = predictor.standardize_item_name(item)
                        mapped_item = getattr(predictor, 'Item_name_mapping', {}).get(std_item, item)

                        if mapped_item in predictor.item_metadata:
                            category = predictor.item_metadata[mapped_item].category
                        else:
                            category = predictor.guess_item_category(item)

                        # Calculate price
                        try:
                            total_price, base_price_per_unit, price_per_person = predictor.calculate_price(
                                total_qty_val, category, guest_count, item, unit=unit
                            )

                            # Make sure we have numeric values
                            if hasattr(total_price, "item"):
                                total_price = total_price.item()
                            if hasattr(base_price_per_unit, "item"):
                                base_price_per_unit = base_price_per_unit.item()
                            if hasattr(price_per_person, "item"):
                                price_per_person = price_per_person.item()

                            # Calculate per-person quantity
                            per_person_qty = total_qty_val / guest_count
                            per_person_qty_str = f"{per_person_qty:.2f}{unit}"

                            # Add to event cost
                            total_event_cost += total_price

                            results.append({
                                'Item': item,
                                'Category': category,
                                'Per Person Quantity': per_person_qty_str,
                                'Total Quantity': qty_str,
                                f'Price per {unit}': f"₹{base_price_per_unit:.2f}",
                                'Price per Person': f"₹{price_per_person:.2f}",
                                'Total Price': f"₹{total_price:.2f}"
                            })

                        except Exception as e:
                            st.warning(f"Error calculating price for {item}: {e}")
                            results.append({
                                'Item': item,
                                'Category': category,
                                'Per Person Quantity': "N/A",
                                'Total Quantity': qty_str,
                                f'Price per {unit}': "N/A",
                                'Price per Person': "N/A",
                                'Total Price': "N/A"
                            })

                    # Create DataFrame for display
                    results_df = pd.DataFrame(results)

                    # Display fancy header
                    st.markdown('<h3 class="subheader">🍲 Quantity and Price Predictions</h3>', unsafe_allow_html=True)

                    # Display as a table
                    st.dataframe(results_df, use_container_width=True)

                    # Summary metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Items", f"{len(menu_items)}")
                    with col2:
                        st.metric("Total Event Cost", f"₹{total_event_cost:.2f}")
                    with col3:
                        st.metric("Cost per Guest", f"₹{(total_event_cost / guest_count):.2f}")

                except Exception as e:
                    st.error(f"An error occurred during prediction: {e}")
                    st.exception(e)

    # Footer
    st.markdown("""
    <div class="footer">
        <p>Food Quantity Prediction System • Developed by Your Name/Company</p>
        <p>© 2025 All Rights Reserved</p>
    </div>
    """, unsafe_allow_html=True)


# Run the app
if __name__ == "__main__":
    main()