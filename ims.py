import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import streamlit as st
import datetime
import os
import matplotlib.pyplot as plt

class DemandPredictor:
    def __init__(self, inventory_data):
        self.inventory_data = inventory_data
        self.model = None
        self.scaler = StandardScaler()

    def prepare_data(self):
        # Feature engineering
        df = self.inventory_data.copy()
        df['days_since_added'] = (pd.Timestamp.now() - df['added_date']).dt.days
        
        # Select relevant features
        features = ['quantity', 'price', 'days_since_added']
        target = 'quantity'  # Predicting future quantity
        
        X = df[features]
        y = df[target]
        
        return train_test_split(X, y, test_size=0.2, random_state=42)

    def train_model(self):
        X_train, X_test, y_train, y_test = self.prepare_data()
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Random Forest Regressor
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X_train_scaled, y_train)
        
        # Calculate and display model performance
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)
        
        st.write(f"Model Training Score: {train_score:.2f}")
        st.write(f"Model Testing Score: {test_score:.2f}")

    def predict_demand(self, item_features):
        if self.model is None:
            st.warning("Model not trained. Please train the model first.")
            return None
        
        # Scale input features
        scaled_features = self.scaler.transform([item_features])
        
        # Predict demand
        predicted_demand = self.model.predict(scaled_features)[0]
        
        return max(0, round(predicted_demand, 2))

# Path to the CSV file for saving/loading inventory data
INVENTORY_FILE = "inventory_data.csv"
CRITICAL_STOCK_THRESHOLD = 10  # Set critical stock threshold

class Inventory:
    def __init__(self):
        self.inventory_df = self.load_inventory()

    def load_inventory(self):
        """Load inventory data from CSV file if it exists."""
        if os.path.exists(INVENTORY_FILE):
            inventory_df = pd.read_csv(INVENTORY_FILE)
            inventory_df['expiration_date'] = pd.to_datetime(inventory_df['expiration_date'], errors='coerce')
            inventory_df['added_date'] = pd.to_datetime(inventory_df['added_date'], errors='coerce')
            return inventory_df
        return pd.DataFrame(columns=["name", "quantity", "price", "expiration_date", "added_date", "total_value", "category"])

    def save_inventory(self):
        """Save inventory data to CSV file."""
        self.inventory_df.to_csv(INVENTORY_FILE, index=False)
        st.success("Inventory saved successfully!")

    def add_item(self, name, quantity, price, expiration_date, category):
        """Add or update an item in the inventory."""
        # Ensure the expiration date has time set to 00:00:00
        if isinstance(expiration_date, datetime.date):
            expiration_date = datetime.datetime.combine(expiration_date, datetime.time(0,0))  # type: ignore

        total_value = quantity * price
        if name in self.inventory_df["name"].values:
            self.inventory_df.loc[self.inventory_df["name"] == name, "quantity"] += quantity
            self.inventory_df.loc[self.inventory_df["name"] == name, "price"] = price
            self.inventory_df.loc[self.inventory_df["name"] == name, "total_value"] = self.inventory_df.loc[self.inventory_df["name"] == name, "quantity"] * price
            self.inventory_df.loc[self.inventory_df["name"] == name, "category"] = category
        else:
            new_item = pd.DataFrame({
                "name": [name],
                "quantity": [quantity],
                "price": [price],
                "expiration_date": [expiration_date],
                "added_date": [datetime.datetime.now()],
                "total_value": [total_value],
                "category": [category]
            })
            self.inventory_df = pd.concat([self.inventory_df, new_item], ignore_index=True)
        self.save_inventory()

    def remove_item(self, name):
        """Remove an item from the inventory."""
        self.inventory_df = self.inventory_df[self.inventory_df["name"] != name]
        self.save_inventory()

    def reduce_quantity(self, name, quantity_to_reduce):
        """Reduce the quantity of an item in the inventory."""
        if name in self.inventory_df["name"].values:
            current_quantity = self.inventory_df.loc[self.inventory_df["name"] == name, "quantity"].values[0]  # type: ignore
            new_quantity = max(0, current_quantity - quantity_to_reduce)
            self.inventory_df.loc[self.inventory_df["name"] == name, "quantity"] = new_quantity
            self.inventory_df.loc[self.inventory_df["name"] == name, "total_value"] = new_quantity * self.inventory_df.loc[self.inventory_df["name"] == name, "price"]
        self.save_inventory()

    def get_inventory(self):
        """Return the current inventory DataFrame."""
        return self.inventory_df

    def check_critical_stock(self):
        """Check if any item has a quantity below the critical threshold."""
        critical_items = self.inventory_df[self.inventory_df["quantity"] < CRITICAL_STOCK_THRESHOLD]
        return critical_items

    def calculate_days_to_expire(self, expiration_date):
        """Calculate the days remaining until the expiration date."""
        if pd.isna(expiration_date):
            return None
        return (expiration_date - datetime.datetime.now()).days

    def sell_item(self, name, quantity_to_sell):
        """Sell a certain quantity of an item from the inventory."""
        if name in self.inventory_df["name"].values:
            current_quantity = self.inventory_df.loc[self.inventory_df["name"] == name, "quantity"].values[0]  # type: ignore
            new_quantity = max(0, current_quantity - quantity_to_sell)
            self.inventory_df.loc[self.inventory_df["name"] == name, "quantity"] = new_quantity
            self.inventory_df.loc[self.inventory_df["name"] == name, "total_value"] = new_quantity * self.inventory_df.loc[self.inventory_df["name"] == name, "price"]
        self.save_inventory()

class User:
    def __init__(self, username, password):
        self.username = username
        self.password = password

    def authenticate(self, entered_username, entered_password):
        """Authenticate user credentials."""
        return self.username == entered_username and self.password == entered_password

# Custom CSS for enhanced styling
def set_custom_style():
    st.markdown("""
    <style>
    .stApp { background-color: #2f2f2f; color: #ffffff; }
    </style>
    """, unsafe_allow_html=True)

set_custom_style()

# Initialize the inventory system
inventory = Inventory()

# Login credentials
username = "admin"
password = "admin123"

# Streamlit app layout
st.title("Inventory Management System (IMS)")

# User login screen
st.sidebar.title("Login")
entered_username = st.sidebar.text_input("Username")
entered_password = st.sidebar.text_input("Password", type="password")

# Authenticate user
user = User(username, password)

if user.authenticate(entered_username, entered_password):
    st.sidebar.success("Logged in successfully!")
    
    # Display inventory management options
    menu = st.sidebar.radio("Menu", ["Home", "Add Item", "Remove Item", "Sell Item", "Check Inventory", "Visualizations"])

    if menu == "Home":
        st.header("Welcome to the Inventory Management System")
        st.write("Manage your inventory efficiently.")
    
    elif menu == "Add Item":
        st.header("Add Item")
        name = st.text_input("Item Name")
        quantity = st.number_input("Quantity", min_value=0)
        price = st.number_input("Price", min_value=0.0, format="%.2f")
        expiration_date = st.date_input("Expiration Date")
        category = st.selectbox("Category", ['furniture', 'food', 'clothing', 'stationary', 'other'])
        
        if st.button("Add Item"):
            inventory.add_item(name, quantity, price, expiration_date, category)
            st.success("Item added successfully!")
    
    elif menu == "Remove Item":
        st.header("Remove Item")
        name = st.text_input("Item Name")
        
        if st.button("Remove Item"):
            inventory.remove_item(name)
            st.success(f"Item '{name}' removed successfully!")
    
    elif menu == "Sell Item":
        st.header("Sell Item")
        name = st.text_input("Item Name")
        quantity_to_sell = st.number_input("Quantity to Sell", min_value=1)
        
        if st.button("Sell Item"):
            inventory.sell_item(name, quantity_to_sell)
            st.success(f"{quantity_to_sell} units of '{name}' sold successfully!")
    
    elif menu == "Check Inventory":
        st.header("Inventory")
        inventory_df = inventory.get_inventory()
        st.write(inventory_df)
    
    elif menu == "Visualizations":
        st.header("Inventory Visualizations")
        chart_type = st.selectbox("Choose Chart Type", ["Bar Chart", "Pie Chart", "Line Chart"])
    
        inventory_df = inventory.get_inventory()
        if chart_type == "Bar Chart":
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(inventory_df["name"], inventory_df["quantity"], color='skyblue')
            ax.set_title("Inventory Quantities by Item", fontsize=16)
            ax.set_xlabel("Item Name", fontsize=14)
            ax.set_ylabel("Quantity", fontsize=14)
            ax.tick_params(axis='x', rotation=45)
            st.pyplot(fig)
        elif chart_type == "Pie Chart":
            fig, ax = plt.subplots(figsize=(8, 8))
            inventory_df.set_index("name")["quantity"].plot.pie(autopct="%1.1f%%", ax=ax)
            ax.set_ylabel("")
            ax.set_title("Inventory Distribution", fontsize=16)
            st.pyplot(fig)
        elif chart_type == "Line Chart":
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(inventory_df["name"], inventory_df["quantity"], marker='o', color='purple')
            ax.set_title("Inventory Quantities Over Time", fontsize=16)
            ax.set_xlabel("Item Name", fontsize=14)
            ax.set_ylabel("Quantity", fontsize=14)
            ax.tick_params(axis='x', rotation=45)
            st.pyplot(fig)
else:
    st.sidebar.error("Invalid username or password.")
