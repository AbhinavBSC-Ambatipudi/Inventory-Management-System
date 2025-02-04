# Inventory Management System (IMS)

## Overview
The **Inventory Management System (IMS)** is a user-friendly web application built using **Streamlit** and **Machine Learning** to help businesses efficiently manage their inventory. It provides functionalities such as adding, removing, selling items, tracking inventory levels, and predicting future demand using a **Random Forest Regressor**.

## Features
- **User Authentication**: Secure login system to prevent unauthorized access.
- **Add, Remove, and Sell Items**: Easily manage inventory data with intuitive UI components.
- **Inventory Overview**: View and track items with details like quantity, price, category, and expiration date.
- **Critical Stock Alerts**: Get alerts for items below a critical stock threshold.
- **Demand Prediction**: AI-powered forecasting to predict future demand based on past inventory data.
- **Data Persistence**: Stores inventory data in a CSV file for easy access and retrieval.
- **Visual Analytics**: Graphical representations of inventory levels using bar charts, pie charts, and line charts.

## Technologies Used
- **Frontend & Backend**: [Streamlit](https://streamlit.io/)
- **Data Processing**: [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
- **Machine Learning Model**: [Scikit-learn](https://scikit-learn.org/)
- **Data Visualization**: [Matplotlib](https://matplotlib.org/)

## Installation & Setup
### Prerequisites
Ensure you have Python (>=3.7) installed. Install required dependencies using:
```sh
pip install -r requirements.txt
```

### Running the Application
```sh
streamlit run app.py
```
This will start the Streamlit application, and you can access it in your browser.

## Usage
1. **Login**: Enter username (`admin`) and password (`admin123`).
2. **Manage Inventory**: Use the sidebar to add, remove, or sell items.
3. **Check Inventory**: View and analyze stock levels in tabular and graphical formats.
4. **Predict Demand**: The system provides demand predictions based on historical data.

## Project Structure
```
📂 Inventory-Management-System
├── 📄 app.py                # Main application file
├── 📄 inventory.py          # Inventory class for data management
├── 📄 demand_predictor.py   # Machine Learning model for demand prediction
├── 📄 requirements.txt      # Python dependencies
├── 📄 README.md             # Project documentation
└── 📂 data
    ├── inventory_data.csv   # Stored inventory records
```

## Future Enhancements
- **Cloud Database Integration**
- **Role-Based Access Control**
- **Automated Restocking Suggestions**
- **Mobile App Version**

## Contributions
Contributions are welcome! Feel free to fork this repository and submit a pull request.

