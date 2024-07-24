import streamlit as st
import pickle
import pandas as pd

# Memuat model
with open('restaurant_model.sav', 'rb') as file:
    restaurant_model = pickle.load(file)

# Judul
st.title("Analisis Profitabilitas Menu Restoran")

# Input dari pengguna
MenuCategory = st.text_input("Input MenuCategory")
MenuItem = st.text_input("Input MenuItem")
Price = st.text_input("Input Price")

# Kode untuk prediksi
restaurant_menu = ''

# Membuat tombol prediksi
if st.button('Prediksi'):
    try:
        # Konversi Price ke float
        Price = float(Price)
        
        # Menyiapkan data input sebagai DataFrame dengan nama kolom yang sesuai
        input_data = pd.DataFrame({
            'MenuCategory': [MenuCategory],
            'MenuItem': [MenuItem],
            'Price': [Price]
        })
        
        # Transformasi data input
        input_data_transformed = restaurant_model.named_steps['preprocessor'].transform(input_data)
        
        # Melakukan prediksi
        restaurant_prediction = restaurant_model.named_steps['classifier'].predict(input_data_transformed)

        # Memetakan prediksi ke kategori
        def map_prediction_to_category(prediction):
            categories = ['Low', 'Medium', 'High']
            if isinstance(prediction, str):
                return prediction
            else:
                return categories[int(prediction)]
        
        # Mendapatkan hasil prediksi
        restaurant_menu = map_prediction_to_category(restaurant_prediction[0])
        
        st.success(f"Predicted Profitability: {restaurant_menu}")
    except ValueError:
        st.error("Silakan masukkan nilai yang valid.")
    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")
