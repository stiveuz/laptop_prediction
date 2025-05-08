import streamlit as st  # type: ignore
import numpy as np
import joblib
import pandas as pd

# Kurs ma'lumotlari
som_value = 9294.31
dollar_value = 375
conversion_rate = dollar_value / som_value

# Faylni yuklash
df = pd.read_csv("Laptop_price.csv")

# Narxni dollarga o‘zgartirish
df["Price_in_dollars"] = (df["Price"] * conversion_rate).round(2)


# Modelni yuklash
model = joblib.load("fr_model.pkl")

# Ilova sarlavhasi
st.title("💸 Noutbuk narxini bashorat qilish dasturi")

st.divider()

st.write(
    "Ushbu ilova orqali protsessor tezligi, operativ xotira va doimiy xotira hajmini kiritgan holda, taxminiy narxini bilib olishingiz mumkin."
)


st.divider()

# Foydalanuvchi inputlari
processor_speed = st.number_input("🧠 Protsessor tezligi (GHz)", value=2.50, step=0.50)
ram_size = st.number_input("💾 Operativ xotira hajmi (GB)", value=16, step=8)
storage_capacity = st.number_input("🗃️ Doimiy xotira hajmi (GB)", value=512, step=256)

X = [processor_speed, ram_size, storage_capacity]

st.divider()

# Narxni hisoblash
if st.button("📊 Narxni hisoblash"):
    st.balloons()
    prediction_som = model.predict([np.array(X)])[0]
    prediction_usd = prediction_som * conversion_rate
    st.success(f"💰 Noutbuk uchun taxminiy narx: {prediction_usd:,.2f} $")
else:
    st.info("Iltimos, taxminiy narxni ko‘rish uchun tugmani bosing.")
