import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt
import gdown
import os

# =====================================================
# LOAD MODELS
# =====================================================
# Logistic model tetap dari file lokal
try:
    logistic_model = joblib.load("logistic_model.pkl")
except Exception as e:
    st.error(f"Gagal memuat logistic_model: {e}")
    st.stop()

# Random Forest: download otomatis dari Google Drive
RF_FILE = "randomforest.pkl"
RF_DRIVE_ID = "1Cq3Cj3yUyj18sXysJpUdmMKPAYGxb6Pw"

if not os.path.exists(RF_FILE):
    url = f"https://drive.google.com/uc?id={RF_DRIVE_ID}"
    gdown.download(url, RF_FILE, quiet=False)

try:
    rf_model = joblib.load(RF_FILE)
except Exception as e:
    st.error(f"Gagal memuat randomforest model: {e}")
    st.stop()

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Financial ML Dashboard",
    layout="centered"
)

st.title("📊 Financial Spending Analysis (ML-Based)")
st.write(
    "Analisis kondisi keuangan berbasis Machine Learning "
    "menggunakan rasio pengeluaran dan tabungan."
)

# =====================================================
# INPUT USER
# =====================================================
st.subheader("🧾 Input Keuangan Bulanan")

income = st.number_input("💰 Pendapatan Bulanan (Rp)", min_value=0, step=100_000)

food = st.number_input("🍽️ Makan & Minum", min_value=0, step=50_000)
transport = st.number_input("🚗 Transportasi", min_value=0, step=50_000)
entertainment = st.number_input("🎮 Hiburan", min_value=0, step=50_000)
shopping = st.number_input("🛍️ Belanja", min_value=0, step=50_000)
others = st.number_input("📦 Sewa", min_value=0, step=50_000)

# =====================================================
# PROCESS
# =====================================================
if st.button("🔍 Analisis Keuangan"):

    if income <= 0:
        st.error("Pendapatan harus lebih dari 0.")
        st.stop()

    total_expense = food + transport + entertainment + shopping + others
    remaining = income - total_expense
    saving_ratio_raw = remaining / income

    X = np.array([[
        food / income,
        transport / income,
        entertainment / income,
        shopping / income,
        others / income,
        saving_ratio_raw
    ]])

    # Logistic Regression
    proba = logistic_model.predict_proba(X)[0]
    classes = logistic_model.classes_
    pred_class = classes[np.argmax(proba)]
    status = "HEMAT" if pred_class == 1 else "BOROS"

    # Random Forest
    ideal_saving_ratio = rf_model.predict(X)[0]
    ideal_saving_ratio = max(min(ideal_saving_ratio, 1), 0)
    gap_ratio = ideal_saving_ratio - saving_ratio_raw
    gap_amount = max(gap_ratio * income, 0)

    # =================================================
    st.subheader("📌 Status Keuangan")
    if status == "HEMAT":
        st.success("✅ HEMAT — Berdasarkan prediksi Machine Learning")
    else:
        st.error("❌ BOROS — Berdasarkan prediksi Machine Learning")

    # Ringkasan
    st.markdown("### 📄 Ringkasan Keuangan")
    st.write(f"- **Total Pengeluaran:** Rp {total_expense:,.0f}")
    st.write(f"- **Sisa Pendapatan:** Rp {remaining:,.0f}")

    st.caption(
        f"📊 Probabilitas ML → BOROS: {proba[classes.tolist().index(0)]:.2f} | "
        f"HEMAT: {proba[classes.tolist().index(1)]:.2f}"
    )

    # Rekomendasi
    st.subheader("💡 Rekomendasi Pengeluaran")
    categories = {
        "Makan & Minum": food,
        "Transportasi": transport,
        "Hiburan": entertainment,
        "Belanja": shopping,
        "Sewa": others
    }
    sorted_categories = sorted(categories.items(), key=lambda x: x[1], reverse=True)

    if status == "BOROS" and gap_amount > 0:
        st.warning(
            f"🔻 Untuk lebih hemat, disarankan mengurangi pengeluaran sekitar Rp {gap_amount:,.0f}."
        )
        total_expense = sum(categories.values())
        for cat, val in sorted_categories:
            proporsi = val / total_expense
            suggested_cut = proporsi * gap_amount
            if suggested_cut >= 10_000:
                st.write(f"• **{cat}** → kurangi sekitar **Rp {suggested_cut:,.0f}**")
    else:
        st.success("✅ Pengeluaran Anda sudah hemat. Tidak perlu pengurangan.")

    st.caption(
        "📌 Status BOROS/HEMAT ditentukan oleh Logistic Regression. "
        "Random Forest digunakan untuk estimasi target tabungan ideal."
    )

    # Pie Chart
    st.subheader("📊 Komposisi Pengeluaran vs Tabungan")
    saving_amount = max(remaining, 0)
    labels = ["Pengeluaran", "Tabungan"]
    values = [total_expense, saving_amount]

    fig, ax = plt.subplots()
    ax.pie(values, labels=labels, autopct="%1.1f%%", startangle=90, wedgeprops={"edgecolor": "white"})
    ax.axis("equal")
    st.pyplot(fig)
