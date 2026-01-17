import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt

# =====================================================
# LOAD MODELS
# =====================================================
try:
    logistic_model = joblib.load("model/logistic_model.pkl")
    rf_model = joblib.load("model/randomforest.pkl")
except Exception as e:
    st.error(f"Gagal memuat model: {e}")
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
    "Analisis kondisi keuangan sepenuhnya berbasis Machine Learning "
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

    # -------------------------------------------------
    # PERHITUNGAN DASAR
    # -------------------------------------------------
    total_expense = food + transport + entertainment + shopping + others
    remaining = income - total_expense

    saving_ratio_raw = remaining / income
    saving_ratio_ui = max(min(saving_ratio_raw, 1), 0)

    # -------------------------------------------------
    # FEATURE VECTOR
    # -------------------------------------------------
    X = np.array([[
        food / income,
        transport / income,
        entertainment / income,
        shopping / income,
        others / income,
        saving_ratio_raw
    ]])

    # -------------------------------------------------
    # LOGISTIC REGRESSION
    # -------------------------------------------------
    proba = logistic_model.predict_proba(X)[0]
    classes = logistic_model.classes_

    pred_class = classes[np.argmax(proba)]
    status = "HEMAT" if pred_class == 1 else "BOROS"

    # -------------------------------------------------
    # RANDOM FOREST (TARGET TABUNGAN IDEAL)
    # -------------------------------------------------
    ideal_saving_ratio = rf_model.predict(X)[0]
    ideal_saving_ratio = max(min(ideal_saving_ratio, 1), 0)

    gap_ratio = ideal_saving_ratio - saving_ratio_raw
    gap_amount = max(gap_ratio * income, 0)

    # =================================================
    # OUTPUT
    # =================================================
    st.subheader("📌 Status Keuangan")

    if status == "HEMAT":
        st.success("✅ HEMAT — Berdasarkan prediksi Machine Learning")
    else:
        st.error("❌ BOROS — Berdasarkan prediksi Machine Learning")

    # -------------------------------------------------
    # RINGKASAN
    # -------------------------------------------------
    st.markdown("### 📄 Ringkasan Keuangan")
    st.write(f"""
    - **Total Pengeluaran:** Rp {total_expense:,.0f}  
    - **Sisa Pendapatan:** Rp {remaining:,.0f}  
    """)

    st.caption(
        f"📊 Probabilitas ML → BOROS: {proba[classes.tolist().index(0)]:.2f} | "
        f"HEMAT: {proba[classes.tolist().index(1)]:.2f}"
    )

    # -------------------------------------------------
    # REKOMENDASI (LOGIS & KONSISTEN)
    # -------------------------------------------------
    st.subheader("💡 Rekomendasi Pengeluaran")

    categories = {
        "Makan & Minum": food,
        "Transportasi": transport,
        "Hiburan": entertainment,
        "Belanja": shopping,
        "Sewa": others
    }

    sorted_categories = sorted(
        categories.items(),
        key=lambda x: x[1],
        reverse=True
    )

    # =========================
    # JIKA BOROS → ADA POTONGAN
    # =========================
    if status == "BOROS" and gap_amount > 0:
        st.warning(
            f"🔻 Untuk mencapai kondisi lebih hemat, "
            f"disarankan mengurangi pengeluaran sekitar "
            f"**Rp {gap_amount:,.0f}**."
        )

        st.write("Distribusi pengurangan yang disarankan:")

        total_expense = sum(categories.values())

        for cat, val in sorted_categories:
            proporsi = val / total_expense
            suggested_cut = proporsi * gap_amount

            if suggested_cut >= 10_000:
                st.write(
                    f"• **{cat}** → kurangi sekitar **Rp {suggested_cut:,.0f}**"
                )

    # =========================
    # JIKA HEMAT → TIDAK ADA POTONGAN
    # =========================
    else:
        st.success(
            "✅ Pengeluaran Anda sudah berada dalam kondisi **HEMAT**. "
            "Tidak diperlukan pengurangan pengeluaran saat ini."
        )

    st.caption(
        "📌 Status BOROS/HEMAT ditentukan oleh Logistic Regression. "
        "Random Forest digunakan untuk estimasi target tabungan ideal."
    )
        # =================================================
    # PIE CHART: KOMPOSISI KEUANGAN (100%)
    # =================================================
    st.subheader("📊 Komposisi Pengeluaran vs Tabungan")

    saving_amount = max(remaining, 0)

    labels = ["Pengeluaran", "Tabungan"]
    values = [total_expense, saving_amount]

    fig, ax = plt.subplots()
    ax.pie(
        values,
        labels=labels,
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops={"edgecolor": "white"}
    )
    ax.axis("equal")

    st.pyplot(fig)
