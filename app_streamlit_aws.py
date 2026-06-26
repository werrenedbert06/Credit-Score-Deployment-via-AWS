"""
Streamlit UI untuk Credit Score Classifier yang di-host di SageMaker.

Reads endpoint name dan region dari environment variables.
boto3 picks up AWS credentials dari:
  - EC2 instance profile (ketika running on EC2 dengan LabInstanceProfile), OR
  - ~/.aws/credentials (ketika running locally)
"""

import json
import os

import boto3
import pandas as pd
import streamlit as st
from botocore.exceptions import ClientError, NoCredentialsError

ENDPOINT_NAME = os.environ.get("ENDPOINT_NAME", "credit-score-endpoint")
REGION        = os.environ.get("AWS_REGION",    "us-east-1")

st.set_page_config(
    page_title="Credit Score Predictor",
    page_icon="💳",
    layout="centered"
)


@st.cache_resource
def get_runtime_client():
    return boto3.client("sagemaker-runtime", region_name=REGION)

def invoke_endpoint(input_dict: dict) -> dict:
    runtime = get_runtime_client()
    payload = {"instances": [input_dict]}
    response = runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="application/json",
        Accept="application/json",
        Body=json.dumps(payload),
    )
    return json.loads(response["Body"].read().decode("utf-8"))

# Header
st.title("💳 Credit Score Predictor")
st.caption(f"Credit Classification: **Poor**, **Standard**, atau **Good** via SageMaker endpoint `{ENDPOINT_NAME}`")
st.divider()

# Form — identik dengan versi lokal
with st.form("credit_form"):

    st.subheader("👤 Profil Pribadi")
    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.number_input("Usia (tahun)", min_value=18, max_value=100, value=30)
    with c2:
        month = st.selectbox("Bulan", ["Januari","Februari","Maret","April","Mei","Juni",
                                       "Juli","Agustus","September","Oktober","November","Desember"])
    with c3:
        occupation = st.selectbox("Pekerjaan", ["Akuntan","Arsitek","Developer","Dokter","Insinyur",
                                                "Pengusaha","Jurnalis","Pengacara","Manajer","Mekanik",
                                                "Manajer Media","Musisi","Ilmuwan","Guru","Penulis"])

    st.subheader("💰 Pendapatan & Rekening")
    c1, c2 = st.columns(2)
    with c1:
        annual_income           = st.number_input("Pendapatan Tahunan (USD)", min_value=0, value=50000, step=1000)
        monthly_inhand_salary   = st.number_input("Gaji Bersih Bulanan (USD)", min_value=0.0, value=3000.0, step=100.0)
        monthly_balance         = st.number_input("Saldo Bulanan (USD)", min_value=0.0, value=500.0, step=10.0)
        amount_invested_monthly = st.number_input("Investasi Bulanan (USD)", min_value=0.0, value=200.0, step=10.0)
    with c2:
        num_bank_accounts         = st.number_input("Jumlah Rekening Bank", min_value=0, max_value=20, value=3)
        num_credit_card           = st.number_input("Jumlah Kartu Kredit", min_value=0, max_value=20, value=4)
        total_emi_per_month       = st.number_input("Total Angsuran per Bulan (USD)", min_value=0.0, value=100.0, step=10.0)
        credit_history_age_months = st.number_input("Usia Riwayat Kredit (bulan)", min_value=0, value=120)

    st.subheader("🏦 Utang & Risiko Kredit")
    c1, c2 = st.columns(2)
    with c1:
        outstanding_debt         = st.number_input("Sisa Utang (USD)", min_value=0.0, value=1000.0, step=100.0)
        credit_utilization_ratio = st.number_input("Rasio Penggunaan Kartu Kredit (%)", min_value=0.0, max_value=100.0, value=30.0)
        interest_rate            = st.number_input("Suku Bunga (%)", min_value=0, max_value=100, value=15)
        num_of_loan              = st.number_input("Jumlah Pinjaman", min_value=0, max_value=50, value=2)
    with c2:
        changed_credit_limit = st.number_input("Perubahan Limit Kartu Kredit (%)", min_value=0.0, value=10.0, step=0.1)
        num_credit_inquiries = st.number_input("Jumlah Permintaan Kredit", min_value=0, value=5)
        type_of_loan         = st.selectbox("Jenis Pinjaman", ["Pinjaman Pribadi","Pinjaman Mahasiswa","KPR",
                                                               "Pinjaman Gaji","Pinjaman Mobil","Pinjaman Pembangun Kredit",
                                                               "Konsolidasi Utang","Pinjaman Ekuitas Rumah","Tidak Ditentukan"])
        credit_mix           = st.selectbox("Komposisi Kredit", ["Bad","Standard","Good"])

    st.subheader("📅 Perilaku Pembayaran")
    c1, c2, c3 = st.columns(3)
    with c1:
        delay_from_due_date    = st.number_input("Rata-rata Keterlambatan (hari)", min_value=0, value=10)
    with c2:
        num_of_delayed_payment = st.number_input("Jumlah Pembayaran Terlambat", min_value=0, value=5)
    with c3:
        payment_of_min_amount = st.selectbox("Hanya Bayar Minimum?", ["No","Yes"])

    payment_behaviour = st.selectbox("Pola Belanja & Pembayaran", [
        "Low_spent_Small_value_payments",
        "Low_spent_Medium_value_payments",
        "Low_spent_Large_value_payments",
        "High_spent_Small_value_payments",
        "High_spent_Medium_value_payments",
        "High_spent_Large_value_payments",
    ])

    st.divider()
    submitted = st.form_submit_button("🔍 Prediksi", type="primary", use_container_width=True)

# Hasil prediksi — struktur identik versi lokal, bedanya pakai invoke_endpoint bukan joblib
if submitted:
    input_dict = {
        "Age": age, "Annual_Income": annual_income,
        "Monthly_Inhand_Salary": monthly_inhand_salary,
        "Num_Bank_Accounts": num_bank_accounts, "Num_Credit_Card": num_credit_card,
        "Interest_Rate": interest_rate, "Num_of_Loan": num_of_loan,
        "Delay_from_due_date": delay_from_due_date, "Num_of_Delayed_Payment": num_of_delayed_payment,
        "Changed_Credit_Limit": changed_credit_limit, "Num_Credit_Inquiries": num_credit_inquiries,
        "Outstanding_Debt": outstanding_debt, "Credit_Utilization_Ratio": credit_utilization_ratio,
        "Total_EMI_per_month": total_emi_per_month, "Amount_invested_monthly": amount_invested_monthly,
        "Monthly_Balance": monthly_balance, "Credit_History_Age_Months": credit_history_age_months,
        "Month": month, "Occupation": occupation, "Type_of_Loan": type_of_loan,
        "Credit_Mix": credit_mix, "Payment_of_Min_Amount": payment_of_min_amount,
        "Payment_Behaviour": payment_behaviour,
    }

    try:
        result = invoke_endpoint(input_dict)
        pred   = result["predictions"][0]
        labels = {0: "Poor", 1: "Standard", 2: "Good"}
        result_label = labels[int(pred)]

        st.divider()
        if result_label == "Good":
            st.success(f"### ✅ Credit Score: **{result_label}**")
            st.caption("Risiko rendah: Layak mendapat produk kredit premium")
            st.balloons()
        elif result_label == "Standard":
            st.info(f"### 📊 Credit Score: **{result_label}**")
            st.caption("Risiko sedang: Memenuhi persyaratan dasar")
        else:
            st.error(f"### ⚠️ Credit Score: **{result_label}**")
            st.caption("Risiko tinggi: Perbaikan finansial disarankan")

        # Ringkasan input — identik versi lokal
        with st.expander("📋 Ringkasan Input", expanded=False):
            st.dataframe(pd.DataFrame({
                "Fitur": ["Pendapatan Tahunan","Saldo Bulanan","Sisa Utang",
                          "Rasio Penggunaan Kredit","Pembayaran Terlambat","Usia Riwayat Kredit"],
                "Nilai": [f"${annual_income:,.0f}", f"${monthly_balance:,.0f}", f"${outstanding_debt:,.0f}",
                          f"{credit_utilization_ratio:.1f}%", str(num_of_delayed_payment), f"{credit_history_age_months} bulan"]
            }), use_container_width=True, hide_index=True)

    except NoCredentialsError:
        st.error(
            "❌ AWS credentials tidak ditemukan. "
            "Jika running di EC2, pastikan LabInstanceProfile sudah di-attach. "
            "Jika running lokal, jalankan `aws configure` terlebih dahulu."
        )
    except ClientError as e:
        st.error(f"❌ AWS error: {e.response['Error'].get('Message', str(e))}")
    except Exception as e:
        st.error(f"❌ Prediksi gagal: {e}")
        st.caption("Pastikan endpoint aktif dan nama endpoint sudah benar.")
