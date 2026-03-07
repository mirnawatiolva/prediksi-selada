import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import lime
import lime.lime_tabular
import pickle
import os

# Konfigurasi halaman
st.set_page_config(
    page_title="Prediksi Selada",
    page_icon="🥬",
    layout="wide"
)

# Judul aplikasi
st.title("🥬 Sistem Prediksi Selada")
st.markdown("---")

# Sidebar untuk navigasi
st.sidebar.header("Menu")
menu = st.sidebar.selectbox(
    "Pilih Model Prediksi",
    ["Distribusi Penjualan", "Jumlah Tanam"]
)

# Upload file
st.sidebar.header("Upload Data")
st.sidebar.markdown("Upload file CSV (opsional)")
uploaded_file = st.sidebar.file_uploader("Upload file CSV", type=['csv'], label_visibility="collapsed")
st.sidebar.info("💡 Jika tidak upload, akan menggunakan dataset default")

# Fungsi evaluasi model
def evaluate_model(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    # MAPE: Hindari pembagian dengan nol jika ada nilai y_true yang nol
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    mape = mape if not (np.isinf(mape) or np.isnan(mape)) else 0  # Handle cases where y_true is 0, resulting in inf or nan
    return mse, mae, mape

# Fungsi untuk load data
@st.cache_data
def load_data(file):
    if file is not None:
        df = pd.read_csv(file, sep=';')
    else:
        # Load default dataset dari folder dataset
        default_path = os.path.join('dataset', 'data_selada.csv')
        if os.path.exists(default_path):
            df = pd.read_csv(default_path, sep=';')
            st.info(f"📂 Menggunakan dataset default: {default_path}")
        else:
            st.warning("⚠️ Tidak ada file yang diupload dan file default tidak ditemukan. Silakan upload file CSV data selada.")
            return None
    
    # Bersihkan kolom
    df.columns = df.columns.str.strip()
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    
    # Isi nilai NaN di kolom Pasar jika ada
    if 'Pasar' in df.columns:
        df['Pasar'].fillna(0, inplace=True)
    
    return df

# ===== MODEL DISTRIBUSI PENJUALAN =====
if menu == "Distribusi Penjualan":
    st.header("📊 Prediksi Distribusi Penjualan")
    
    df = load_data(uploaded_file)
    
    if df is not None:
        st.subheader("Preview Data")
        st.dataframe(df.head(10))
        
        # Definisikan fitur dan target
        features = ['Jumlah Tanam', 'Data Stok', 'Data Transaksi']
        targets = ['Rumah Sendiri', 'Rt 1', 'Rt 2', 'Rt 3', 'Rt 4', 'Rt 5', 'Rt 6', 'Rt 7', 'Pasar']
        
        # Cek apakah semua kolom ada
        missing_cols = [col for col in features + targets if col not in df.columns]
        if missing_cols:
            st.error(f"Kolom yang hilang: {missing_cols}")
        else:
            X = df[features]
            y = df[targets]
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            st.success(f"Data berhasil dimuat! Total data: {len(df)}, Training: {len(X_train)}, Testing: {len(X_test)}")
            
            # Tampilkan data training dan testing
            with st.expander("📊 Lihat Data Training dan Testing"):
                st.markdown("### Data Training")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Fitur (X_train)**")
                    st.dataframe(X_train, height=300)
                with col2:
                    st.markdown("**Target (y_train)**")
                    st.dataframe(y_train, height=300)
                
                st.markdown("---")
                st.markdown("### Data Testing")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Fitur (X_test)**")
                    st.dataframe(X_test, height=300)
                with col2:
                    st.markdown("**Target (y_test)**")
                    st.dataframe(y_test, height=300)
            
            # Tabs untuk berbagai fungsi
            tab1, tab2, tab3 = st.tabs(["🎯 Prediksi", "📈 Model Training", "💡 LIME Explanation"])
            
            with tab1:
                st.subheader("Prediksi Distribusi Penjualan")
                
                # Hitung periode terakhir dari data
                if 'Periode' in df.columns:
                    last_period = int(df['Periode'].max())
                else:
                    last_period = len(df)
                
                # Input untuk periode prediksi
                st.markdown("### 📅 Periode Prediksi")
                st.info(f"📊 Data historis tersedia sampai periode **{last_period}**. Prediksi akan dimulai dari periode **{last_period + 1}**.")
                
                # Pilihan mode prediksi
                prediction_mode = st.radio(
                    "Mode Prediksi:",
                    ["Jumlah Periode", "Sampai Periode Tertentu"],
                    horizontal=True
                )
                
                if prediction_mode == "Jumlah Periode":
                    num_periods = st.number_input(
                        "Jumlah Periode untuk Diprediksi", 
                        min_value=1, 
                        max_value=100, 
                        value=1, 
                        step=1,
                        help=f"Masukkan jumlah periode ke depan yang ingin diprediksi"
                    )
                else:  # Sampai Periode Tertentu
                    target_period = st.number_input(
                        "Prediksi Sampai Periode", 
                        min_value=last_period + 1, 
                        max_value=last_period + 100, 
                        value=53 if last_period < 53 else last_period + 1, 
                        step=1,
                        help=f"Masukkan periode akhir yang ingin diprediksi (minimal {last_period + 1})"
                    )
                    num_periods = target_period - last_period
                
                st.markdown("### 🔢 Input Fitur")
                col1, col2, col3 = st.columns(3)
                with col1:
                    jumlah_tanam = st.number_input("Jumlah Tanam", value=float(df['Jumlah Tanam'].mean()), step=10.0)
                with col2:
                    data_stok = st.number_input("Data Stok", value=float(df['Data Stok'].mean()), step=10.0)
                with col3:
                    data_transaksi = st.number_input("Data Transaksi", value=float(df['Data Transaksi'].mean()), step=5.0)
                
                model_choice = st.selectbox("Pilih Model", ["Random Forest", "SVM", "Voting Regressor"])
                
                if st.button("🚀 Prediksi"):
                    with st.spinner("Melatih model dan melakukan prediksi untuk semua periode..."):
                        new_data = pd.DataFrame({
                            'Jumlah Tanam': [jumlah_tanam],
                            'Data Stok': [data_stok],
                            'Data Transaksi': [data_transaksi]
                        })
                        
                        # Dictionary untuk menyimpan prediksi per target dan periode
                        predictions_by_period = []
                        
                        # Train dan prediksi untuk setiap target
                        progress_bar = st.progress(0)
                        models = {}
                        
                        for idx, target in enumerate(targets):
                            y_train_target = y_train[target]
                            
                            if model_choice == "Random Forest":
                                model = RandomForestRegressor(n_estimators=100, random_state=42)
                                model.fit(X_train, y_train_target)
                            elif model_choice == "SVM":
                                model = SVR(kernel='rbf')
                                model.fit(X_train, y_train_target)
                            else:  # Voting Regressor
                                rf = RandomForestRegressor(n_estimators=100, random_state=42)
                                svm = SVR(kernel='rbf')
                                model = VotingRegressor([('rf', rf), ('svm', svm)])
                                model.fit(X_train, y_train_target)
                            
                            models[target] = model
                            progress_bar.progress((idx + 1) / len(targets))
                        
                        # Prediksi untuk multiple periode
                        for period in range(1, int(num_periods) + 1):
                            actual_period = last_period + period
                            period_predictions = {'Periode': actual_period}
                            
                            for target in targets:
                                pred = models[target].predict(new_data)[0]
                                period_predictions[target] = max(0, round(pred))
                            
                            predictions_by_period.append(period_predictions)
                        
                        predictions_df = pd.DataFrame(predictions_by_period)
                        
                        # Prediksi untuk periode pertama (untuk display utama)
                        first_period_pred = predictions_by_period[0]
                        
                        st.success(f"✅ Prediksi untuk {int(num_periods)} periode berhasil dibuat! (Periode {last_period + 1} - {last_period + int(num_periods)})")
                        
                        # Tampilkan hasil periode pertama
                        st.markdown(f"### 📊 Hasil Prediksi Periode {last_period + 1}")
                        
                        col1, col2 = st.columns([1, 2])
                        
                        with col1:
                            st.subheader("Hasil per Saluran")
                            for target in targets:
                                st.metric(target, f"{first_period_pred[target]}")
                            
                            st.metric("Total Prediksi", f"{sum([first_period_pred[t] for t in targets])}")
                        
                        with col2:
                            st.subheader("Visualisasi Bar Chart")
                            fig, ax = plt.subplots(figsize=(10, 6))
                            positions = np.arange(len(targets))
                            colors = plt.cm.viridis(np.linspace(0, 1, len(targets)))
                            values = [first_period_pred[t] for t in targets]
                            bars = ax.bar(positions, values, color=colors)
                            ax.set_xlabel('Saluran Penjualan')
                            ax.set_ylabel('Jumlah Prediksi')
                            ax.set_title(f'Prediksi Distribusi Penjualan Periode {last_period + 1} ({model_choice})')
                            ax.set_xticks(positions)
                            ax.set_xticklabels(targets, rotation=45, ha='right')
                            ax.grid(axis='y', alpha=0.3)
                            
                            # Tambahkan nilai di atas bar
                            for i, v in enumerate(values):
                                ax.text(i, v, str(v), ha='center', va='bottom')
                            
                            plt.tight_layout()
                            st.pyplot(fig)
                        
                        # Tampilkan tabel prediksi semua periode
                        if int(num_periods) > 1:
                            st.markdown("### 📋 Tabel Prediksi Semua Periode")
                            st.dataframe(predictions_df, use_container_width=True)
                            
                            # Visualisasi trend per saluran
                            st.markdown("### 📈 Trend Prediksi per Saluran")
                            
                            fig, axes = plt.subplots(3, 3, figsize=(15, 12))
                            axes = axes.flatten()
                            
                            for idx, target in enumerate(targets):
                                ax = axes[idx]
                                ax.plot(predictions_df['Periode'], predictions_df[target], marker='o', linewidth=2, markersize=6)
                                ax.set_xlabel('Periode')
                                ax.set_ylabel('Jumlah')
                                ax.set_title(target, fontweight='bold')
                                ax.grid(True, alpha=0.3)
                            
                            plt.tight_layout()
                            st.pyplot(fig)
                        
                        # Download hasil prediksi
                        st.markdown("### 💾 Download Hasil Prediksi")
                        csv = predictions_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download CSV",
                            data=csv,
                            file_name=f"prediksi_distribusi_penjualan_{int(num_periods)}_periode.csv",
                            mime="text/csv"
                        )
            
            with tab2:
                st.subheader("Training dan Evaluasi Model")
                
                if st.button("🔧 Train All Models"):
                    with st.spinner("Melatih semua model... (ini mungkin memakan waktu)"):
                        results = {}
                        
                        for target in targets:
                            st.write(f"**Training untuk {target}...**")
                            y_train_target = y_train[target]
                            y_test_target = y_test[target]
                            
                            # Random Forest
                            rf = RandomForestRegressor(n_estimators=100, random_state=42)
                            rf.fit(X_train, y_train_target)
                            rf_pred = rf.predict(X_test)
                            rf_mse, rf_mae, rf_mape = evaluate_model(y_test_target, rf_pred)
                            
                            # SVM
                            svm = SVR(kernel='rbf')
                            svm.fit(X_train, y_train_target)
                            svm_pred = svm.predict(X_test)
                            svm_mse, svm_mae, svm_mape = evaluate_model(y_test_target, svm_pred)
                            
                            # Voting
                            voting = VotingRegressor([('rf', rf), ('svm', svm)])
                            voting.fit(X_train, y_train_target)
                            voting_pred = voting.predict(X_test)
                            voting_mse, voting_mae, voting_mape = evaluate_model(y_test_target, voting_pred)
                            
                            results[target] = {
                                'RF': {'MSE': rf_mse, 'MAE': rf_mae, 'MAPE': rf_mape},
                                'SVM': {'MSE': svm_mse, 'MAE': svm_mae, 'MAPE': svm_mape},
                                'Voting': {'MSE': voting_mse, 'MAE': voting_mae, 'MAPE': voting_mape}
                            }
                        
                        st.success("Training selesai!")
                        
                        # Tampilkan hasil
                        for target, metrics in results.items():
                            st.write(f"### {target}")
                            df_metrics = pd.DataFrame(metrics).T
                            st.dataframe(df_metrics.style.format("{:.2f}"))
            
            with tab3:
                st.subheader("LIME Explanation")
                
                # Penjelasan tentang LIME dan fitur
                st.markdown("""
                ### 📖 Tentang LIME (Local Interpretable Model-agnostic Explanations)
                
                LIME menjelaskan kontribusi setiap fitur terhadap prediksi distribusi penjualan untuk satu instance data.
                
                **Penjelasan Fitur yang Digunakan:**
                
                - **Jumlah Tanam** 🌱: Total jumlah tanaman selada yang ditanam
                - **Data Stok** 📦: Jumlah stok selada yang tersedia saat ini
                - **Data Transaksi** 💳: Jumlah transaksi penjualan yang terjadi
                
                **Target Penjualan (9 Saluran):**
                - Rumah Sendiri, Rt 1, Rt 2, Rt 3, Rt 4, Rt 5, Rt 6, Rt 7, Pasar
                
                **Kenapa Periode Tertentu Dipilih?**
                
                Data dibagi menjadi **Training (80%)** dan **Testing (20%)** secara **random** menggunakan `train_test_split` dengan `random_state=42`.
                Periode yang muncul di LIME adalah **instance pertama dari data testing** (bukan urutan periode kronologis).
                """)
                
                st.info("💡 LIME akan menjelaskan prediksi untuk SEMUA saluran penjualan sekaligus menggunakan Voting Regressor")
                
                if st.button("🚀 Generate LIME Explanation"):
                    with st.spinner("Melatih model dan membuat penjelasan LIME untuk semua saluran penjualan..."):
                        # Train model untuk setiap target
                        best_models = {}
                        
                        for target in targets:
                            y_train_target = y_train[target]
                            
                            # Train Voting Regressor untuk setiap target
                            rf = RandomForestRegressor(n_estimators=100, random_state=42)
                            svm = SVR(kernel='rbf')
                            model = VotingRegressor([('rf', rf), ('svm', svm)])
                            model.fit(X_train, y_train_target)
                            best_models[f'voting_{target}'] = model
                        
                        # Inisialisasi LIME Explainer
                        explainer = lime.lime_tabular.LimeTabularExplainer(
                            training_data=X_train.values,
                            feature_names=X_train.columns.tolist(),
                            class_names=targets,  # Gunakan nama target untuk multi-output
                            mode='regression'
                        )
                        
                        # Pilih satu instance dari X_test untuk dijelaskan
                        idx_to_explain = 0
                        instance_to_explain = X_test.iloc[idx_to_explain].values
                        
                        st.success(f"✅ Instance dari Data Testing dipilih: Periode **{X_test.index[idx_to_explain]}**")
                        
                        # Tampilkan informasi detail instance
                        st.markdown("### 📊 Detail Instance yang Dijelaskan")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**Nilai Fitur:**")
                            instance_df = pd.DataFrame({
                                'Fitur': X_train.columns.tolist(),
                                'Nilai': instance_to_explain
                            })
                            st.dataframe(instance_df, hide_index=True)
                        
                        with col2:
                            st.metric("Periode", X_test.index[idx_to_explain])
                            st.info(f"Instance ini adalah baris ke-{idx_to_explain + 1} dari {len(X_test)} data testing")
                        
                        st.markdown("---")
                        st.markdown("### 🎯 Penjelasan LIME untuk Setiap Saluran Penjualan")
                        st.markdown("""
                        **Cara Membaca Grafik LIME:**
                        - Warna **orange (positif)**: Fitur yang meningkatkan prediksi penjualan
                        - Warna **biru (negatif)**: Fitur yang menurunkan prediksi penjualan
                        - Semakin panjang bar, semakin besar pengaruh fitur tersebut terhadap prediksi
                        """)
                        
                        # Untuk setiap target, jelaskan prediksi menggunakan Voting Regressor
                        for target in targets:
                            st.markdown(f"#### 📍 Target: **{target}**")
                            
                            model_to_explain = best_models[f'voting_{target}']
                            
                            exp = explainer.explain_instance(
                                data_row=instance_to_explain,
                                predict_fn=lambda x: model_to_explain.predict(pd.DataFrame(x, columns=X_train.columns)),
                                num_features=len(features)
                            )
                            
                            predicted_val = model_to_explain.predict(pd.DataFrame([instance_to_explain], columns=X_train.columns))[0]
                            actual_val = y_test.iloc[idx_to_explain][target]
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Prediksi Voting", f"{predicted_val:.2f}")
                            with col2:
                                st.metric("Nilai Aktual", f"{actual_val:.0f}")
                            with col3:
                                selisih = abs(predicted_val - actual_val)
                                st.metric("Selisih", f"{selisih:.2f}")
                            
                            # Tampilkan kontribusi fitur dalam tabel
                            contributions = exp.as_list()
                            contrib_df = pd.DataFrame(contributions, columns=['Fitur', 'Kontribusi'])
                            st.dataframe(contrib_df, hide_index=True)
                            
                            # Visualisasi penjelasan LIME
                            fig = exp.as_pyplot_figure()
                            fig.set_size_inches(10, 6)
                            plt.title(f'LIME Explanation for {target} (Actual: {actual_val}, Predicted: {predicted_val:.2f})')
                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.close()
                            
                            st.markdown("---")

# ===== MODEL JUMLAH TANAM =====
elif menu == "Jumlah Tanam":
    st.header("🌱 Prediksi Jumlah Tanam")
    
    df = load_data(uploaded_file)
    
    if df is not None:
        st.subheader("Preview Data")
        st.dataframe(df.head(10))
        
        # Definisikan fitur dan target
        features = ['Data Stok', 'Data Transaksi', 'Sudah Packing', 'Stok Belum Panen']
        target = 'Jumlah Tanam'
        
        # Cek kolom
        missing_cols = [col for col in features + [target] if col not in df.columns]
        if missing_cols:
            st.error(f"Kolom yang hilang: {missing_cols}")
        else:
            X = df[features]
            y = df[target]
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            st.success(f"Data berhasil dimuat! Total data: {len(df)}, Training: {len(X_train)}, Testing: {len(X_test)}")
            
            # Tampilkan data training dan testing
            with st.expander("📊 Lihat Data Training dan Testing"):
                st.markdown("### Data Training")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Fitur (X_train)**")
                    st.dataframe(X_train, height=300)
                with col2:
                    st.markdown("**Target (y_train)**")
                    st.dataframe(y_train, height=300)
                
                st.markdown("---")
                st.markdown("### Data Testing")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Fitur (X_test)**")
                    st.dataframe(X_test, height=300)
                with col2:
                    st.markdown("**Target (y_test)**")
                    st.dataframe(y_test, height=300)
            
            # Tabs
            tab1, tab2, tab3 = st.tabs(["🎯 Prediksi", "📈 Model Training", "💡 LIME Explanation"])
            
            with tab1:
                st.subheader("Prediksi Jumlah Tanam")
                
                # Hitung periode terakhir dari data
                if 'Periode' in df.columns:
                    last_period = int(df['Periode'].max())
                else:
                    last_period = len(df)
                
                # Input untuk periode prediksi
                st.markdown("### 📅 Periode Prediksi")
                st.info(f"📊 Data historis tersedia sampai periode **{last_period}**. Prediksi akan dimulai dari periode **{last_period + 1}**.")
                
                # Pilihan mode prediksi
                prediction_mode = st.radio(
                    "Mode Prediksi:",
                    ["Jumlah Periode", "Sampai Periode Tertentu"],
                    horizontal=True
                )
                
                if prediction_mode == "Jumlah Periode":
                    num_periods = st.number_input(
                        "Jumlah Periode untuk Diprediksi", 
                        min_value=1, 
                        max_value=100, 
                        value=1, 
                        step=1,
                        help=f"Masukkan jumlah periode ke depan yang ingin diprediksi"
                    )
                else:  # Sampai Periode Tertentu
                    target_period = st.number_input(
                        "Prediksi Sampai Periode", 
                        min_value=last_period + 1, 
                        max_value=last_period + 100, 
                        value=53 if last_period < 53 else last_period + 1, 
                        step=1,
                        help=f"Masukkan periode akhir yang ingin diprediksi (minimal {last_period + 1})"
                    )
                    num_periods = target_period - last_period
                
                st.markdown("### 🔢 Input Fitur")
                col1, col2 = st.columns(2)
                with col1:
                    data_stok = st.number_input("Data Stok", value=float(df['Data Stok'].mean()), step=10.0)
                    data_transaksi = st.number_input("Data Transaksi", value=float(df['Data Transaksi'].mean()), step=5.0)
                with col2:
                    sudah_packing = st.number_input("Sudah Packing", value=float(df['Sudah Packing'].mean()), step=10.0)
                    stok_belum_panen = st.number_input("Stok Belum Panen", value=float(df['Stok Belum Panen'].mean()), step=10.0)
                
                model_choice = st.selectbox("Pilih Model", ["Random Forest", "SVM", "Voting Regressor"])
                
                if st.button("🚀 Prediksi"):
                    with st.spinner("Melatih model dan melakukan prediksi..."):
                        # Train model
                        if model_choice == "Random Forest":
                            model = RandomForestRegressor(n_estimators=100, random_state=42)
                            model.fit(X_train, y_train)
                        elif model_choice == "SVM":
                            model = SVR(kernel='rbf')
                            model.fit(X_train, y_train)
                        else:  # Voting Regressor
                            rf = RandomForestRegressor(n_estimators=100, random_state=42)
                            svm = SVR(kernel='rbf')
                            model = VotingRegressor([('rf', rf), ('svm', svm)])
                            model.fit(X_train, y_train)
                        
                        # Prediksi untuk periode tunggal
                        new_data = pd.DataFrame({
                            'Data Stok': [data_stok],
                            'Data Transaksi': [data_transaksi],
                            'Sudah Packing': [sudah_packing],
                            'Stok Belum Panen': [stok_belum_panen]
                        })
                        prediction = model.predict(new_data)[0]
                        
                        # Prediksi untuk multiple periode
                        predictions_list = []
                        for period in range(1, int(num_periods) + 1):
                            actual_period = last_period + period
                            pred = model.predict(new_data)[0]
                            predictions_list.append({
                                'Periode': actual_period,
                                'Prediksi': pred
                            })
                        
                        predictions_df = pd.DataFrame(predictions_list)
                        
                        st.success(f"✅ Prediksi untuk {int(num_periods)} periode berhasil dibuat! (Periode {last_period + 1} - {last_period + int(num_periods)})")
                        
                        # Tampilkan hasil
                        col1, col2 = st.columns([1, 2])
                        
                        with col1:
                            st.metric(f"Prediksi Jumlah Tanam (Periode {last_period + 1})", f"{prediction:.0f}")
                            st.metric("Rata-rata Prediksi", f"{predictions_df['Prediksi'].mean():.0f}")
                            st.metric("Total Prediksi", f"{predictions_df['Prediksi'].sum():.0f}")
                        
                        with col2:
                            # Grafik trend dengan prediksi multiple periode
                            fig, ax = plt.subplots(figsize=(12, 6))
                            
                            # Plot data historis
                            if 'Periode' in df.columns:
                                historical_periods = df['Periode']
                            else:
                                historical_periods = range(1, len(df) + 1)
                            
                            ax.plot(historical_periods, df[target], marker='o', 
                                   label='Data Historis', color='blue', linewidth=2, markersize=5)
                            
                            # Plot prediksi menggunakan periode aktual dari dataframe
                            ax.plot(predictions_df['Periode'], predictions_df['Prediksi'], 
                                   marker='s', label=f'Prediksi (Periode {last_period + 1}-{last_period + int(num_periods)})', 
                                   color='red', linewidth=2, linestyle='--', markersize=5)
                            
                            ax.set_xlabel('Periode', fontsize=12)
                            ax.set_ylabel('Jumlah Tanam', fontsize=12)
                            ax.set_title(f'Prediksi Jumlah Tanam ({model_choice})', fontsize=14, fontweight='bold')
                            ax.legend(fontsize=10)
                            ax.grid(True, alpha=0.3)
                            plt.tight_layout()
                            st.pyplot(fig)
                        
                        # Tabel prediksi detail
                        st.subheader("📊 Detail Prediksi per Periode")
                        
                        # Buat tabel dengan 5 kolom
                        predictions_display = predictions_df.copy()
                        predictions_display['Prediksi'] = predictions_display['Prediksi'].round(0).astype(int)
                        
                        # Tampilkan dalam format yang rapi
                        col_count = 5
                        cols = st.columns(col_count)
                        for idx, row in predictions_display.iterrows():
                            col_idx = idx % col_count
                            with cols[col_idx]:
                                st.metric(f"Periode {row['Periode']}", f"{row['Prediksi']}")
                        
                        # Tambahkan opsi download
                        st.markdown("### 💾 Download Hasil Prediksi")
                        csv = predictions_display.to_csv(index=False)
                        st.download_button(
                            label="📥 Download CSV",
                            data=csv,
                            file_name=f"prediksi_jumlah_tanam_{int(num_periods)}_periode.csv",
                            mime="text/csv"
                        )
            
            with tab2:
                st.subheader("Training dan Evaluasi Model")
                
                if st.button("🔧 Train All Models"):
                    with st.spinner("Melatih semua model..."):
                        results = {}
                        
                        # Random Forest
                        rf = RandomForestRegressor(n_estimators=100, random_state=42)
                        rf.fit(X_train, y_train)
                        rf_pred = rf.predict(X_test)
                        rf_mse, rf_mae, rf_mape = evaluate_model(y_test, rf_pred)
                        results['Random Forest'] = {'MSE': rf_mse, 'MAE': rf_mae, 'MAPE': rf_mape}
                        
                        # SVM
                        svm = SVR(kernel='rbf')
                        svm.fit(X_train, y_train)
                        svm_pred = svm.predict(X_test)
                        svm_mse, svm_mae, svm_mape = evaluate_model(y_test, svm_pred)
                        results['SVM'] = {'MSE': svm_mse, 'MAE': svm_mae, 'MAPE': svm_mape}
                        
                        # Voting
                        voting = VotingRegressor([('rf', rf), ('svm', svm)])
                        voting.fit(X_train, y_train)
                        voting_pred = voting.predict(X_test)
                        voting_mse, voting_mae, voting_mape = evaluate_model(y_test, voting_pred)
                        results['Voting Regressor'] = {'MSE': voting_mse, 'MAE': voting_mae, 'MAPE': voting_mape}
                        
                        st.success("Training selesai!")
                        
                        # Tampilkan hasil
                        df_results = pd.DataFrame(results).T
                        st.dataframe(df_results.style.format("{:.2f}").background_gradient(cmap='RdYlGn_r'))
                        
                        # Visualisasi perbandingan
                        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                        
                        for idx, (name, preds) in enumerate([('RF', rf_pred), ('SVM', svm_pred), ('Voting', voting_pred)]):
                            axes[idx].scatter(y_test, preds, alpha=0.6)
                            axes[idx].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
                            axes[idx].set_xlabel('Actual')
                            axes[idx].set_ylabel('Predicted')
                            axes[idx].set_title(f'{name}')
                            axes[idx].grid(True, alpha=0.3)
                        
                        plt.tight_layout()
                        st.pyplot(fig)
            
            with tab3:
                st.subheader("LIME Explanation")
                
                # Penjelasan tentang LIME dan fitur
                st.markdown("""
                ### 📖 Tentang LIME (Local Interpretable Model-agnostic Explanations)
                
                LIME menjelaskan kontribusi setiap fitur terhadap prediksi jumlah tanam untuk satu instance data.
                
                **Penjelasan Fitur yang Digunakan:**
                
                - **Data Stok** 📦: Jumlah stok selada yang tersedia saat ini
                - **Data Transaksi** 💳: Jumlah transaksi penjualan yang terjadi
                - **Sudah Packing** 📦: Jumlah selada yang sudah dikemas dan siap dijual
                - **Stok Belum Panen** 🌱: Jumlah tanaman selada yang masih dalam proses pertumbuhan
                
                **Kenapa Periode Tertentu Dipilih?**
                
                Data dibagi menjadi **Training (80%)** dan **Testing (20%)** secara **random** menggunakan `train_test_split` dengan `random_state=42`.
                Periode yang muncul di LIME adalah **instance pertama dari data testing** (bukan urutan periode kronologis).
                Ini berarti periode tersebut **dipilih secara acak** saat model membagi data, dan menjadi baris pertama di data testing.
                """)
                
                st.info("💡 Klik tombol di bawah untuk melihat penjelasan LIME dari ketiga model (Random Forest, SVM, dan Voting Regressor)")
                
                if st.button("🚀 Generate LIME Explanation"):
                    with st.spinner("Melatih model dan membuat penjelasan LIME..."):
                        # Train semua model
                        # Random Forest
                        best_rf = RandomForestRegressor(n_estimators=100, random_state=42)
                        best_rf.fit(X_train, y_train)
                        
                        # SVM
                        best_svm = SVR(kernel='rbf')
                        best_svm.fit(X_train, y_train)
                        
                        # Voting Regressor
                        voting_regressor = VotingRegressor([
                            ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
                            ('svm', SVR(kernel='rbf'))
                        ])
                        voting_regressor.fit(X_train, y_train)
                        
                        # Inisialisasi LIME Explainer
                        # Pastikan fitur_names sesuai dengan kolom yang digunakan dalam X_train
                        explainer = lime.lime_tabular.LimeTabularExplainer(
                            training_data=X_train.values,
                            feature_names=X_train.columns.tolist(),
                            class_names=['Jumlah Tanam'],  # Untuk regresi, kita bisa menggunakan nama target
                            mode='regression'
                        )
                        
                        # Pilih satu instance dari X_test untuk dijelaskan
                        # Misalnya, instance pertama dari X_test
                        idx_to_explain = 0
                        instance_to_explain = X_test.iloc[idx_to_explain].values
                        actual_value = y_test.iloc[idx_to_explain]
                        
                        st.success(f"✅ Instance dari Data Testing dipilih: Periode **{X_test.index[idx_to_explain]}**")
                        
                        # Tampilkan informasi detail instance
                        st.markdown("### 📊 Detail Instance yang Dijelaskan")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**Nilai Fitur:**")
                            instance_df = pd.DataFrame({
                                'Fitur': X_train.columns.tolist(),
                                'Nilai': instance_to_explain
                            })
                            st.dataframe(instance_df, hide_index=True)
                        
                        with col2:
                            st.metric("Periode", X_test.index[idx_to_explain])
                            st.metric("Nilai Aktual Jumlah Tanam", f"{actual_value:.0f}")
                            st.info(f"Instance ini adalah baris ke-{idx_to_explain + 1} dari {len(X_test)} data testing")
                        
                        st.markdown("---")
                        
                        # Penjelasan untuk Random Forest
                        st.markdown("### 🌲 Penjelasan Random Forest")
                        st.markdown("""
                        **Cara Membaca Grafik LIME:**
                        - Warna **orange (positif)**: Fitur yang meningkatkan prediksi jumlah tanam
                        - Warna **biru (negatif)**: Fitur yang menurunkan prediksi jumlah tanam
                        - Semakin panjang bar, semakin besar pengaruh fitur tersebut terhadap prediksi
                        """)
                        
                        exp_rf = explainer.explain_instance(
                            data_row=instance_to_explain,
                            predict_fn=lambda x: best_rf.predict(pd.DataFrame(x, columns=X_train.columns)),
                            num_features=len(features)
                        )
                        pred_rf_instance = best_rf.predict(pd.DataFrame([instance_to_explain], columns=X_train.columns))[0]
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Prediksi RF", f"{pred_rf_instance:.2f}")
                        with col2:
                            st.metric("Nilai Aktual", f"{actual_value:.0f}")
                        with col3:
                            selisih = abs(pred_rf_instance - actual_value)
                            st.metric("Selisih", f"{selisih:.2f}")
                        
                        # Visualisasi penjelasan LIME untuk Random Forest
                        fig_rf = exp_rf.as_pyplot_figure()
                        fig_rf.set_size_inches(10, 6)
                        plt.title(f'LIME Explanation for Random Forest (Actual: {actual_value}, Predicted: {pred_rf_instance:.2f})')
                        plt.tight_layout()
                        st.pyplot(fig_rf)
                        plt.close()
                        
                        st.markdown("---")
                        
                        # Penjelasan untuk SVM
                        st.markdown("### 🔵 Penjelasan SVM")
                        
                        exp_svm = explainer.explain_instance(
                            data_row=instance_to_explain,
                            predict_fn=lambda x: best_svm.predict(pd.DataFrame(x, columns=X_train.columns)),
                            num_features=len(features)
                        )
                        pred_svm_instance = best_svm.predict(pd.DataFrame([instance_to_explain], columns=X_train.columns))[0]
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Prediksi SVM", f"{pred_svm_instance:.2f}")
                        with col2:
                            st.metric("Nilai Aktual", f"{actual_value:.0f}")
                        with col3:
                            selisih = abs(pred_svm_instance - actual_value)
                            st.metric("Selisih", f"{selisih:.2f}")
                        
                        # Visualisasi penjelasan LIME untuk SVM
                        fig_svm = exp_svm.as_pyplot_figure()
                        fig_svm.set_size_inches(10, 6)
                        plt.title(f'LIME Explanation for SVM (Actual: {actual_value}, Predicted: {pred_svm_instance:.2f})')
                        plt.tight_layout()
                        st.pyplot(fig_svm)
                        plt.close()
                        
                        st.markdown("---")
                        
                        # Penjelasan untuk Voting Regressor
                        st.markdown("### 🗳️ Penjelasan Voting Regressor")
                        st.info("Voting Regressor menggabungkan prediksi dari Random Forest dan SVM untuk hasil yang lebih stabil")
                        
                        exp_voting = explainer.explain_instance(
                            data_row=instance_to_explain,
                            predict_fn=lambda x: voting_regressor.predict(pd.DataFrame(x, columns=X_train.columns)),
                            num_features=len(features)
                        )
                        pred_voting_instance = voting_regressor.predict(pd.DataFrame([instance_to_explain], columns=X_train.columns))[0]
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Prediksi Voting", f"{pred_voting_instance:.2f}")
                        with col2:
                            st.metric("Nilai Aktual", f"{actual_value:.0f}")
                        with col3:
                            selisih = abs(pred_voting_instance - actual_value)
                            st.metric("Selisih", f"{selisih:.2f}")
                        
                        # Visualisasi penjelasan LIME untuk Voting Regressor
                        fig_voting = exp_voting.as_pyplot_figure()
                        fig_voting.set_size_inches(10, 6)
                        plt.title(f'LIME Explanation for Voting Regressor (Actual: {actual_value}, Predicted: {pred_voting_instance:.2f})')
                        plt.tight_layout()
                        st.pyplot(fig_voting)
                        plt.close()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>🥬 Sistem Prediksi Selada | Dibuat dengan Streamlit</p>
</div>
""", unsafe_allow_html=True)
