import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import subprocess
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, StackingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# CẤU HÌNH TRANG WEB
# ==========================================
st.set_page_config(page_title="AI MLOps | Nhận diện Khách Mua Hàng", page_icon="🚀", layout="wide")
MODEL_PATH = "stacking_purchase_model.joblib"

# ==========================================
# GIAO DIỆN ĐIỀU HƯỚNG (SIDEBAR)
# ==========================================
st.sidebar.title("⚙️ Bảng Điều Khiển MLOps")
menu = st.sidebar.radio(
    "Lựa chọn Chức năng:",
    ("🎯 1. Dự đoán Khách hàng (Inference)", 
     "📊 2. Huấn luyện Mô hình (AutoML)", 
     "☁️ 3. Đẩy lên Git (Deploy)")
)

# ==========================================
# CHỨC NĂNG 1: DỰ ĐOÁN TRỰC TIẾP
# ==========================================
if menu == "🎯 1. Dự đoán Khách hàng (Inference)":
    st.title("🎯 Công cụ Dự đoán Xác suất Chốt Đơn")
    
    if not os.path.exists(MODEL_PATH):
        st.warning("⚠️ Chưa tìm thấy mô hình. Vui lòng sang tab 'Huấn luyện' để tải dữ liệu và tạo mô hình trước.")
    else:
        model = joblib.load(MODEL_PATH)
        st.success("✅ Đã nạp mô hình Stacking AI thành công!")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📝 Nhập thông tin hành vi duyệt Web:")
            prod_duration = st.slider("Thời gian xem Sản phẩm (giây)", 0.0, 5000.0, 1500.0)
            admin_duration = st.slider("Thời gian trang Quản lý", 0.0, 1000.0, 50.0)
            bounce_rate = st.slider("Tỷ lệ thoát (Bounce Rate)", 0.0, 0.2, 0.01)
            exit_rate = bounce_rate + 0.02
            page_values = st.number_input("Giá trị trang (Page Values)", min_value=0.0, value=10.5)
            month = st.selectbox("Tháng truy cập", ["Feb", "Mar", "May", "Oct", "Nov", "Dec"], index=4)
            visitor_type = st.selectbox("Loại khách hàng", ["New_Visitor", "Returning_Visitor", "Other"])
            
        with col2:
            st.subheader("🤖 Phân tích từ hệ thống AI:")
            if st.button("🚀 Xử lý ngay", use_container_width=True):
                # 1. Tính toán feature phái sinh (Feature Engineering)
                total_duration = admin_duration + 0 + prod_duration
                prod_ratio = prod_duration / (total_duration + 1e-5)
                bounce_exit = bounce_rate * exit_rate
                has_pv = 1 if page_values > 0 else 0
                intensity = 10 / (prod_duration + 1e-5) 
                
                # 2. Đóng gói dữ liệu đầu vào
                input_df = pd.DataFrame([{
                    "Administrative": 0, "Administrative_Duration": float(admin_duration),
                    "Informational": 0, "Informational_Duration": 0.0,
                    "ProductRelated": 10, "ProductRelated_Duration": float(prod_duration),
                    "BounceRates": float(bounce_rate), "ExitRates": float(exit_rate),
                    "PageValues": float(page_values), "SpecialDay": 0.0,
                    "Month": month, "OperatingSystems": 2, "Browser": 2,
                    "Region": 1, "TrafficType": 2, "VisitorType": visitor_type, "Weekend": False,
                    "Total_Page_Duration": total_duration, "Product_Duration_Ratio": prod_ratio,
                    "Bounce_Exit_Interaction": bounce_exit, "Has_PageValues": has_pv,
                    "Session_Intensity": intensity
                }])
                
                # Ép kiểu cho mô hình
                cat_cols = ['Month', 'OperatingSystems', 'Browser', 'Region', 'TrafficType', 'VisitorType', 'Weekend']
                for c in cat_cols: input_df[c] = input_df[c].astype(str)
                
                # 3. Chạy suy luận
                prob = model.predict_proba(input_df)[0][1]
                st.metric(label="Xác suất mua hàng", value=f"{prob * 100:.2f}%")
                st.progress(float(prob))
                
                # 4. Kịch bản kinh doanh
                if prob >= 0.6: 
                    st.success("🟢 **KHÁCH HÀNG TIỀM NĂNG CAO!** \n\n Không cần gửi mã giảm giá để bảo toàn lợi nhuận.")
                elif prob >= 0.3: 
                    st.warning("🟠 **KHÁCH ĐANG PHÂN VÂN!** \n\n Tung ngay Voucher giảm 10% để kích thích chốt sale.")
                else: 
                    st.error("🔴 **KHÁCH VÃNG LAI.** \n\n Xác suất quá thấp. Hủy các chiến dịch bám đuôi (Retargeting) để tiết kiệm Ads.")

# ==========================================
# CHỨC NĂNG 2: HUẤN LUYỆN (TRAINING)
# ==========================================
elif menu == "📊 2. Huấn luyện Mô hình (AutoML)":
    st.title("📊 Huấn luyện AI với Dữ liệu Mới")
    st.info("Upload file `online_shoppers.csv` để khởi chạy tiến trình huấn luyện lại (Retrain) cho hệ thống.")
    
    uploaded_file = st.file_uploader("Tải lên file Dataset (CSV)", type=["csv"])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write(f"Đã tải lên **{df.shape[0]}** dòng dữ liệu.")
        
        if st.button("⚙️ Bắt đầu Huấn luyện Stacking AI", type="primary"):
            with st.spinner("Đang chạy Feature Engineering, chống Data Leakage và Huấn luyện mô hình..."):
                try:
                    # 1. Feature Engineering tự động
                    df = df.drop_duplicates().reset_index(drop=True)
                    df['Total_Page_Duration'] = df['Administrative_Duration'] + df['Informational_Duration'] + df['ProductRelated_Duration']
                    df['Product_Duration_Ratio'] = df['ProductRelated_Duration'] / (df['Total_Page_Duration'] + 1e-5)
                    df['Bounce_Exit_Interaction'] = df['BounceRates'] * df['ExitRates']
                    df['Has_PageValues'] = (df['PageValues'] > 0).astype(int)
                    df['Session_Intensity'] = df['ProductRelated'] / (df['ProductRelated_Duration'] + 1e-5)
                    df['Session_Intensity'] = df['Session_Intensity'].clip(upper=df['Session_Intensity'].quantile(0.99))
                    
                    X = df.drop(columns=['Revenue'])
                    y = df['Revenue'].astype(int)
                    
                    cat_cols = ['Month', 'OperatingSystems', 'Browser', 'Region', 'TrafficType', 'VisitorType', 'Weekend']
                    for c in cat_cols: X[c] = X[c].astype(str)
                    num_cols = [c for c in X.columns if c not in cat_cols]
                    
                    # 2. Stratified Split 
                    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
                    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
                    
                    preprocessor = ColumnTransformer([
                        ('num', StandardScaler(), num_cols),
                        ('cat', OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False), cat_cols)
                    ])
                    
                    # 3. Khởi tạo Stacking Classifier
                    rf = RandomForestClassifier(n_estimators=100, class_weight='balanced_subsample', max_depth=10, random_state=42, n_jobs=-1)
                    hgb = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.05, max_leaf_nodes=31, random_state=42)
                    meta = LogisticRegression(class_weight='balanced', random_state=42)
                    
                    stacking_clf = StackingClassifier(estimators=[('rf', rf), ('hgb', hgb)], final_estimator=meta, n_jobs=-1)
                    
                    # 4. Huấn luyện qua Pipeline an toàn
                    pipeline = Pipeline([('preprocessor', preprocessor), ('stacking', stacking_clf)])
                    pipeline.fit(X_train, y_train)
                    
                    # 5. Đánh giá và lưu
                    y_test_proba = pipeline.predict_proba(X_test)[:, 1]
                    pr_auc = average_precision_score(y_test, y_test_proba)
                    
                    joblib.dump(pipeline, MODEL_PATH)
                    st.success(f"✅ Huấn luyện hoàn tất! File trọng số AI đã lưu tại `{MODEL_PATH}`.")
                    st.metric("Chỉ số đo lường hiệu năng cốt lõi (PR-AUC)", f"{pr_auc:.3f}")
                    
                except Exception as e:
                    st.error(f"❌ Có lỗi trong quá trình xử lý: {e}")

# ==========================================
# CHỨC NĂNG 3: GIT DEPLOYMENT (CÓ XÁC THỰC PAT)
# ==========================================
elif menu == "☁️ 3. Đẩy lên Git (Deploy)":
    st.title("☁️ Đẩy Mô hình lên GitHub (CD/CD)")
    
    # --- CẤU HÌNH TÀI KHOẢN ---
    GITHUB_TOKEN = "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"
    GITHUB_USER = "phatpt1"
    
    st.markdown(f"Đang sử dụng cấu hình tài khoản GitHub: **@{GITHUB_USER}**")
    REPO_NAME = st.text_input("Vui lòng nhập chính xác Tên Repository của anh (ví dụ: online-shoppers-ai):", "")
    
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Không tìm thấy file `{MODEL_PATH}`. Hãy quay lại Tab 'Huấn luyện' để tạo file trước.")
    else:
        st.success(f"✅ Đã quét thấy file `{MODEL_PATH}` trong hệ thống.")
        commit_message = st.text_input("📝 Nội dung bản cập nhật (Commit):", "Cập nhật mô hình Stacking AI dữ liệu tháng mới")
        
        if st.button("🚀 Xác thực & Đẩy lên GitHub", type="primary"):
            if not REPO_NAME:
                st.warning("⚠️ Anh cần nhập tên Repository để hệ thống biết chỗ đẩy code lên!")
            else:
                with st.spinner("Đang đóng gói và gửi dữ liệu qua GitHub API..."):
                    try:
                        # 1. URL chứa Token xác thực
                        remote_url = f"https://{GITHUB_USER}:{GITHUB_TOKEN}@github.com/{GITHUB_USER}/{REPO_NAME}.git"
                        
                        # 2. Khởi tạo Git config cơ bản để tránh lỗi định danh
                        subprocess.run(["git", "config", "--global", "user.email", "mlops-bot@tam-anh.local"], check=True)
                        subprocess.run(["git", "config", "--global", "user.name", "AI Ops Bot"], check=True)
                        
                        # 3. Ghi đè URL remote để chèn Token vào
                        subprocess.run(["git", "remote", "remove", "origin"], capture_output=True) # Xóa origin cũ nếu có
                        subprocess.run(["git", "remote", "add", "origin", remote_url], check=True)
                        
                        # 4. Git Add & Commit
                        subprocess.run(["git", "add", MODEL_PATH], check=True)
                        subprocess.run(["git", "commit", "-m", commit_message], check=True)
                        
                        # 5. Push lên nhánh main (Sửa thành 'master' nếu repo của anh dùng nhánh master)
                        push_result = subprocess.run(["git", "push", "-u", "origin", "main"], check=True, capture_output=True, text=True)
                        
                        st.success("🎉 Đã tích hợp (Deploy) mô hình mới lên GitHub thành công!")
                        st.code(push_result.stdout)
                        
                    except subprocess.CalledProcessError as e:
                        st.error("❌ Lỗi cấu hình Git. Vui lòng kiểm tra kỹ lại Tên Repository hoặc xem nhánh mặc định là 'main' hay 'master'.")
                        st.code(e.stderr)