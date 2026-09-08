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
from sklearn.metrics import roc_auc_score, average_precision_score
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# CẤU HÌNH TRANG WEB
# ==========================================
st.set_page_config(
    page_title="Hệ Thống Phân Tích & Dự Báo Mua Hàng | MLOps Platform",
    page_icon="🛒",
    layout="wide"
)

MODEL_PATH = "stacking_purchase_model.joblib"
DATA_PATH = "online_shoppers.csv"

# Danh sách chuẩn 17 cột gốc của dataset
CAT_COLS = ['Month', 'OperatingSystems', 'Browser', 'Region', 'TrafficType', 'VisitorType', 'Weekend']
NUM_COLS = [
    'Administrative', 'Administrative_Duration',
    'Informational', 'Informational_Duration',
    'ProductRelated', 'ProductRelated_Duration',
    'BounceRates', 'ExitRates', 'PageValues', 'SpecialDay'
]

# ==========================================
# GIAO DIỆN ĐIỀU HƯỚNG (SIDEBAR)
# ==========================================
st.sidebar.title("🛒 MLOps Control Panel")
st.sidebar.caption("Nền tảng Quản trị Mô hình & Dự báo Ý định Mua hàng")
menu = st.sidebar.radio(
    "Lựa chọn Chức năng:",
    ("🎯 1. Dự đoán Khách hàng (Inference)", 
     "🎛️ 2. Tinh chỉnh & Huấn luyện (AutoML)", 
     "☁️ 3. Đẩy lên Git (Deploy CI/CD)")
)

# ==========================================
# CHỨC NĂNG 1: DỰ ĐOÁN TOÀN DIỆN (17 FEATURES)
# ==========================================
if menu == "🎯 1. Dự đoán Khách hàng (Inference)":
    st.title("🎯 Dự Báo Ý Định Chốt Đơn (Full Feature Inference)")
    st.markdown("""
    Phân tích toàn diện **17 thuộc tính hành vi & bối cảnh** từ phiên truy cập người dùng theo chuẩn bộ dữ liệu `online_shoppers.csv`.
    """)
    
    if not os.path.exists(MODEL_PATH):
        st.warning("⚠️ Chưa phát hiện file mô hình `stacking_purchase_model.joblib`. Vui lòng chuyển sang tab 'Tinh chỉnh & Huấn luyện' để khởi tạo mô hình.")
    else:
        model = joblib.load(MODEL_PATH)
        
        # Chia layout: Bên trái nhập liệu (rộng hơn một chút), Bên phải kết quả
        col_input, col_output = st.columns([1.2, 0.8], gap="large")
        
        with col_input:
            st.subheader("📋 Thông số Phiên Truy cập")
            
            # --- Nhóm 1: Sản phẩm ---
            with st.container(border=True):
                st.markdown("##### 🛍️ Trang Sản Phẩm")
                st.caption("Đo lường mức độ tương tác với danh mục hàng hóa")
                col_p1, col_p2 = st.columns(2)
                with col_p1:
                    prod_related = st.number_input(
                        "Số trang sản phẩm (ProductRelated)", 
                        min_value=0, max_value=700, value=25, step=1
                    )
                with col_p2:
                    prod_duration = st.number_input(
                        "Thời gian xem sản phẩm (giây)", 
                        min_value=0.0, max_value=60000.0, value=1200.0, step=30.0
                    )
            
            # --- Nhóm 2: Quản trị & Thông tin ---
            with st.container(border=True):
                st.markdown("##### 📑 Quản Trị & Thông Tin")
                st.caption("Hành vi tra cứu tài khoản, hóa đơn, chính sách")
                col_a1, col_a2 = st.columns(2)
                with col_a1:
                    admin = st.number_input(
                        "Số trang quản trị (Admin)", 
                        min_value=0, max_value=30, value=2, step=1
                    )
                    admin_duration = st.number_input(
                        "Thời gian trang quản trị (giây)", 
                        min_value=0.0, max_value=5000.0, value=45.0, step=5.0
                    )
                with col_a2:
                    info = st.number_input(
                        "Số trang thông tin (Info)", 
                        min_value=0, max_value=30, value=0, step=1
                    )
                    info_duration = st.number_input(
                        "Thời gian trang thông tin (giây)", 
                        min_value=0.0, max_value=3000.0, value=0.0, step=5.0
                    )

            # --- Nhóm 3: Chỉ số Kỹ thuật ---
            with st.container(border=True):
                st.markdown("##### 📉 Chỉ Số Kỹ Thuật (Page Metrics)")
                st.caption("Đo lường rủi ro rời bỏ và giá trị lịch sử của trang")
                page_values = st.number_input(
                    "⭐ Giá trị trang (PageValues)", 
                    min_value=0.0, max_value=400.0, value=12.5, step=0.5,
                    help="Biến quan trọng nhất: Phản ánh trang mà khách duyệt qua có lịch sử đóng góp vào doanh thu hay không."
                )
                col_m1, col_m2 = st.columns(2)
                with col_m1:
                    bounce_rate = st.slider(
                        "Tỷ lệ thoát (BounceRates)", 
                        min_value=0.0, max_value=0.2, value=0.01, step=0.005, format="%.3f"
                    )
                with col_m2:
                    exit_rate = st.slider(
                        "Tỷ lệ rời (ExitRates)", 
                        min_value=0.0, max_value=0.2, value=0.03, step=0.005, format="%.3f"
                    )
                special_day = st.select_slider(
                    "Hệ số gần ngày lễ (SpecialDay)", 
                    options=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0], value=0.0
                )

            # --- Nhóm 4: Bối cảnh & Thiết bị ---
            with st.container(border=True):
                st.markdown("##### 🌐 Thiết Bị & Thời Gian")
                st.caption("Định danh môi trường và thời điểm truy cập")
                col_c1, col_c2 = st.columns(2)
                with col_c1:
                    month = st.selectbox(
                        "Tháng truy cập (Month)", 
                        ["Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], index=8
                    )
                    visitor_type = st.selectbox(
                        "Phân loại khách", 
                        ["Returning_Visitor", "New_Visitor", "Other"], index=0
                    )
                    weekend = st.checkbox("Truy cập cuối tuần (Weekend)", value=False)
                with col_c2:
                    os_type = st.selectbox("Hệ điều hành", list(range(1, 9)), index=1)
                    browser = st.selectbox("Trình duyệt", list(range(1, 14)), index=1)
                    region = st.selectbox("Vùng địa lý", list(range(1, 10)), index=0)
                    traffic_type = st.selectbox("Nguồn truy cập", list(range(1, 21)), index=1)

        # ==========================================
        # CỘT KẾT QUẢ ĐẦU RA (BÊN PHẢI)
        # ==========================================
        with col_output:
            st.subheader("🤖 Kết Quả Đánh Giá & Lý Giải")
            
            with st.container(border=True):
                btn_predict = st.button("🚀 Chạy Phân Tích Ý Định Mua Hàng", type="primary", use_container_width=True)
            
            if btn_predict:
                # 1. Tính toán 5 đặc trưng phái sinh kỹ thuật
                total_duration = float(admin_duration + info_duration + prod_duration)
                prod_duration_ratio = float(prod_duration / (total_duration + 1e-5))
                bounce_exit_interaction = float(bounce_rate * exit_rate)
                has_page_values = 1 if page_values > 0 else 0
                session_intensity = float(prod_related / (prod_duration + 1e-5))
                
                # 2. Đóng gói đầy đủ 17 cột gốc + 5 cột phái sinh = 22 cột chuẩn pipeline
                payload = {
                    'Administrative': admin,
                    'Administrative_Duration': float(admin_duration),
                    'Informational': info,
                    'Informational_Duration': float(info_duration),
                    'ProductRelated': prod_related,
                    'ProductRelated_Duration': float(prod_duration),
                    'BounceRates': float(bounce_rate),
                    'ExitRates': float(exit_rate),
                    'PageValues': float(page_values),
                    'SpecialDay': float(special_day),
                    'Month': str(month),
                    'OperatingSystems': str(os_type),
                    'Browser': str(browser),
                    'Region': str(region),
                    'TrafficType': str(traffic_type),
                    'VisitorType': str(visitor_type),
                    'Weekend': str(weekend),
                    'Total_Page_Duration': total_duration,
                    'Product_Duration_Ratio': prod_duration_ratio,
                    'Bounce_Exit_Interaction': bounce_exit_interaction,
                    'Has_PageValues': has_page_values,
                    'Session_Intensity': session_intensity
                }
                input_df = pd.DataFrame([payload])
                
                # 3. Dự đoán xác suất
                prob = model.predict_proba(input_df)[0][1]
                prob_pct = prob * 100
                
                # 4. Hiển thị điểm số
                st.metric("Xác suất Chốt đơn (Purchase Probability)", f"{prob_pct:.2f}%")
                st.progress(float(prob))
                
                # 5. Quyết định kinh doanh
                st.markdown("### 🏷️ Đề Xuất Nghiệp Vụ Tự Động")
                if prob >= 0.70:
                    st.success("🟢 **PHÂN KHÚC: KHÁCH HÀNG TIỀM NĂNG CAO (High Intent)**")
                    st.write("• **Hành động:** Giữ nguyên giá niêm yết, không hiển thị thêm voucher giảm giá để tối ưu hóa biên lợi nhuận ròng.")
                    st.write("• **Kỹ thuật:** Kích hoạt tính năng thanh toán nhanh (1-Click Checkout) hoặc gợi ý sản phẩm mua kèm (Cross-sell).")
                elif prob >= 0.40:
                    st.warning("🟠 **PHÂN KHÚC: KHÁCH HÀNG CÂN NHẮC / PHÂN VÂN (Hesitant Shopper)**")
                    st.write("• **Hành động:** Tự động kích hoạt pop-up tặng voucher 10% hoặc miễn phí vận chuyển kèm đếm ngược 15 phút.")
                    st.write("• **Kỹ thuật:** Bổ sung social proof (ví dụ: 'Đã có 42 người mua sản phẩm này trong tuần').")
                else:
                    st.error("🔴 **PHÂN KHÚC: KHÁCH HÀNG VÃNG LAI (Low Intent / Bounce)**")
                    st.write("• **Hành động:** Loại trừ khỏi danh sách chiến dịch Retargeting trả phí của Google/Facebook để cắt giảm chi phí quảng cáo rác.")
                    st.write("• **Kỹ thuật:** Thu thập dữ liệu tìm kiếm nhằm tối ưu hóa danh mục tìm kiếm SEO cơ bản.")
                
                st.divider()
                
                # 6. Lý giải mô hình
                st.markdown("### 🔍 Phân Tích Đóng Góp Yếu Tố (Feature Drivers)")
                positive_factors = []
                negative_factors = []
                
                if page_values > 0:
                    positive_factors.append(f"**Giá trị trang cao (`PageValues = {page_values}`):** Khách lướt qua các trang sinh lời mạnh, đóng góp hơn 50% vào khả năng ra quyết định mua.")
                else:
                    negative_factors.append("**Không có `PageValues` (bằng 0):** Không có tín hiệu chuyển đổi tài chính từ các trang đã ghé thăm.")
                    
                if prod_duration_ratio > 0.8:
                    positive_factors.append(f"**Tập trung vào sản phẩm (`{prod_duration_ratio*100:.1f}%` thời lượng):** Khách không bị phân tâm bởi các trang thông tin phụ.")
                    
                if month in ["Nov", "Dec"]:
                    positive_factors.append(f"**Mùa mua sắm cao điểm (`{month}`):** Tỷ lệ chuyển đổi tự nhiên của thị trường tăng mạnh.")
                    
                if bounce_rate > 0.05 or exit_rate > 0.08:
                    negative_factors.append(f"**Tín hiệu rời trang cao (`Bounce = {bounce_rate:.3f}`, `Exit = {exit_rate:.3f}`):** Nguy cơ đóng trang trước khi tới bước giỏ hàng rất lớn.")
                    
                if total_duration < 120:
                    negative_factors.append(f"**Thời gian ở lại quá ngắn (`{total_duration:.0f}s`):** Chưa đủ thời gian để tìm hiểu và hình thành ý định mua hàng.")
                
                col_drv1, col_drv2 = st.columns(2)
                with col_drv1:
                    st.markdown("##### 📈 Tác nhân thúc đẩy (+)")
                    if positive_factors:
                        for item in positive_factors:
                            st.write(item)
                    else:
                        st.caption("Không ghi nhận tín hiệu thúc đẩy nổi bật.")
                        
                with col_drv2:
                    st.markdown("##### 📉 Tác nhân kéo giảm (-)")
                    if negative_factors:
                        for item in negative_factors:
                            st.write(item)
                    else:
                        st.caption("Không phát hiện tín hiệu rủi ro nghiêm trọng.")

# ==========================================
# CHỨC NĂNG 2: AUTO-ML TINH CHỈNH ĐỦ ĐẶC TRƯNG
# ==========================================
elif menu == "🎛️ 2. Tinh chỉnh & Huấn luyện (AutoML)":
    st.title("🎛️ Phòng Thí Nghiệm Huấn Luyện AI (AutoML Pipeline)")
    st.markdown("""
    Cấu trúc mạng học máy đang sử dụng là **Stacking Ensemble**. Bằng cách kết hợp sức mạnh của *Rừng ngẫu nhiên* (Random Forest) và *Cây tăng cường* (HistGradientBoosting), sau đó dùng một *Mạng học Meta* (Logistic Regression) để quyết định cuối cùng, hệ thống sẽ tối đa hóa khả năng nhận diện ý định mua hàng.
    """)
    
    if not os.path.exists(DATA_PATH):
        st.error(f"❌ Không tìm thấy file dữ liệu gốc `{DATA_PATH}` trong thư mục ứng dụng.")
    else:
        st.subheader("⚙️ Bảng Điều Khiển Siêu Tham Số (Hyperparameter Tuning)")
        
        # Tạo khung chứa toàn bộ cài đặt trên 1 màn hình phẳng
        with st.container(border=True):
            col_rf, col_gb = st.columns(2, gap="large")
            
            # --- Cột 1: Random Forest ---
            with col_rf:
                st.markdown("#### 🌲 Thuật toán Random Forest (Cơ chế Bagging)")
                st.caption("Chiến thuật: Bắt nhầm hơn bỏ sót. Ưu tiên độ phủ (Recall).")
                
                rf_trees = st.slider(
                    "Quần thể cây quyết định (n_estimators)", 
                    min_value=50, max_value=300, value=100, step=25
                )
                st.markdown(f"> *Gồm {rf_trees} cây học độc lập. Số lượng càng lớn, kết quả bầu chọn càng dân chủ và chống nhiễu tốt, tuy nhiên sẽ làm nghẽn RAM.*")
                
                rf_depth = st.slider(
                    "Giới hạn độ sâu phân nhánh (max_depth)", 
                    min_value=5, max_value=25, value=12, step=1
                )
                st.markdown(f"> *Khống chế cây chỉ được hỏi tối đa {rf_depth} tầng câu hỏi. Sâu quá 20 sẽ dẫn đến **Overfitting** (Học vẹt dữ liệu cũ, đoán sai dữ liệu mới).*")

            # --- Cột 2: HistGradientBoosting ---
            with col_gb:
                st.markdown("#### ⚡ Thuật toán HistGradientBoosting")
                st.caption("Chiến thuật: Sai đâu sửa đó. Tối ưu độ chuẩn xác (Precision).")
                
                gb_iters = st.slider(
                    "Số vòng lặp sửa sai (max_iter)", 
                    min_value=50, max_value=300, value=120, step=25
                )
                st.markdown(f"> *Xây dựng {gb_iters} cây nối tiếp nhau. Cây sau sinh ra chuyên để sửa sai cho phần dư (Residuals) của cây trước.*")
                
                gb_lr = st.slider(
                    "Tốc độ hội tụ (learning_rate)", 
                    min_value=0.01, max_value=0.20, value=0.05, step=0.01, format="%.2f"
                )
                st.markdown(f"> *Bước nhảy Gradient Descent: {gb_lr}. Nhảy quá nhanh (0.2) dễ trượt qua điểm tối ưu. Nhảy chậm (0.01) thì học kỹ nhưng cần tăng vòng lặp lên bù lại.*")

        # --- Nút Thực thi ---
        st.write("") 
        if st.button("🧠 Áp Dụng Trọng Số & Huấn Luyện Toàn Bộ Pipeline", type="primary", use_container_width=True):
            with st.spinner("Đang tiền xử lý toàn diện 17 thuộc tính, trích xuất đặc trưng và huấn luyện Stacking..."):
                try:
                    df = pd.read_csv(DATA_PATH).drop_duplicates().reset_index(drop=True)
                    
                    # 1. Feature Engineering chuẩn hóa
                    df['Total_Page_Duration'] = df['Administrative_Duration'] + df['Informational_Duration'] + df['ProductRelated_Duration']
                    df['Product_Duration_Ratio'] = df['ProductRelated_Duration'] / (df['Total_Page_Duration'] + 1e-5)
                    df['Bounce_Exit_Interaction'] = df['BounceRates'] * df['ExitRates']
                    df['Has_PageValues'] = (df['PageValues'] > 0).astype(int)
                    df['Session_Intensity'] = df['ProductRelated'] / (df['ProductRelated_Duration'] + 1e-5)
                    df['Session_Intensity'] = df['Session_Intensity'].clip(upper=df['Session_Intensity'].quantile(0.99))
                    
                    X = df.drop(columns=['Revenue'])
                    y = df['Revenue'].astype(int)
                    
                    for c in CAT_COLS:
                        X[c] = X[c].astype(str)
                    
                    # 2. Stratified Train-Test Split (Tránh rò rỉ dữ liệu)
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.20, random_state=42, stratify=y
                    )
                    
                    # 3. Pipeline tiền xử lý hoàn chỉnh
                    num_features = [c for c in X.columns if c not in CAT_COLS]
                    preprocessor = ColumnTransformer([
                        ('num', StandardScaler(), num_features),
                        ('cat', OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False), CAT_COLS)
                    ])
                    
                    # 4. Cấu hình Stacking Classifier
                    rf = RandomForestClassifier(
                        n_estimators=rf_trees, 
                        max_depth=rf_depth, 
                        class_weight='balanced_subsample', 
                        random_state=42, 
                        n_jobs=-1
                    )
                    hgb = HistGradientBoostingClassifier(
                        max_iter=gb_iters, 
                        learning_rate=gb_lr, 
                        max_leaf_nodes=31, 
                        random_state=42
                    )
                    meta = LogisticRegression(class_weight='balanced', random_state=42)
                    
                    stacking_clf = StackingClassifier(
                        estimators=[('rf', rf), ('hgb', hgb)], 
                        final_estimator=meta, 
                        n_jobs=-1
                    )
                    
                    pipeline = Pipeline([
                        ('preprocessor', preprocessor),
                        ('stacking', stacking_clf)
                    ])
                    
                    pipeline.fit(X_train, y_train)
                    
                    # 5. Đánh giá trên tập kiểm tra độc lập
                    y_test_proba = pipeline.predict_proba(X_test)[:, 1]
                    pr_auc = average_precision_score(y_test, y_test_proba)
                    roc_auc = roc_auc_score(y_test, y_test_proba)
                    
                    joblib.dump(pipeline, MODEL_PATH)
                    
                    st.success("🎉 Cập nhật Não bộ AI thành công! File trọng số mới đã lưu đè vào `stacking_purchase_model.joblib`.")
                    
                    # Khung hiển thị điểm số
                    with st.container(border=True):
                        st.markdown("#### 📊 Báo Cáo Hiệu Năng Khách Quan (Trên tập Test ẩn)")
                        m1, m2, m3 = st.columns(3)
                        m1.metric("PR-AUC (Độ chuẩn xác nhóm chốt đơn)", f"{pr_auc:.4f}", help="Thước đo cực kỳ quan trọng đối với dữ liệu lệch nhãn (Chỉ có 15% người mua).")
                        m2.metric("ROC-AUC (Khả năng phân tách tổng quát)", f"{roc_auc:.4f}", help="Mô hình có khả năng phân loại chung đạt bao nhiêu điểm trên thang 1.0.")
                        m3.metric("Số mẫu kiểm thử độc lập (Test Samples)", f"{len(y_test)}", help="Dữ liệu này mô hình chưa từng được thấy trong quá trình học.")
                    
                except Exception as e:
                    st.error(f"❌ Quá trình huấn luyện gặp lỗi: {e}")

# ==========================================
# CHỨC NĂNG 3: DEPLOY GIT SECRETS
# ==========================================
elif menu == "☁️ 3. Đẩy lên Git (Deploy CI/CD)":
    st.title("☁️ Đẩy Mô Hình & Mã Nguồn Lên GitHub")
    
    GITHUB_USER = "phatpt1"
    REPO_NAME = "online_shopers_learning_python"
    
    st.info(f"📌 Kho lưu trữ đồng bộ: **github.com/{GITHUB_USER}/{REPO_NAME}**")
    
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Không tìm thấy file `{MODEL_PATH}`. Vui lòng chạy tab Huấn luyện trước.")
    else:
        st.success(f"✅ Đã định vị file `{MODEL_PATH}` sẵn sàng triển khai.")
        
        github_token = ""
        if "GITHUB_TOKEN" in st.secrets:
            github_token = st.secrets["GITHUB_TOKEN"]
            st.success("🔒 Đã nhận diện GITHUB_TOKEN bảo mật từ Streamlit Secrets.")
        else:
            st.warning("⚠️ Chưa cấu hình Secrets. Nhập Token tạm thời bên dưới (không ghi vào file):")
            github_token = st.text_input("🔑 GitHub Personal Access Token (classic):", type="password")

        commit_message = st.text_input("📝 Nội dung bản cập nhật (Commit message):", "Deploy: Cập nhật giao diện 17 thuộc tính và tuning siêu tham số")
        
        if st.button("🚀 Bắt Đầu Đẩy Dữ Liệu Lên GitHub", type="primary"):
            if not github_token:
                st.error("❌ Thiếu Token xác thực! Tiến trình bị hủy.")
            else:
                with st.spinner("Đang khởi tạo kết nối an toàn và đẩy dữ liệu..."):
                    try:
                        remote_url = f"https://{GITHUB_USER}:{github_token}@github.com/{GITHUB_USER}/{REPO_NAME}.git"
                        
                        subprocess.run(["git", "config", "--global", "user.email", "mlops-bot@system.local"], check=True)
                        subprocess.run(["git", "config", "--global", "user.name", "MLOps Automation Bot"], check=True)
                        
                        subprocess.run(["git", "init"], check=True, capture_output=True)
                        subprocess.run(["git", "remote", "remove", "origin"], capture_output=True) 
                        subprocess.run(["git", "remote", "add", "origin", remote_url], check=True)
                        
                        subprocess.run(["git", "add", MODEL_PATH, "app_streamlit.py"], check=True)
                        subprocess.run(["git", "commit", "-m", commit_message], check=True, capture_output=True)
                        push_result = subprocess.run(["git", "push", "-u", "origin", "main", "--force"], check=True, capture_output=True, text=True)
                        
                        st.success(f"🎉 Triển khai thành công lên kho `{REPO_NAME}`!")
                        st.code(push_result.stdout)
                        
                    except subprocess.CalledProcessError as e:
                        st.error("❌ Lỗi khi thực thi lệnh Git. Vui lòng kiểm tra quyền hạn của Token.")
                        st.code(e.stderr)
