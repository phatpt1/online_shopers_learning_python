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
     "☁️ 3. Đẩy lên Git (Deploy CI/CD)",
     "📊 4. Phân tích Dữ liệu (EDA)")
)

# ==========================================
# CHỨC NĂNG 1: DỰ ĐOÁN TOÀN DIỆN (17 FEATURES)
# ==========================================
if menu == "🎯 1. Dự đoán Khách hàng (Inference)":
    st.title("🎯 Dự Báo Ý Định Chốt Đơn (Full Feature Inference)")
    st.markdown("Phân tích toàn diện **17 thuộc tính hành vi & bối cảnh** từ phiên truy cập người dùng theo chuẩn bộ dữ liệu `online_shoppers.csv`.")
    
    if not os.path.exists(MODEL_PATH):
        st.warning("⚠️ Chưa phát hiện file mô hình `stacking_purchase_model.joblib`. Vui lòng chuyển sang tab 'Tinh chỉnh & Huấn luyện' để khởi tạo mô hình.")
    else:
        model = joblib.load(MODEL_PATH)
        
        col_input, col_output = st.columns([1.2, 0.8], gap="large")
        
        with col_input:
            st.subheader("📋 Thông số Phiên Truy cập")
            
            with st.container(border=True):
                st.markdown("##### 🛍️ Trang Sản Phẩm")
                col_p1, col_p2 = st.columns(2)
                with col_p1:
                    prod_related = st.number_input("Số trang sản phẩm (ProductRelated)", min_value=0, max_value=700, value=25, step=1)
                with col_p2:
                    prod_duration = st.number_input("Thời gian xem sản phẩm (giây)", min_value=0.0, max_value=60000.0, value=1200.0, step=30.0)
            
            with st.container(border=True):
                st.markdown("##### 📑 Quản Trị & Thông Tin")
                col_a1, col_a2 = st.columns(2)
                with col_a1:
                    admin = st.number_input("Số trang quản trị (Admin)", min_value=0, max_value=30, value=2, step=1)
                    admin_duration = st.number_input("Thời gian trang quản trị (giây)", min_value=0.0, max_value=5000.0, value=45.0, step=5.0)
                with col_a2:
                    info = st.number_input("Số trang thông tin (Info)", min_value=0, max_value=30, value=0, step=1)
                    info_duration = st.number_input("Thời gian trang thông tin (giây)", min_value=0.0, max_value=3000.0, value=0.0, step=5.0)

            with st.container(border=True):
                st.markdown("##### 📉 Chỉ Số Kỹ Thuật (Page Metrics)")
                page_values = st.number_input("⭐ Giá trị trang (PageValues)", min_value=0.0, max_value=400.0, value=12.5, step=0.5)
                col_m1, col_m2 = st.columns(2)
                with col_m1:
                    bounce_rate = st.slider("Tỷ lệ thoát (BounceRates)", min_value=0.0, max_value=0.2, value=0.01, step=0.005, format="%.3f")
                with col_m2:
                    exit_rate = st.slider("Tỷ lệ rời (ExitRates)", min_value=0.0, max_value=0.2, value=0.03, step=0.005, format="%.3f")
                special_day = st.select_slider("Hệ số gần ngày lễ (SpecialDay)", options=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0], value=0.0)

            with st.container(border=True):
                st.markdown("##### 🌐 Thiết Bị & Thời Gian")
                col_c1, col_c2 = st.columns(2)
                with col_c1:
                    month = st.selectbox("Tháng truy cập (Month)", ["Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], index=8)
                    visitor_type = st.selectbox("Phân loại khách", ["Returning_Visitor", "New_Visitor", "Other"], index=0)
                    weekend = st.checkbox("Truy cập cuối tuần (Weekend)", value=False)
                with col_c2:
                    os_type = st.selectbox("Hệ điều hành", list(range(1, 9)), index=1)
                    browser = st.selectbox("Trình duyệt", list(range(1, 14)), index=1)
                    region = st.selectbox("Vùng địa lý", list(range(1, 10)), index=0)
                    traffic_type = st.selectbox("Nguồn truy cập", list(range(1, 21)), index=1)

        with col_output:
            st.subheader("🤖 Kết Quả Đánh Giá & Lý Giải")
            
            with st.container(border=True):
                btn_predict = st.button("🚀 Chạy Phân Tích Ý Định Mua Hàng", type="primary", use_container_width=True)
            
            if btn_predict:
                total_duration = float(admin_duration + info_duration + prod_duration)
                prod_duration_ratio = float(prod_duration / (total_duration + 1e-5))
                bounce_exit_interaction = float(bounce_rate * exit_rate)
                has_page_values = 1 if page_values > 0 else 0
                session_intensity = float(prod_related / (prod_duration + 1e-5))
                
                payload = {
                    'Administrative': admin, 'Administrative_Duration': float(admin_duration),
                    'Informational': info, 'Informational_Duration': float(info_duration),
                    'ProductRelated': prod_related, 'ProductRelated_Duration': float(prod_duration),
                    'BounceRates': float(bounce_rate), 'ExitRates': float(exit_rate),
                    'PageValues': float(page_values), 'SpecialDay': float(special_day),
                    'Month': str(month), 'OperatingSystems': str(os_type),
                    'Browser': str(browser), 'Region': str(region),
                    'TrafficType': str(traffic_type), 'VisitorType': str(visitor_type),
                    'Weekend': str(weekend), 'Total_Page_Duration': total_duration,
                    'Product_Duration_Ratio': prod_duration_ratio, 'Bounce_Exit_Interaction': bounce_exit_interaction,
                    'Has_PageValues': has_page_values, 'Session_Intensity': session_intensity
                }
                input_df = pd.DataFrame([payload])
                
                prob = model.predict_proba(input_df)[0][1]
                prob_pct = prob * 100
                
                st.metric("Xác suất Chốt đơn (Purchase Probability)", f"{prob_pct:.2f}%")
                st.progress(float(prob))
                
                st.markdown("### 🏷️ Đề Xuất Nghiệp Vụ Tự Động")
                if prob >= 0.70:
                    st.success("🟢 **PHÂN KHÚC: KHÁCH HÀNG TIỀM NĂNG CAO (High Intent)**\n\n**Hành động:** Giữ nguyên giá niêm yết, không hiển thị thêm voucher giảm giá để tối ưu hóa biên lợi nhuận ròng.")
                elif prob >= 0.40:
                    st.warning("🟠 **PHÂN KHÚC: KHÁCH HÀNG CÂN NHẮC / PHÂN VÂN (Hesitant Shopper)**\n\n**Hành động:** Tự động kích hoạt pop-up tặng voucher 10% hoặc miễn phí vận chuyển kèm đếm ngược 15 phút.")
                else:
                    st.error("🔴 **PHÂN KHÚC: KHÁCH HÀNG VÃNG LAI (Low Intent / Bounce)**\n\n**Hành động:** Loại trừ khỏi danh sách chiến dịch Retargeting trả phí của Google/Facebook để tiết kiệm ngân sách.")
                
                st.divider()
                st.markdown("### 🔍 Phân Tích Đóng Góp Yếu Tố")
                positive_factors = []
                negative_factors = []
                
                if page_values > 0: positive_factors.append(f"**Giá trị trang (`PageValues = {page_values}`):** Khách lướt qua các trang sinh lời mạnh.")
                else: negative_factors.append("**Không có `PageValues` (bằng 0):** Không có tín hiệu chuyển đổi từ trang.")
                if prod_duration_ratio > 0.8: positive_factors.append(f"**Tập trung sản phẩm (`{prod_duration_ratio*100:.1f}%` thời lượng):** Không bị phân tâm.")
                if month in ["Nov", "Dec"]: positive_factors.append(f"**Mùa mua sắm cao điểm (`{month}`):** Tỷ lệ chuyển đổi tự nhiên cao.")
                if bounce_rate > 0.05 or exit_rate > 0.08: negative_factors.append(f"**Tín hiệu rời trang cao (`Bounce = {bounce_rate:.3f}`, `Exit = {exit_rate:.3f}`):** Nguy cơ đóng trang lớn.")
                if total_duration < 120: negative_factors.append(f"**Thời gian lướt web ngắn (`{total_duration:.0f}s`):** Khó hình thành ý định mua.")
                
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("##### 📈 Tác nhân thúc đẩy (+)")
                    for item in positive_factors: st.write(item) if positive_factors else st.caption("Không có.")
                with c2:
                    st.markdown("##### 📉 Tác nhân kéo giảm (-)")
                    for item in negative_factors: st.write(item) if negative_factors else st.caption("Không có.")

# ==========================================
# CHỨC NĂNG 2: AUTO-ML TINH CHỈNH ĐỦ ĐẶC TRƯNG
# ==========================================
elif menu == "🎛️ 2. Tinh chỉnh & Huấn luyện (AutoML)":
    st.title("🎛️ Phòng Thí Nghiệm Huấn Luyện AI (AutoML Pipeline)")
    st.markdown("Cấu trúc mạng học máy đang sử dụng là **Stacking Ensemble** (Random Forest + HistGradientBoosting $\\rightarrow$ Logistic Regression).")
    
    if not os.path.exists(DATA_PATH):
        st.error(f"❌ Không tìm thấy file dữ liệu gốc `{DATA_PATH}` trong thư mục ứng dụng.")
    else:
        st.subheader("⚙️ Bảng Điều Khiển Siêu Tham Số (Hyperparameter Tuning)")
        with st.container(border=True):
            col_rf, col_gb = st.columns(2, gap="large")
            with col_rf:
                st.markdown("#### 🌲 Random Forest (Cơ chế Bagging)")
                rf_trees = st.slider("Quần thể cây quyết định (n_estimators)", 50, 300, 100, 25)
                st.markdown(f"> *Số lượng càng lớn, kết quả bầu chọn càng dân chủ và chống nhiễu tốt.*")
                rf_depth = st.slider("Giới hạn độ sâu phân nhánh (max_depth)", 5, 25, 12, 1)
                st.markdown(f"> *Sâu quá 20 sẽ dễ dẫn đến **Overfitting** (Học vẹt dữ liệu cũ).*")
            with col_gb:
                st.markdown("#### ⚡ HistGradientBoosting")
                gb_iters = st.slider("Số vòng lặp sửa sai (max_iter)", 50, 300, 120, 25)
                st.markdown(f"> *Cây sau sinh ra chuyên để sửa phần sai số (Residuals) của cây trước.*")
                gb_lr = st.slider("Tốc độ hội tụ (learning_rate)", 0.01, 0.20, 0.05, 0.01, format="%.2f")
                st.markdown(f"> *Nhảy chậm (0.01) thì học kỹ nhưng cần tăng vòng lặp lên bù lại.*")

        st.write("") 
        if st.button("🧠 Áp Dụng Trọng Số & Huấn Luyện Toàn Bộ Pipeline", type="primary", use_container_width=True):
            with st.spinner("Đang tiền xử lý toàn diện 17 thuộc tính, trích xuất đặc trưng và huấn luyện Stacking..."):
                try:
                    df = pd.read_csv(DATA_PATH).drop_duplicates().reset_index(drop=True)
                    
                    df['Total_Page_Duration'] = df['Administrative_Duration'] + df['Informational_Duration'] + df['ProductRelated_Duration']
                    df['Product_Duration_Ratio'] = df['ProductRelated_Duration'] / (df['Total_Page_Duration'] + 1e-5)
                    df['Bounce_Exit_Interaction'] = df['BounceRates'] * df['ExitRates']
                    df['Has_PageValues'] = (df['PageValues'] > 0).astype(int)
                    df['Session_Intensity'] = df['ProductRelated'] / (df['ProductRelated_Duration'] + 1e-5)
                    df['Session_Intensity'] = df['Session_Intensity'].clip(upper=df['Session_Intensity'].quantile(0.99))
                    
                    X = df.drop(columns=['Revenue'])
                    y = df['Revenue'].astype(int)
                    
                    for c in CAT_COLS: X[c] = X[c].astype(str)
                    
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
                    
                    num_features = [c for c in X.columns if c not in CAT_COLS]
                    preprocessor = ColumnTransformer([
                        ('num', StandardScaler(), num_features),
                        ('cat', OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False), CAT_COLS)
                    ])
                    
                    rf = RandomForestClassifier(n_estimators=rf_trees, max_depth=rf_depth, class_weight='balanced_subsample', random_state=42, n_jobs=-1)
                    hgb = HistGradientBoostingClassifier(max_iter=gb_iters, learning_rate=gb_lr, max_leaf_nodes=31, random_state=42)
                    meta = LogisticRegression(class_weight='balanced', random_state=42)
                    
                    stacking_clf = StackingClassifier(estimators=[('rf', rf), ('hgb', hgb)], final_estimator=meta, n_jobs=-1)
                    
                    pipeline = Pipeline([('preprocessor', preprocessor), ('stacking', stacking_clf)])
                    pipeline.fit(X_train, y_train)
                    
                    y_test_proba = pipeline.predict_proba(X_test)[:, 1]
                    pr_auc = average_precision_score(y_test, y_test_proba)
                    roc_auc = roc_auc_score(y_test, y_test_proba)
                    
                    joblib.dump(pipeline, MODEL_PATH)
                    
                    st.success("🎉 Cập nhật Não bộ AI thành công! File trọng số mới đã lưu đè vào `stacking_purchase_model.joblib`.")
                    with st.container(border=True):
                        st.markdown("#### 📊 Báo Cáo Hiệu Năng Khách Quan (Trên tập Test ẩn)")
                        m1, m2, m3 = st.columns(3)
                        m1.metric("PR-AUC (Chuẩn xác chốt đơn)", f"{pr_auc:.4f}")
                        m2.metric("ROC-AUC (Phân tách tổng quát)", f"{roc_auc:.4f}")
                        m3.metric("Số mẫu kiểm thử độc lập", f"{len(y_test)}")
                except Exception as e:
                    st.error(f"❌ Quá trình huấn luyện gặp lỗi: {e}")

# ==========================================
# CHỨC NĂNG 3: DEPLOY GIT SECRETS (BẢN VÁ LỖI NÂNG CAO)
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
        
        # --- QUẢN LÝ TOKEN ---
        github_token = ""
        if "GITHUB_TOKEN" in st.secrets:
            github_token = st.secrets["GITHUB_TOKEN"]
            st.success("🔒 Đã nhận diện GITHUB_TOKEN bảo mật từ Streamlit Secrets.")
        else:
            st.warning("⚠️ Chưa cấu hình Secrets. Nhập Token tạm thời bên dưới (không ghi vào file):")
            github_token = st.text_input("🔑 GitHub Personal Access Token:", type="password")

        commit_message = st.text_input("📝 Nội dung cập nhật:", "Auto-deploy: Cập nhật hệ thống MLOps")
        
        if st.button("🚀 Bắt Đầu Đẩy Dữ Liệu Lên GitHub", type="primary"):
            if not github_token: 
                st.error("❌ Thiếu Token xác thực! Tiến trình bị hủy.")
            else:
                with st.spinner("Đang khởi tạo kết nối an toàn và đẩy dữ liệu... (Vui lòng đợi)"):
                    try:
                        remote_url = f"https://{GITHUB_USER}:{github_token}@github.com/{GITHUB_USER}/{REPO_NAME}.git"
                        
                        # 1. Thiết lập danh tính
                        subprocess.run(["git", "config", "--global", "user.email", "mlops-bot@system.local"], check=True)
                        subprocess.run(["git", "config", "--global", "user.name", "MLOps Automation Bot"], check=True)
                        
                        # 2. Khởi tạo và thiết lập remote (Chặn xuất log thừa)
                        subprocess.run(["git", "init"], check=True, capture_output=True)
                        subprocess.run(["git", "remote", "remove", "origin"], capture_output=True) # Bỏ qua nếu lỗi
                        subprocess.run(["git", "remote", "add", "origin", remote_url], check=True)
                        
                        # 3. Gom file để chuẩn bị đẩy
                        subprocess.run(["git", "add", MODEL_PATH, "app_streamlit.py"], check=True)
                        
                        # 4. Kiểm tra xem có gì thay đổi để commit không (Tránh lỗi commit rỗng)
                        status_check = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
                        if not status_check.stdout.strip():
                            st.warning("⚠️ Không có sự thay đổi nào ở mã nguồn hoặc mô hình so với bản trên Git. Hệ thống hủy thao tác đẩy.")
                        else:
                            subprocess.run(["git", "commit", "-m", commit_message], check=True, capture_output=True)
                            
                            # 5. TÌM TÊN NHÁNH (TRÁNH LỖI NHẦM MAIN/MASTER)
                            branch_process = subprocess.run(["git", "branch", "--show-current"], capture_output=True, text=True)
                            current_branch = branch_process.stdout.strip()
                            if not current_branch:
                                current_branch = "main" # Nếu máy chưa có nhánh mặc định thì gán là main
                            
                            st.info(f"Đang đẩy dữ liệu qua luồng chính: nhánh `{current_branch}`")
                            
                            # 6. Đẩy file lên (Force push)
                            push_result = subprocess.run(
                                ["git", "push", "-u", "origin", current_branch, "--force"], 
                                check=True, capture_output=True, text=True
                            )
                            
                            st.success(f"🎉 Triển khai thành công lên kho `{REPO_NAME}`!")
                            st.code(push_result.stdout)
                            
                    except subprocess.CalledProcessError as e:
                        st.error("❌ Xảy ra sự cố khi kết nối với GitHub!")
                        
                        # An toàn hóa kiểu dữ liệu: ép chuyển bytes sang string (nếu có)
                        raw_err = e.stderr if e.stderr else e.stdout
                        if isinstance(raw_err, bytes):
                            error_log = raw_err.decode('utf-8', errors='ignore')
                        else:
                            error_log = str(raw_err) if raw_err else "Unknown subprocess error"
                        
                        # Hiển thị thông báo thân thiện bằng tiếng Việt
                        if "Authentication failed" in error_log or "Invalid username or token" in error_log:
                            st.error("🔑 Mật khẩu/Token của anh bị sai hoặc đã hết hạn. Hãy tạo Token mới.")
                        elif "Repository not found" in error_log:
                            st.error(f"📁 Không tìm thấy kho `{REPO_NAME}` trên tài khoản của anh. Hãy kiểm tra lại tên kho.")
                        else:
                            st.write("Chi tiết mã lỗi hệ thống:")
                            st.code(error_log)

# ==========================================
# CHỨC NĂNG 4: PHÂN TÍCH DỮ LIỆU (EDA)
# ==========================================
elif menu == "📊 4. Phân tích Dữ liệu (EDA)":
    st.title("📊 Phân Tích Dữ Liệu Khám Phá (EDA)")
    st.markdown("Chức năng này hỗ trợ đội ngũ Data hiểu rõ đặc tính của tập dữ liệu khách hàng, phát hiện các điểm bất thường và xác nhận mức độ mất cân bằng trước khi đưa vào huấn luyện AI.")
    
    if not os.path.exists(DATA_PATH):
        st.error(f"❌ Không tìm thấy file dữ liệu gốc `{DATA_PATH}` trong hệ thống.")
    else:
        df = pd.read_csv(DATA_PATH)
        
        # --- 1. Tổng quan ---
        st.subheader("1. Tổng quan Tập dữ liệu (Dataset Overview)")
        with st.container(border=True):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Tổng số dòng (Sessions)", f"{df.shape[0]:,}")
            c2.metric("Số lượng thuộc tính", df.shape[1])
            c3.metric("Dữ liệu trùng lặp (Duplicates)", df.duplicated().sum())
            c4.metric("Giá trị khuyết thiếu (Nulls)", df.isnull().sum().sum())
            
            with st.expander("👁️ Xem trước dữ liệu thô (Raw Data)"):
                st.dataframe(df.head(50))

        # --- 2. Phân phối Target ---
        st.subheader("2. Phân phối Nhãn Mục tiêu (Class Imbalance)")
        with st.container(border=True):
            # SỬA LỖI PANDAS: Ghép chung vào bảng trước, sau đó mới đổi tên index
            rev_counts = df['Revenue'].value_counts()
            rev_pct = df['Revenue'].value_counts(normalize=True) * 100
            
            df_rev = pd.DataFrame({'Số lượng': rev_counts, 'Tỷ lệ (%)': rev_pct}).round(2)
            
            # Khắc phục lỗi TypeError bằng cách gán index sau khi đã tạo DataFrame
            df_rev.index = ['Không Mua (False)', 'Có Mua (True)'] 
            
            col_chart, col_table = st.columns([2, 1], gap="large")
            with col_chart:
                # Vẽ biểu đồ dựa trên số lượng gốc để tránh lỗi
                st.bar_chart(df['Revenue'].value_counts(), color="#ff4b4b")
            with col_table:
                st.dataframe(df_rev)
            
            st.error("🚨 **Nhận xét cốt lõi:** Dữ liệu bị lệch cực kỳ nghiêm trọng (Tỷ lệ Không mua lên đến ~84.5%). Thuật toán bắt buộc phải dùng tham số phạt `class_weight='balanced'` và theo dõi bằng `PR-AUC` thay vì Accuracy (Độ chính xác chung).")

        # --- 3. Hành vi Mùa vụ & Tương tác ---
        st.subheader("3. Phân tích Hành vi & Tỷ lệ Chuyển đổi")
        tab_month, tab_metrics = st.tabs(["📅 Tính mùa vụ (Seasonality)", "⭐ Điểm giá trị & Rời bỏ (Page Metrics)"])
        
        with tab_month:
            st.markdown("**Tỷ lệ chuyển đổi (Conversion Rate) theo Tháng**")
            month_conv = df.groupby('Month')['Revenue'].mean() * 100
            month_conv = month_conv.sort_values(ascending=False)
            st.bar_chart(month_conv, color="#2e9bf5")
            st.info("💡 **Phân tích:** Tháng 11 (Nov) chứng kiến lượng chốt đơn bùng nổ do hiệu ứng mùa mua sắm cuối năm (Black Friday). Ngược lại, các tháng đầu năm (Feb, Mar) khách vào web chủ yếu để xem (Window Shopping) chứ tỷ lệ chốt đơn rất thấp.")
            
        with tab_metrics:
            st.markdown("**Sự khác biệt rõ rệt về Hành vi giữa Khách mua và Khách vãng lai**")
            mean_metrics = df.groupby('Revenue')[['BounceRates', 'ExitRates', 'PageValues']].mean()
            
            # Gán lại index cho đẹp mắt
            mean_metrics.index = ['Không Mua (False)', 'Có Mua (True)']
            
            st.dataframe(mean_metrics, use_container_width=True)
            st.success("💡 **Phân tích:** Dễ dàng nhận thấy nhóm Khách Mua Hàng có **PageValues cao gấp hàng chục lần**, đồng thời Tỷ lệ thoát/rời trang (Bounce/Exit Rates) thấp hơn hẳn. Đây chính là các `Top Drivers` (Thuộc tính đóng vai trò quyết định) giúp AI phân loại khách hàng.")
