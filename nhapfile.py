import streamlit as st
import pandas as pd
from io import StringIO
from sklearn.preprocessing import LabelEncoder
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler

# Khởi tạo session_state
if "page" not in st.session_state:
    st.session_state["page"] = "Xử lý dữ liệu"
if "data_processing" not in st.session_state:
    st.session_state["data_processing"] = pd.DataFrame()
if "data_transfer" not in st.session_state:
    st.session_state["data_transfer"] = pd.DataFrame()
def Export_Data(_data,_labels,_k, _colums):
    clustered_data = pd.DataFrame(_data)
    clustered_data['Cluster'] = _labels
    for i in range(_k):
        # Lấy dữ liệu của cụm thứ i
        cluster_i_data = clustered_data[clustered_data['Cluster'] == i].copy()
        # Chuyển đổi dữ liệu của cụm thứ i trở lại thành dữ liệu gốc
        cluster_i_data.iloc[:, :-1] = std_scaler.inverse_transform(cluster_i_data.iloc[:, :-1])
        # Đặt lại tên cột cho cluster_i_data
        cluster_i_data.columns = list(_colums) + ['Cluster']
        filename = f'cluster_{i}.csv'
        # Xuất dữ liệu ra tệp CSV
        cluster_i_data.to_csv(filename, index=False)
    st.success(f'Đã xuất dữ liệu của cụm {k} thành công.')

# Phần sidebar
st.sidebar.title("Chuyển đổi trang")
page = st.sidebar.radio("Chọn trang", ("Xử lý dữ liệu", "Chọn K cụm","Kết quả phân cụm"))
# Phần hiển thị nội dung của trang
if page == "Xử lý dữ liệu":
    st.header("Ứng dụng thuật toán K-means phân loại kết quả học tập")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        if not uploaded_file.name.endswith(".csv"):
            st.write("Xin hãy nhập tệp CSV.")
        else:
            stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
            df = pd.read_csv(stringio)
            st.dataframe(df)
            st.markdown("<h3 style='text-align: center;'>Thông tin tệp</h3>", unsafe_allow_html=True)
            st.markdown("<h5 style='text-align: left;'>Thông tin tổng quan về tệp </h5>", unsafe_allow_html=True)
            # Tạo một DataFrame mới từ thông tin cột và giá trị không rỗng của df
            info_df = pd.DataFrame({'Column': df.columns, 'Non-Null Count': df.count(), 'Dtype': df.dtypes})
            # Thêm thông tin về số lượng hàng và cột vào info_df
            info_df.loc['RangeIndex'] = [f'{len(df)} entries, 0 to {len(df) - 1}', f'total {len(df.columns)} columns', '']
            # Hiển thị info_df dưới dạng bảng trên Streamlit
            st.dataframe(info_df)
            st.markdown("<h5 style='text-align: left;'> Số lượng giá trị null trong các cột </h5>", unsafe_allow_html=True)
            # Tính toán số lượng giá trị null trong mỗi cột
            null_counts = df.isnull().sum()
            positive_values = null_counts
            positive_values_df = positive_values.to_frame(name='Số lượng null')
            # Tính phần trăm các giá trị null
            percentage = ((positive_values / len(df)) * 100)
            percentage_str = percentage.apply(lambda percentage: "{:.2f}".format(percentage))
            positive_values_df['Phần trăm'] = percentage_str
            # Định dạng bảng
            styled_table = positive_values_df.style.set_properties(**{'text-align': 'center'})
            # Hiển thị bảng
            st.write(styled_table)
            st.markdown("<br><h3 style='text-align: center;'>Xử lý dữ liệu - Tiền xử lý</h3>", unsafe_allow_html=True)
            st.markdown("<br><h6 style='text-align: left; color:#19A7CE'>&#9679; Bước 1. Xóa bỏ các cột không quan trọng </h6>", unsafe_allow_html=True)
            data_processing = df.copy()
            # Lấy danh sách tất cả các cột
            cols = list(data_processing.columns)
            # Cho phép người dùng chọn nhiều cột để xóa
            cols_to_remove = st.multiselect('Chọn cột để xóa', cols)
            data_processing = data_processing.drop(cols_to_remove, axis=1)
            # Hiển thị dữ liệu tệp
            st.dataframe(data_processing.head(5))
            st.markdown("<br><h6 style='text-align: left; color:#19A7CE'>&#9679;"
                        " Bước 2. Xử lý các cột chứa giá trị null </h6>", unsafe_allow_html=True)
            if st.button('Xử lý'):
                # Tính tỷ lệ giá trị null
                null_ratio = data_processing.isnull().sum() / len(data_processing)
                # Xóa các cột có tỷ lệ giá trị null vượt quá 25%
                cols_to_drop = null_ratio[null_ratio > 0.25].index
                data_processing = data_processing.drop(cols_to_drop, axis=1)
                # Xóa các hàng chứa giá trị null
                data_processing.dropna(axis=0, inplace=True)  # Xóa các hàng chứa giá trị NaN
                st.dataframe(data_processing.head(5))
                if data_processing.isnull().values.any() or data_processing.isna().values.any():
                    st.write('\u274C Chưa được xử lý !')
                else:
                    st.success('\u2705 Các giá trị null đã được xử lý xong.')
            st.markdown("<br><h6 style='text-align: left; color:#19A7CE'>&#9679; "
                        "Bước 3. Xử lý các cột dạng Object - cột phân loại </h6>", unsafe_allow_html=True)
            if st.button('Xử lý!'):
                # Danh sách cột thuộc kiểu categorical columns
                cat_cols = data_processing.select_dtypes(include=['object']).columns.tolist()
                encoder = LabelEncoder()
                for col in cat_cols:
                    data_processing[col + "_en"] = encoder.fit_transform(data_processing[col])
                data_processing = data_processing.drop(cat_cols, axis=1)
                cat_cols_check = data_processing.select_dtypes(include=['object']).columns.tolist()
                if len(cat_cols_check) > 0:
                    st.write('\u274C Chưa được xử lý !')
                else:
                    st.success('\u2705 Các cột dạng Object đã được xử lý xong.')
                    st.markdown("<br><h6 style='text-align: left; color:#19A7CE'>&#9679; "
                                "Dữ liệu sau khi xử lý - đây sẽ là dữ liệu dùng để vào thuật toán K-Means </h6>",
                                unsafe_allow_html=True)
                    st.dataframe(data_processing)
            # Lưu dữ liệu vào session_state
            st.session_state["df"] = df
            st.session_state["data_processing"] = data_processing
            st.session_state["page"] = "Chọn K cụm"
elif page == "Chọn K cụm":
    #Lấy dữ liệu từ session_state
    data_elbow = st.session_state["data_processing"]
    if data_elbow.empty:
        st.warning("Bạn cần chọn file dữ liệu trước")
    else:
        # Lấy danh sách các cột trong tệp dữ liệu
        columns = data_elbow.columns.tolist()
        st.markdown("<h3 style='text-align: center;'>Phương pháp Elbow - Khuỷu tay</h3>", unsafe_allow_html=True)
        st.markdown("<h6 style='text-align: left; color:#19A7CE'>&#9679; Biểu đồ khuỷu tay để tìm số cụm tối ưu</h6>", unsafe_allow_html=True)
        cols_to_selected = st.multiselect('Chọn các cột dùng để phân cụm', columns)
        if len(cols_to_selected) >= 2:
            selected_data = data_elbow.loc[:, cols_to_selected]
            data_transfer = selected_data.copy()
            data_use = selected_data.copy()
            std_scaler = StandardScaler()
            data_use[data_use.columns] = std_scaler.fit_transform(data_use)
            st.session_state["data_transfer"] = data_transfer
            try:
                WCSS = []
                for i in range(1, 11):
                    km = KMeans(n_clusters=i, init="k-means++", random_state=42)
                    km.fit(data_use)
                    WCSS.append(km.inertia_)
                fig, ax = plt.subplots(figsize=(7, 5))
                ax.plot(range(1, 11), WCSS, marker='o', color='orange')
                ax.grid(color='gray', linestyle='-', linewidth=0.25, alpha=0.5)
                ax.set_title("Elbow Plot", fontsize=16)
                ax.set_xlabel("Number of Clusters", fontsize=14)
                ax.set_ylabel("WCSS", fontsize=14)
                ax.tick_params(axis='both', which='major', labelsize=12)
                st.pyplot(fig)
                k = st.text_input('Nhập số k cụm:')
                if k:
                    try:
                        k = int(k)
                        st.success("Nhập thành công !")
                        # Lưu k vào session_state
                        st.session_state["k"] = k
                        st.session_state["page"] = "Kết quả phân cụm"
                    except ValueError:
                        st.warning("Số k cụm không hợp lệ. Vui lòng nhập lại!")
            except ValueError as e:
                st.error("Bạn phải 'xử lý dữ liệu' trước khi thực hiện thao tác này: {}".format(str(e)))
        else:
            st.write("Danh sách columns bạn chọn phải có ít nhất 2 phần tử.")

elif page == "Kết quả phân cụm":
    # Lấy dữ liệu từ session_state
    data_cluster = st.session_state["data_transfer"]
    k = st.session_state.get("k")
    if data_cluster.empty:
        st.warning("Bạn cần chọn file dữ liệu trước")
    elif k is None:
        st.warning("Bạn cần nhập số k cụm trước")
    else:
        st.markdown("<h3 style='text-align: center;'>Trực quan hóa K-means</h3>", unsafe_allow_html=True)
        # Lấy danh sách các cột trong tệp dữ liệu
        columns = data_cluster.columns.tolist()
        num_cols = st.radio("Chọn số lượng cột để biểu diễn", [2, 3], index=0)
        if num_cols == 2:
            try:
                if len(columns) >= 2:
                    selected_cols = st.multiselect("Chọn 2 cột", columns, default=[columns[0], columns[1]])
                    if len(selected_cols) < 2:
                        raise ValueError('Bạn cần chọn ít nhất 2 cột.')
                    # Trích xuất dữ liệu của các cột đã chọn
                    selected_data = data_cluster[selected_cols]
                    data = data_cluster[selected_cols].values
                    columns_ex = selected_data.columns
                    std_scaler = StandardScaler()
                    data_std = std_scaler.fit_transform(data)
                    # Áp dụng thuật toán k-means để phân cụm dữ liệu
                    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
                    kmeans.fit(data_std)
                    centers = kmeans.cluster_centers_
                    # Lấy nhãn của từng điểm dữ liệu
                    labels = kmeans.labels_

                    colors = ['blue', 'green', 'yellow', 'orange','purple', 'pink', 'gray']
                    fig, ax = plt.subplots(figsize=(8, 6))
                    for i in range(k):
                        ax.scatter(data_std[labels == i, 0], data_std[labels == i, 1], c=colors[i], s=50,
                                   label=f"Cụm {i + 1}")  # thêm label cho mỗi cụm
                    ax.scatter(centers[:, 0], centers[:, 1], c='red', s=100, alpha=0.4)
                    ax.scatter(centers[:, 0], centers[:, 1], c='red', s=110, alpha=0.4, edgecolor='red')
                    ax.set_xlabel(selected_cols[0])
                    ax.set_ylabel(selected_cols[1])
                    ax.set_title(f"Phân cụm KMeans với {k} cụm")
                    ax.legend()  # hiển thị chú thích các cụm với màu tương ứng
                    st.pyplot(fig)
                    if st.button("Xuất dữ liệu theo từng cụm"):
                        Export_Data(data_std, labels, k, columns_ex)
                else:
                    st.write("Danh sách columns phải có ít nhất 2 phần tử.")
            except ValueError as e:
                st.error(str(e))
        elif num_cols == 3:
            try:
                if len(columns) >= 3:
                    selected_cols = st.multiselect("Chọn 3 cột", columns, default=[columns[0], columns[1], columns[2]])
                    if len(selected_cols) < 3:
                        raise ValueError('Bạn cần chọn ít nhất 3 cột.')
                    # Trích xuất dữ liệu của các cột đã chọn
                    selected_data = data_cluster[selected_cols]
                    data = data_cluster[selected_cols].values
                    columns_ex = selected_data.columns
                    std_scaler = StandardScaler()
                    data_std = std_scaler.fit_transform(data)
                    # Áp dụng thuật toán k-means để phân cụm dữ liệu
                    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
                    kmeans.fit(data_std)
                    centers = kmeans.cluster_centers_
                    # Lấy nhãn của từng điểm dữ liệu
                    labels = kmeans.labels_
                    # Tính toán tọa độ của các điểm đầu và điểm cuối của các đường nối
                    x_lines = []
                    y_lines = []
                    z_lines = []
                    for i in range(len(labels)):
                        x_lines.extend([data_std[i, 0], centers[labels[i], 0], None])
                        y_lines.extend([data_std[i, 1], centers[labels[i], 1], None])
                        z_lines.extend([data_std[i, 2], centers[labels[i], 2], None])
                    # Vẽ biểu đồ
                    fig_cluster = go.Figure()
                    fig_cluster.add_trace(
                        go.Scatter3d(x=data_std[:, 0], y=data_std[:, 1], z=data_std[:, 2], mode='markers',
                                     marker=dict(size=4, color=labels, colorscale='turbo'), name='Dữ liệu')
                    )
                    fig_cluster.add_trace(
                        go.Scatter3d(x=centers[:, 0], y=centers[:, 1], z=centers[:, 2], mode='markers',
                        marker=dict(size=5, color='red', opacity=0.6), name='Tâm cụm')
                    )
                    fig_cluster.update_layout(
                        scene=dict(
                            xaxis=dict(title=selected_cols[0], showgrid=True, gridcolor='gray', gridwidth=0.6,
                                       showbackground=True, backgroundcolor='rgb(236,242,255)'),
                            yaxis=dict(title=selected_cols[1], showgrid=True, gridcolor='gray', gridwidth=0.6,
                                       showbackground=True, backgroundcolor='rgb(236,242,255)'),
                            zaxis=dict(title=selected_cols[2], showgrid=True, gridcolor='gray', gridwidth=0.6,
                                       showbackground=True, backgroundcolor='rgb(236,242,255)'),
                            camera=dict(up=dict(x=0, y=0, z=1), center=dict(x=0, y=0, z=0),
                                        eye=dict(x=-1.5, y=1.5, z=0.5)), bgcolor='#262A56'
                        ),
                        width=750, height=750
                    )
                    fig_cluster.add_trace(
                        go.Scatter3d(x=x_lines, y=y_lines, z=z_lines, mode='lines',
                                     line=dict(width=1, color='gray'), name='Đường nối')
                    )
                    st.plotly_chart(fig_cluster)
                    if st.button("Xuất dữ liệu theo từng cụm"):
                        Export_Data(data_std, labels, k, columns_ex)
                else:
                    st.write("Danh sách columns phải có ít nhất 3 phần tử.")
            except ValueError as e:
                st.error(str(e))