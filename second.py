import streamlit as st
import pandas as pd
import plotly.express as px
from PIL import Image
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import colorsys

final_data_path = "C:/Users/sking/Desktop/Stream/final_data.csv"
color_data_path = "C:/Users/sking/Desktop/Stream/color_data.csv"
image_file_path = "C:/Users/sking/Desktop/Stream/dyeing.png"

st.set_page_config(layout="wide")

@st.cache_data
def load_data(file_path, encoding):
    data = pd.read_csv(file_path, encoding=encoding)
    return data

dyeing_img = Image.open(image_file_path)

def hsl_to_rgb(h, s, l):
    return tuple(round(c * 255) for c in colorsys.hls_to_rgb(h / 360, l / 100, s / 100))

def main():
    final_data = load_data(final_data_path, encoding='cp949')
    color_data = load_data(color_data_path, encoding='cp949')

    menu = st.sidebar.selectbox("Menu", ["메인", "대시보드", "모델링"])

    if menu == "메인":
        col1, col2 = st.columns([2, 3])

        with col1:
            st.title('섬유 염색')
            st.title('공정 데이터')

        with col2:
            st.image(dyeing_img, use_column_width=True)
            st.title('팀 구성')
            st.subheader('손희연, 송도훈, 표수연, 홍여림')

    elif menu == "대시보드":
        st.title("대시보드")
        st.subheader("전체 데이터 프레임")
        st.write(final_data)

        st.subheader("산점도:")
        numeric_columns = final_data.select_dtypes(include='number').columns.tolist()
        x_column = st.selectbox("x축 변수 선택", options=numeric_columns)
        y_column = st.selectbox("Y 축 변수 선택", options=['염색색차 DE'])

        if x_column and y_column:
            fig = px.scatter(final_data, x=x_column, y=y_column, title=f"Scatter Plot of {y_column} vs {x_column}")
            st.plotly_chart(fig)
    
    elif menu == "모델링":
        st.title("선형 회귀 모델")

        st.subheader("변수 선택:")
        x_variables = st.multiselect("X 축 변수 선택:", options=final_data.select_dtypes(include=['float64', 'int64']).columns.tolist())
        y_variable = '염색색차 DE'

        if st.button("모델링 시작"):
            X = final_data[x_variables]
            y = final_data[y_variable]

            model = LinearRegression()
            model.fit(X, y)

            y_pred = model.predict(X)

            fig = px.scatter(final_data, x=x_variables, y=y_variable, title=f'{y_variable} vs. {", ".join(x_variables)}')
            fig.add_scatter(x=final_data[x_variables[0]], y=y_pred, mode='lines', name='선형 회귀 예측')
            st.plotly_chart(fig)

        st.title('염색 색상 변화 시각화')

        dl_value = st.slider("염색색차 DL", min_value=color_data['염색색차 DL'].min(), max_value=color_data['염색색차 DL'].max(), value=0.0, step=0.1)
        da_value = st.slider("염색색차 DA", min_value=color_data['염색색차 DA'].min(), max_value=color_data['염색색차 DA'].max(), value=0.0, step=0.1)
        db_value = st.slider("염색색차 DB", min_value=color_data['염색색차 DB'].min(), max_value=color_data['염색색차 DB'].max(), value=0.0, step=0.1)
        dc_value = st.slider("염색색차 DC", min_value=color_data['염색색차 DC'].min(), max_value=color_data['염색색차 DC'].max(), value=0.0, step=0.1)
        dh_value = st.slider("염색색차 DH", min_value=color_data['염색색차 DH'].min(), max_value=color_data['염색색차 DH'].max(), value=0.0, step=0.1)

        def calculate_color(dl, da, db, dc, dh):
            brightness = dl / color_data['염색색차 DL'].max() * 100

            if da > 0:
                r = 255
                g = int(255 - da / color_data['염색색차 DA'].max() * 255)
            else:
                r = int(255 + da / color_data['염색색차 DA'].min() * 255)
                g = 255

            if db > 0:
                r = int(255 - db / color_data['염색색차 DB'].max() * 255)
                b = 255
            else:
                r = 255
                b = int(255 + db / color_data['염색색차 DB'].min() * 255)

            saturation = abs(dc / color_data['염색색차 DC'].max())

            hue_shift = dh / color_data['염색색차 DH'].max() * 360

            rgb_color = hsl_to_rgb(hue_shift, saturation * 100, brightness)

            hex_color = f"#{rgb_color[0]:02x}{rgb_color[1]:02x}{rgb_color[2]:02x}"

            return hex_color

        color = calculate_color(dl_value, da_value, db_value, dc_value, dh_value)

        fig_color = go.Figure(data=go.Scatter(
            x=[0], y=[0],
            marker=dict(color=color, size=200),
            mode='markers',
            hoverinfo='skip'
        ))
        st.plotly_chart(fig_color)

if __name__ == "__main__":
    main()