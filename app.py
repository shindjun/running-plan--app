import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# 페이지 설정
st.set_page_config(page_title="Advanced Running Plan App", page_icon="🏃‍♂️", layout="wide")

st.title("🏃‍♂️ Advanced Running Training Plan Generator")
st.write("사용자 데이터를 입력해 개인화된 계획을 생성하세요. Zone 2, VO2max, 속도 향상, 예상 레이스 타임 포함!")

# 사이드바
with st.sidebar:
    st.header("사용법")
    st.write("1. 기본 정보 입력 (Max HR 추가)")
    st.write("2. 최근 레이스 데이터 입력 (VO2max/예상 타임용)")
    st.write("3. 러닝 데이터 입력 (ML 학습용)")
    st.write("4. 계획 생성!")

# 사용자 입력
st.header("1. 기본 정보 입력")
col1, col2, col3, col4 = st.columns(4)
with col1:
    age = st.number_input("나이", min_value=18, max_value=80, value=30)
with col2:
    goal_distance = st.number_input("목표 거리 (km)", min_value=5.0, max_value=42.0, value=10.0)
with col3:
    current_pace = st.number_input("현재 페이스 (분/km)", min_value=3.0, max_value=10.0, value=6.0)
with col4:
    max_hr = st.number_input("Max Heart Rate (bpm)", min_value=140, max_value=220, value=180)

# 최근 레이스 입력
st.header("1.5. 최근 레이스 정보 (VO2max/예상 타임 계산용)")
recent_distance = st.number_input("최근 레이스 거리 (km, 예: 5 또는 10)", min_value=1.0, max_value=42.0, value=5.0)
recent_time_min = st.number_input("최근 레이스 타임 (분)", min_value=10.0, max_value=300.0, value=25.0)

# 러닝 데이터 입력
st.header("2. 러닝 데이터 입력 (모델 정확도 향상을 위해)")
st.write("과거 러닝 로그를 입력하세요. 형식: 거리, 페이스, 심박수")
uploaded_file = st.file_uploader("CSV 파일 업로드 (선택)", type="csv")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
else:
    num_entries = st.number_input("입력할 데이터 개수", min_value=1, max_value=10, value=3)
    running_data = []
    for i in range(num_entries):
        dist = st.number_input(f"거리 {i+1} (km)", value=5.0)
        pace = st.number_input(f"페이스 {i+1} (분/km)", value=6.0)
        heart_rate = st.number_input(f"심박수 {i+1} (bpm)", value=150)
        running_data.append([dist, pace, heart_rate])
    if running_data:
        data = pd.DataFrame(running_data, columns=["distance", "pace", "heart_rate"])

# 함수: Zone 2 Pace 계산
def calculate_zone2_pace(max_hr, current_pace):
    zone2_hr_low = max_hr * 0.60
    zone2_hr_high = max_hr * 0.70
    zone2_pace = current_pace + 1.5  # Zone 2는 느린 페이스
    return zone2_hr_low, zone2_hr_high, zone2_pace

# 함수: VO2max 추정
def estimate_vo2max(recent_distance_km, recent_time_min, age):
    distance_m = recent_distance_km * 1000
    speed_m_per_min = distance_m / recent_time_min
    vo2max = (speed_m_per_min * 0.172) + 33.3 - (0.17 * age)
    return vo2max

# 함수: Race Time 예측
def predict_race_time(recent_distance, recent_time_min, target_distance):
    exponent = 1.06
    predicted_time_min = recent_time_min * (target_distance / recent_distance) ** exponent
    return predicted_time_min

# ML 모델
@st.cache_data
def train_model(data):
    if data is None or len(data) < 2:
        np.random.seed(42)
        sample_data = pd.DataFrame({
            'distance': np.random.uniform(3, 15, 100),
            'heart_rate': np.random.uniform(120, 180, 100),
            'pace': np.random.uniform(4, 8, 100)
        })
        X = sample_data[['distance', 'heart_rate']]
        y = sample_data['pace']
    else:
        X = data[['distance', 'heart_rate']]
        y = data['pace']
    
    model = LinearRegression()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    return model, mse

if st.button("모델 학습 및 계획 생성"):
    if 'data' in locals() and len(data) > 0:
        model, mse = train_model(data)
        st.write(f"모델 정확도 (MSE): {mse:.2f}")

        # 예측 페이스
        input_features = np.array([[goal_distance, 150]])
        predicted_pace = model.predict(input_features)[0]
        st.success(f"예상 페이스 (속도 향상 후): {predicted_pace:.2f} 분/km")

        # Zone 2
        zone2_hr_low, zone2_hr_high, zone2_pace = calculate_zone2_pace(max_hr, current_pace)
        st.subheader("Zone 2 계산")
        st.write(f"Zone 2 HR: {zone2_hr_low:.0f} - {zone2_hr_high:.0f} bpm")
        st.write(f"Zone 2 Pace: {zone2_pace:.2f} 분/km")

        # VO2max
        vo2max = estimate_vo2max(recent_distance, recent_time_min, age)
        st.subheader("VO2max 추정")
        st.write(f"현재 VO2max: {vo2max:.2f} ml/kg/min")

        # 예상 레이스 타임
        st.subheader("예상 레이스 타임")
        distances = {'5K': 5, '10K': 10, 'Half Marathon': 21.1, 'Full Marathon': 42.2}
        race_predictions = {}
        for name, dist in distances.items():
            pred_time = predict_race_time(recent_distance, recent_time_min, dist)
            hours, mins = divmod(pred_time, 60)
            race_predictions[name] = f"{int(hours):02d}:{int(mins):02d}"
        pred_df = pd.DataFrame.from_dict(race_predictions, orient='index', columns=['예상 타임 (HH:MM)'])
        st.dataframe(pred_df)

        # 주간 계획
        plan_data = {
            'Day': ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
            'Type': ['Rest', 'Easy Run (Zone 2)', 'Interval (VO2max)', 'Tempo (속도)', 'Easy Run (Zone 2)', 'Long Run', 'Rest'],
            'Distance (km)': [0, 5, 8, 6, 5, goal_distance, 0],
            'Pace (분/km)': [0, zone2_pace, predicted_pace - 0.5, predicted_pace, zone2_pace, predicted_pace + 0.5, 0]
        }
        plan_df = pd.DataFrame(plan_data)
        st.subheader("개인화된 주간 계획")
        st.dataframe(plan_df)

        # 그래프
        fig, ax = plt.subplots()
        ax.bar(plan_df['Day'], plan_df['Distance (km)'])
        ax.set_title("Weekly Distance Plan")
        ax.set_ylabel("Distance (km)")
        st.pyplot(fig)

        st.write("모델이 업데이트되었습니다!")
else:
    st.info("버튼을 클릭해 계획을 생성하세요.")