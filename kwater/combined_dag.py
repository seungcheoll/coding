import pandas as pd
import psycopg2
from datetime import datetime
from airflow import DAG
from airflow.models.variable import Variable
from airflow.operators.python import PythonOperator
from airflow.operators.python import BranchPythonOperator
from joblib import load
import numpy as np
import logging
import random
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model


def generate_random_data(**kwargs):
    conn = psycopg2.connect(
        host="192.168.22.241",
        user="postgres",
        password="12345678"
    )
    query = "SELECT * FROM kwater_db4"
    df = pd.read_sql(query, conn)
    conn.close()
    #---------------------------------------------------------------lstm적용을 위한 처리#
    lstm_df=df[['탁도']]
    cutoff = {'탁도' : 0.1}
    cols = ['탁도']
    for col in cols:
        # 푸리에 변환
        frequencies = np.fft.fftfreq(len(lstm_df[col]))
        fft_values = np.fft.fft(lstm_df[col])

        # 컷오프 주파수 설정 및 필터 적용
        cutoff_frequency = cutoff.get(col, 0.1)
        fft_values[np.abs(frequencies) > cutoff_frequency] = 0

        # 역 푸리에 변환
        low_val = np.fft.ifft(fft_values).real
        lstm_df[col] = low_val
    scaler=MinMaxScaler(feature_range=(0,1)).fit(lstm_df[['탁도']].astype(float))
    #---------------------------------------------------------------lstm적용을 위한 처리#
    index_list = list(df.index)
    rand_index = random.sample(index_list, k=1)
    new_data = df.iloc[rand_index, [1, 2, 3, 4, 5, 8]].values * 0.9

    if rand_index[0]>=2:
        new_ntu_data=[[df.iloc[[rand_index[0]-2], 1].values[0]*0.99],[df.iloc[[rand_index[0]-1], 1].values[0]*1.01],[df.iloc[rand_index, 1].values[0]*0.98]]

    return new_data.tolist(), new_ntu_data, scaler


def predict_ntu(**kwargs):
    ti = kwargs['ti']
    new_ntu_data = ti.xcom_pull(task_ids="generate_random_data_task")[1]
    print(new_ntu_data)
    logging.info(new_ntu_data)
    #----------------------------------------------------------------lstm적용을 위한 처리#
    ntu_model=load_model('/home/user/airflow/models/lstm_ntu.h5')
    scaler=ti.xcom_pull(task_ids="generate_random_data_task")[2]
    ss_ntu_data=scaler.transform(new_ntu_data)
    seq_len=3
    input_ntu_data=[ss_ntu_data[-seq_len:]]
    input_ntu_data=np.array(input_ntu_data)
    ntu_pred_scaled=ntu_model.predict(input_ntu_data)
    ntu_pred=scaler.inverse_transform(ntu_pred_scaled)[0]
    #----------------------------------------------------------------lstm적용을 위한 처리#
    # ntu_model = load('/home/user/airflow/models/reg_ntu.pkl')
    # ntu_pred = ntu_model.predict(new_ntu_data)
    print('new data :', new_ntu_data)
    print('t+1 ntu :', ntu_pred)
    print(type(ntu_pred))
    return ntu_pred

    


def insert_to_db(**kwargs):
    ti = kwargs['ti']
    new_data = ti.xcom_pull(task_ids='generate_random_data_task')[0]
    ntu_pred = ti.xcom_pull(task_ids='predict_ntu_task')[0]
    new_pred_pacs = ti.xcom_pull(task_ids='load_modeling')[0]
    new_pred_cluster = ti.xcom_pull(task_ids='load_modeling')[1]
    print(new_data)
    logging.info(new_data)

     # 데이터와 타입을 로깅
    # logging.info(f"new_data: {new_data}, type: {type(new_data)}")
    # logging.info(f"ntu_pred: {ntu_pred}, type: {type(ntu_pred)}")
    # logging.info(f"new_pred_pacs: {new_pred_pacs}, type: {type(new_pred_pacs)}")
    # logging.info(f"new_pred_cluster: {new_pred_cluster}, type: {type(new_pred_cluster)}")

    conn = psycopg2.connect(
        host="192.168.22.241",
        user="postgres",
        password="12345678"
    )
    cursor = conn.cursor()
    insert_query = """
    INSERT INTO kwater_db4 ("logTime", 탁도, "pH", 수온, 전기전도도, 알칼리도, "PACS투입률", cluster, 원수유입유량, 예측탁도)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
    """

    current_time = datetime.now().strftime('%Y/%m/%d %H:%M')
    cursor.execute(insert_query, (current_time, new_data[0][0], new_data[0][1], new_data[0][2], new_data[0][3], new_data[0][4], float(new_pred_pacs[0]), int(new_pred_cluster[0]), new_data[0][5], float(ntu_pred)))

    conn.commit()
    conn.close()

def replace_missing_update_time():
    conn = psycopg2.connect(
        host="192.168.22.241",
        user="postgres",
        password="12345678"
    )

    # `update_time`에 비어있는 값이 있는지 확인합니다.
    check_missing_query = """
    SELECT update_time
    FROM kwater_db4
    WHERE update_time IS NULL OR update_time = '';
    """
    cursor = conn.cursor()
    cursor.execute(check_missing_query)
    missing_records = cursor.fetchall()

    # 비어있는 `update_time` 값이 있다면, 그 직전의 값으로 대체합니다.
    if missing_records:
        fetch_prev_query = """
        SELECT update_time
        FROM kwater_db4
        WHERE update_time IS NOT NULL AND update_time != ''
        ORDER BY update_time DESC
        LIMIT 1;
        """
        cursor.execute(fetch_prev_query)
        prev_value = cursor.fetchone()[0]

        update_missing_query = """
        UPDATE kwater_db4
        SET update_time = %s
        WHERE update_time IS NULL OR update_time = '';
        """
        cursor.execute(update_missing_query, (prev_value,))
        conn.commit()

    conn.close()


def load_modeling(**kwargs):
    ti = kwargs['ti']
    new_data = ti.xcom_pull(task_ids='generate_random_data_task')[0]
    cl_model = load('/home/user/airflow/models/classify_model.pkl')
    new_pred_cluster = cl_model.predict([new_data[0][0:5]])
    reg_model = load(f'/home/user/airflow/models/reg_clust_{new_pred_cluster[0]}.pkl')
    new_pred_pacs = reg_model.predict([new_data[0][0:5]])

    return new_pred_pacs, new_pred_cluster


default_args = {
    'start_date': datetime(2021, 1, 1),
}

with DAG(dag_id='combined_dag',
         schedule_interval="@once",  
         default_args=default_args,
         tags=['fetch_modeling'],
         catchup=False) as dag:
    
    generate_data_task = PythonOperator(
        task_id='generate_random_data_task',
        python_callable=generate_random_data
    )
    
    predict_ntu_task = PythonOperator(
        task_id='predict_ntu_task',
        python_callable=predict_ntu,
        provide_context=True
    )

    modeling_task = PythonOperator(
        task_id='load_modeling',
        python_callable=load_modeling,
        provide_context=True
    )

    insert_task = PythonOperator(
        task_id='insert_to_db',
        python_callable=insert_to_db,
        provide_context=True
    )

    apply_update_time_task = PythonOperator(
        task_id='apply_last_update_time_to_new_data',
        python_callable=replace_missing_update_time,
    )

    generate_data_task >> [predict_ntu_task, modeling_task]
    [predict_ntu_task, modeling_task] >> insert_task
    insert_task >> apply_update_time_task