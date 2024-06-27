from flask import Flask, render_template, request, jsonify, send_file, abort, redirect, url_for
from io import BytesIO
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph, Image, Table, TableStyle
from PIL import Image
from openai import OpenAI
from datetime import time, datetime, timedelta
import numpy as np
import pandas as pd
import re
import os
import plotly.express as px
import plotly.io as pio
import time
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import and_
from pathlib import Path
import calendar
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout

basedir = os.path.abspath(os.path.dirname(__file__))

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] =\
        'sqlite:///' + os.path.join(basedir, 'database', 'cust_data.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

CORS(app)

plot_figure = None

statistics = {
    'OilTemp': {'nama': 'Suhu Oli'},
    'BusTemp1': {'nama': 'Suhu Busbar Fase U'},
    'BusTemp2': {'nama': 'Suhu Busbar Fase V'},
    'BusTemp3': {'nama': 'Suhu Busbar Fase W'},
    'WTITemp1': {'nama': 'Winding Temperature Fase U'},
    'WTITemp2': {'nama': 'Winding Temperature Fase V'},
    'WTITemp3': {'nama': 'Winding Temperature Fase W'},
    'Press': {'nama': 'Tank Pressure'},
    'Level': {'nama': 'Oil Level'},
    'Van': {'nama': 'Tegangan dari fase U ke fase netral'},
    'Vbn': {'nama': 'Tegangan dari fase V ke fase netral'},
    'Vcn': {'nama': 'Tegangan dari fase W ke fase netral'},
    # 'Vab': {'nama': 'Tegangan dari Fase U ke Fase V'},
    # 'Vbc': {'nama': 'Tegangan dari Fase V ke Fase W'},
    # 'Vca': {'nama': 'Tegangan dari Fase W ke Fase U'},
    'Ia': {'nama': 'Arus Fase U'},
    'Ib': {'nama': 'Arus Fase V'},
    'Ic': {'nama': 'Arus Fase W'},
    'Pa': {'nama': 'Daya Aktif Fase U'},
    'Pb': {'nama': 'Daya Aktif Fase V'},
    'Pc': {'nama': 'Daya Aktif Fase W'},
    'Qa': {'nama': 'Daya Reaktif Fase U'},
    'Qb': {'nama': 'Daya Reaktif Fase V'},
    'Qc': {'nama': 'Daya Reaktif Fase W'},
    'Sa': {'nama': 'Daya Semu Fase U'},
    'Sb': {'nama': 'Daya Semu Fase V'},
    'Sc': {'nama': 'Daya Semu Fase W'},
    'PFa': {'nama': 'Power Factor Fase U'},
    'PFb': {'nama': 'Power Factor Fase V'},
    'PFc': {'nama': 'Power Factor Fase W'},
    'Freq': {'nama': 'Frekuensi'},
    'Ineutral': {'nama': 'Arus Netral'},
    'kWhInp': {'nama': 'Active Energy'},
    'kVARhinp': {'nama': 'Reactive Energy'},
    'THDV1': {'nama': 'Total Harmonic Distortion Tegangan Fase U'},
    'THDV2': {'nama': 'Total Harmonic Distortion Tegangan Fase V'},
    'THDV3': {'nama': 'Total Harmonic Distortion Tegangan Fase W'},
    'THDI1': {'nama': 'Total Harmonic Distortion Arus Fase U'},
    'THDI2': {'nama': 'Total Harmonic Distortion Arus Fase V'},
    'THDI3': {'nama': 'Total Harmonic Distortion Arus Fase W'},
    'KRateda': {'nama': 'K-Rated Fase U'},
    'KRatedb': {'nama': 'K-Rated Fase V'},
    'KRatedc': {'nama': 'K-Rated Fase W'},
    'deRatinga': {'nama': 'deRating Fase U'},
    'deRatingb': {'nama': 'deRating Fase V'},
    'deRatingc': {'nama': 'deRating Fase W'},
}

not_used = []

press_type = 0 # 0 = konservator (0 semua), 1 konservator (ada yg > 0), 2
press_value = {}

progress = 0


class TransformerData(db.Model):
    nama = db.Column(db.String(255), primary_key=True)
    serial_number = db.Column(db.String(255))
    impedance = db.Column(db.Float)
    rated_power = db.Column(db.Float)
    frequency = db.Column(db.Float)
    rated_high_voltage = db.Column(db.Float)
    rated_low_voltage = db.Column(db.Float)
    rated_current_high_voltage = db.Column(db.Float)
    rated_current_low_voltage = db.Column(db.Float)
    vector_group = db.Column(db.String(255))
    phase = db.Column(db.Integer)
    no_load_loss = db.Column(db.Float)
    full_load_loss = db.Column(db.Float)
    top_oil_temp_rise_lv = db.Column(db.Float)
    top_oil_temp_rise_hv = db.Column(db.Float)
    avg_winding_temp_rise_lv = db.Column(db.Float)
    avg_winding_temp_rise_hv = db.Column(db.Float)
    gradient_lv = db.Column(db.Float)
    gradient_hv = db.Column(db.Float)
    cooling_mode = db.Column(db.String(255))
    ct_ratio = db.Column(db.Float)
    hotspot_factor = db.Column(db.Float)
    k_rated = db.Column(db.Integer)

    def __repr__(self):
        return f'<{self.nama}>'

class TransformerSettings(db.Model):
    nama = db.Column(db.String(255), primary_key=True)
    voltage_low_trip = db.Column(db.Float)
    voltage_low_alarm = db.Column(db.Float)
    voltage_high_alarm = db.Column(db.Float)
    voltage_high_trip = db.Column(db.Float)
    freq_low_trip = db.Column(db.Float)
    freq_low_alarm = db.Column(db.Float)
    freq_high_alarm = db.Column(db.Float)
    freq_high_trip = db.Column(db.Float)
    thdi_trip = db.Column(db.Float)
    thdi_alarm = db.Column(db.Float)
    thdv_alarm = db.Column(db.Float)
    thdv_trip = db.Column(db.Float)
    top_oil_high_alarm = db.Column(db.Float)
    top_oil_high_trip = db.Column(db.Float)
    wti_high_alarm = db.Column(db.Float)
    wti_high_trip = db.Column(db.Float)
    pf_low_alarm = db.Column(db.Float)
    pf_low_trip = db.Column(db.Float)
    current_high_alarm = db.Column(db.Float)
    current_high_trip = db.Column(db.Float)
    i_neutral_high_alarm = db.Column(db.Float)
    i_neutral_high_trip = db.Column(db.Float)
    bustemp_high_alarm = db.Column(db.Float)
    bustemp_high_trip = db.Column(db.Float)
    pressure_high_alarm = db.Column(db.Float)
    pressure_high_trip = db.Column(db.Float)
    unbalance_high_alarm = db.Column(db.Float)
    unbalance_high_trip = db.Column(db.Float)

    def __repr__(self):
        pass

def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def create_new_page(pdf_canvas, page):
    if page == 0:
        add_bg(pdf_canvas, 0)
        add_page_number(pdf_canvas)
    elif page == 2:
        pdf_canvas.showPage()
        pdf_canvas.setPageSize(landscape(A4))
        add_bg(pdf_canvas, 1)
    else:
        pdf_canvas.showPage()
        add_bg(pdf_canvas, 0)
        add_page_number(pdf_canvas)

def draw_image_on_canvas(p, plot_data, x, y, width, height):
    img_data = pio.to_image(plot_data, format='png')
    p.drawImage(ImageReader(BytesIO(img_data)), x, y, width=width, height=height)
    p.setStrokeColor(colors.Color(0, 0, 0, alpha=0.5))
    p.rect(x, y, width, height)

def generate_answers(p, p_answer, p_sum_answer, y):
    text_y = y

    style = ParagraphStyle('CustomStyle')
    style.alignment = 4  # 4 = justify
    style.fontSize = 12
    style.leading = 16
    
    p_text = Paragraph(str(p_answer), style)
    par_height = p_text.wrap(410, 999999)[1]
    text_y -= par_height
    p_text.drawOn(p, 90, text_y)

    summary_style = ParagraphStyle('CustomStyle')
    summary_style.alignment = 1 # 1 = center
    summary_style.fontSize = 12
    summary_style.leading = 16
    summary_style.fontName = 'Helvetica-Oblique'

    if p_sum_answer != ' ':
        p_sum_answer_text = str(p_sum_answer)
        p_text2 = Paragraph(f'"{p_sum_answer_text}"', summary_style)
        par_height = p_text2.wrap(410, 999999)[1]
        text_y -= par_height + 12
        p_text2.drawOn(p, 90, text_y)

def draw_judul_img(p, text, image_x, image_width, judulimg_y):
    text_width = p.stringWidth(text, "Helvetica-Oblique", 10)
    judulimg_x = image_x + (image_width - text_width) / 2
    p.setFont("Helvetica-Oblique", 10)
    p.drawString(judulimg_x, judulimg_y, text)

def draw_table(p, min, max, avg, y):
    data = [
        ['Max', 'Min', 'Average'],
        [max, min, avg]
    ]
    col_widths = [50, 50, 50]
    t = Table(data, colWidths=col_widths)
    style = TableStyle([('BACKGROUND', (0, 0), (-1, 0), '#4472C4'),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                    ('INNERGRID', (0, 0), (-1, -1), 0.25, colors.black),
                    ('BOX', (0, 0), (-1, -1), 0.25, colors.black),
                    ('FONT', (0, 0), (-1, -1), 'Helvetica', 8),
                    ('LEADING', (0, 0), (-1, -1), 10)
                    ])
    t.setStyle(style)
    t.wrapOn(p, 0, 0)
    table_width = sum(t._colWidths)
    x_center = (p._pagesize[0] - table_width) / 2
    t.drawOn(p, x_center, y)
        
def draw_table3(p, min1, min2, min3, max1, max2, max3, avg1, avg2, avg3, y):
    data = [
        ['Max (Fase U)', 'Min (Fase U)', 'Avg (Fase U)',
            'Max (Fase V)','Min (Fase V)', 'Avg (Fase V)',
            'Max (Fase W)', 'Min (Fase W)', 'Avg (Fase W)'],
        [max1, min1, avg1, max2, min2, avg2, max3, min3, avg3]
    ]
    col_widths = [45, 45, 47, 45, 45, 47, 45, 45, 47]
    t = Table(data, colWidths=col_widths)
    style = TableStyle([('BACKGROUND', (0, 0), (-1, 0), '#4472C4'),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                    ('INNERGRID', (0, 0), (-1, -1), 0.25, colors.black),
                    ('BOX', (0, 0), (-1, -1), 0.25, colors.black),
                    ('FONT', (0, 0), (-1, -1), 'Helvetica', 7),
                    ('LEADING', (0, 0), (-1, -1), 10)
                    ])
    t.setStyle(style)
    t.wrapOn(p, 0, 0)
    table_width = sum(t._colWidths)
    x_center = (p._pagesize[0] - table_width) / 2
    t.drawOn(p, 90, y)

def add_page_number(canvas):
    page_num = canvas.getPageNumber()
    text = f"{page_num}"
    canvas.setFont("Helvetica", 8)
    canvas.drawCentredString(A4[0] / 2, 25, text)

def add_bg(p, page):
    if page == 0:
        bg_width, bg_height = 595, 842 
        image_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bg", "bg_update.png")
        with Image.open(image_path) as img:
            img.thumbnail((120, 120))
            p.drawImage(image_path, 0, 0, width=bg_width, height=bg_height, preserveAspectRatio=True)
    else:
        bg_width, bg_height = 842, 595 
        image_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bg", "bg_lampiran.png")
        with Image.open(image_path) as img:
            img.thumbnail((120, 120))
            p.drawImage(image_path, 0, 0, width=bg_width, height=bg_height, preserveAspectRatio=True)

def get_days_in_month(date_str):
    year, month = map(int, date_str.split('-'))
    return calendar.monthrange(year, month)[1]

def add_one_month(date_str):
    year, month = map(int, date_str.split('-'))
    if month == 12:
        year += 1
        month = 1
    else:
        month += 1
    return f"{year:04d}-{month:02d}"

def forecast_data(name, date):
    current_path = os.getcwd()
    directory = current_path + '/uploads'
    folder_path = f'{directory}/{name}/{date}'
    all_data = pd.DataFrame()
    for file in sorted(os.listdir(folder_path)):
        if file.endswith('.xlsx') or file.endswith('.xls'):
            file_path = os.path.join(folder_path, file)
            data = pd.read_excel(file_path)
            data.rename(columns={data.columns[-1]: "Status"}, inplace=True)
            tanggal_file = re.findall(r'\d{4}\d{2}\d{2}', file)[0]
            data['timestamp'] = pd.to_datetime(tanggal_file, format='%Y%m%d') + pd.to_timedelta(data['timestamp'].astype(str))
            all_data = pd.concat([all_data, data], ignore_index=True)
    daily_data = all_data[['timestamp', 'kWhInp']]
    daily_data['usage'] = daily_data['kWhInp'] - daily_data['kWhInp'].shift(1)
    daily_data = daily_data[['timestamp', 'usage']]
    daily_data.set_index('timestamp', inplace=True)
    daily_data = daily_data.resample('D').sum()
    
    forecast_result = forecast(name, daily_data, 10, get_days_in_month(add_one_month(date)), add_one_month(date))
    total_usage = forecast_result['Predicted'].sum()
    
    return total_usage

def preprocess_data(data, look_back):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    X, y = [], []
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i-look_back:i, 0])
        y.append(scaled_data[i, 0])
        
    X, y = np.array(X), np.array(y)
    if X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError("Not enough data points to create X and y.")
    
    return X, y, scaler

def build_model(look_back):
    model = Sequential()
    
    model.add(LSTM(units = 50, return_sequences = True, input_shape = (look_back, 1)))
    model.add(Dropout(0.2))

    # Adding a second LSTM layer and some Dropout regularisation
    model.add(LSTM(units = 50, return_sequences = True))
    model.add(Dropout(0.2))

    # Adding a third LSTM layer and some Dropout regularisation
    model.add(LSTM(units = 50, return_sequences = True))
    model.add(Dropout(0.2))

    # Adding a fourth LSTM layer and some Dropout regularisation
    model.add(LSTM(units = 50))
    model.add(Dropout(0.2))

    # Adding the output layer
    model.add(Dense(units = 1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_and_save_model(name):
    current_path = os.getcwd()
    directory = current_path + '/uploads'
    all_data = pd.DataFrame()
    if name in sorted(os.listdir(directory)):
        for folder in sorted(os.listdir(os.path.join(directory, name))):
            folder_path = os.path.join(directory, name, folder)
            for file in sorted(os.listdir(folder_path)):
                if file.endswith('.xlsx') or file.endswith('.xls'):
                    file_path = os.path.join(folder_path, file)
                    data = pd.read_excel(file_path)
                    data.rename(columns={data.columns[-1]: "Status"}, inplace=True)
                    tanggal_file = re.findall(r'\d{4}\d{2}\d{2}', file)[0]
                    data['timestamp'] = pd.to_datetime(tanggal_file, format='%Y%m%d') + pd.to_timedelta(data['timestamp'].astype(str))
                    all_data = pd.concat([all_data, data], ignore_index=True)
        daily_data = all_data[['timestamp', 'kWhInp']]
        daily_data['usage'] = daily_data['kWhInp'] - daily_data['kWhInp'].shift(1)
        daily_data = daily_data[['timestamp', 'usage']]
        daily_data.set_index('timestamp', inplace=True)
        daily_data = daily_data.resample('D').sum()
        
        X, y, scaler = preprocess_data(daily_data, 10)
    
        model = build_model(10)
        model.fit(X, y, epochs=25, batch_size=32)
        models_directory = current_path + '/models'
        model_save_path = os.path.join(models_directory, f'{name}_model.keras')
        model.save(model_save_path)
    else:
        print('Tidak ada')

def forecast(name, recent_data, look_back, forecast_days, date):
    current_path = os.getcwd()
    models_directory = current_path + '/models'
    model_load_path = os.path.join(models_directory, f'{name}_model.keras')
    model = load_model(model_load_path)
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    recent_data_scaled = scaler.fit_transform(recent_data)
    
    X_new = []
    for i in range(look_back, len(recent_data_scaled)):
        X_new.append(recent_data_scaled[i-look_back:i])
    
    X_new = np.array(X_new)
    X_new = X_new.reshape((X_new.shape[0], look_back, 1))

    predictions = []

    for _ in range(forecast_days):
        predicted_price = model.predict(X_new[-1].reshape(1, look_back, 1))
        predictions.append(predicted_price[0, 0])
        
        new_input = np.append(X_new[-1, 1:, 0], predicted_price[0, 0])
        X_new = np.vstack([X_new, new_input.reshape(1, look_back, 1)])

    predictions = np.array(predictions).reshape(-1, 1)
    predicted = scaler.inverse_transform(predictions)
    
    start_date = datetime.strptime(date, "%Y-%m")
    date_range = [start_date + timedelta(days=i) for i in range(forecast_days)]

    forecast_dates = pd.to_datetime(date_range)
    df_result_new = pd.DataFrame({'Date': forecast_dates.strftime('%Y-%m-%d'), 'Predicted': predicted.flatten()})
    
    return df_result_new


@app.route('/')
def home():
    tdata = TransformerData.query.with_entities(TransformerData.nama).distinct().all()

    return render_template('home.html', tdata=tdata)

@app.route('/upload')
def upload():
    tdata = TransformerData.query.with_entities(TransformerData.nama).distinct().all()
    return render_template('upload.html', tdata=tdata)

@app.route('/download')
def download():
    tdata = TransformerData.query.with_entities(TransformerData.nama).distinct().all()
    return render_template('download.html', tdata=tdata)

@app.route('/list_files', methods=['GET'])
def list_files():
    files = []
    selected_company = request.args.get('selectedCompany')
    date = request.args.get('date')
    current_path = os.getcwd()
    folder_path = current_path + '/uploads'
    customer_folder_path = os.path.join(folder_path, selected_company)
    date_folder_path = os.path.join(customer_folder_path, date)
    if os.path.exists(date_folder_path):
        files = [f for f in os.listdir(date_folder_path) if os.path.isfile(os.path.join(date_folder_path, f))]
    return jsonify({'files': files})

@app.route('/upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        selected_company = request.form.get('selectedCompany')
        date = request.form.get('date')
        uploaded_files = request.files.getlist('excel_file')
        if uploaded_files != []:
            current_path = os.getcwd()
            folder_path = current_path + '/uploads'
            company_folder_path = os.path.join(folder_path, selected_company)
            date_folder_path = os.path.join(company_folder_path, date)
            if not os.path.exists(date_folder_path):
                os.makedirs(date_folder_path)

            for uploaded_file in uploaded_files:
                tanggal_file = re.findall(r'\d{4}\d{2}\d{2}', uploaded_file.filename)[0]
                file_date = f"{tanggal_file[:4]}-{tanggal_file[4:6]}"
                if file_date != date:
                    return jsonify({'message': 'date_error'})
                file_path = os.path.join(date_folder_path, uploaded_file.filename)
                uploaded_file.save(file_path)
                
            train_and_save_model(selected_company)

            return jsonify({'message': 'success'})

        else:
            return jsonify({'message': 'error'})
        
@app.route('/get-data', methods=['POST'])
def get_data():
    selected_company = request.form.get('selectedCompany')
    # date = request.form.get('date')
    # transformer_data = TransformerData.query.filter(and_(TransformerData.nama == selected_company, TransformerData.date == date)).order_by(TransformerData.no.desc()).first()
    # transformer_settings = TransformerSettings.query.filter(and_(TransformerSettings.nama == selected_company, TransformerSettings.date == date)).order_by(TransformerSettings.no.desc()).first()
    transformer_data = TransformerData.query.filter(TransformerData.nama == selected_company).first()
    transformer_settings = TransformerSettings.query.filter(TransformerSettings.nama == selected_company).first()
    if transformer_data and transformer_settings:
        data_dict = transformer_data.__dict__
        settings_dict = transformer_settings.__dict__
        data_dict.pop('_sa_instance_state', None)
        settings_dict.pop('_sa_instance_state', None)
        combined_data = {'tdata': data_dict, 'tsettings': settings_dict}
        return jsonify({'status': 'success', 'data': combined_data})
    else:
        return jsonify({'status': 'error'})
    
@app.route('/get-latest-data', methods=['POST'])
def get_latest_data():
    selected_company = request.form.get('selectedCompany')
    transformer_data = TransformerData.query.filter(TransformerData.nama == selected_company).order_by(TransformerData.date.desc(), TransformerData.no.desc()).first()
    transformer_settings = TransformerSettings.query.filter(TransformerSettings.nama == selected_company).order_by(TransformerSettings.date.desc(), TransformerSettings.no.desc()).first()
    if transformer_data and transformer_settings:
        date = transformer_data.date
        data_dict = transformer_data.__dict__
        settings_dict = transformer_settings.__dict__
        data_dict.pop('_sa_instance_state', None)
        settings_dict.pop('_sa_instance_state', None)
        combined_data = {'tdata': data_dict, 'tsettings': settings_dict}
        return jsonify({'status': 'success', 'data': combined_data, 'date': date})
    else:
        return jsonify({'status': 'error'})
    
@app.route('/update-data', methods=['POST'])
def update_data():
    changed_values = request.json.get('changedValues')
    selected_company = request.json.get('selectedCompany')
    
    for item in changed_values:
        if item['data'] == 'tdata':
            transformer_data = TransformerData.query.filter_by(nama=selected_company).first()
            setattr(transformer_data, item['par'], item['value'])
        elif item['data'] == 'tsettings':
            transformer_settings = TransformerSettings.query.filter_by(nama=selected_company).first()
            setattr(transformer_settings, item['par'], item['value'])

    db.session.commit() 
    
    transformer_data = TransformerData.query.filter_by(nama=selected_company).first()
    transformer_settings = TransformerSettings.query.filter_by(nama=selected_company).first()

    if transformer_data and transformer_settings:
        data_dict = transformer_data.__dict__
        settings_dict = transformer_settings.__dict__
        data_dict.pop('_sa_instance_state', None)
        settings_dict.pop('_sa_instance_state', None)
        combined_data = {'tdata': data_dict, 'tsettings': settings_dict}
        return jsonify(combined_data)
    else:
        return jsonify({'error': 'Data not found'}), 404

@app.route('/insert-data', methods=['POST'])
def insert_data():
    selected_company = request.json.get('selectedCompany')
    type = request.json.get('type')

    tdata = request.json.get('tdata')
    tsettings = request.json.get('tsettings')

    new_transformer_data = TransformerData(
        nama=selected_company,
        avg_winding_temp_rise_hv=tdata.get('avg_winding_temp_rise_hv'),
        avg_winding_temp_rise_lv=tdata.get('avg_winding_temp_rise_lv'),
        cooling_mode=tdata.get('cooling_mode'),
        ct_ratio=tdata.get('ct_ratio'),
        frequency=tdata.get('frequency'),
        full_load_loss=tdata.get('full_load_loss'),
        gradient_hv=tdata.get('gradient_hv'),
        gradient_lv=tdata.get('gradient_lv'),
        hotspot_factor=tdata.get('hotspot_factor'),
        impedance=tdata.get('impedance'),
        k_rated=tdata.get('k_rated'),
        no_load_loss=tdata.get('no_load_loss'),
        phase=tdata.get('phase'),
        rated_current_high_voltage=tdata.get('rated_current_high_voltage'),
        rated_current_low_voltage=tdata.get('rated_current_low_voltage'),
        rated_high_voltage=tdata.get('rated_high_voltage'),
        rated_low_voltage=tdata.get('rated_low_voltage'),
        rated_power=tdata.get('rated_power'),
        serial_number=tdata.get('serial_number'),
        top_oil_temp_rise_hv=tdata.get('top_oil_temp_rise_hv'),
        top_oil_temp_rise_lv=tdata.get('top_oil_temp_rise_lv'),
        vector_group=tdata.get('vector_group')
    )
    new_transformer_settings = TransformerSettings(
        nama=selected_company,
        voltage_low_trip=tsettings.get('voltage_low_trip'),
        voltage_low_alarm=tsettings.get('voltage_low_alarm'),
        voltage_high_alarm=tsettings.get('voltage_high_alarm'),
        voltage_high_trip=tsettings.get('voltage_high_trip'),
        freq_low_trip=tsettings.get('freq_low_trip'),
        freq_low_alarm=tsettings.get('freq_low_alarm'),
        freq_high_alarm=tsettings.get('freq_high_alarm'),
        freq_high_trip=tsettings.get('freq_high_trip'),
        thdi_trip=tsettings.get('thdi_trip'),
        thdi_alarm=tsettings.get('thdi_alarm'),
        thdv_alarm=tsettings.get('thdv_alarm'),
        thdv_trip=tsettings.get('thdv_trip'),
        top_oil_high_alarm=tsettings.get('top_oil_high_alarm'),
        top_oil_high_trip=tsettings.get('top_oil_high_trip'),
        wti_high_alarm=tsettings.get('wti_high_alarm'),
        wti_high_trip=tsettings.get('wti_high_trip'),
        pf_low_alarm=tsettings.get('pf_low_alarm'),
        pf_low_trip=tsettings.get('pf_low_trip'),
        current_high_alarm=tsettings.get('current_high_alarm'),
        current_high_trip=tsettings.get('current_high_trip'),
        i_neutral_high_alarm=tsettings.get('i_neutral_high_alarm'),
        i_neutral_high_trip=tsettings.get('i_neutral_high_trip'),
        bustemp_high_alarm=tsettings.get('bustemp_high_alarm'),
        bustemp_high_trip=tsettings.get('bustemp_high_trip'),
        pressure_high_alarm=tsettings.get('pressure_high_alarm'),
        pressure_high_trip=tsettings.get('pressure_high_trip'),
        unbalance_high_alarm=tsettings.get('unbalance_high_alarm'),
        unbalance_high_trip=tsettings.get('unbalance_high_trip')
    )
    
    status = ''

    if type == 'new':
        transformer_data = TransformerData.query.filter_by(nama=selected_company).first()
        transformer_settings = TransformerSettings.query.filter_by(nama=selected_company).first()

        if transformer_data and transformer_settings:
            status = 'edit'
        else:
            db.session.add(new_transformer_data)
            db.session.commit()

            db.session.add(new_transformer_settings)
            db.session.commit()
            
            status = 'success'
    # else:
    #     db.session.add(new_transformer_data)
    #     db.session.commit()

    #     db.session.add(new_transformer_settings)
    #     db.session.commit()
    #     status = 'success'

    return jsonify({'status': status})
        
@app.route('/delete_files', methods=['POST'])
def delete_files():
    files_to_delete = request.json.get('files')
    selected_company = request.json.get('selectedCompany')
    date = request.json.get('date')
    
    current_path = os.getcwd()
    folder_path = current_path + '/uploads'
    company_folder_path = os.path.join(folder_path, selected_company)
    date_folder_path = os.path.join(company_folder_path, date)
    for file_name in files_to_delete:
        try:
            os.remove(os.path.join(date_folder_path, file_name))
        except FileNotFoundError:
            pass
    return jsonify({'message': 'success'})

@app.route('/delete_all_files', methods=['POST'])
def delete_all_files():
    selected_company = request.json.get('selectedCompany')
    date = request.json.get('date')
    current_path = os.getcwd()
    folder_path = current_path + '/uploads'
    company_folder_path = os.path.join(folder_path, selected_company)
    date_folder_path = os.path.join(company_folder_path, date)
    for file_name in os.listdir(date_folder_path):
        file_path = os.path.join(date_folder_path, file_name)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f"Error deleting file {file_path}: {e}")
            return jsonify({'message': 'error'})
    
    return jsonify({'message': 'success'})

@app.route('/check', methods=['POST'])
def check():
    selected_company = request.form.get('selectedCompany')
    date = request.form.get('date')

    transformer_data = TransformerData.query.filter(TransformerData.nama == selected_company).first()
    transformer_settings = TransformerSettings.query.filter(TransformerSettings.nama == selected_company).first()

    if not transformer_data and not transformer_settings:
        return jsonify({'message': 'Tranformer data and settings have not been inputed'})
    
    current_path = os.getcwd()
    folder_path = current_path + '/uploads'
    customer_folder_path = os.path.join(folder_path, selected_company)
    date_folder_path = os.path.join(customer_folder_path, date)
    if not os.path.exists(date_folder_path):
        return jsonify({'message': 'Data have not been uploaded'})
    files_in_folder = os.listdir(date_folder_path)
    if not files_in_folder:
        return jsonify({'message': 'Data have not been uploaded'})
    
    return jsonify({'message': 'success'})

@app.route('/make-pdf', methods=['POST'])
def make_pdf():
    global progress
    progress = 0
    total_task = 50
    current_progress = (1/total_task)*100
    start_time = time.time()
    selected_company = request.form.get('selectedCompany')
    date = request.form.get('date')
    current_path = os.getcwd()

    folder_path = current_path + '/uploads'
    customer_folder_path = os.path.join(folder_path, selected_company)
    date_folder_path = os.path.join(customer_folder_path, date)

    transformer_data = TransformerData.query.filter(TransformerData.nama == selected_company).first()
    transformer_settings = TransformerSettings.query.filter(TransformerSettings.nama == selected_company).first()

    # Datalog
    status = ''
    directory = os.path.join(os.path.dirname(__file__), date_folder_path)
    excel_files = [file for file in os.listdir(directory) if file.endswith('.xlsx')]
    excel_files.sort()
    combined_data = pd.DataFrame()
    start_time_read_data = time.time()
    for file in excel_files:
        try:
            data = pd.read_excel(os.path.join(directory, file))
            data.rename(columns={data.columns[-1]: "Status"}, inplace=True)
            tanggal_file = re.findall(r'\d{4}\d{2}\d{2}', file)[0]
            data['timestamp'] = pd.to_datetime(tanggal_file, format='%Y%m%d') + pd.to_timedelta(data['timestamp'].astype(str))
            if data['timestamp'].iloc[0].strftime('%Y-%m') != date:
                status = 'error'
                break
            combined_data = pd.concat([combined_data, data], ignore_index=True)
            progress += current_progress/len(excel_files)
        except Exception as e:
            print(f"Error reading {file}: {e}")
    end_time_read_data = time.time()

    data = combined_data

    if status == 'error':
        return abort(400, description=f"File {file} does not match the given date {date}")

    app.bulan = calendar.month_name[int(date.split('-')[1])]
    app.tahun = str(date.split('-')[0])

    # Statistics Data
    start_time_calculate_data = time.time()
    
    for column in statistics.keys():
        if ((data[column] == 0).all() == False):
            if column == 'Press':
                total_count = len(data[column])
                zero_count = (data[column] == 0.0).sum()
                if zero_count > total_count/2:
                    press_type = 1
                    filtered_values = data[data[column] > 0.0]['Press']
                    press_value = filtered_values.tolist()
                    not_used.append(column)
                    statistics[column].update({
                        'total_times': '-',
                        'hour_max_freq': '-',
                        'mean': '-',
                        'std': '-',
                        'avg_diff': '-',
                        'min': '-',
                        'max': '-',
                        'mode': '-',
                        'q1': '-',
                        'q2': '-',
                        'q3': '-',
                        'skewness': '-',
                        'cv': '-',
                        'threshold': '-',
                        'proporsi': '-',
                        'persentase': '-',
                    })
                else:
                    press_type = 2
                    data['selisih_waktu'] = data['timestamp'].diff()
                    data['total_time'] = data['selisih_waktu'].where(data[column].shift(-1) == data[column].max()).fillna(pd.Timedelta(0)).cumsum()
                    total_times = data['total_time'].iloc[-1]
                    max_value = data[column].max()
                    data_max = data[data[column] == max_value]
                    freq = data_max['timestamp'].dt.hour.value_counts()
                    jam_max = freq.idxmax()
                    mean_value = round(data[column].mean(), 2)
                    std_value = round(data[column].std(), 2)
                    avg_diff = round(np.mean(np.abs(data[column] - np.mean(data[column]))), 2)
                    min_value = round(data[column].min(), 2)
                    max_value = round(data[column].max(), 2)
                    mode_value = round(data[column].mode()[0], 2)
                    q1_value = round(data[column].quantile(0.25), 2)
                    q2_value = round(data[column].quantile(0.50), 2)
                    q3_value = round(data[column].quantile(0.75), 2)
                    skewness = round(data[column].skew(), 2)
                    cv = round((data[column].std() / data[column].mean()) * 100, 2)
                    threshold = q3_value
                    proporsi = round((data[column] >= threshold).sum() / len(data), 2)
                    persentase = round(proporsi * 100, 2)
                    print('calculating...')

                    statistics[column].update({
                        'total_times': total_times,
                        'hour_max_freq': jam_max,
                        'mean': mean_value,
                        'std': std_value,
                        'avg_diff': avg_diff,
                        'min': min_value,
                        'max': max_value,
                        'mode': mode_value,
                        'q1': q1_value,
                        'q2': q2_value,
                        'q3': q3_value,
                        'skewness': skewness,
                        'cv': cv,
                        'threshold': threshold,
                        'proporsi': proporsi,
                        'persentase': persentase,
                    })
            else:
                data['selisih_waktu'] = data['timestamp'].diff()
                data['total_time'] = data['selisih_waktu'].where(data[column].shift(-1) == data[column].max()).fillna(pd.Timedelta(0)).cumsum()

                total_times = data['total_time'].iloc[-1]
                max_value = data[column].max()
                data_max = data[data[column] == max_value]
                freq = data_max['timestamp'].dt.hour.value_counts()
                jam_max = freq.idxmax()
                mean_value = round(data[column].mean(), 2)
                std_value = round(data[column].std(), 2)
                avg_diff = round(np.mean(np.abs(data[column] - np.mean(data[column]))), 2)
                min_value = round(data[column].min(), 2)
                max_value = round(data[column].max(), 2)
                mode_value = round(data[column].mode()[0], 2)
                q1_value = round(data[column].quantile(0.25), 2)
                q2_value = round(data[column].quantile(0.50), 2)
                q3_value = round(data[column].quantile(0.75), 2)
                skewness = round(data[column].skew(), 2)
                cv = round((data[column].std() / data[column].mean()) * 100, 2)
                threshold = q3_value
                proporsi = round((data[column] >= threshold).sum() / len(data), 2)
                persentase = round(proporsi * 100, 2)
                print('calculating...')

                statistics[column].update({
                    'total_times': total_times,
                    'hour_max_freq': jam_max,
                    'mean': mean_value,
                    'std': std_value,
                    'avg_diff': avg_diff,
                    'min': min_value,
                    'max': max_value,
                    'mode': mode_value,
                    'q1': q1_value,
                    'q2': q2_value,
                    'q3': q3_value,
                    'skewness': skewness,
                    'cv': cv,
                    'threshold': threshold,
                    'proporsi': proporsi,
                    'persentase': persentase,
                })

        else:
            not_used.append(column)
            statistics[column].update({
                'total_times': '-',
                'hour_max_freq': '-',
                'mean': '-',
                'std': '-',
                'avg_diff': '-',
                'min': '-',
                'max': '-',
                'mode': '-',
                'q1': '-',
                'q2': '-',
                'q3': '-',
                'skewness': '-',
                'cv': '-',
                'threshold': '-',
                'proporsi': '-',
                'persentase': '-',
            })
            if column == 'Press':
                press_type = 0
        progress += current_progress/len(statistics.keys())
        
    end_time_calculate_data = time.time()

    start_time_making_plot = time.time()
    # Oil Temp Plot
    data['Smoothed_OilTemp'] = data['OilTemp'].rolling(window=100, min_periods=1).mean()
    fig_oiltemp = px.line(data, x='timestamp', y='Smoothed_OilTemp', title=f'{app.bulan} Oil Temperature Chart')
    fig_oiltemp.update_layout(title_x=0.5, title_font=dict(size=20))
    fig_oiltemp.update_traces(mode='markers+lines', marker=dict(size=5))
    fig_oiltemp.update_xaxes(
        title_text='timestamp',
        tickvals=[data['timestamp'].min(), data['timestamp'].max()],
        range=[data['timestamp'].min(), data['timestamp'].max()],
        dtick='D1',
        tickformat='%Y-%m-%d'
    )
    fig_oiltemp.update_yaxes(title_text='temperature')
    app.plot_oiltemp = fig_oiltemp
    progress += current_progress

    # BusTemp Plot
    fig_bustemp = px.line(data, x='timestamp', y=['BusTemp1', 'BusTemp2', 'BusTemp3'], title=f'{app.bulan} Busbar Temperature Chart',
              labels={'value': 'temperature', 'variable': 'BusBar'})
    fig_bustemp.update_layout(title_x=0.5, title_font=dict(size=20))
    fig_bustemp.update_traces(mode='lines')
    fig_bustemp.update_xaxes(
        title_text='timestamp',
        tickvals=[data['timestamp'].min(), data['timestamp'].max()],
        range=[data['timestamp'].min(), data['timestamp'].max()],
        dtick='D1',
        tickformat='%Y-%m-%d'
    )
    app.plot_bustemp = fig_bustemp
    progress += current_progress

    # WTITemp Plot
    fig_wtitemp = px.line(data, x='timestamp', y=['WTITemp1', 'WTITemp2', 'WTITemp3'], title=f'{app.bulan} Winding Temperature Chart',
              labels={'value': 'temperature', 'variable': 'WTI'})
    fig_wtitemp.update_layout(title_x=0.5, title_font=dict(size=20))
    fig_wtitemp.update_traces(mode='lines')
    fig_wtitemp.update_xaxes(
        title_text='timestamp',
        tickvals=[data['timestamp'].min(), data['timestamp'].max()],
        range=[data['timestamp'].min(), data['timestamp'].max()],
        dtick='D1',
        tickformat='%Y-%m-%d'
    )
    app.plot_wtitemp = fig_wtitemp
    progress += current_progress

    # Pressure Plot
    data['Smoothed_Press'] = data['Press'].rolling(window=100, min_periods=1).mean()
    fig_press = px.line(data, x='timestamp', y='Smoothed_Press', title=f'{app.bulan} Tank Pressure Chart')
    fig_press.update_layout(title_x=0.5, title_font=dict(size=20))
    fig_press.update_traces(mode='lines', marker=dict(size=5))
    fig_press.update_xaxes(
        title_text='timestamp',
        tickvals=[data['timestamp'].min(), data['timestamp'].max()],
        range=[data['timestamp'].min(), data['timestamp'].max()],
        dtick='D1',
        tickformat='%Y-%m-%d'
    )
    fig_press.update_yaxes(title_text='tank pressure')
    app.plot_press = fig_press
    # data.drop(columns=['Smoothed_Press'], inplace=True)
    progress += current_progress

    # Level Plot
    data['Smoothed_Level'] = data['Level'].rolling(window=100, min_periods=1).mean()
    fig_level = px.line(data, x='timestamp', y='Level', title=f'{app.bulan} Oil Level Chart')
    fig_level.update_layout(title_x=0.5, title_font=dict(size=20))
    fig_level.update_traces(mode='lines', marker=dict(size=5))
    fig_level.update_xaxes(
        title_text='timestamp',
        tickvals=[data['timestamp'].min(), data['timestamp'].max()],
        range=[data['timestamp'].min(), data['timestamp'].max()],
        dtick='D1',
        tickformat='%Y-%m-%d'
    )
    fig_level.update_yaxes(title_text='oil level')
    app.plot_level = fig_level
    #data.drop(columns=['Smoothed_Level'], inplace=True)
    progress += current_progress

    # Voltage Plot
    fig_voltage = px.line(data, x='timestamp', y=['Van', 'Vbn', 'Vcn'], title=f'{app.bulan} Voltage Chart',
              labels={'value': 'voltage', 'variable': 'voltage'})
    fig_voltage.update_layout(title_x=0.5, title_font=dict(size=20))
    fig_voltage.update_traces(mode='lines')
    fig_voltage.update_xaxes(
        title_text='timestamp',
        tickvals=[data['timestamp'].min(), data['timestamp'].max()],
        range=[data['timestamp'].min(), data['timestamp'].max()],
        dtick='D1',
        tickformat='%Y-%m-%d'
    )
    app.plot_voltage = fig_voltage
    progress += current_progress

    # Arus Plot
    fig_arus = px.line(data, x='timestamp', y=['Ia', 'Ib', 'Ic'], title=f'{app.bulan} Current Chart',
              labels={'value': 'current', 'variable': 'current'})
    fig_arus.update_layout(title_x=0.5, title_font=dict(size=20))
    fig_arus.update_traces(mode='lines')
    fig_arus.update_xaxes(
        title_text='timestamp',
        tickvals=[data['timestamp'].min(), data['timestamp'].max()],
        range=[data['timestamp'].min(), data['timestamp'].max()],
        dtick='D1',
        tickformat='%Y-%m-%d'
    )
    app.plot_arus = fig_arus
    progress += current_progress

    # Active Power Plot
    fig_p = px.line(data, x='timestamp', y=['Pa', 'Pb', 'Pc'], title=f'{app.bulan} Active Power Chart',
              labels={'value': 'power', 'variable': 'active power'})
    fig_p.update_layout(title_x=0.5, title_font=dict(size=20))
    fig_p.update_traces(mode='lines')
    fig_p.update_xaxes(
        title_text='timestamp',
        tickvals=[data['timestamp'].min(), data['timestamp'].max()],
        range=[data['timestamp'].min(), data['timestamp'].max()],
        dtick='D1',
        tickformat='%Y-%m-%d'
    )
    app.plot_p = fig_p
    progress += current_progress

    # Reactive Power Plot
    fig_q = px.line(data, x='timestamp', y=['Qa', 'Qb', 'Qc'], title=f'{app.bulan} Reactive Power Chart',
              labels={'value': 'power', 'variable': 'reactive power'})
    fig_q.update_layout(title_x=0.5, title_font=dict(size=20))
    fig_q.update_traces(mode='lines')
    fig_q.update_xaxes(
        title_text='timestamp',
        tickvals=[data['timestamp'].min(), data['timestamp'].max()],
        range=[data['timestamp'].min(), data['timestamp'].max()],
        dtick='D1',
        tickformat='%Y-%m-%d'
    )
    app.plot_q = fig_q
    progress += current_progress

    # Apparent Power Plot
    fig_s = px.line(data, x='timestamp', y=['Sa', 'Sb', 'Sc'], title=f'{app.bulan} Apparent Power Chart',
              labels={'value': 'power', 'variable': 'apparent power'})
    fig_s.update_layout(title_x=0.5, title_font=dict(size=20))
    fig_s.update_traces(mode='lines')
    fig_s.update_xaxes(
        title_text='timestamp',
        tickvals=[data['timestamp'].min(), data['timestamp'].max()],
        range=[data['timestamp'].min(), data['timestamp'].max()],
        dtick='D1',
        tickformat='%Y-%m-%d'
    )
    app.plot_s = fig_s
    progress += current_progress

    # Power Factor Plot
    fig_pf = px.line(data, x='timestamp', y=['PFa', 'PFb', 'PFc'], title=f'{app.bulan} Power Factor Chart',
              labels={'value': 'power', 'variable': ' '})
    fig_pf.update_layout(title_x=0.5, title_font=dict(size=20))
    fig_pf.update_traces(mode='lines')
    fig_pf.update_xaxes(
        title_text='timestamp',
        tickvals=[data['timestamp'].min(), data['timestamp'].max()],
        range=[data['timestamp'].min(), data['timestamp'].max()],
        dtick='D1',
        tickformat='%Y-%m-%d'
    )
    app.plot_pf = fig_pf
    progress += current_progress

    # Freq Plot
    fig_freq = px.line(data, x='timestamp', y='Freq', title=f'{app.bulan} Frequency Chart')
    fig_freq.update_layout(title_x=0.5, title_font=dict(size=20))
    fig_freq.update_traces(mode='lines', marker=dict(size=5))
    fig_freq.update_xaxes(
        title_text='timestamp',
        tickvals=[data['timestamp'].min(), data['timestamp'].max()],
        range=[data['timestamp'].min(), data['timestamp'].max()],
        dtick='D1',
        tickformat='%Y-%m-%d'
    )
    fig_freq.update_yaxes(title_text='freq')
    app.plot_freq = fig_freq
    progress += current_progress

    # Arus Netral Plot
    fig_ineutral = px.line(data, x='timestamp', y='Ineutral', title=f'{app.bulan} Neutral Current Chart')
    fig_ineutral.update_layout(title_x=0.5, title_font=dict(size=20))
    fig_ineutral.update_traces(mode='lines', marker=dict(size=5))
    fig_ineutral.update_xaxes(
        title_text='timestamp',
        tickvals=[data['timestamp'].min(), data['timestamp'].max()],
        range=[data['timestamp'].min(), data['timestamp'].max()],
        dtick='D1',
        tickformat='%Y-%m-%d'
    )
    fig_ineutral.update_yaxes(title_text='neutral current')
    app.plot_ineutral = fig_ineutral
    progress += current_progress

    # Active Energy Plot
    fig_kwh = px.line(data, x='timestamp', y='kWhInp', title=f'{app.bulan} Active Energy Chart')
    fig_kwh.update_layout(title_x=0.5, title_font=dict(size=20))
    fig_kwh.update_traces(mode='lines', marker=dict(size=5))
    fig_kwh.update_xaxes(
        title_text='timestamp',
        tickvals=[data['timestamp'].min(), data['timestamp'].max()],
        range=[data['timestamp'].min(), data['timestamp'].max()],
        dtick='D1',
        tickformat='%Y-%m-%d'
    )
    fig_kwh.update_yaxes(title_text='energy')
    app.plot_kwh = fig_kwh
    progress += current_progress


    # Reactive Energy Plot
    fig_kvarh = px.line(data, x='timestamp', y='kVARhinp', title=f'{app.bulan} Reactive Energy Chart')
    fig_kvarh.update_layout(title_x=0.5, title_font=dict(size=20))
    fig_kvarh.update_traces(mode='lines', marker=dict(size=5))
    fig_kvarh.update_xaxes(
        title_text='timestamp',
        tickvals=[data['timestamp'].min(), data['timestamp'].max()],
        range=[data['timestamp'].min(), data['timestamp'].max()],
        dtick='D1',
        tickformat='%Y-%m-%d'
    )
    fig_kvarh.update_yaxes(title_text='energy')
    app.plot_kvarh = fig_kvarh
    progress += current_progress


    # THDV Plot
    fig_thdv = px.line(data, x='timestamp', y=['THDV1', 'THDV2', 'THDV3'], title=f'{app.bulan} Total Harmonic Distortion Voltage Chart',
              labels={'value': 'thdv', 'variable': ' '})
    fig_thdv.update_layout(title_x=0.5, title_font=dict(size=20))
    fig_thdv.update_traces(mode='lines')
    fig_thdv.update_xaxes(
        title_text='timestamp',
        tickvals=[data['timestamp'].min(), data['timestamp'].max()],
        range=[data['timestamp'].min(), data['timestamp'].max()],
        dtick='D1',
        tickformat='%Y-%m-%d'
    )
    app.plot_thdv = fig_thdv
    progress += current_progress

    # THDI Plot
    fig_thdi = px.line(data, x='timestamp', y=['THDI1', 'THDI2', 'THDI3'], title=f'{app.bulan} Total Harmonic Distortion Current Chart',
              labels={'value': 'thdi', 'variable': ' '})
    fig_thdi.update_layout(title_x=0.5, title_font=dict(size=20))
    fig_thdi.update_traces(mode='lines')
    fig_thdi.update_xaxes(
        title_text='timestamp',
        tickvals=[data['timestamp'].min(), data['timestamp'].max()],
        range=[data['timestamp'].min(), data['timestamp'].max()],
        dtick='D1',
        tickformat='%Y-%m-%d'
    )
    app.plot_thdi = fig_thdi
    progress += current_progress

    # kRated Plot
    fig_krated = px.line(data, x='timestamp', y=['KRateda', 'KRatedb', 'KRatedc'], title=f'{app.bulan} kRated Chart',
              labels={'value': 'krated', 'variable': ' '})
    fig_krated.update_layout(title_x=0.5, title_font=dict(size=20))
    fig_krated.update_traces(mode='lines')
    fig_krated.update_xaxes(
        title_text='timestamp',
        tickvals=[data['timestamp'].min(), data['timestamp'].max()],
        range=[data['timestamp'].min(), data['timestamp'].max()],
        dtick='D1',
        tickformat='%Y-%m-%d'
    )
    app.plot_krated = fig_krated
    progress += current_progress

    # deRating Plot
    fig_derating = px.line(data, x='timestamp', y=['deRatinga', 'deRatingb', 'deRatingc'], title=f'{app.bulan} deRating Chart',
              labels={'value': 'derating', 'variable': ' '})
    fig_derating.update_layout(title_x=0.5, title_font=dict(size=20))
    fig_derating.update_traces(mode='lines')
    fig_derating.update_xaxes(
        title_text='timestamp',
        tickvals=[data['timestamp'].min(), data['timestamp'].max()],
        range=[data['timestamp'].min(), data['timestamp'].max()],
        dtick='D1',
        tickformat='%Y-%m-%d'
    )
    app.plot_derating = fig_derating
    progress += current_progress

    end_time_making_plot = time.time()

    parameters = ['OilTemp', 'BusTemp', 'WTITemp', 'Press', 'Level', 'V', 'I', 'P', 'Q', 'S', 'PF', 'Freq', 'Ineutral', 'kVARhinp', 'THDV', 'THDI', 'KRated', 'deRating']
    # parameters = ['OilTemp', 'BusTemp', 'WTITemp', 'Press', 'Level', 'V', 'I', 'P', 'Q', 'S', 'PF', 'Freq', 'Ineutral', 'kWhInp', 'kVARhinp', 'THDV', 'THDI', 'KRated', 'deRating']

    start_time_making_prompt = time.time()

    prompts = {}

    for parameter in parameters:
        print(parameter)
        if parameter in not_used or parameter+'1' in not_used or parameter+'2' in not_used or parameter+'3' in not_used or parameter+'a' in not_used or parameter+'b' in not_used or parameter+'c' in not_used or parameter+'ab' in not_used or parameter+'bc' in not_used or parameter+'ca' in not_used:
            prompts[parameter] = ' '
        else:
            if parameter == 'BusTemp' or parameter == 'WTITemp' or parameter == 'THDV' or parameter == 'THDI':
                if parameter == 'BusTemp':
                    par_prompt = f"Trafo tersebut memiliki setting busbar temperature high alarm sebesar {transformer_settings.bustemp_high_alarm} dan busbar temperature high trip sebesar {transformer_settings.bustemp_high_trip}. "
                elif parameter == 'WTITemp':
                    par_prompt = f"Trafo tersebut memiliki setting WTI high alarm sebesar {transformer_settings.wti_high_alarm} dan WTI high trip sebesar {transformer_settings.wti_high_trip}. "
                elif parameter == 'THDV':
                    par_prompt = f"Trafo tersebut memiliki setting THDV alarm sebesar {transformer_settings.thdv_alarm} dan THDV trip sebesar {transformer_settings.thdv_trip}. "
                elif parameter == 'THDI':
                    par_prompt = f"Trafo tersebut memiliki setting THDI alarm sebesar {transformer_settings.thdi_alarm} dan THDI trip sebesar {transformer_settings.thdi_trip}. "
                else:
                    par_prompt = f"Trafo tersebut memiliki setting {parameter} maksimal sebagai berikut: {statistics[parameter+'1']['nama']} sebesar {statistics[parameter+'1']['max']}, {statistics[parameter+'2']['nama']} sebesar {statistics[parameter+'2']['max']}, {statistics[parameter+'3']['nama']} sebesar {statistics[parameter+'3']['max']}."
                prompt = (
                    f"Trafo ini memiliki nilai rata-rata {statistics[parameter+'1']['nama']} sebesar {statistics[parameter+'1']['mean']} selama sebulan, nilai maksimal {statistics[parameter+'1']['nama']} sebesar {statistics[parameter+'1']['max']}, dan nilai minimum {statistics[parameter+'1']['nama']} sebesar {statistics[parameter+'1']['min']}. "
                    f"Lalu, nilai rata-rata {statistics[parameter+'2']['nama']} sebesar {statistics[parameter+'2']['mean']} selama sebulan, nilai maksimal {statistics[parameter+'2']['nama']} sebesar {statistics[parameter+'2']['max']}, dan nilai minimum {statistics[parameter+'2']['nama']} sebesar {statistics[parameter+'2']['min']}. "
                    f"Dan, nilai rata-rata {statistics[parameter+'3']['nama']} sebesar {statistics[parameter+'3']['mean']} selama sebulan, nilai maksimal {statistics[parameter+'3']['nama']} sebesar {statistics[parameter+'3']['max']}, dan nilai minimum {statistics[parameter+'3']['nama']} sebesar {statistics[parameter+'3']['min']}. "
                    f"{par_prompt}"
                    f"Buatlah analisis deskriptif dalam satu paragraf terkait keadaan trafo tersebut, serta berikan statement apakah kondisi trafo tersebut normal atau tidak berdasarkan data tersebut."
                )
                prompts[parameter] = prompt
            elif parameter == 'I' or parameter == 'P' or parameter == 'Q' or parameter == 'S' or parameter == 'PF' or parameter == 'KRated' or parameter == 'deRating':
                if parameter == 'I':
                    par_prompt = f"Trafo tersebut memiliki setting current high alarm sebesar {transformer_settings.current_high_alarm} dan current high trip sebesar {transformer_settings.current_high_trip}."
                elif parameter == 'S':
                    par_prompt = f"Trafo tersebut memiliki setting rated power sebesar {transformer_data.rated_power}."
                elif parameter == 'PF':
                    par_prompt = f"Trafo tersebut memiliki setting power factor low alarm sebesar {transformer_settings.pf_low_alarm} dan power factor low trip sebesar {transformer_settings.pf_low_trip}."
                else:
                    par_prompt = f"Trafo tersebut memiliki setting {parameter} maksimal sebagai berikut: {statistics[parameter+'a']['nama']} sebesar {statistics[parameter+'a']['max']}, {statistics[parameter+'b']['nama']} sebesar {statistics[parameter+'b']['max']}, {statistics[parameter+'c']['nama']} sebesar {statistics[parameter+'c']['max']}."
                    

                prompt = (
                    f"Trafo ini memiliki nilai rata-rata {statistics[parameter+'a']['nama']} sebesar {statistics[parameter+'a']['mean']} selama sebulan, nilai maksimal {statistics[parameter+'a']['nama']} sebesar {statistics[parameter+'a']['max']}, dan nilai minimum {statistics[parameter+'a']['nama']} sebesar {statistics[parameter+'a']['min']}. "
                    f"Lalu, nilai rata-rata {statistics[parameter+'b']['nama']} sebesar {statistics[parameter+'b']['mean']} selama sebulan, nilai maksimal {statistics[parameter+'b']['nama']} sebesar {statistics[parameter+'b']['max']}, dan nilai minimum {statistics[parameter+'b']['nama']} sebesar {statistics[parameter+'b']['min']}. "
                    f"Dan, nilai rata-rata {statistics[parameter+'c']['nama']} sebesar {statistics[parameter+'c']['mean']} selama sebulan, nilai maksimal {statistics[parameter+'c']['nama']} sebesar {statistics[parameter+'c']['max']}, dan nilai minimum {statistics[parameter+'c']['nama']} sebesar {statistics[parameter+'c']['min']}. "
                    f"{par_prompt}"
                    f"Buatlah analisis deskriptif dalam satu paragraf terkait keadaan trafo tersebut, serta berikan statement apakah kondisi trafo tersebut normal atau tidak berdasarkan data tersebut."
                
                )
                prompts[parameter] = prompt
            elif parameter == 'V':

                prompt = (
                    f"Trafo ini memiliki nilai rata-rata {statistics[parameter+'an']['nama']} sebesar {statistics[parameter+'an']['mean']} selama sebulan, nilai maksimal {statistics[parameter+'an']['nama']} sebesar {statistics[parameter+'an']['max']}, dan nilai minimum {statistics[parameter+'an']['nama']} sebesar {statistics[parameter+'an']['min']}. "
                    f"Lalu, nilai rata-rata {statistics[parameter+'bn']['nama']} sebesar {statistics[parameter+'bn']['mean']} selama sebulan, nilai maksimal {statistics[parameter+'bn']['nama']} sebesar {statistics[parameter+'bn']['max']}, dan nilai minimum {statistics[parameter+'bn']['nama']} sebesar {statistics[parameter+'bn']['min']}. "
                    f"Dan, nilai rata-rata {statistics[parameter+'cn']['nama']} sebesar {statistics[parameter+'cn']['mean']} selama sebulan, nilai maksimal {statistics[parameter+'cn']['nama']} sebesar {statistics[parameter+'cn']['max']}, dan nilai minimum {statistics[parameter+'cn']['nama']} sebesar {statistics[parameter+'cn']['min']}."
                    f"Trafo tersebut memiliki setting voltage low trip sebesar {transformer_settings.voltage_low_trip}, voltage low alarm sebesar {transformer_settings.voltage_low_alarm}, voltage high trip sebesar {transformer_settings.voltage_high_trip}, dan voltage high alarm sebesar {transformer_settings.voltage_high_alarm}."
                    f"Buatlah analisis deskriptif dalam satu paragraf terkait keadaan trafo tersebut, serta berikan statement apakah kondisi trafo tersebut normal atau tidak berdasarkan data tersebut."
                )
                prompts[parameter] = prompt
            else:
                if parameter == 'OilTemp':
                    par_prompt = f"Trafo tersebut memiliki setting top oil high alarm sebesar {transformer_settings.top_oil_high_alarm} dan top oil high trip sebesar {transformer_settings.top_oil_high_trip}."
                elif parameter == 'Press':
                    par_prompt = f"Trafo tersebut memiliki setting pressure high alarm sebesar {transformer_settings.pressure_high_alarm} dan pressure high trip sebesar {transformer_settings.pressure_high_trip}."
                elif parameter == 'Ineutral':
                    par_prompt = f"Trafo tersebut memiliki setting I neutral high alarm sebesar {transformer_settings.i_neutral_high_alarm} dan I neutral high trip sebesar {transformer_settings.i_neutral_high_trip}."
                elif parameter == 'Freq':
                    par_prompt = f"Trafo tersebut memiliki setting freq low alarm sebesar {transformer_settings.freq_low_alarm} dan freq low trip sebesar {transformer_settings.freq_low_trip}."
                else:
                    par_prompt = f"Trafo tersebut memiliki setting {parameter} maksimal sebesar {statistics[parameter]['max']}."

                prompt = (
                    f"Trafo ini memiliki nilai rata-rata {statistics[parameter]['nama']} sebesar {statistics[parameter]['mean']} selama sebulan, nilai maksimal {statistics[parameter]['nama']} sebesar {statistics[parameter]['max']}, dan nilai minimum {statistics[parameter]['nama']} sebesar {statistics[parameter]['min']}. "
                    f"{par_prompt}"
                    f"Buatlah analisis deskriptif dalam satu paragraf terkait keadaan trafo tersebut, serta berikan statement apakah kondisi trafo tersebut normal atau tidak berdasarkan data tersebut."
                )
                prompts[parameter] = prompt
            print(prompt)
        progress += current_progress/len(parameters)

        
    end_time_making_prompt = time.time()

    def delayed_completion(delay_in_seconds: float = 1, **kwargs):
        time.sleep(delay_in_seconds)
        return client.chat.completions.create(**kwargs)

    rate_limit_per_minute = 3
    delay = 100.0 / rate_limit_per_minute
    
    client = OpenAI(api_key="sk-proj-XzX4jdYT5lVFvi75a09xT3BlbkFJ2boQYyIuoxJ0uke1CpEg")

    answers = {}
    summary_answers = {}


    start_time_generate_answer = time.time()

    for parameter in parameters:
        if prompts[parameter] == ' ':
            answers[parameter] = 'Tidak ada penjelasan'
        else:
            completion = delayed_completion(
                delay_in_seconds=delay,
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a smart assistant"},
                    {"role": "user", "content": prompts[parameter]}
                ]
            )
            answers[parameter] = completion.choices[0].message.content
            print('generating answer...')
            print(parameter)
            print(answers[parameter])
            # answers[parameter] = 'Ada penjelasan'

        if answers[parameter] == 'Tidak ada penjelasan':
            summary_answers[parameter] = ' '
        else:
            completion = delayed_completion(
                delay_in_seconds=delay,
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a smart assistant"},
                    {"role": "user", "content": f"Jelaskan dalam 1 kalimat singkat mengenai kondisi transformer dari paragraf berikut: {answers[parameter]}"}
                ]
            )
            summary_answers[parameter] = completion.choices[0].message.content
            print('generating one sentence answer')
            # summary_answers[parameter] = 'Penjelasan 1 kalimat'
        progress += current_progress/len(parameters)
    
    if press_type == 0:
        answers['Press'] = 'Trafo ini berjenis konservator.'
    elif press_type == 1:
        answers['Press'] = 'Trafo ini berjenis konservator namun, terdapat beberapa kali nilai tekanan lebih dari 0. Hal itu mengindikasikan adanya kondisi tidak normal pada trafo. Nilai tekanan yang melebihi 0 dapat menandakan adanya masalah seperti penyumbatan pada pipa penghubung, kerusakan pada sistem ventilasi, atau kebocoran pada tangki, yang memerlukan inspeksi dan penanganan lebih lanjut untuk mencegah kerusakan lebih parah pada trafo.'

    end_time_generate_answer = time.time()

    prompt_kesimpulan = "Dari informasi berikut, berikan kesimpulan dalam 1 paragraf"
    for parameter in parameters:
        prompt_kesimpulan += f"\n- {answers[parameter]}"


    start_time_generate_conclusion = time.time()

    completion = delayed_completion(
        delay_in_seconds=delay,
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a smart assistant"},
            {"role": "user", "content": prompt_kesimpulan}
        ]
    )
    conclusion = completion.choices[0].message.content
    # conclusion = 'Berdasarkan informasi yang diberikan mengenai kondisi trafo PT Bambang Djaja selama bulan November, kesimpulannya adalah bahwa trafo perusahaan menunjukkan tren suhu oli, suhu busbar, suhu winding, tegangan antar fase, arus fase, daya aktif, daya reaktif, daya semu, power factor, frekuensi, arus netral, penggunaan energi listrik, reactive energy, total harmonic distortion, K-rated, dan deRating berada dalam rentang yang stabil dan aman. Meskipun terdapat sedikit variasi atau deviasi dari nilai rata-rata atau setting maksimal yang telah ditentukan, kondisi trafo secara keseluruhan dapat dikatakan baik dan operasionalnya berjalan efisien serta sesuai dengan standar kapasitas yang diinginkan. Perusahaan perlu terus memantau dan melakukan pemeliharaan secara berkala untuk memastikan performa trafo tetap optimal dan aman untuk kebutuhan operasional.'
    print(conclusion)
    end_time_generate_conclusion = time.time()

    start_time_generate_suggestion = time.time()
    prompt_saran = "Dari informasi berikut, berikan 3 saran singkat untuk perusahaan"
    for parameter in parameters:
        prompt_saran += f"\n- {answers[parameter]}"

    completion = delayed_completion(
        delay_in_seconds=delay,
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a smart assistant"},
            {"role": "user", "content": prompt_saran}
        ]
    )
    suggestion = completion.choices[0].message.content
    # suggestion = 'Beberapa saran yang dapat diberikan adalah:\n\n1. Optimalkan Efisiensi Energi: Meskipun data menunjukkan efisiensi yang baik dalam penggunaan energi, perusahaan dapat terus memantau dan mengoptimalkan penggunaan energi untuk mengurangi potensi pemborosan energi yang tidak perlu. Langkah-langkah seperti melakukan audit energi dan memperbarui sistem untuk meningkatkan efisiensi dapat membantu perusahaan menghemat biaya dan mendukung keberlanjutan.\n2. Perawatan dan Monitoring Harmonisa: Karena terdapat variasi distorsi harmonik yang signifikan pada beberapa data, perusahaan disarankan untuk meningkatkan perawatan dan monitoring terhadap harmonisa pada trafo. Hal ini dapat dilakukan dengan mengikuti pedoman perawatan yang direkomendasikan dan menggunakan perangkat pemantauan yang tepat untuk memastikan kualitas tegangan dan arus yang optimal.\n3. Pemantauan dan Peningkatan Faktor Daya: Meskipun faktor daya pada fasa-fasa tertentu cenderung stabil, perusahaan dapat memantau dan mengoptimalkan faktor daya pada setiap fasa untuk meningkatkan efisiensi sistem kelistrikan mereka. Langkah-langkah seperti instalasi peralatan yang lebih efisien atau penyempurnaan sistem distribusi dapat membantu meningkatkan faktor daya secara keseluruhan.'
    print(suggestion)
    suggestion_poin = re.findall(r'\d+\.\s+.*?(?=\n\d+\.|\Z)', suggestion, re.DOTALL)
    
    forecast_kwh = forecast_data(selected_company, date)
    answers['kWhInp'] = f"Berdasarkan data historis penggunaan sebelumnya, diperkirakan penggunaan kWh dari trafo pada bulan depan akan mencapai {forecast_kwh} kWh."
    summary_answers['kWhInp'] = " "
    end_time_generate_suggestion = time.time()


    # CREATING PDF
    
    buffer = BytesIO()
    p = canvas.Canvas(buffer, pagesize=A4)

    # Sumbu
    bab_x, bab_y = 60, 755
    subbab_x, subbab_y = 90, bab_y-20
    image_x, image_y = 90, bab_y-270
    table_x, table_y = 90, bab_y-393
    image_width, image_height = 410, 240
    judulimg_x, judulimg_y = image_x+130, image_y-15
    table2_x, table2_y = 90, judulimg_y-43
    text_x, text_y = 90, judulimg_y-48
    logo_x, logo_y = 438, 777
    logo_width, logo_height = 70, 30
    bg_width, bg_height = 595, 842 

    start_time_generate_cover = time.time()

    # COVER
    create_new_page(p, 0)

    image_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bg", "cover_update.png")
    with Image.open(image_path) as img:
        img.thumbnail((120, 120))
        p.drawImage(image_path, 0, 0, width=bg_width, height=bg_height, preserveAspectRatio=True)

    text = f"{app.bulan} {app.tahun}"
    text_width = p.stringWidth(text, "Helvetica-Bold", 15)
    x = (A4[0] - text_width) / 2
    p.setFont("Helvetica-Bold", 15)
    p.drawString(x, 90, text)

    create_new_page(p, 1)
    progress += current_progress

    end_time_generate_cover = time.time()



    start_time_generate_toc = time.time()

    # TABLE OF CONTENTS
    p.setFont("Helvetica-Bold", 15)
    text = "TABLE OF CONTENTS"
    text_width = p.stringWidth(text, "Helvetica-Bold", 15)
    x = (A4[0] - text_width) / 2
    p.drawString(x, 775, text)
    x1, y1 = 90, 725

    p.setFont("Helvetica-Bold", 12)
    p.drawString(x1, y1, "Transformer Data & Settings")
    p.setFont("Helvetica", 12)
    p.drawString(100, y1-15, "1.1   Data Transformer")
    p.drawString(100, y1-30, "1.2   Data Settings Threshold")
    p.setFont("Helvetica-Bold", 12)
    p.drawString(90, y1-50, "Physical Deep Analysis")
    p.setFont("Helvetica", 12)
    p.drawString(100, y1-65, "2.1   Oil Temperature")
    p.drawString(100, y1-80, "2.2   Busbar Temperature")
    p.drawString(100, y1-95, "2.3   Winding Temperature")
    p.drawString(100, y1-110, "2.4   Tank Pressure")
    p.drawString(100, y1-125, "2.5   Oil Level")
    p.setFont("Helvetica-Bold", 12)
    p.drawString(90, y1-145, "Electrical Deep Analysis")
    p.setFont("Helvetica", 12)
    p.drawString(100, y1-160, "3.1   Voltage Analysis")
    p.drawString(100, y1-175, "3.2   Current Analysis")
    p.drawString(100, y1-190, "3.3   Power Usage Analysis")
    p.drawString(110, y1-205, "3.3.1   Active Power")
    p.drawString(110, y1-220, "3.3.2   Reactive Power")
    p.drawString(110, y1-235, "3.3.3   Apparent Power")

    p.drawString(100, y1-250, "3.4   Power Quality Analysis")
    p.drawString(110, y1-265, "3.4.1   Power Factor")
    p.drawString(110, y1-280, "3.4.2   Frequency")
    p.drawString(110, y1-295, "3.4.3   Arus Netral")
    p.drawString(100, y1-310, "3.5   Energy Consumption Analysis")
    p.drawString(110, y1-325, "3.5.1   Active Energy")
    p.drawString(110, y1-340, "3.5.2   Reactive Energy")
    p.drawString(100, y1-355, "3.6   Harmonic Analysis")
    p.drawString(110, y1-370, "3.6.1   Total Harmonic Distortion Voltage")
    p.drawString(110, y1-385, "3.6.2   Total Harmonic Distortion Arus")
    p.drawString(110, y1-400, "3.6.3   KRated")
    p.drawString(110, y1-415, "3.6.4   DeRating")
    p.setFont("Helvetica-Bold", 12)
    p.drawString(90, y1-435, "Conclusion & Suggestion")
    p.setFont("Helvetica", 12)
    p.drawString(100, y1-450, "4.1   Kesimpulan")
    p.drawString(100, y1-465, "4.2   Saran")
    p.setFont("Helvetica-Bold", 12)
    p.drawString(90, y1-485, "Lampiran")
    print('toc')
    end_time_generate_toc = time.time()
    progress += current_progress



    # BAB 1
    create_new_page(p, 1)

    start_time_generate_bab1 = time.time()


    add_bg(p, 0)

    p.setFont("Helvetica-BoldOblique", 15)
    p.drawString(bab_x, bab_y, "1   Transformer Data & Settings")

    # Data Transformer
    p.setFont("Helvetica-Bold", 12)
    p.drawString(subbab_x, subbab_y, "1.1. Data Transformer")

    if transformer_data:
        columns = transformer_data.__table__.columns.keys()
        data = [['No', 'Parameter', 'Data', 'Units']]
        parameter_name = ['Serial Number', 'Impedance', 'Rated Power', 'Frequency', 'Rated High Voltage', 'Rated Low Voltage', 'Rated Current High Voltage',
                     'Rated Current Low Voltage', 'Vector Group', 'Phase', 'No Load Loss', 'Full Load Loss', 'Top Oil Temp. Rise LV', 'Top Oil Temp. Rise HV',
                     'Average Winding Temp. Rise LV', 'Average Winding Temp. Rise HV', 'Gradient LV', 'Gradient HV', 'Cooling Mode', 'CT Rasio', 'Hotspot Factor', 'K-Rated']
        units = ['', '%', 'kVA', 'Hz', 'V', 'V', 'A', 'A', '', '', 'Watt', 'Watt', '°C', '°C', '°C', '°C', 'gr', 'gr', '', '', '', '']

        i = 1
        for col in columns[1:]:
            parameter_value = getattr(transformer_data, col)
            data.append([i, parameter_name[i-1], parameter_value, units[i-1]])
            i += 1

    col_widths = [35, 180, 120, 80]
    t = Table(data, colWidths=col_widths)

    style = TableStyle([('BACKGROUND', (0, 0), (-1, 0), '#4472C4'),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                        ('INNERGRID', (0, 0), (-1, -1), 0.25, colors.black),
                        ('BOX', (0, 0), (-1, -1), 0.25, colors.black),
                        ('FONT', (0, 0), (-1, -1), 'Helvetica', 8),
                        ('LEADING', (0, 0), (-1, -1), 10)
                        ])

    t.setStyle(style)
    t.wrapOn(p, 0, 0)

    table_width = sum(t._colWidths)
    x_center = (p._pagesize[0] - table_width) / 2
    t.drawOn(p, x_center, table_y)
    progress += current_progress
    
    # Data Settings
    p.setFont("Helvetica-Bold", 12)
    p.drawString(90, table_y-25, "1.2. Data Settings Threshold")

    columns_name = [
        "Voltage Low Trip", "Voltage Low Alarm", "Voltage High Alarm", "Voltage High Trip",
        "Freq. Low Trip", "Freq. Low Alarm", "Freq. High Alarm", "Freq. High Trip",
        "THDI Trip", "TDHI Alarm", "THDV Alarm", "THDV Trip",
        "Top Oil High Alarm", "Top Oil High Trip", "WTI High Alarm", "WTI High Trip",
        "PF Low Alarm", "PF Low Trip", "Current High Alarm", "Current High Trip",
        "I Neutral High Alarm", "I Neutral High Trip", "BusTemp. High Alarm", "BusTemp. High Trip",
        "Pressure High Alarm", "Pressure High Trip", "Unbalance High Alarm", "Unbalance High Trip"
    ]

    if transformer_settings:
        columns = list(transformer_settings.__table__.columns.keys())[1:]
        values = [getattr(transformer_settings, col) for col in columns]

        data = []
        i = 0
        while i < len(columns):
            data.append(columns_name[i:i+4])
            data.append(values[i:i+4]) 
            i += 4

    col_widths = [103.75, 103.75, 103.75, 103.75]
    t = Table(data, colWidths=col_widths)

    style = [('BACKGROUND', (0, 0), (-1, 0), '#4472C4'),
         ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
         ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
         ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
         ('INNERGRID', (0, 0), (-1, -1), 0.25, colors.black),
         ('BOX', (0, 0), (-1, -1), 0.25, colors.black),
         ('FONT', (0, 0), (-1, -1), 'Helvetica', 8)]

    for i in range(2, len(data), 2):
        style.append(('BACKGROUND', (0, i), (-1, i), '#4472C4'))
        style.append(('TEXTCOLOR', (0, i), (-1, i), colors.white))

    t.setStyle(TableStyle(style))
    t.wrapOn(p, 0, 0)
    t.drawOn(p, 90, 110)
    progress += current_progress

    add_page_number(p)

    print('bab1')

    end_time_generate_bab1 = time.time()


    # BAB 2

    create_new_page(p, 1)

    start_time_generate_bab2 = time.time()


    add_bg(p, 0)

    p.setFont("Helvetica-BoldOblique", 15)
    p.drawString(bab_x, bab_y, "2   Physical Deep Analysis")

    # Oil Temp
    p.setFont("Helvetica-Bold", 12)
    p.drawString(subbab_x, subbab_y, "2.1. Oil Temperature")
    draw_image_on_canvas(p, app.plot_oiltemp, image_x, image_y, image_width, image_height)
    draw_judul_img(p, "Gambar 2.1 Oil Temperature Chart", image_x, image_width, judulimg_y)
    draw_table(p, statistics['OilTemp']['min'], statistics['OilTemp']['max'], statistics['OilTemp']['mean'], table2_y)
    generate_answers(p, answers['OilTemp'], summary_answers['OilTemp'], text_y)
    
    add_page_number(p)
    progress += current_progress

    # BusTemp
    create_new_page(p, 1)

    p.setFont("Helvetica-Bold", 12)
    p.drawString(subbab_x, subbab_y, "2.2. Busbar Temperature")
    draw_image_on_canvas(p, app.plot_bustemp, image_x, image_y, image_width, image_height)
    draw_judul_img(p, "Gambar 2.2 Busbar Temperature Chart", image_x, image_width, judulimg_y)
    draw_table3(p, statistics['BusTemp1']['min'], statistics['BusTemp2']['min'], statistics['BusTemp3']['min'],
            statistics['BusTemp1']['max'], statistics['BusTemp2']['max'], statistics['BusTemp3']['max'],
            statistics['BusTemp1']['mean'], statistics['BusTemp2']['mean'], statistics['BusTemp3']['mean'],
            table2_y)
    generate_answers(p, answers['BusTemp'], summary_answers['BusTemp'], text_y)
    progress += current_progress

    # Winding Temperature
    create_new_page(p, 1)
    

    p.setFont("Helvetica-Bold", 12)
    p.drawString(subbab_x, subbab_y, "2.3. Winding Temperature")
    draw_image_on_canvas(p, app.plot_wtitemp, image_x, image_y, image_width, image_height)
    draw_judul_img(p, "Gambar 2.3 Winding Temperature Chart", image_x, image_width, judulimg_y)
    draw_table3(p, statistics['WTITemp1']['min'], statistics['WTITemp2']['min'], statistics['WTITemp3']['min'],
            statistics['WTITemp1']['max'], statistics['WTITemp2']['max'], statistics['WTITemp3']['max'],
            statistics['WTITemp1']['mean'], statistics['WTITemp2']['mean'], statistics['WTITemp3']['mean'],
            table2_y)
    generate_answers(p, answers['WTITemp'], summary_answers['WTITemp'], text_y)
    progress += current_progress

    # Pressure
    create_new_page(p, 1)

    p.setFont("Helvetica-Bold", 12)
    p.drawString(subbab_x, subbab_y, "2.4. Tank Pressure")
    draw_image_on_canvas(p, app.plot_press, image_x, image_y, image_width, image_height)
    draw_judul_img(p, "Gambar 2.4 Tank Pressure Chart", image_x, image_width, judulimg_y)
    draw_table(p, statistics['Press']['min'], statistics['Press']['max'], statistics['Press']['mean'], table2_y)
    generate_answers(p, answers['Press'], summary_answers['Press'], text_y)
    progress += current_progress

    # Level
    create_new_page(p, 1)

    p.setFont("Helvetica-Bold", 12)
    p.drawString(subbab_x, subbab_y, "2.5. Oil Level")
    draw_image_on_canvas(p, app.plot_level, image_x, image_y, image_width, image_height)
    draw_judul_img(p, "Gambar 2.5 Oil Level Chart", image_x, image_width, judulimg_y)
    draw_table(p, statistics['Level']['min'], statistics['Level']['max'], statistics['Level']['mean'], table2_y)
    generate_answers(p, answers['Level'], summary_answers['Level'], text_y)
    progress += current_progress

    print('bab2')
    end_time_generate_bab2 = time.time()

    
    # BAB 3
    create_new_page(p, 1)
    start_time_generate_bab3 = time.time()

    p.setFont("Helvetica-BoldOblique", 15)
    p.drawString(bab_x, bab_y, "3   Electrical Deep Analysis")

    # Voltage Analysis
    p.setFont("Helvetica-Bold", 12)
    p.drawString(subbab_x, subbab_y, "3.1. Voltage Analysis")
    draw_image_on_canvas(p, app.plot_voltage, image_x, image_y, image_width, image_height)
    draw_judul_img(p, "Gambar 3.1 Voltage Chart", image_x, image_width, judulimg_y)
    draw_table3(p, statistics['Van']['min'], statistics['Vbn']['min'], statistics['Vcn']['min'],
            statistics['Van']['max'], statistics['Vbn']['max'], statistics['Vcn']['max'],
            statistics['Van']['mean'], statistics['Vbn']['mean'], statistics['Vcn']['mean'],
            table2_y)
    generate_answers(p, answers['V'], summary_answers['V'], text_y)
    progress += current_progress

    # Current Analysis
    create_new_page(p, 1)

    p.setFont("Helvetica-Bold", 12)
    p.drawString(subbab_x, subbab_y, "3.2. Current Analysis")
    draw_image_on_canvas(p, app.plot_arus, image_x, image_y, image_width, image_height)
    draw_judul_img(p, "Gambar 3.2 Current Chart", image_x, image_width, judulimg_y)
    draw_table3(p, statistics['Ia']['min'], statistics['Ib']['min'], statistics['Ic']['min'],
            statistics['Ia']['max'], statistics['Ib']['max'], statistics['Ic']['max'],
            statistics['Ia']['mean'], statistics['Ib']['mean'], statistics['Ic']['mean'],
            table2_y)
    generate_answers(p, answers['I'], summary_answers['I'], text_y)
    progress += current_progress

    # Power Usage Analysis
    create_new_page(p, 1)

    # bab_x, bab_y = 60, 780
    # subbab_x, subbab_y = 90, 750
    subsubbab_x, subsubbab_y = 90, subbab_y-20
    image_x, image_y = 90, bab_y-300
    image_width, image_height = 410, 240
    judulimg_x, judulimg_y = 220, image_y-15
    table2_y = judulimg_y-43
    text_x, text_y = 90, judulimg_y-48

    p.setFont("Helvetica-Bold", 12)
    p.drawString(subbab_x, subbab_y, "3.3. Power Usage Analysis")

    # Power Usage Analysis: P
    p.setFont("Helvetica-Bold", 11)
    p.drawString(subsubbab_x, subsubbab_y, "3.3.1. Active Power Factor")
    draw_image_on_canvas(p, app.plot_p, image_x, image_y, image_width, image_height)
    draw_judul_img(p, "Gambar 3.3.1 Active Power Chart", image_x, image_width, judulimg_y)
    draw_table3(p, statistics['Pa']['min'], statistics['Pb']['min'], statistics['Pc']['min'],
            statistics['Pa']['max'], statistics['Pb']['max'], statistics['Pc']['max'],
            statistics['Pa']['mean'], statistics['Pb']['mean'], statistics['Pc']['mean'],
            table2_y)
    generate_answers(p, answers['P'], summary_answers['P'], text_y)
    progress += current_progress

    print('bab3')

    # Gambar 3.4

    create_new_page(p, 1)

    # Power Usage Analysis: Q
    p.setFont("Helvetica-Bold", 11)
    p.drawString(subsubbab_x, subsubbab_y, "3.3.2. Reactive Power")
    draw_image_on_canvas(p, app.plot_q, image_x, image_y, image_width, image_height)
    draw_judul_img(p, "Gambar 3.3.2 Reactive Power Chart", image_x, image_width, judulimg_y)
    draw_table3(p, statistics['Qa']['min'], statistics['Qb']['min'], statistics['Qc']['min'],
            statistics['Qa']['max'], statistics['Qb']['max'], statistics['Qc']['max'],
            statistics['Qa']['mean'], statistics['Qb']['mean'], statistics['Qc']['mean'],
            table2_y)
    generate_answers(p, answers['Q'], summary_answers['Q'], text_y)
    progress += current_progress

    create_new_page(p, 1)

    # Power Usage Analysis: S
    p.setFont("Helvetica-Bold", 11)
    p.drawString(subsubbab_x, subsubbab_y, "3.3.3. Apparent Power")
    draw_image_on_canvas(p, app.plot_s, image_x, image_y, image_width, image_height)
    draw_judul_img(p, "Gambar 3.3.3 Apparent Power Chart", image_x, image_width, judulimg_y)
    draw_table3(p, statistics['Sa']['min'], statistics['Sb']['min'], statistics['Sc']['min'],
            statistics['Sa']['max'], statistics['Sb']['max'], statistics['Sc']['max'],
            statistics['Sa']['mean'], statistics['Sb']['mean'], statistics['Sc']['mean'],
            table2_y)
    generate_answers(p, answers['S'], summary_answers['S'], text_y)
    progress += current_progress

    create_new_page(p, 1)

    # Power Quality Analysis
    p.setFont("Helvetica-Bold", 12)
    p.drawString(subbab_x, subbab_y, "3.4. Power Quality Analysis")

    p.setFont("Helvetica-Bold", 11)
    p.drawString(subsubbab_x, subsubbab_y, "3.4.1. Power Factor")

    draw_image_on_canvas(p, app.plot_pf, image_x, image_y, image_width, image_height)
    draw_judul_img(p, "Gambar 3.4.1 Power Factor Chart", image_x, image_width, judulimg_y)
    draw_table3(p, statistics['PFa']['min'], statistics['PFb']['min'], statistics['PFc']['min'],
            statistics['PFa']['max'], statistics['PFb']['max'], statistics['PFc']['max'],
            statistics['PFa']['mean'], statistics['PFb']['mean'], statistics['PFc']['mean'],
            table2_y)
    generate_answers(p, answers['PF'], summary_answers['PF'], text_y)
    progress += current_progress

    create_new_page(p, 1)

    p.setFont("Helvetica-Bold", 11)
    p.drawString(subsubbab_x, subsubbab_y, "3.4.2. Frequency")
    draw_image_on_canvas(p, app.plot_freq, image_x, image_y, image_width, image_height)
    draw_judul_img(p, "Gambar 3.4.2 Frequency Chart", image_x, image_width, judulimg_y)
    draw_table(p, statistics['Freq']['min'], statistics['Freq']['max'], statistics['Freq']['mean'], table2_y)
    generate_answers(p, answers['Freq'], summary_answers['Freq'], text_y)
    progress += current_progress

    create_new_page(p, 1)

    p.setFont("Helvetica-Bold", 11)
    p.drawString(subsubbab_x, subsubbab_y, "3.4.3. Neutral Current")
    draw_image_on_canvas(p, app.plot_ineutral, image_x, image_y, image_width, image_height)
    draw_judul_img(p, "Gambar 3.4.3 Neutral Current Chart", image_x, image_width, judulimg_y)
    draw_table(p, statistics['Ineutral']['min'], statistics['Ineutral']['max'], statistics['Ineutral']['mean'], table2_y)
    generate_answers(p, answers['Ineutral'], summary_answers['Ineutral'], text_y)
    progress += current_progress

    create_new_page(p, 1)

    print('bab3')

    # Energy Consumption Analysis
    p.setFont("Helvetica-Bold", 12)
    p.drawString(subbab_x, subbab_y, "3.5. Energy Consumption Analysis")

    p.setFont("Helvetica-Bold", 11)
    p.drawString(subsubbab_x, subsubbab_y, "3.5.1. Active Energy")
    draw_image_on_canvas(p, app.plot_kwh, image_x, image_y, image_width, image_height)
    draw_judul_img(p, "Gambar 3.5.1 Active Energy Chart", image_x, image_width, judulimg_y)
    draw_table(p, statistics['kWhInp']['min'], statistics['kWhInp']['max'], statistics['kWhInp']['mean'], table2_y)
    generate_answers(p, answers['kWhInp'], summary_answers['kWhInp'], text_y)
    progress += current_progress

    create_new_page(p, 1)

    p.setFont("Helvetica-Bold", 11)
    p.drawString(subsubbab_x, subsubbab_y, "3.5.2. Reactive Energy")
    draw_image_on_canvas(p, app.plot_kvarh, image_x, image_y, image_width, image_height)
    draw_judul_img(p, "Gambar 3.5.2 Reactive Energy Chart", image_x, image_width, judulimg_y)
    draw_table(p, statistics['kVARhinp']['min'], statistics['kVARhinp']['max'], statistics['kVARhinp']['mean'], table2_y)
    generate_answers(p, answers['kVARhinp'], summary_answers['kVARhinp'], text_y)
    progress += current_progress
    
    create_new_page(p, 1)

    # Harmonic Analysis
    p.setFont("Helvetica-Bold", 12)
    p.drawString(subbab_x, subbab_y, "3.6. Harmonic Analysis")

    p.setFont("Helvetica-Bold", 11)
    p.drawString(subsubbab_x, subsubbab_y, "3.6.1. Total Harmonic Distortion Voltage")
    draw_image_on_canvas(p, app.plot_thdv, image_x, image_y, image_width, image_height)
    draw_judul_img(p, "Gambar 3.6.1 Total Harmonic Distortion Voltage Chart", image_x, image_width, judulimg_y)
    draw_table3(p, statistics['THDV1']['min'], statistics['THDV2']['min'], statistics['THDV3']['min'],
            statistics['THDV1']['max'], statistics['THDV2']['max'], statistics['THDV3']['max'],
            statistics['THDV1']['mean'], statistics['THDV2']['mean'], statistics['THDV3']['mean'],
            table2_y)
    generate_answers(p, answers['THDV'], summary_answers['THDV'], text_y)
    progress += current_progress

    create_new_page(p, 1)

    p.setFont("Helvetica-Bold", 11)
    p.drawString(subsubbab_x, subsubbab_y, "3.6.2. Total Harmonic Distortion Current")
    draw_image_on_canvas(p, app.plot_thdi, image_x, image_y, image_width, image_height)
    draw_judul_img(p, "Gambar 3.6.2 Total Harmonic Distortion Current Chart", image_x, image_width, judulimg_y)
    draw_table3(p, statistics['THDI1']['min'], statistics['THDI2']['min'], statistics['THDI3']['min'],
            statistics['THDI1']['max'], statistics['THDI2']['max'], statistics['THDI3']['max'],
            statistics['THDI1']['mean'], statistics['THDI2']['mean'], statistics['THDI3']['mean'],
            table2_y)
    generate_answers(p, answers['THDI'], summary_answers['THDI'], text_y)
    progress += current_progress

    create_new_page(p, 1)

    p.setFont("Helvetica-Bold", 11)
    p.drawString(subsubbab_x, subsubbab_y, "3.6.3. KRated")
    draw_image_on_canvas(p, app.plot_krated, image_x, image_y, image_width, image_height)
    draw_judul_img(p, "Gambar 3.6.3 KRated Chart", image_x, image_width, judulimg_y)
    draw_table3(p, statistics['KRateda']['min'], statistics['KRatedb']['min'], statistics['KRatedc']['min'],
            statistics['KRateda']['max'], statistics['KRatedb']['max'], statistics['KRatedc']['max'],
            statistics['KRateda']['mean'], statistics['KRatedb']['mean'], statistics['KRatedc']['mean'],
            table2_y)
    generate_answers(p, answers['KRated'], summary_answers['KRated'], text_y)
    progress += current_progress

    create_new_page(p, 1)

    p.setFont("Helvetica-Bold", 11)
    p.drawString(subsubbab_x, subsubbab_y, "3.6.4. DeRating")
    draw_image_on_canvas(p, app.plot_derating, image_x, image_y, image_width, image_height)
    draw_judul_img(p, "Gambar 3.6.4 DeRating Chart", image_x, image_width, judulimg_y)
    draw_table3(p, statistics['deRatinga']['min'], statistics['deRatingb']['min'], statistics['deRatingc']['min'],
            statistics['deRatinga']['max'], statistics['deRatingb']['max'], statistics['deRatingc']['max'],
            statistics['deRatinga']['mean'], statistics['deRatingb']['mean'], statistics['deRatingc']['mean'],
            table2_y)
    generate_answers(p, answers['deRating'], summary_answers['deRating'], text_y)
    progress += current_progress

    print('bab3')
    end_time_generate_bab3 = time.time()


    # BAB 4
    create_new_page(p, 1)

    start_time_generate_bab4 = time.time()


    p.setFont("Helvetica-BoldOblique", 15)
    p.drawString(bab_x, bab_y, "4   Conclusion & Suggestion")

    p.setFont("Helvetica-Bold", 12)
    p.drawString(subbab_x, subbab_y, "4.1. Kesimpulan")

    text_y = subbab_y

    style = ParagraphStyle('CustomStyle')
    style.alignment = 4
    style.fontSize = 12
    style.leading = 16
    
    p_text = Paragraph(str(conclusion), style)
    par_height = p_text.wrap(410, 999999)[1]
    text_y -= par_height + 15
    p_text.drawOn(p, 90, text_y)

    progress += current_progress

    text_y -= 20

    subbab_y=text_y

    p.setFont("Helvetica-Bold", 12)
    p.drawString(subbab_x, subbab_y, "4.2. Saran")

    text_y = subbab_y

    style = ParagraphStyle('CustomStyle')
    style.alignment = 4
    style.fontSize = 12
    style.leading = 16

    for s in suggestion_poin:
        # poin = '- ' + s
        poin = s
        p_text = Paragraph(str(poin), style)
        par_height = p_text.wrap(410, 999999)[1]
        text_y -= par_height + 15
        p_text.drawOn(p, 90, text_y)    

    progress += current_progress

    print('bab4')
    end_time_generate_bab4 = time.time()


    # LAMPIRAN
    create_new_page(p, 2)

    start_time_generate_lampiran = time.time()


    p.setFont("Helvetica-Bold", 15)
    text = "LAMPIRAN"
    text_width = p.stringWidth(text, "Helvetica-Bold", 15)
    x = (landscape(A4)[0] - text_width) / 2
    p.drawString(x, 500, text)

    table_data = []
    header_row = ['Parameters']
    parameter_row = []
    keys = []

    for key, value in statistics.items():
        header_row.append(key)
        for key2 in statistics[key].keys():
            if key2 != 'nama' and key2 != 'threshold':
                if key2 not in keys:
                    keys.append(key2)
    table_data.append(header_row)
    stats = ["Total waktu di\nsekitar nilai\nmaksimal", "Jam di sekitar\nnilai maksimal", "Mean", "Standar\ndeviasi", "Simpangan\nrata-rata", "Min", "Max", "Mode", "Q1", "Q2", "Q3", "Skewness", "Koefisien\nvariasi", "Proporsi\n(>= Q3)", "Persentase\n(>= Q3)"]

    i = 0
    total_times_values = []
    for k in keys:
        total_times_values.append(stats[i])
        i += 1
        for key, value in statistics.items():
            total_times_value = value.get(k, None)
            total_times_values.append(total_times_value)
        parameter_row.append(total_times_values)
        total_times_values = []

    for i in range(len(parameter_row)):
        table_data.append(parameter_row[i])
    
    tdata = [row[0:22] for row in table_data]
    tdata2 = [row[0:1] + row[22:44] for row in table_data]

    style = TableStyle([('BACKGROUND', (0, 0), (-1, 0), '#4472C4'),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black),
                    ('FONTSIZE', (0, 0), (-1, -1), 3),
                    ('WORDWRAP', (0, 0), (-1, -1), 1),
                    ('LEADING', (0, 0), (-1, -1), 5),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                    ])
    t = Table(tdata)
    t.setStyle(style)

    table_width = len(table_data) * 19
    table_height = len(table_data[0]) * 20
    t.wrapOn(p, table_width, table_height)
    
    t.drawOn(p, 60, table_width-70)

    progress += current_progress

    create_new_page(p, 2)

    t = Table(tdata2)
    t.setStyle(style)

    col_widths = [max(len(str(cell)) for row in table_data for cell in row)]
    t.setStyle(TableStyle([('COLWIDTHS', (0, 0), (-1, -1), col_widths)]))

    table_width = len(table_data) * 19
    table_height = len(table_data[0]) * 20
    t.wrapOn(p, table_width, table_height)

    t.drawOn(p, 40, table_width-70)
    progress += current_progress



    end_time_generate_lampiran = time.time()


    download_name = f'Report {selected_company} {app.bulan} {app.tahun}'
    p.setTitle(download_name)
    
    p.save()
    buffer.seek(0)

    end_time = time.time()
    execution_time = end_time - start_time
    execution_time_read = end_time_read_data - start_time_read_data
    execution_time_calculate = end_time_calculate_data - start_time_calculate_data
    execution_time_generate_ans = end_time_generate_answer - start_time_generate_answer
    execution_time_generate_conclusion = end_time_generate_conclusion - start_time_generate_conclusion
    execution_time_generate_suggestion = end_time_generate_suggestion - start_time_generate_suggestion
    execution_time_making_plot = end_time_making_plot - start_time_making_plot
    execution_time_making_prompt = end_time_making_prompt - start_time_making_prompt
    execution_time_generate_bab1 = end_time_generate_bab1 - start_time_generate_bab1
    execution_time_generate_bab2 = end_time_generate_bab2 - start_time_generate_bab2
    execution_time_generate_bab3 = end_time_generate_bab3 - start_time_generate_bab3
    execution_time_generate_bab4 = end_time_generate_bab4 - start_time_generate_bab4
    execution_time_generate_toc = end_time_generate_toc - start_time_generate_toc
    execution_time_generate_cover = end_time_generate_cover - start_time_generate_cover
    execution_time_generate_lampiran = end_time_generate_lampiran - start_time_generate_lampiran

    print(f"Program selesai dalam waktu: {execution_time} detik")
    print("Execution time read data:", execution_time_read)
    print("Execution time calculate data:", execution_time_calculate)
    print("Execution time generate answer:", execution_time_generate_ans)
    print("Execution time generate conclusion:", execution_time_generate_conclusion)
    print("Execution time generate suggestion:", execution_time_generate_suggestion)
    print("Execution time making plot:", execution_time_making_plot)
    print("Execution time making prompt:", execution_time_making_prompt)
    print("Execution time generate bab 1:", execution_time_generate_bab1)
    print("Execution time generate bab 2:", execution_time_generate_bab2)
    print("Execution time generate bab 3:", execution_time_generate_bab3)
    print("Execution time generate bab 4:", execution_time_generate_bab4)
    print("Execution time generate TOC:", execution_time_generate_toc)
    print("Execution time generate cover:", execution_time_generate_cover)
    print("Execution time generate lampiran:", execution_time_generate_lampiran)


    return send_file(buffer, as_attachment=True, download_name=f'{download_name}.pdf')

@app.route('/make-pdf-progress')
def make_pdf_progress():
    # floored_progress = int(progress)
    # return jsonify(progress=floored_progress)
    return jsonify(progress=progress)

if __name__ == '__main__':
    app.run(debug=True)
