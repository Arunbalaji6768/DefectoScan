from flask import Blueprint, request, jsonify, session
from twilio.rest import Client
import os
import random
import time

otp_bp = Blueprint('otp', __name__)
TWILIO_SID = os.environ.get('TWILIO_SID')
TWILIO_AUTH = os.environ.get('TWILIO_AUTH')
TWILIO_PHONE = os.environ.get('TWILIO_PHONE')
OTP_EXPIRY = 300

@otp_bp.route('/send-otp', methods=['POST'])
def send_otp():
    data = request.get_json()
    phone = data.get('phone')
    if not phone:
        return jsonify({'error': 'Phone number required'}), 400
    otp = str(random.randint(100000, 999999))
    session['otp'] = otp
    session['otp_time'] = int(time.time())
    if TWILIO_SID and TWILIO_AUTH and TWILIO_PHONE:
        client = Client(TWILIO_SID, TWILIO_AUTH)
        try:
            client.messages.create(
                body=f'Your OTP is {otp}',
                from_=TWILIO_PHONE,
                to=phone
            )
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    print(f"OTP for {phone}: {otp}")
    return jsonify({'message': 'OTP sent'})

@otp_bp.route('/verify-otp', methods=['POST'])
def verify_otp():
    data = request.get_json()
    otp = data.get('otp')
    if not otp:
        return jsonify({'error': 'OTP required'}), 400
    if 'otp' not in session or 'otp_time' not in session:
        return jsonify({'error': 'No OTP sent'}), 400
    if int(time.time()) - session['otp_time'] > OTP_EXPIRY:
        return jsonify({'error': 'OTP expired'}), 400
    if otp == session['otp']:
        session.pop('otp')
        session.pop('otp_time')
        return jsonify({'message': 'OTP verified'})
    return jsonify({'error': 'Invalid OTP'}), 400 