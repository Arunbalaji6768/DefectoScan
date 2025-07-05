from flask import Blueprint, redirect, request, session, jsonify
import os
import requests
import json

oauth_bp = Blueprint('oauth', __name__)
GOOGLE_CLIENT_ID = os.environ.get('GOOGLE_CLIENT_ID')
GOOGLE_CLIENT_SECRET = os.environ.get('GOOGLE_CLIENT_SECRET')
GOOGLE_REDIRECT_URI = os.environ.get('GOOGLE_REDIRECT_URI')

@oauth_bp.route('/google/login')
def google_login():
    google_auth_url = (
        'https://accounts.google.com/o/oauth2/v2/auth?'
        f'client_id={GOOGLE_CLIENT_ID}&'
        'response_type=code&'
        f'redirect_uri={GOOGLE_REDIRECT_URI}&'
        'scope=openid%20email%20profile&'
        'access_type=offline&'
        'prompt=consent'
    )
    return redirect(google_auth_url)

@oauth_bp.route('/google/callback')
def google_callback():
    code = request.args.get('code')
    if not code:
        return jsonify({'error': 'No code provided'}), 400
    token_url = 'https://oauth2.googleapis.com/token'
    data = {
        'code': code,
        'client_id': GOOGLE_CLIENT_ID,
        'client_secret': GOOGLE_CLIENT_SECRET,
        'redirect_uri': GOOGLE_REDIRECT_URI,
        'grant_type': 'authorization_code',
    }
    r = requests.post(token_url, data=data)
    if r.status_code != 200:
        return jsonify({'error': 'Failed to get token'}), 400
    tokens = r.json()
    id_token = tokens.get('id_token')
    userinfo_url = 'https://openidconnect.googleapis.com/v1/userinfo'
    headers = {'Authorization': f'Bearer {tokens.get("access_token")}' }
    userinfo = requests.get(userinfo_url, headers=headers).json()
    session['user'] = userinfo
    return redirect(f'/oauth-success?name={userinfo.get("name")}&email={userinfo.get("email")}') 