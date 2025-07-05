# DefectoScan Deployment Configuration

## Environment Variables Required

Create a `.env` file or set these environment variables in your deployment platform:

```bash
# Flask Configuration
SECRET_KEY=your_super_secret_key_here
FLASK_ENV=production

# MongoDB Configuration
MONGO_URI=mongodb+srv://username:password@cluster.mongodb.net/?retryWrites=true&w=majority

# Twilio Configuration (for SMS OTP)
TWILIO_SID=your_twilio_sid
TWILIO_AUTH=your_twilio_auth_token
TWILIO_PHONE=your_twilio_phone_number

# Google OAuth Configuration
GOOGLE_CLIENT_ID=your_google_client_id
GOOGLE_CLIENT_SECRET=your_google_client_secret
GOOGLE_REDIRECT_URI=https://your-domain.com/google/callback

# Server Configuration
PORT=5000
```

## Deployment Steps

1. **Set up environment variables** in your deployment platform
2. **Build the frontend**: `cd DefectoScan/frontend && npm run build`
3. **Deploy to Heroku/Railway/Render** using the Procfile
4. **Configure CORS** if needed for your domain
5. **Set up MongoDB Atlas** database
6. **Configure Twilio** for SMS functionality
7. **Set up Google OAuth** credentials

## Current Issues Fixed

- ✅ Fixed Twilio SID environment variable typo
- ✅ Added missing dependencies to requirements.txt
- ✅ Made secret key configurable via environment variable
- ✅ Made MongoDB URI configurable via environment variable
- ✅ Fixed API endpoint mismatch between frontend and backend
- ✅ Added proxy route for API compatibility

## Remaining Tasks

- [ ] Set up environment variables in deployment platform
- [ ] Configure Google OAuth credentials
- [ ] Set up Twilio account and credentials
- [ ] Test OTP functionality
- [ ] Test OAuth flow
- [ ] Configure CORS for production domain
- [ ] Set up proper SSL certificates
- [ ] Test complete user flow in production 