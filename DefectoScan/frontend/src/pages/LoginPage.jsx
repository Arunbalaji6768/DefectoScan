import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { GoogleLogin, googleLogout } from '@react-oauth/google';
import { jwtDecode } from 'jwt-decode';

export default function LoginPage() {
  const [step, setStep] = useState(0); // 0: input, 1: OTP or password
  const [input, setInput] = useState('');
  const [isMobile, setIsMobile] = useState(false);
  const [otp, setOtp] = useState('');
  const [password, setPassword] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [showFAQ, setShowFAQ] = useState(false);
  const navigate = useNavigate();

  useEffect(() => {
    const listener = (event) => {
      if (event.data && event.data.type === 'oauth-success') {
        localStorage.setItem('user', JSON.stringify(event.data.user));
        navigate('/upload');
      }
    };
    window.addEventListener('message', listener);
    // Fallback: Poll localStorage for oauth-user
    const interval = setInterval(() => {
      const user = localStorage.getItem('oauth-user');
      if (user) {
        localStorage.setItem('user', user);
        localStorage.removeItem('oauth-user');
        navigate('/upload');
      }
    }, 500);
    return () => {
      window.removeEventListener('message', listener);
      clearInterval(interval);
    };
  }, []);

  const handleContinue = async () => {
    setError('');
    if (/^\d{10}$/.test(input)) {
      setIsMobile(true);
      setLoading(true);
      try {
        const res = await fetch('/api/send-otp', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ phone: input })
        });
        if (!res.ok) throw new Error('Failed to send OTP');
        setStep(1);
      } catch (err) {
        setError('Failed to send OTP. Try again.');
      } finally {
        setLoading(false);
      }
    } else if (/^[^@\s]+@[^@\s]+\.[^@\s]+$/.test(input)) {
      setIsMobile(false);
      setStep(1);
    } else {
      setError('Please enter a valid email or 10-digit mobile number.');
    }
  };

  const handleVerifyOtp = async () => {
    setError('');
    setLoading(true);
    try {
      const res = await fetch('/api/verify-otp', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ phone: input, otp })
      });
      if (res.ok) {
        setStep(2);
        setError("");
        localStorage.setItem('user', JSON.stringify({ phone: input }));
        navigate('/upload');
        return;
      }
      throw new Error('Invalid OTP');
    } catch (err) {
      setError('Invalid OTP. Try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleLogin = () => {
    // Remove mock login logic for email
    setError('Please use OTP or Google login.');
  };

  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-gradient-to-br from-white via-blue-100 to-purple-100 p-8 animate-fade-in relative">
      {/* Back Button - OUTSIDE the card, top left of viewport */}
      <button
        className="fixed top-6 left-6 flex items-center gap-2 text-blue-700 hover:text-blue-900 font-semibold text-lg bg-white/80 rounded-full px-4 py-2 shadow-lg transition-all duration-150 border border-blue-100 hover:bg-blue-50 z-50"
        onClick={() => navigate('/')} style={{zIndex: 50}}
        aria-label="Back to landing page"
      >
        <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor" className="w-5 h-5">
          <path strokeLinecap="round" strokeLinejoin="round" d="M15.75 19.5L8.25 12l7.5-7.5" />
        </svg>
        Back
      </button>
      <div className="bg-white/80 rounded-3xl shadow-2xl p-10 w-full max-w-md flex flex-col gap-8 items-center relative">
        {/* Heading only, no logo */}
        <h2 className="text-3xl font-bold text-center text-blue-900 mb-4">Sign in to DefectoScan</h2>
        {/* Error Message */}
        {error && (
          <div className="w-full bg-red-100 text-red-700 rounded-xl px-4 py-2 text-center text-sm font-semibold animate-fade-in">
            {error}
          </div>
        )}
        <div className="flex flex-col items-center gap-4 mt-4 w-full">
          {/* OTP Login Section */}
          {step === 0 && (
            <>
              <input
                type="text"
                className="w-full px-4 py-3 rounded-xl border border-gray-300 focus:outline-none focus:ring-2 focus:ring-blue-400 text-lg bg-white/80"
                placeholder="Enter your 10-digit mobile number"
                value={input}
                onChange={e => setInput(e.target.value)}
                maxLength={10}
                disabled={loading}
              />
              <button
                className="w-full bg-blue-600 hover:bg-blue-700 py-3 px-4 rounded-xl text-white font-semibold shadow-lg text-lg transition-all duration-150"
                onClick={handleContinue}
                disabled={loading}
              >
                {loading ? 'Sending OTP...' : 'Send OTP'}
              </button>
            </>
          )}
          {step === 1 && isMobile && (
            <>
              <input
                type="text"
                className="w-full px-4 py-3 rounded-xl border border-gray-300 focus:outline-none focus:ring-2 focus:ring-blue-400 text-lg bg-white/80"
                placeholder="Enter OTP"
                value={otp}
                onChange={e => setOtp(e.target.value)}
                maxLength={6}
                disabled={loading}
              />
              <button
                className="w-full bg-green-600 hover:bg-green-700 py-3 px-4 rounded-xl text-white font-semibold shadow-lg text-lg transition-all duration-150"
                onClick={handleVerifyOtp}
                disabled={loading}
              >
                {loading ? 'Verifying...' : 'Verify OTP'}
              </button>
              <button
                className="w-full bg-gray-300 hover:bg-gray-400 py-2 px-4 rounded-xl text-gray-800 font-semibold shadow text-sm mt-2 transition-all duration-150"
                onClick={() => { setStep(0); setOtp(''); }}
                disabled={loading}
              >
                Back
              </button>
            </>
          )}
          {/* Divider */}
          <div className="flex items-center w-full my-4">
            <div className="flex-grow h-px bg-gray-300"></div>
            <span className="mx-2 bg-white rounded-full px-3 py-1 text-gray-500 font-semibold shadow">or</span>
            <div className="flex-grow h-px bg-gray-300"></div>
          </div>
          {/* Google Login Section */}
          <div className="w-full flex flex-col items-center">
            <GoogleLogin
              onSuccess={credentialResponse => {
                try {
                  const decoded = jwtDecode(credentialResponse.credential);
                  localStorage.setItem('user', JSON.stringify({
                    email: decoded.email,
                    name: decoded.name,
                    picture: decoded.picture
                  }));
                  navigate('/upload');
                } catch (e) {
                  setError('Google login failed.');
                }
              }}
              onError={() => setError('Google login failed.')}
              width="100%"
              useOneTap
            />
          </div>
        </div>
        <button
          onClick={() => setShowFAQ(true)}
          className="mt-2 text-blue-700 underline text-base hover:text-blue-900 transition-colors duration-150 focus:outline-none focus:ring-2 focus:ring-blue-300"
          aria-label="Open Help or FAQ"
        >
          Help / FAQ
        </button>
      </div>
      {/* Footer */}
      <footer className="absolute bottom-0 left-0 w-full py-4 bg-black/60 text-white text-center text-sm z-20 flex flex-col md:flex-row items-center justify-between px-6 gap-2">
        <span>&copy; {new Date().getFullYear()} DefectoScan. All rights reserved.</span>
        <button
          onClick={() => setShowFAQ(true)}
          className="text-blue-200 underline hover:text-white transition-colors duration-150 focus:outline-none focus:ring-2 focus:ring-blue-300"
          aria-label="Open Help or FAQ from footer"
        >
          Help / FAQ
        </button>
      </footer>
      {/* FAQ Modal */}
      {showFAQ && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60">
          <div className="bg-white rounded-2xl shadow-2xl p-8 max-w-lg w-full relative animate-fade-in">
            <button
              className="absolute top-4 right-4 text-gray-500 hover:text-blue-700 text-2xl font-bold focus:outline-none"
              onClick={() => setShowFAQ(false)}
              aria-label="Close Help or FAQ"
            >
              &times;
            </button>
            <h2 className="text-2xl font-bold text-blue-900 mb-4">Help & FAQ</h2>
            <ul className="list-disc pl-6 text-blue-900 text-base space-y-2">
              <li><strong>What is DefectoScan?</strong> <br/>DefectoScan is an AI-powered tool for analyzing chest X-ray images to assist in detecting pneumonia and other abnormalities.</li>
              <li><strong>How do I use it?</strong> <br/>Sign in, upload your chest X-ray image, and view the AI's prediction in seconds.</li>
              <li><strong>Is my data private?</strong> <br/>Yes, your images are processed securely and are not shared with third parties.</li>
              <li><strong>Who can I contact for support?</strong> <br/>Email: support@defectoscan.com</li>
            </ul>
          </div>
        </div>
      )}
    </div>
  );
} 