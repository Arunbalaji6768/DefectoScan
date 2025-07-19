import React, { useState, useEffect } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:5000";
function useQuery() {
  return new URLSearchParams(useLocation().search);
}

export default function UploadPage() {
  console.log('UploadPage component is being rendered');
  
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState('');
  const [rawResponse, setRawResponse] = useState(null);
  const [user, setUser] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const query = useQuery();
  const navigate = useNavigate();
  const [showFAQ, setShowFAQ] = useState(false);

  useEffect(() => {
    // Get user from localStorage
    try {
      const userData = localStorage.getItem('user');
      console.log('UploadPage: Retrieved user data from localStorage:', userData);
      if (userData) {
        const parsedUser = JSON.parse(userData);
        console.log('UploadPage: Parsed user data:', parsedUser);
        setUser(parsedUser);
      } else {
        console.log('UploadPage: No user data found in localStorage');
      }
    } catch (e) {
      console.error('Error parsing user data:', e);
    }
    setIsLoading(false);
  }, []);

  useEffect(() => {
    console.log('UploadPage: Authentication check - isLoading:', isLoading, 'user:', user);
    if (!isLoading && (!user || (!user.email && !user.phone && !user.name))) {
      console.log('UploadPage: Redirecting to login - user not authenticated');
      navigate('/login');
    } else if (!isLoading && user) {
      console.log('UploadPage: User authenticated successfully:', user);
    }
  }, [user, isLoading, navigate]);

  // Show loading state while checking authentication
  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-white via-blue-100 to-purple-100">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
          <p className="text-blue-800 text-lg">Loading...</p>
        </div>
      </div>
    );
  }

  // Don't render the main content if user is not authenticated
  if (!user || (!user.email && !user.phone && !user.name)) {
    return null;
  }

  const email = user?.email || query.get('email');
  const name = user?.name || query.get('name');

  const handleFileChange = (e) => {
    const f = e.target.files[0];
    setFile(f);
    if (f) {
      setPreview(URL.createObjectURL(f));
      setResult('');
    } else {
      setPreview(null);
      setResult('');
    }
  };

  const handleUpload = async () => {
    if (!file) return;
    const formData = new FormData();
    formData.append('file', file);
    setResult('Predicting...');
    setRawResponse(null);
    console.log('Using API_URL:', API_URL);
    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        body: formData,
      });
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        setResult(`Error: ${errorData.error || 'Prediction failed (status ' + response.status + ')'}\nCheck your backend and API URL.`);
        return;
      }
      const data = await response.json();
      setRawResponse(data);
      setResult(`Prediction: ${data.label} (Confidence: ${data.confidence})`);
    } catch (err) {
      setResult('Error: Could not connect to backend. Please check your API URL, backend deployment, and network.');
      console.error('Error connecting to backend:', err);
    }
  };

  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-gradient-to-br from-white via-blue-100 to-purple-100 p-8 animate-fade-in relative">
      {/* Back Button - OUTSIDE the card, top left of viewport */}
      <button
        className="fixed top-6 left-6 flex items-center gap-2 text-blue-700 hover:text-blue-900 font-semibold text-lg bg-white/80 rounded-full px-4 py-2 shadow-lg transition-all duration-150 border border-blue-100 hover:bg-blue-50 z-50"
        onClick={() => navigate(-1)}
        style={{zIndex: 50}}
        aria-label="Back to previous page"
      >
        <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor" className="w-5 h-5">
          <path strokeLinecap="round" strokeLinejoin="round" d="M15.75 19.5L8.25 12l7.5-7.5" />
        </svg>
        Back
      </button>
      <div className="bg-white/80 rounded-3xl shadow-2xl p-10 w-full max-w-xl flex flex-col gap-8 items-center">
        {/* Home & Logout Buttons */}
        <div className="w-full flex justify-end mb-2 gap-2 mt-2">
          <button
            className="text-blue-700 underline text-sm font-semibold hover:text-blue-900 focus:outline-none focus:ring-2 focus:ring-blue-300 transition-colors duration-150"
            onClick={() => navigate('/')}
            aria-label="Go to Home"
          >
            Home
          </button>
          <button
            className="text-red-600 underline text-sm font-semibold hover:text-red-800 focus:outline-none focus:ring-2 focus:ring-red-300 transition-colors duration-150"
            onClick={() => { localStorage.removeItem('user'); navigate('/login'); }}
            aria-label="Logout"
          >
            Logout
          </button>
        </div>
        <h2 className="text-3xl font-bold text-center text-blue-900 mb-2">Upload X-ray Image</h2>
        <p className="text-center text-blue-800/80 mb-4 text-lg">Select a chest X-ray image to analyze for abnormalities using our AI model.</p>
        <label htmlFor="file-upload" className="w-full cursor-pointer flex flex-col items-center justify-center border-2 border-dashed border-blue-400 bg-white/70 hover:bg-blue-50 transition-all duration-150 rounded-2xl py-10 px-4 shadow-inner mb-2 focus-within:ring-2 focus-within:ring-blue-300">
          <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-12 h-12 text-blue-400 mb-2">
            <path strokeLinecap="round" strokeLinejoin="round" d="M3 16.5V19a2.003 2.003 0 002 2h14a2.003 2.003 0 002-2v-2.5M16.5 12.75L12 17.25m0 0l-4.5-4.5M12 17.25V4.5" />
          </svg>
          <span className="text-blue-700 font-medium">Click or drag & drop to upload</span>
          <input
            id="file-upload"
            type="file"
            accept="image/*"
            className="hidden"
            onChange={handleFileChange}
            aria-label="Upload X-ray image"
          />
        </label>
        {preview && (
          <img src={preview} alt="Preview" className="w-64 h-64 object-contain rounded-xl border-2 border-blue-200 shadow-lg mb-2" />
        )}
        <button
          className="w-full bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 py-3 px-4 rounded-xl text-white font-semibold shadow-lg text-lg transition-all duration-150 mt-2 focus:outline-none focus:ring-2 focus:ring-blue-300"
          onClick={handleUpload}
          disabled={!file}
          aria-label="Upload image"
        >
          Upload
        </button>
      </div>
      {/* Floating result card at bottom right */}
      {result && (
        <div className="fixed bottom-8 right-8 z-50 animate-fade-in">
          <div className="backdrop-blur-xl bg-white/70 border border-blue-200 rounded-2xl shadow-2xl px-8 py-6 flex flex-col items-center min-w-[260px]">
            <span className="text-xl font-bold text-blue-900 mb-2">Result</span>
            <span className="text-lg text-blue-800 font-semibold">{result}</span>
          </div>
        </div>
      )}
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