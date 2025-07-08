import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';

export default function LandingPage() {
  const navigate = useNavigate();
  const [showFAQ, setShowFAQ] = useState(false);

  return (
    <div className="relative h-screen w-screen overflow-hidden flex flex-col items-center justify-center bg-black">
      <video
        autoPlay
        loop
        muted
        playsInline
        className="absolute inset-0 w-full h-full object-cover opacity-60 animate-spin-slow"
        style={{ zIndex: 0 }}
        aria-label="Background video of human lungs"
      >
        <source src="/human lungs.mp4" type="video/mp4" />
      </video>
      <div className="absolute inset-0 bg-gradient-to-t from-black/90 via-black/60 to-transparent z-5" />
      <div className="relative z-10 flex flex-col items-center justify-center w-full px-4">
        <div className="backdrop-blur-xl bg-white/10 rounded-3xl shadow-2xl p-10 md:p-16 flex flex-col items-center max-w-2xl w-full border border-white/20">
          <h1 className="text-5xl md:text-7xl font-extrabold text-white drop-shadow-lg mb-4 tracking-tight animate-fade-in">
            DefectoScan
          </h1>
          <p className="text-lg md:text-2xl text-white/90 font-medium mb-8 text-center animate-fade-in delay-100">
            AI-powered Chest X-ray Analysis for Fast, Accurate Diagnosis
          </p>
          <button
            onClick={() => navigate('/login')}
            className="px-10 py-4 bg-gradient-to-r from-blue-600 to-purple-600 text-white text-2xl rounded-full shadow-2xl focus:outline-none focus:ring-4 focus:ring-blue-300 hover:scale-105 hover:bg-blue-700 transition-transform duration-300 font-semibold animate-fade-in delay-200"
            aria-label="Get Started with DefectoScan"
          >
            Get Started
          </button>
          <button
            onClick={() => setShowFAQ(true)}
            className="mt-6 text-blue-200 underline text-base hover:text-white transition-colors duration-150 focus:outline-none focus:ring-2 focus:ring-blue-300"
            aria-label="Open Help or FAQ"
          >
            Help / FAQ
          </button>
        </div>
        <p className="mt-10 text-white/70 text-center text-base md:text-lg max-w-xl animate-fade-in delay-300">
          Upload your chest X-ray and let our advanced AI model assist in detecting pneumonia and other abnormalities in seconds. Secure, private, and easy to use.
        </p>
      </div>
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