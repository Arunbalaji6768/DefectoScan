import { useEffect } from 'react';

export default function OAuthSuccess() {
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const email = params.get('email');
    const name = params.get('name');
    if (window.opener && email && name) {
      window.opener.postMessage({ type: 'oauth-success', user: { email, name } }, '*');
      setTimeout(() => window.close(), 100);
    } else if (email && name) {
      // Fallback: store in localStorage for polling
      localStorage.setItem('oauth-user', JSON.stringify({ email, name }));
      setTimeout(() => window.close(), 100);
    } else {
      setTimeout(() => window.close(), 1000);
    }
  }, []);
  return (
    <div className="flex items-center justify-center min-h-screen">
      <div className="bg-white p-8 rounded shadow text-center">
        <h2 className="text-2xl font-bold mb-4">Login Successful!</h2>
        <p>You can close this window.</p>
      </div>
    </div>
  );
} 