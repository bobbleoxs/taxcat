import Head from 'next/head';
import Link from 'next/link';
import { useState } from 'react';
import { useRouter } from 'next/router';

export default function Classify() {
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const router = useRouter();
  const exampleSuggestions = [
    "children's books",
    "restaurant meal",
    "office supplies"
  ];

  const handleSubmit = async (e, overrideText) => {
    e.preventDefault();
    setError('');
    const queryText = overrideText || input;
    if (!queryText.trim()) {
      setError('Please enter a description.');
      return;
    }
    setLoading(true);
    try {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
      const res = await fetch(`${apiUrl}/classify`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: queryText })
      });
      if (!res.ok) throw new Error('API error');
      const data = await res.json();
      // Redirect to /results with response data (pass as query param for now)
      router.push({
        pathname: '/results',
        query: { response: data.response, input: queryText }
      });
    } catch (err) {
      setError('Sorry, something went wrong. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleExampleClick = (ex) => {
    setInput(ex);
    // Auto-submit with the example
    handleSubmit({ preventDefault: () => {} }, ex);
  };

  return (
    <div className="min-h-screen bg-white flex flex-col">
      <Head>
        <title>TaxCat - VAT Categorisation</title>
        <meta name="description" content="Describe your product or service and get the correct VAT rate instantly." />
      </Head>

      {/* Header */}
      <header className="border-b border-gray-100">
        <div className="max-w-2xl mx-auto px-4 py-4 flex items-center">
          <Link href="/">
            <div className="flex items-center cursor-pointer">
              <img src="/taxcat-logo.png" alt="TaxCat logo" className="h-9 w-auto mr-2" />
              <span className="text-xl font-bold text-yellow-500">TaxCat</span>
            </div>
          </Link>
        </div>
      </header>

      {/* Main Search UI */}
      <main className="flex-1 flex flex-col items-center justify-center px-4">
        <form
          className="w-full max-w-xl mt-12 mb-8"
          onSubmit={handleSubmit}
        >
          <div className="flex flex-col items-center">
            {/* Country Selector */}
            <div className="mb-6 flex items-center space-x-2">
              <span className="inline-block bg-gray-100 text-gray-700 px-3 py-1 rounded-full text-sm font-medium border border-gray-200">
                🇬🇧 United Kingdom
              </span>
            </div>
            {/* Search Box */}
            <input
              type="text"
              className="w-full text-lg md:text-xl px-6 py-4 border border-gray-200 rounded-lg shadow focus:outline-none focus:ring-2 focus:ring-yellow-400 focus:border-yellow-400 transition mb-4"
              placeholder="Describe your product or service..."
              value={input}
              onChange={e => setInput(e.target.value)}
              autoFocus
              disabled={loading}
            />
            <button
              className={`w-full bg-yellow-500 text-gray-900 font-semibold text-lg py-3 rounded-lg shadow hover:bg-yellow-400 transition-colors mb-2 ${loading ? 'opacity-60 cursor-not-allowed' : ''}`}
              type="submit"
              disabled={loading}
            >
              {loading ? 'Getting VAT Rate...' : 'Get VAT Rate'}
            </button>
            {error && <div className="text-red-500 text-sm mt-2">{error}</div>}
          </div>
          {/* Example Suggestions */}
          <div className="mt-8 flex flex-col items-center space-y-2">
            {exampleSuggestions.map((ex, idx) => (
              <button
                key={idx}
                className="text-gray-500 hover:text-yellow-600 text-base md:text-lg transition-colors underline underline-offset-2"
                onClick={e => { e.preventDefault(); handleExampleClick(ex); }}
                type="button"
                disabled={loading}
              >
                Try: {ex}
              </button>
            ))}
          </div>
        </form>
      </main>
    </div>
  );
}
