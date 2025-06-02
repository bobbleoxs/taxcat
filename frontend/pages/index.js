import Head from 'next/head';
import Link from 'next/link';

export default function Home() {
  const exampleQueries = [
    {
      query: "children's books",
      rate: "0%",
      description: "Zero-rated items for children"
    },
    {
      query: "restaurant meal",
      rate: "20%",
      description: "Standard rate for food served in restaurants"
    },
    {
      query: "office supplies",
      rate: "20%",
      description: "Standard rate for business supplies"
    },
    {
      query: "domestic fuel",
      rate: "5%",
      description: "Reduced rate for home energy"
    }
  ];

  return (
    <div className="min-h-screen bg-white">
      <Head>
        <title>TaxCat - Instant VAT Rate Categorisation for UK Businesses</title>
        <meta name="description" content="Get instant VAT rate categorisation for your UK business transactions. Based on official HMRC guidance." />
        <link rel="icon" href="/favicon.ico" />
      </Head>

      {/* Header */}
      <header className="border-b border-gray-100">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex justify-between items-center">
            <div className="flex items-center">
              <img src="/taxcat-logo.png" alt="TaxCat logo" className="h-11 w-auto mr-3" />
              <span className="text-2xl font-bold text-yellow-500">TaxCat</span>
            </div>
            <nav className="hidden md:flex space-x-8">
              <Link href="/classify" className="text-gray-600 hover:text-yellow-500">
                Try Now
              </Link>
              <a href="#examples" className="text-gray-600 hover:text-yellow-500">
                Examples
              </a>
            </nav>
          </div>
        </div>
      </header>

      <main>
        {/* Hero Section */}
        <section className="py-20 bg-gradient-to-b from-gray-50 to-white relative overflow-hidden">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 relative z-10">
            <div className="text-center">
              <h1 className="text-4xl md:text-5xl font-bold text-gray-900 mb-6">
                Instant VAT Rate Categorisation
                <br />
                for UK Businesses
              </h1>
              <p className="text-xl text-gray-600 mb-8 max-w-3xl mx-auto">
                Describe your product or service in plain English - get the correct VAT rate in seconds
              </p>
              <Link href="/classify">
                <button className="bg-yellow-500 text-gray-900 px-8 py-4 rounded-lg text-lg font-semibold hover:bg-yellow-400 transition-colors shadow-lg hover:shadow-xl">
                  Try TaxCat Free
                </button>
              </Link>
            </div>
          </div>
          {/* Hero Accent Logo */}
          <img
            src="/taxcat-logo.png"
            alt="TaxCat hero accent"
            className="hidden md:block absolute right-8 top-8 lg:right-24 lg:top-10 w-64 lg:w-96 opacity-10 pointer-events-none select-none z-0"
            aria-hidden="true"
          />
        </section>

        {/* Trust Section */}
        <section className="py-16 bg-white">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="text-center">
              <h2 className="text-2xl font-semibold text-gray-900 mb-4">
                Based on Official HMRC Guidance
              </h2>
              <p className="text-lg text-gray-600">
                96.2% accuracy on standard cases - we'll flag complex scenarios that need professional review
              </p>
            </div>
          </div>
        </section>

        {/* Example Queries Section */}
        <section id="examples" className="py-16 bg-gray-50">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <h2 className="text-3xl font-bold text-center text-gray-900 mb-12">
              Example Queries
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
              {exampleQueries.map((example, index) => (
                <div key={index} className="bg-white p-6 rounded-lg shadow-md hover:shadow-lg transition-shadow">
                  <h3 className="text-lg font-semibold text-gray-900 mb-2">
                    {example.query}
                  </h3>
                  <p className="text-2xl font-bold text-yellow-500 mb-2">
                    {example.rate}
                  </p>
                  <p className="text-gray-600">
                    {example.description}
                  </p>
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* Disclaimer */}
        <section className="py-8 bg-white">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <p className="text-center text-gray-500 text-sm">
              For guidance only - always verify with HMRC for complex cases
            </p>
          </div>
        </section>
      </main>

      {/* Footer */}
      <footer className="bg-gray-50 border-t border-gray-100">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            <div>
              <h3 className="text-lg font-semibold text-gray-900 mb-4">TaxCat</h3>
              <p className="text-gray-600">
                Making VAT categorisation simple for UK businesses
              </p>
            </div>
            <div>
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Quick Links</h3>
              <ul className="space-y-2">
                <li>
                  <Link href="/classify" className="text-gray-600 hover:text-yellow-500">
                    Try Now
                  </Link>
                </li>
                <li>
                  <a href="#examples" className="text-gray-600 hover:text-yellow-500">
                    Examples
                  </a>
                </li>
              </ul>
            </div>
            <div>
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Legal</h3>
              <ul className="space-y-2">
                <li>
                  <a href="#" className="text-gray-600 hover:text-yellow-500">
                    Terms of Service
                  </a>
                </li>
                <li>
                  <a href="#" className="text-gray-600 hover:text-yellow-500">
                    Privacy Policy
                  </a>
                </li>
              </ul>
            </div>
          </div>
          <div className="mt-8 pt-8 border-t border-gray-100">
            <p className="text-center text-gray-500">
              © {new Date().getFullYear()} TaxCat. All rights reserved.
            </p>
          </div>
        </div>
      </footer>
    </div>
  );
}
