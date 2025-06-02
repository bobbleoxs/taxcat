import Head from 'next/head';
import Link from 'next/link';
import { useRouter } from 'next/router';
import { useEffect, useState } from 'react';

function cleanText(text) {
  if (!text) return '';
  // Remove asterisks, excessive whitespace, and format as paragraphs
  return text
    .replace(/\*\*/g, '')
    .replace(/\n{2,}/g, '\n')
    .replace(/\n/g, '\n\n')
    .replace(/\s{2,}/g, ' ')
    .trim();
}

function getRateColor(rate) {
  if (!rate) return 'bg-gray-200 text-gray-800';
  if (/needs review|unknown/i.test(rate)) return 'bg-red-500 text-white';
  if (/exempt/i.test(rate)) return 'bg-violet-500 text-white';
  if (/out[_\s-]?of[_\s-]?scope/i.test(rate)) return 'bg-gray-400 text-white';
  if (rate.startsWith('0')) return 'bg-green-500 text-white';
  if (rate.startsWith('5')) return 'bg-blue-500 text-white';
  if (rate.startsWith('20')) return 'bg-orange-500 text-white';
  return 'bg-gray-200 text-gray-800';
}

function getRateLabel(rate, rationale) {
  // Use explicit rate if available
  if (rate && rate !== 'N/A' && !/needs review|unknown/i.test(rate)) {
    if (/exempt/i.test(rate)) return 'Exempt';
    if (/out[_\s-]?of[_\s-]?scope/i.test(rate)) return 'Out of Scope';
    if (rate.startsWith('0')) return '0%';
    if (rate.startsWith('5')) return '5%';
    if (rate.startsWith('20')) return '20%';
    return rate;
  }
  // Fallback: analyze rationale/explanation
  const text = (rationale || '').toLowerCase();
  if (/out[-\s]?of[-\s]?scope|outside the scope/.test(text)) return 'Out of Scope';
  if (/reverse charge/.test(text)) return 'Out of Scope';
  if (/exempt/.test(text)) return 'Exempt';
  if (/zero[-\s]?rated|0%/.test(text)) return '0%';
  if (/reduced rate|5%/.test(text)) return '5%';
  if (/standard rate|20%/.test(text)) return '20%';
  // Only show Needs Review if truly unclear
  return 'Needs Review';
}

function shouldShowMerchantNote(rate) {
  if (!rate) return false;
  const label = getRateLabel(rate);
  return label === '0%' || label === '5%' || label === '20%';
}

function confidenceToPercent(conf) {
  if (!conf) return 50;
  if (typeof conf === 'number') return conf;
  if (/100/.test(conf)) return 100;
  if (/95/.test(conf)) return 95;
  if (/90/.test(conf)) return 90;
  if (/85/.test(conf)) return 85;
  if (/80/.test(conf)) return 80;
  if (/75/.test(conf)) return 75;
  if (/70/.test(conf)) return 70;
  if (/60/.test(conf)) return 60;
  if (/50/.test(conf)) return 50;
  if (/high/i.test(conf)) return 95;
  if (/medium/i.test(conf)) return 70;
  if (/low/i.test(conf)) return 40;
  return 50;
}

function rationaleToBullets(rationale) {
  if (!rationale) return [];
  // Split on periods, semicolons, or newlines, filter short/empty, trim
  return rationale
    .replace(/\n/g, ' ')
    .split(/[.;!?]\s+/)
    .map(s => s.trim())
    .filter(s => s.length > 2 && !/^\d+\)/.test(s));
}

function getContextualTripUps(input, parsed) {
  const lower = (input || '').toLowerCase();
  const tripUps = [];

  // Service-related
  if (/service|consulting|advice|project management|design|marketing|legal|accounting/.test(lower)) {
    tripUps.push(
      {
        main: "If provided to EU businesses:",
        detail: "May be subject to reverse charge (customer accounts for VAT)",
      },
      {
        main: "If includes physical deliverables:",
        detail: "Mixed supply rules may apply (different VAT rates for each component)",
      },
      {
        main: "Location of customer:",
        detail: "Place of supply rules determine VAT treatment for international clients",
      }
    );
  } else if (/food|meal|ketchup|condiment|restaurant|takeaway|bread|cake|biscuit|snack|drink|beverage/.test(lower)) {
    tripUps.push(
      {
        main: "Hot vs cold food:",
        detail: "Hot takeaway food is usually standard rated (20%), cold food may be zero-rated (0%)",
      },
      {
        main: "Takeaway vs dine-in:",
        detail: "Dine-in meals are standard rated, some takeaway food can be zero-rated",
      },
      {
        main: "Luxury vs essential:",
        detail: "Luxury food items (e.g. confectionery, crisps) are standard rated, essentials (bread, milk) are often zero-rated",
      }
    );
  } else if (/digital|ebook|e-book|software|download|app|online|subscription/.test(lower)) {
    tripUps.push(
      {
        main: "Digital vs physical:",
        detail: "Digital products (e-books, software) are usually standard rated (20%), physical books may be zero-rated (0%)",
      },
      {
        main: "Place of supply:",
        detail: "Digital services to EU consumers may require VAT registration in each country (OSS/MOSS rules)",
      },
      {
        main: "B2B vs B2C:",
        detail: "Sales to businesses abroad may be out of scope or reverse charge applies",
      }
    );
  } else if (/export|international|abroad|overseas|outside uk/.test(lower)) {
    tripUps.push(
      {
        main: "Exports to non-UK countries:",
        detail: "Goods exported outside the UK are usually zero-rated, but strict evidence rules apply",
      },
      {
        main: "B2B vs B2C:",
        detail: "Sales to VAT-registered businesses abroad may be out of scope or reverse charge applies",
      },
      {
        main: "Place of supply:",
        detail: "VAT treatment depends on where the customer belongs and the nature of the supply",
      }
    );
  } else if (/book|magazine|publication|journal/.test(lower)) {
    tripUps.push(
      {
        main: "Physical vs digital:",
        detail: "Physical books are usually zero-rated, digital publications are standard rated (20%)",
      },
      {
        main: "Educational vs entertainment:",
        detail: "Some educational materials may be zero-rated, entertainment publications are standard rated",
      },
      {
        main: "Children's vs adult:",
        detail: "Children's books are zero-rated, some adult magazines may be standard rated",
      }
    );
  } else {
    // General trip-ups
    tripUps.push(
      {
        main: "Mixed supplies:",
        detail: "If your sale includes items with different VAT rates, each part may need to be treated separately",
      },
      {
        main: "Cross-border transactions:",
        detail: "International sales often have different VAT rules (reverse charge, out of scope, zero-rating)",
      },
      {
        main: "Unusual business models:",
        detail: "Special VAT rules may apply for vouchers, agents, or new digital services",
      }
    );
  }

  // Always add professional advice prompt
  tripUps.push({
    main: "VAT can be complex:",
    detail: "Consider professional advice for: multiple supply components, cross-border elements, or unusual business models",
    advice: true,
  });

  return tripUps;
}

export default function Results() {
  const router = useRouter();
  const { response, input } = router.query;
  const [parsed, setParsed] = useState(null);

  useEffect(() => {
    let parsedData = null;
    try {
      if (response) {
        // Try to parse as JSON first
        parsedData = typeof response === 'string' && response.trim().startsWith('{')
          ? JSON.parse(response)
          : null;
      }
    } catch (e) {
      parsedData = null;
    }
    if (!parsedData && response) {
      // Clean the response string before regex
      const cleanedResponse = response.replace(/\*/g, '').replace(/\r/g, '');
      // Improved regex for VAT Rate (handles numbers and percent)
      const rateMatch = cleanedResponse.match(/VAT Rate\s*[:\-]?\s*([0-9]+%|exempt|out_of_scope|unknown|needs review)/i);
      const rationaleMatch = cleanedResponse.match(/Short Rationale\s*[:\-]?\s*([\s\S]*?)(?:Source|Confidence|$)/i);
      const confidenceMatch = cleanedResponse.match(/Confidence\s*[:\-]?\s*([\w%]+)/i);
      const sourceMatch = cleanedResponse.match(/Source URL\s*[:\-]?\s*([\S ]+)/i);
      parsedData = {
        rate: rateMatch ? rateMatch[1].trim() : null,
        rationale: rationaleMatch ? rationaleMatch[1].replace(/\n/g, ' ').trim() : null,
        confidence: confidenceMatch ? confidenceMatch[1].trim() : null,
        source: sourceMatch ? sourceMatch[1].replace(/N\/A/i, '').trim() : null,
      };
    }
    console.log('Raw response:', response);
    console.log('Parsed data:', parsedData);
    setParsed(parsedData);
  }, [response]);

  // Handle missing data
  if (!response || !input) {
    return (
      <div className="min-h-screen flex flex-col items-center justify-center bg-white">
        <Head>
          <title>TaxCat - VAT Categorisation Result</title>
        </Head>
        <div className="max-w-md w-full p-8 bg-white rounded-lg shadow text-center">
          <img src="/taxcat-logo.png" alt="TaxCat logo" className="h-12 mx-auto mb-4" />
          <h2 className="text-xl font-bold mb-2">No result to display</h2>
          <p className="mb-4 text-gray-600">Please start a new search.</p>
          <Link href="/classify">
            <button className="bg-yellow-500 text-gray-900 px-6 py-2 rounded-lg font-semibold hover:bg-yellow-400 transition-colors">New Search</button>
          </Link>
        </div>
      </div>
    );
  }

  const tripUps = getContextualTripUps(input, parsed);
  const rateLabel = getRateLabel(parsed?.rate, parsed?.rationale);
  const sourceDisplay = parsed?.source && parsed?.source.length > 2
    ? `Source: HMRC VAT Notice (${parsed.source})`
    : 'Source: HMRC VAT Guidance';
  const bullets = rationaleToBullets(parsed?.rationale);

  // Confidence bar width (default 70 for medium)
  let confidencePercent = 70;
  if (parsed?.confidence) {
    if (/high/i.test(parsed.confidence)) confidencePercent = 95;
    else if (/medium/i.test(parsed.confidence)) confidencePercent = 70;
    else if (/low/i.test(parsed.confidence)) confidencePercent = 40;
  }

  return (
    <div className="min-h-screen bg-white flex flex-col">
      <Head>
        <title>TaxCat - VAT Categorisation Result</title>
      </Head>
      {/* Header */}
      <header className="border-b border-gray-100">
        <div className="max-w-2xl mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center">
            <img src="/taxcat-logo.png" alt="TaxCat logo" className="h-9 w-auto mr-2" />
            <span className="text-xl font-bold text-yellow-500">TaxCat</span>
          </div>
          <Link href="/classify">
            <button className="bg-yellow-500 text-gray-900 px-4 py-2 rounded-lg font-semibold hover:bg-yellow-400 transition-colors">New Search</button>
          </Link>
        </div>
      </header>

      <main className="flex-1 flex flex-col items-center px-4 py-8">
        {/* Query Card */}
        <div className="w-full max-w-xl mb-6">
          <div className="bg-gray-50 border border-gray-100 rounded-lg shadow-sm p-4 text-center text-gray-700 text-base">
            <span className="font-medium text-gray-500">You searched for:</span>
            <div className="mt-1 text-lg text-gray-900">{input}</div>
          </div>
        </div>

        {/* Main Result Card */}
        <div className="w-full max-w-xl mb-10">
          <div className="rounded-xl shadow-lg p-6 flex flex-col items-center">
            {/* VAT Rate */}
            <div className={`text-4xl font-extrabold rounded-lg px-8 py-4 mb-4 ${getRateColor(parsed?.rate)}`}>{rateLabel}</div>
            {/* Merchant/Supplier Perspective Note */}
            {shouldShowMerchantNote(parsed?.rate) && (
              <div className="text-xs text-gray-500 mb-2 text-center">Rate from merchant/supplier perspective (what you should charge customers)</div>
            )}
            {/* Rationale as bullet points */}
            <ul className="text-gray-800 text-left mb-4 w-full max-w-lg mx-auto list-disc pl-6">
              {bullets.length > 0 ? bullets.map((b, i) => (
                <li key={i} className="mb-1">{b}</li>
              )) : <li>No rationale provided.</li>}
            </ul>
            {/* Confidence Bar */}
            <div className="w-full mb-2">
              <div className="flex justify-between text-xs text-gray-500 mb-1">
                <span>Less Confident</span>
                <span>More Confident</span>
              </div>
              <div className="w-full bg-yellow-100 rounded-full h-3">
                <div
                  className="h-3 rounded-full transition-all duration-300"
                  style={{
                    width: `${confidencePercent}%`,
                    background: 'linear-gradient(90deg, #fde68a 0%, #f59e42 100%)',
                  }}
                ></div>
              </div>
            </div>
            {/* Source */}
            <div className="mt-2 text-xs text-gray-500">
              {sourceDisplay}
            </div>
          </div>
        </div>

        {/* Common Trip-ups Section */}
        <div className="w-full max-w-xl">
          <h3 className="text-lg font-semibold mb-4 text-gray-900">Common Trip-ups for Similar Items</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {tripUps.map((ex, idx) => (
              <div key={idx} className={`bg-white border border-gray-100 rounded-lg shadow p-4 ${ex.advice ? 'border-yellow-400' : ''}`}>
                <div className={`font-medium text-gray-800 mb-1 ${ex.advice ? 'text-yellow-600' : ''}`}>{ex.main}</div>
                <div className="text-sm text-gray-500">{ex.detail}</div>
              </div>
            ))}
          </div>
        </div>
      </main>
    </div>
  );
}
