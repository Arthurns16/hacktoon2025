import { useState } from 'react';
import { Upload, FileText, Download, AlertCircle, CheckCircle2 } from 'lucide-react';

function App() {
  const [files, setFiles] = useState<File[]>([]);
  const [pdfUrls, setPdfUrls] = useState<string[]>([]);
  const [extracted, setExtracted] = useState<string[]>([]);
  const [loading, setLoading] = useState<boolean[]>([]);
  const [error, setError] = useState<(string | null)[]>([]);
  const [fields, setFields] = useState<string[]>([]);
  const [fieldInput, setFieldInput] = useState<string>('');

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFiles = Array.from(e.target.files || []).filter(f => f.type === 'application/pdf');
    if (selectedFiles.length > 0) {
      setFiles(selectedFiles);
      setPdfUrls(selectedFiles.map(f => URL.createObjectURL(f)));
      setExtracted(Array(selectedFiles.length).fill(''));
      setError(Array(selectedFiles.length).fill(null));
      setLoading(Array(selectedFiles.length).fill(false));
    } else {
      setFiles([]);
      setPdfUrls([]);
      setExtracted([]);
      setError(['Please select PDF files.']);
      setLoading([]);
    }
  };

  const [combinedCsv, setCombinedCsv] = useState<string>('');
  const [extracting, setExtracting] = useState(false);
  const handleExtractAll = async () => {
    if (!files.length) return;
    setExtracting(true);
    setError([]);
    setExtracted(Array(files.length).fill(''));
    setCombinedCsv('');
    try {
      const formData = new FormData();
      files.forEach(file => formData.append('files', file));
      formData.append('fields', fields.join(','));
      const res = await fetch('http://localhost:5000/extract-multi-file-data', {
        method: 'POST',
        body: formData,
      });
      if (!res.ok) throw new Error('Failed to extract PDFs');
      const data = await res.json();
      if (data.results) {
        setExtracted(data.results.map((r: any) => r.result || r.error || ''));
      }
      if (data.combinedCsv) {
        setCombinedCsv(data.combinedCsv);
      }
    } catch (err) {
      setError([err instanceof Error ? err.message : 'Error extracting PDFs']);
    } finally {
      setExtracting(false);
    }
  };



  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-100">
      <div className="container mx-auto px-4 py-8 max-w-6xl">
        {/* Header */}
        <div className="text-center mb-8">
          <div className="inline-flex items-center justify-center w-24 h-24  mb-4 ">
          <img src="./logo.png" alt="Logo" />
          </div>
          <h1 className="text-4xl font-bold text-gray-900 mb-2">Forja de Matéria AI</h1>
          <p className="text-lg text-gray-600">Extract and analyze data from your PDF documents with ease</p>
        </div>

        <div className="grid lg:grid-cols-2 gap-8">
          {/* Upload & Preview Section */}
          <div className="space-y-6">
            {/* Upload Card */}
            <div className="bg-white/70 backdrop-blur-sm rounded-3xl p-8 shadow-xl border border-white/50">
              <h2 className="text-xl font-semibold text-gray-800 mb-6 flex items-center gap-3">
                <Upload size={24} className="text-blue-600" />
                Upload Document
              </h2>
              
              <div className="relative">
                <input
                  type="file"
                  accept="application/pdf"
                  multiple
                  onChange={handleFileChange}
                  className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                  id="pdf-upload"
                />
                <label
                  htmlFor="pdf-upload"
                  className={`
                    flex flex-col items-center justify-center p-8 border-2 border-dashed rounded-2xl
                    transition-all duration-200 cursor-pointer
                    ${files.length > 0
                      ? 'border-green-300 bg-green-50'
                      : 'border-gray-300 bg-gray-50 hover:border-blue-400 hover:bg-blue-50'
                    }
                  `}
                >
                  {files.length > 0 ? (
                    <>
                      <CheckCircle2 size={48} className="text-green-600 mb-4" />
                      <p className="text-lg font-medium text-green-700 mb-2">{files.length} PDF{files.length > 1 ? 's' : ''} selected</p>
                      <ul className="text-sm text-green-600 mb-2">
                        {files.map(f => <li key={f.name}>{f.name}</li>)}
                      </ul>
                      <p className="text-sm text-green-600">Ready for extraction</p>
                    </>
                  ) : (
                    <>
                      <Upload size={48} className="text-gray-400 mb-4" />
                      <p className="text-lg font-medium text-gray-700 mb-2">Drop your PDFs here</p>
                      <p className="text-sm text-gray-500">or click to browse</p>
                    </>
                  )}
                </label>
              </div>

              {error && error.some(e => e) && (
                <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded-xl flex items-start gap-3">
                  <AlertCircle size={20} className="text-red-600 flex-shrink-0 mt-0.5" />
                  <ul className="text-red-700">
                    {error.map((e, i) => e && <li key={i}>{e}</li>)}
                  </ul>
                </div>
              )}

              {/* Input for specifying fields/data points */}
              <div className="mt-6">
                <label htmlFor="fields-input" className="block text-gray-700 font-medium mb-2">
                  Specify the exact data columns or fields to extract
                </label>
                <div className="flex flex-wrap gap-2 mb-2">
                  {fields.map((field, idx) => (
                    <span
                      key={field + idx}
                      className="inline-flex items-center px-3 py-1 bg-blue-100 text-blue-800 rounded-full text-sm font-medium mr-2 mb-2"
                    >
                      {field}
                      <button
                        type="button"
                        className="ml-2 text-blue-500 hover:text-red-500 focus:outline-none"
                        onClick={() => setFields(fields.filter((_, i) => i !== idx))}
                        aria-label={`Remove ${field}`}
                      >
                        ×
                      </button>
                    </span>
                  ))}
                </div>
                <div className="flex gap-2">
                  <input
                    id="fields-input"
                    type="text"
                    value={fieldInput}
                    onChange={e => setFieldInput(e.target.value)}
                    onKeyDown={e => {
                      if ((e.key === 'Enter' || e.key === ',') && fieldInput.trim()) {
                        e.preventDefault();
                        if (!fields.includes(fieldInput.trim())) {
                          setFields([...fields, fieldInput.trim()]);
                        }
                        setFieldInput('');
                      }
                    }}
                    placeholder="Type a field and press Enter or Comma"
                    className="w-full px-4 py-3 rounded-xl border border-gray-300 focus:border-blue-500 focus:ring-2 focus:ring-blue-100 outline-none text-gray-800 bg-white"
                  />
                  <button
                    type="button"
                    className="px-4 py-2 bg-blue-600 text-white rounded-xl font-medium hover:bg-blue-700"
                    onClick={() => {
                      if (fieldInput.trim() && !fields.includes(fieldInput.trim())) {
                        setFields([...fields, fieldInput.trim()]);
                        setFieldInput('');
                      }
                    }}
                  >
                    Add
                  </button>
                </div>
                <p className="text-xs text-gray-500 mt-1">Press Enter or Comma to add multiple fields.</p>
              </div>

              {files.length > 0 && (
                <div className="mt-6">
                  <button
                    onClick={handleExtractAll}
                    disabled={extracting}
                    className={`w-full px-6 py-4 rounded-2xl font-semibold text-white transition-all duration-200 transform
                      ${extracting
                        ? 'bg-gray-300 cursor-not-allowed'
                        : 'bg-blue-600 hover:bg-blue-700 hover:scale-[1.02] active:scale-[0.98] shadow-lg hover:shadow-xl'
                      }`}
                  >
                    {extracting ? (
                      <div className="flex items-center justify-center gap-3">
                        <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin"></div>
                        Extracting All PDFs...
                      </div>
                    ) : (
                      'Extract Data from All PDFs'
                    )}
                  </button>
                </div>
              )}
            </div>

            {/* PDF Previews */}
            {pdfUrls.length > 0 && (
              <div className="bg-white/70 backdrop-blur-sm rounded-3xl p-8 shadow-xl border border-white/50">
                <h2 className="text-xl font-semibold text-gray-800 mb-6">Document Previews</h2>
                <div className="grid gap-6">
                  {pdfUrls.map((url, idx) => (
                    <div key={url} className="bg-white rounded-2xl p-4 shadow-inner">
                      <div className="font-semibold mb-2">{files[idx]?.name}</div>
                      <embed
                        src={url}
                        type="application/pdf"
                        width="100%"
                        height="300px"
                        className="rounded-xl"
                      />
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* Results Section */}
          <div>
            <div className="bg-white rounded-2xl p-6 shadow-lg border border-gray-100 h-full">
              <h2 className="text-xl font-semibold text-gray-900 flex items-center gap-3 mb-6">
                <FileText size={24} className="text-blue-600" />
                Extracted Data
              </h2>
              {/* Combined CSV Download */}
              {combinedCsv && (
                <div className="mb-6">
                  <button
                    onClick={() => {
                      const blob = new Blob([combinedCsv], { type: 'text/csv' });
                      const url = URL.createObjectURL(blob);
                      const a = document.createElement('a');
                      a.href = url;
                      a.download = 'combined-extracted.csv';
                      document.body.appendChild(a);
                      a.click();
                      setTimeout(() => {
                        document.body.removeChild(a);
                        URL.revokeObjectURL(url);
                      }, 0);
                    }}
                    className="px-4 py-2 bg-green-700 hover:bg-green-800 text-white rounded-lg text-sm font-medium transition-colors duration-200 flex items-center gap-2 shadow-sm"
                  >
                    <Download size={16} />
                    Download Combined CSV
                  </button>
                  <pre className="mt-4 text-xs text-gray-800 whitespace-pre-wrap font-mono leading-relaxed max-h-[300px] overflow-y-auto bg-white rounded p-2 border">
                    {combinedCsv}
                  </pre>
                </div>
              )}
              {files.length === 0 ? (
                <div className="flex flex-col items-center justify-center py-20 text-center">
                  <div className="w-16 h-16 bg-gray-100 rounded-full flex items-center justify-center mb-4">
                    <FileText size={24} className="text-gray-400" />
                  </div>
                  <h3 className="text-lg font-medium text-gray-700 mb-2">No data extracted yet</h3>
                  <p className="text-gray-500 max-w-sm">Upload PDF files and click extract to see the results here</p>
                </div>
              ) : (
                <div className="grid gap-6">
                  {files.map((f, idx) => (
                    <div key={f.name} className="border border-gray-200 rounded-xl p-4 bg-gray-50">
                      <div className="flex items-center justify-between mb-2">
                        <div className="font-semibold text-gray-800">{f.name}</div>
                        <div className="flex gap-2">
                          {extracted[idx] && (
                            <>
                              <button
                                onClick={() => {
                                  const csv = (extracted[idx].split("```")[1] || extracted[idx]).trim();
                                  const blob = new Blob([csv], { type: 'text/csv' });
                                  const url = URL.createObjectURL(blob);
                                  const a = document.createElement('a');
                                  a.href = url;
                                  a.download = f.name.replace(/\.pdf$/i, '') + '-extracted.csv';
                                  document.body.appendChild(a);
                                  a.click();
                                  setTimeout(() => {
                                    document.body.removeChild(a);
                                    URL.revokeObjectURL(url);
                                  }, 0);
                                }}
                                className="px-3 py-1 bg-green-600 hover:bg-green-700 text-white rounded-lg text-xs font-medium transition-colors duration-200 flex items-center gap-2 shadow-sm"
                              >
                                <Download size={14} />
                                Download CSV
                              </button>
                            </>
                          )}
                        </div>
                      </div>
                      {error[idx] && (
                        <div className="text-red-600 text-xs mb-2">{error[idx]}</div>
                      )}
                      {extracted[idx]
                        && (
                          <pre className="text-xs text-gray-800 whitespace-pre-wrap font-mono leading-relaxed max-h-[300px] overflow-y-auto bg-white rounded p-2 border">
                            {(extracted[idx].split("```")[1] || extracted[idx]).trim()}
                          </pre>
                        )
                      }
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;