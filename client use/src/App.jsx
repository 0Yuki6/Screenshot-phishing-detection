import { useState, useRef } from 'react';
import axios from 'axios';
import { 
  ShieldAlert, 
  ShieldCheck, 
  UploadCloud, 
  Image as ImageIcon, 
  Loader2, 
  AlertTriangle, 
  Smartphone, 
  Mail, 
  MessageCircle, 
  HelpCircle,
  ShieldQuestion
} from 'lucide-react';
import KeyIndicators from './KeyIndicators.jsx';

const API_URL = 'http://127.0.0.1:8000';

function App() {
  const [file, setFile] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');
  
  const fileInputRef = useRef(null);

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const droppedFile = e.dataTransfer.files[0];
      if (droppedFile.type.startsWith('image/')) {
        setFile(droppedFile);
        setError('');
      } else {
        setError('Please upload a valid image file (PNG, JPEG, WebP)');
      }
    }
  };

  const handleFileSelect = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      setFile(e.target.files[0]);
      setError('');
    }
  };

  const analyze = async () => {
    if (!file) {
      setError('Please select an image first.');
      return;
    }

    setIsLoading(true);
    setError('');
    setResult(null);

    try {
      const formData = new FormData();
      formData.append('file', file);

      const url = `${API_URL}/predict`;

      const response = await axios.post(url, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      setResult(response.data);
    } catch (err) {
      console.error(err);
      if (err.response?.data?.detail) {
        setError(err.response.data.detail);
      } else {
        setError('Failed to connect to the analysis server. Ensure the backend is running.');
      }
    } finally {
      setIsLoading(false);
    }
  };

  const reset = () => {
    setFile(null);
    setResult(null);
    setError('');
  };

  const renderChannelIcon = (channel) => {
    switch(channel) {
      case 'email': return <Mail className="w-5 h-5 text-blue-400" />;
      case 'sms': return <Smartphone className="w-5 h-5 text-green-400" />;
      case 'whatsapp': return <MessageCircle className="w-5 h-5 text-green-500" />;
      default: return <HelpCircle className="w-5 h-5 text-slate-400" />;
    }
  };

  const getRiskDetails = (verdict, phishingScore) => {
    if (verdict === 'phishing') {
      return {
        icon: <AlertTriangle className="w-10 h-10" />,
        title: "Phishing Detected",
        colorClass: "text-red-400",
        bgClass: "bg-red-500/20 text-red-500 ring-red-500/30",
        topBorder: "bg-red-500"
      };
    } else if (verdict === 'uncertain') {
      return {
        icon: <ShieldQuestion className="w-10 h-10" />,
        title: "Could Not Decide",
        subtitle: "Verify through the official app or website",
        colorClass: "text-yellow-400",
        bgClass: "bg-yellow-500/20 text-yellow-500 ring-yellow-500/30",
        topBorder: "bg-yellow-500"
      };
    } else {
      return {
        icon: <ShieldCheck className="w-10 h-10" />,
        title: "Looks Safe",
        colorClass: "text-green-400",
        bgClass: "bg-green-500/20 text-green-500 ring-green-500/30",
        topBorder: "bg-green-500"
      };
    }
  };

  return (
    <div className="min-h-screen py-12 px-4 sm:px-6 lg:px-8 flex flex-col items-center">
      {/* Header */}
      <div className="text-center max-w-3xl mb-12">
        <div className="inline-flex items-center justify-center p-3 bg-blue-500/10 rounded-2xl mb-4 ring-1 ring-blue-500/20">
          <ShieldAlert className="w-8 h-8 text-blue-400" />
        </div>
        <h1 className="text-4xl md:text-5xl font-bold tracking-tight mb-4 bg-gradient-to-r from-blue-400 via-indigo-400 to-purple-400 text-transparent bg-clip-text">
          Phishing Detector
        </h1>
        <p className="text-lg text-slate-400 mb-6">
          Upload a screenshot of a suspicious message to analyze it for phishing indicators using AI.
        </p>
        <div className="inline-block p-4 bg-blue-500/10 border border-blue-500/20 rounded-xl text-left max-w-2xl">
          <p className="text-sm text-blue-200/90 leading-relaxed">
            <strong className="text-blue-300">Important:</strong> Please only upload screenshots of <strong>Email, WhatsApp, or SMS</strong> messages. Uploading other types of images will likely result in inaccurate predictions.
          </p>
        </div>
      </div>

      {/* Main Content */}
      <div className="w-full max-w-6xl xl:max-w-7xl grid grid-cols-1 lg:grid-cols-12 gap-8">
        
        {/* Input Section (Left) */}
        <div className="lg:col-span-5 space-y-6">
          <div className="glass-panel p-6">
            
            {/* Input Area */}
            <div
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
              className={`relative border-2 border-dashed rounded-2xl p-12 flex flex-col items-center justify-center text-center cursor-pointer transition-all duration-200 ${
                isDragging 
                  ? 'border-blue-400 bg-blue-500/5' 
                  : file 
                    ? 'border-slate-600 bg-slate-800/30 hover:bg-slate-800/50' 
                    : 'border-slate-700 hover:border-slate-500 hover:bg-slate-800/30'
              }`}
            >
              <input
                type="file"
                ref={fileInputRef}
                onChange={handleFileSelect}
                accept="image/png, image/jpeg, image/webp"
                className="hidden"
              />
              
              {file ? (
                <div className="space-y-3">
                  <div className="w-16 h-16 bg-blue-500/20 rounded-2xl flex items-center justify-center mx-auto ring-1 ring-blue-500/30">
                    <ImageIcon className="w-8 h-8 text-blue-400" />
                  </div>
                  <div>
                    <p className="font-medium text-slate-200 truncate max-w-[200px]">{file.name}</p>
                    <p className="text-xs text-slate-500 mt-1">{(file.size / 1024 / 1024).toFixed(2)} MB</p>
                  </div>
                  <p className="text-xs text-blue-400 mt-2 hover:underline">Click to change file</p>
                </div>
              ) : (
                <div className="space-y-4">
                  <div className="w-16 h-16 bg-slate-800 rounded-full flex items-center justify-center mx-auto shadow-inner ring-1 ring-slate-700">
                    <UploadCloud className="w-8 h-8 text-slate-400" />
                  </div>
                  <div>
                    <p className="text-base font-medium text-slate-200">
                      Drag & drop an Email, WhatsApp, or SMS screenshot here
                    </p>
                    <p className="text-sm text-slate-400 mt-1">
                      or click to browse from your computer
                    </p>
                  </div>
                  <p className="text-xs text-slate-500 mt-4">
                    Supports PNG, JPG, WebP up to 10MB
                  </p>
                </div>
              )}
            </div>

            {/* Action Buttons */}
            <div className="mt-6 flex gap-3">
              <button
                onClick={analyze}
                disabled={isLoading || !file}
                className="flex-1 bg-gradient-to-r from-blue-500 to-indigo-600 hover:from-blue-400 hover:to-indigo-500 text-white font-medium py-3 px-6 rounded-xl shadow-lg shadow-blue-500/25 transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2 group"
              >
                {isLoading ? (
                  <>
                    <Loader2 className="w-5 h-5 animate-spin" />
                    Analyzing...
                  </>
                ) : (
                  <>
                    <ShieldCheck className="w-5 h-5 group-hover:scale-110 transition-transform" />
                    Analyze Message
                  </>
                )}
              </button>
              
              {(file || result) && (
                <button
                  onClick={reset}
                  disabled={isLoading}
                  className="px-6 py-3 bg-slate-800 hover:bg-slate-700 text-slate-300 font-medium rounded-xl border border-slate-700 transition-all disabled:opacity-50"
                >
                  Clear
                </button>
              )}
            </div>
          </div>
        </div>

        {/* Results Section (Right) */}
        <div className="lg:col-span-7">
          {result ? (
            (() => {
              const phishingScore = result.phishing_score ?? 0;
              const verdict = result.verdict || 'uncertain';
              const riskDetails = getRiskDetails(verdict, phishingScore);
              return (
                <div className="glass-panel p-1 relative overflow-hidden h-full flex flex-col">
                  {/* Colored top border based on verdict */}
                  <div className={`absolute top-0 left-0 right-0 h-1.5 ${riskDetails.topBorder}`} />
                  
                  <div className="p-6 flex-1 flex flex-col">
                    <h3 className="text-sm font-semibold uppercase tracking-wider text-slate-500 mb-6">Analysis Results</h3>
                    
                    {/* Main Verdict */}
                    <div className="flex flex-col items-center justify-center py-6 mb-8 border-b border-slate-700/50 text-center">
                      <div className={`w-20 h-20 rounded-full flex items-center justify-center mb-4 ring-4 ring-offset-4 ring-offset-slate-900 ${riskDetails.bgClass}`}>
                        {riskDetails.icon}
                      </div>
                      
                      <h2 className={`text-2xl font-bold tracking-tight ${riskDetails.colorClass} px-2`}>
                        {riskDetails.title}
                      </h2>
                      {riskDetails.subtitle && (
                        <p className="text-sm text-slate-300 mt-2 italic">
                          {riskDetails.subtitle}
                        </p>
                      )}
                      <p className="text-slate-400 mt-2 font-medium">
                        Phishing score: {(phishingScore * 100).toFixed(1)}%
                      </p>
                    </div>

                    {/* Details Grid */}
                    <div className="space-y-4 flex-1">
                      
                      {/* Channel */}
                      <div className="bg-slate-800/40 rounded-xl p-4 flex items-center justify-between border border-slate-700/50">
                        <div className="flex items-center gap-3">
                          <div className="p-2 bg-slate-900 rounded-lg">
                            {renderChannelIcon(result.channel)}
                          </div>
                          <div>
                            <p className="text-sm font-medium text-slate-300">Detected Channel</p>
                            <p className="text-xs text-slate-500 capitalize">{result.channel}</p>
                          </div>
                        </div>
                      </div>

                      {/* Key Indicators */}
                      <KeyIndicators result={result} phishingScore={phishingScore} />

                    </div>
                    
                    <div className="mt-8 pt-4 border-t border-slate-700/50 text-center">
                      <p className="text-xs text-slate-500">ID: {result.prediction_id}</p>
                    </div>
                  </div>
                </div>
              );
            })()
          ) : (
            <div className="glass-panel h-full min-h-[400px] flex flex-col items-center justify-center p-8 text-center border-dashed">
              <div className="w-20 h-20 rounded-full bg-slate-800/50 flex items-center justify-center mb-6 ring-1 ring-slate-700">
                <ShieldCheck className="w-10 h-10 text-slate-600" />
              </div>
              <h3 className="text-xl font-medium text-slate-300 mb-2">Ready for Analysis</h3>
              <p className="text-slate-500 text-sm max-w-[250px]">
                Upload a screenshot on the left to see the AI's verdict.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;
