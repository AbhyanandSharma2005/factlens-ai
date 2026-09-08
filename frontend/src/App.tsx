import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { UploadCloud, FileText, CheckCircle2, AlertOctagon, HelpCircle, Loader2, ArrowRight } from 'lucide-react';

interface Relationship {
  relation_type: 'corroborates' | 'contradicts' | 'reconciled_by_context' | 'insufficient_evidence';
  explanation: string;
  other_fact_text: string;
}

interface Fact {
  id: string;
  subject: string;
  fact_type: string;
  metric_value: any;
  scope_context: any;
  evidence_text: string;
  evidence_page: number;
  filename: string;
  relationships: Relationship[];
}

export default function App() {
  const [facts, setFacts] = useState<Fact[]>([]);
  const [selectedFact, setSelectedFact] = useState<Fact | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState<string | null>(null);

  // Fetch facts from the FastAPI backend
  const fetchFacts = async () => {
    try {
      const response = await axios.get('http://127.0.0.1:8000/api/facts');
      setFacts(response.data);
    } catch (error) {
      console.error("Error fetching facts:", error);
    }
  };

  // Poll for updates when the app loads
  useEffect(() => {
    fetchFacts();
    const interval = setInterval(fetchFacts, 5000); // Auto-refresh every 5 seconds
    return () => clearInterval(interval);
  }, []);

  // Handle PDF Upload
  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    const formData = new FormData();
    formData.append("file", file);

    setIsUploading(true);
    setUploadStatus("Uploading & Analyzing Document...");

    try {
      await axios.post('http://127.0.0.1:8000/api/documents', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      setUploadStatus("Processing in background. Facts will appear shortly.");
      setTimeout(() => setUploadStatus(null), 5000);
    } catch (error) {
      console.error("Upload failed", error);
      setUploadStatus("Upload failed. Check console.");
    } finally {
      setIsUploading(false);
      // Clear the input so you can upload the same file again if needed
      event.target.value = ''; 
    }
  };

  const getRelationStyle = (type: string) => {
    switch (type) {
      case 'corroborates': return { color: 'text-emerald-700', bg: 'bg-emerald-50', border: 'border-emerald-200', icon: <CheckCircle2 className="w-4 h-4 text-emerald-600" /> };
      case 'contradicts': return { color: 'text-rose-700', bg: 'bg-rose-50', border: 'border-rose-200', icon: <AlertOctagon className="w-4 h-4 text-rose-600" /> };
      case 'reconciled_by_context': return { color: 'text-amber-700', bg: 'bg-amber-50', border: 'border-amber-200', icon: <HelpCircle className="w-4 h-4 text-amber-600" /> };
      default: return { color: 'text-slate-600', bg: 'bg-slate-50', border: 'border-slate-200', icon: <FileText className="w-4 h-4 text-slate-400" /> };
    }
  };

  return (
    <div className="flex h-screen w-full bg-slate-50 font-sans overflow-hidden text-slate-900">
      
      {/* LEFT PANEL: Upload & Evidence Inspector */}
      <div className="w-full lg:w-5/12 border-r border-slate-200 bg-white flex flex-col h-full shadow-sm z-10">
        <div className="p-6 border-b border-slate-100 bg-slate-50/50">
          <h1 className="text-2xl font-bold bg-gradient-to-r from-indigo-600 to-blue-500 bg-clip-text text-transparent mb-1">FactLens AI</h1>
          <p className="text-sm text-slate-500 font-medium">Enterprise Knowledge Layer</p>
        </div>

        <div className="p-6">
          <label className="flex flex-col items-center justify-center w-full h-32 border-2 border-slate-300 border-dashed rounded-xl cursor-pointer bg-slate-50 hover:bg-indigo-50 hover:border-indigo-300 transition-colors">
            <div className="flex flex-col items-center justify-center pt-5 pb-6">
              {isUploading ? (
                <Loader2 className="w-8 h-8 text-indigo-500 animate-spin mb-2" />
              ) : (
                <UploadCloud className="w-8 h-8 text-slate-400 mb-2" />
              )}
              <p className="text-sm font-semibold text-slate-700">
                {isUploading ? "Uploading..." : "Click to upload PDF"}
              </p>
            </div>
            <input type="file" className="hidden" accept="application/pdf" onChange={handleFileUpload} disabled={isUploading} />
          </label>
          {uploadStatus && <p className="text-xs text-indigo-600 mt-3 font-medium text-center">{uploadStatus}</p>}
        </div>

        <div className="flex-1 overflow-y-auto p-6 bg-slate-50/50 border-t border-slate-100">
          <h2 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-4">Source Evidence Inspector</h2>
          
          {selectedFact ? (
            <div className="bg-white border border-slate-200 rounded-xl shadow-sm overflow-hidden">
              <div className="px-4 py-3 bg-slate-100 border-b border-slate-200 flex justify-between items-center">
                <span className="text-xs font-semibold text-slate-600 truncate mr-2" title={selectedFact.filename}>
                  📄 {selectedFact.filename}
                </span>
                <span className="text-xs font-bold text-indigo-600 bg-indigo-100 px-2 py-1 rounded-md whitespace-nowrap">
                  Page {selectedFact.evidence_page}
                </span>
              </div>
              <div className="p-5">
                <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">Verbatim Quote</p>
                <div className="p-4 bg-amber-50/50 border-l-4 border-amber-400 text-slate-800 font-serif leading-relaxed text-sm rounded-r-md">
                  "{selectedFact.evidence_text}"
                </div>
                <div className="mt-4 flex gap-2 flex-wrap">
                  {selectedFact.scope_context?.period && (
                    <span className="text-[10px] font-bold bg-slate-100 text-slate-500 px-2 py-1 rounded">
                      📅 {selectedFact.scope_context.period}
                    </span>
                  )}
                  {selectedFact.metric_value?.raw_value && (
                    <span className="text-[10px] font-bold bg-slate-100 text-slate-500 px-2 py-1 rounded">
                      📊 {selectedFact.metric_value.raw_value}
                    </span>
                  )}
                </div>
              </div>
            </div>
          ) : (
            <div className="h-48 flex items-center justify-center border-2 border-dashed border-slate-200 rounded-xl text-slate-400 text-sm">
              Select a fact from the right panel to inspect it here.
            </div>
          )}
        </div>
      </div>

      {/* RIGHT PANEL: The Knowledge Graph Feed */}
      <div className="w-full lg:w-7/12 bg-slate-50 flex flex-col h-full">
        <div className="p-6 border-b border-slate-200 bg-white">
          <h2 className="text-lg font-bold text-slate-800">Extracted Facts & Relationships</h2>
          <p className="text-sm text-slate-500">Auto-refreshing every 5s to show background processing.</p>
        </div>
        
        <div className="flex-1 overflow-y-auto p-6 space-y-4">
          {facts.length === 0 ? (
            <div className="text-center text-slate-400 mt-20">
              <FileText className="w-12 h-12 mx-auto mb-3 opacity-20" />
              <p>No facts extracted yet.</p>
              <p className="text-xs mt-1">Upload a document to begin generating the knowledge layer.</p>
            </div>
          ) : (
            facts.map((fact) => (
              <div 
                key={fact.id}
                onClick={() => setSelectedFact(fact)}
                className={`bg-white p-5 rounded-xl border transition-all cursor-pointer ${
                  selectedFact?.id === fact.id 
                    ? 'border-indigo-400 shadow-md ring-1 ring-indigo-400' 
                    : 'border-slate-200 hover:border-slate-300 hover:shadow-sm'
                }`}
              >
                <div className="flex justify-between items-start mb-3">
                  <div className="pr-4">
                    <span className="text-[10px] font-bold uppercase tracking-wider text-indigo-500 bg-indigo-50 px-2 py-0.5 rounded-full">
                      {fact.fact_type}
                    </span>
                    <h3 className="text-base font-semibold text-slate-900 mt-2 leading-tight">{fact.subject}</h3>
                  </div>
                  <div className="text-right whitespace-nowrap">
                    <span className="text-sm font-bold text-slate-800 bg-slate-100 px-3 py-1.5 rounded-lg border border-slate-200">
                      {fact.metric_value?.raw_value || 'N/A'}
                    </span>
                  </div>
                </div>

                {/* Render Relationships if they exist */}
                {fact.relationships && fact.relationships.length > 0 && (
                  <div className="mt-4 pt-4 border-t border-slate-100 space-y-3">
                    {fact.relationships.map((rel, idx) => {
                      const style = getRelationStyle(rel.relation_type);
                      return (
                        <div key={idx} className={`p-3 rounded-lg border ${style.bg} ${style.border}`}>
                          <div className="flex items-center gap-2 mb-1.5">
                            {style.icon}
                            <span className={`text-xs font-bold uppercase tracking-wider ${style.color}`}>
                              {rel.relation_type.replace(/_/g, ' ')}
                            </span>
                          </div>
                          <p className="text-sm text-slate-700 font-medium mb-1">
                            {rel.explanation}
                          </p>
                          <div className="flex items-start gap-2 mt-2 pt-2 border-t border-black/5">
                            <ArrowRight className="w-3 h-3 text-slate-400 mt-0.5 flex-shrink-0" />
                            <p className="text-xs text-slate-500 italic line-clamp-2">
                              "{rel.other_fact_text}"
                            </p>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
}