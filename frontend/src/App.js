import { useState, useEffect } from 'react';
import '@/App.css';
import axios from 'axios';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { Upload, Download, Sparkles, Trash2 } from 'lucide-react';
import { useToast } from '@/hooks/use-toast';
import { Toaster } from '@/components/ui/sonner';
import { toast } from 'sonner';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

function App() {
  const [ocrResults, setOcrResults] = useState([]);
  const [generatedMemes, setGeneratedMemes] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [generating, setGenerating] = useState(false);
  const [activeTab, setActiveTab] = useState('upload');

  useEffect(() => {
    loadOcrResults();
    loadGeneratedMemes();
  }, []);

  const loadOcrResults = async () => {
    try {
      const response = await axios.get(`${API}/ocr-results`);
      setOcrResults(response.data);
    } catch (error) {
      console.error('Failed to load OCR results:', error);
    }
  };

  const loadGeneratedMemes = async () => {
    try {
      const response = await axios.get(`${API}/generated-memes`);
      setGeneratedMemes(response.data);
    } catch (error) {
      console.error('Failed to load generated memes:', error);
    }
  };

  const handleFileUpload = async (event) => {
    const files = Array.from(event.target.files);
    if (files.length === 0) return;

    setUploading(true);
    const formData = new FormData();
    files.forEach(file => {
      formData.append('files', file);
    });

    try {
      const response = await axios.post(`${API}/upload-memes`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      
      toast.success(`Successfully processed ${response.data.uploaded} meme(s)!`);
      loadOcrResults();
      setActiveTab('results');
    } catch (error) {
      toast.error('Failed to upload memes: ' + error.message);
    } finally {
      setUploading(false);
    }
  };

  const handleGenerateMemes = async () => {
    if (ocrResults.length === 0) {
      toast.error('Please upload memes first!');
      return;
    }

    setGenerating(true);
    try {
      const response = await axios.post(`${API}/generate-new-memes`, { count: 5 });
      toast.success(`Generated ${response.data.generated} new memes!`);
      loadGeneratedMemes();
      setActiveTab('generated');
    } catch (error) {
      toast.error('Failed to generate memes: ' + error.message);
    } finally {
      setGenerating(false);
    }
  };

  const handleDownloadCSV = async () => {
    try {
      const response = await axios.get(`${API}/download-csv`);
      const blob = new Blob([response.data.csv], { type: 'text/csv' });
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'meme-ocr-results.csv';
      a.click();
      toast.success('CSV downloaded!');
    } catch (error) {
      toast.error('Failed to download CSV');
    }
  };

  const handleClearData = async () => {
    if (!window.confirm('Are you sure you want to clear all data?')) return;
    
    try {
      await axios.delete(`${API}/clear-data`);
      toast.success('All data cleared!');
      setOcrResults([]);
      setGeneratedMemes([]);
    } catch (error) {
      toast.error('Failed to clear data');
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
      <Toaster richColors position="top-center" />
      
      {/* Header */}
      <div className="border-b border-purple-500/20 bg-black/30 backdrop-blur-sm">
        <div className="max-w-7xl mx-auto px-4 py-6">
          <h1 className="text-4xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-pink-400 via-red-400 to-purple-400">
            🔞 NSFW Meme OCR & Generator
          </h1>
          <p className="text-gray-400 mt-2">Upload memes, extract keywords, and generate EXPLICIT adult humor content</p>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-4 py-8">
        <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
          <TabsList className="bg-black/40 border border-purple-500/30">
            <TabsTrigger value="upload" className="data-[state=active]:bg-purple-600">
              <Upload className="w-4 h-4 mr-2" />
              Upload
            </TabsTrigger>
            <TabsTrigger value="results" className="data-[state=active]:bg-purple-600">
              OCR Results ({ocrResults.length})
            </TabsTrigger>
            <TabsTrigger value="generated" className="data-[state=active]:bg-purple-600">
              <Sparkles className="w-4 h-4 mr-2" />
              Generated ({generatedMemes.length})
            </TabsTrigger>
          </TabsList>

          {/* Upload Tab */}
          <TabsContent value="upload">
            <Card className="bg-black/40 border-purple-500/30 backdrop-blur-sm">
              <CardHeader>
                <CardTitle className="text-purple-300">Upload Meme Images (NSFW)</CardTitle>
                <CardDescription className="text-gray-400">
                  Upload meme images with text. Low-res OK! Images without text will be auto-skipped. Keywords like "fuck", "sex", "boobs" are KEPT for generation.
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="border-2 border-dashed border-purple-500/50 rounded-lg p-12 text-center hover:border-purple-400 transition-colors">
                  <Upload className="w-16 h-16 mx-auto mb-4 text-purple-400" />
                  <label htmlFor="file-upload" className="cursor-pointer">
                    <span className="text-xl text-purple-300 hover:text-purple-200">
                      Click to upload or drag and drop
                    </span>
                    <p className="text-gray-500 mt-2">PNG, JPG, JPEG (Multiple files supported)</p>
                  </label>
                  <input
                    id="file-upload"
                    type="file"
                    multiple
                    accept="image/*"
                    onChange={handleFileUpload}
                    className="hidden"
                    disabled={uploading}
                    data-testid="file-upload-input"
                  />
                </div>
                
                {uploading && (
                  <div className="text-center text-purple-400">
                    <div className="animate-pulse">Processing images...</div>
                  </div>
                )}

                <div className="flex gap-4">
                  <Button
                    onClick={handleGenerateMemes}
                    disabled={generating || ocrResults.length === 0}
                    className="flex-1 bg-gradient-to-r from-pink-600 to-red-600 hover:from-pink-700 hover:to-red-700"
                    data-testid="generate-memes-button"
                  >
                    <Sparkles className="w-4 h-4 mr-2" />
                    {generating ? 'Generating NSFW...' : 'Generate NSFW Memes'}
                  </Button>
                  
                  <Button
                    onClick={handleClearData}
                    variant="destructive"
                    className="bg-red-600 hover:bg-red-700"
                    data-testid="clear-data-button"
                  >
                    <Trash2 className="w-4 h-4 mr-2" />
                    Clear All
                  </Button>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          {/* OCR Results Tab */}
          <TabsContent value="results">
            <Card className="bg-black/40 border-purple-500/30 backdrop-blur-sm">
              <CardHeader className="flex flex-row items-center justify-between">
                <div>
                  <CardTitle className="text-purple-300">OCR Extraction Results</CardTitle>
                  <CardDescription className="text-gray-400">
                    Text extracted from uploaded meme images
                  </CardDescription>
                </div>
                <Button
                  onClick={handleDownloadCSV}
                  variant="outline"
                  className="border-purple-500/50 hover:bg-purple-600"
                  disabled={ocrResults.length === 0}
                  data-testid="download-csv-button"
                >
                  <Download className="w-4 h-4 mr-2" />
                  Download CSV
                </Button>
              </CardHeader>
              <CardContent>
                {ocrResults.length === 0 ? (
                  <div className="text-center py-12 text-gray-500">
                    No data yet. Upload some memes to get started!
                  </div>
                ) : (
                  <div className="border border-purple-500/30 rounded-lg overflow-hidden">
                    <Table>
                      <TableHeader>
                        <TableRow className="bg-purple-900/20 border-purple-500/30">
                          <TableHead className="text-purple-300">Filename</TableHead>
                          <TableHead className="text-purple-300">Extracted Text</TableHead>
                          <TableHead className="text-purple-300">Keywords</TableHead>
                          <TableHead className="text-purple-300">Word Count</TableHead>
                          <TableHead className="text-purple-300">Preview</TableHead>
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        {ocrResults.map((result, idx) => (
                          <TableRow key={idx} className="border-purple-500/20" data-testid={`ocr-result-row-${idx}`}>
                            <TableCell className="text-gray-300">{result.filename}</TableCell>
                            <TableCell className="text-gray-400 max-w-md truncate">
                              {result.extracted_text || 'No text found'}
                            </TableCell>
                            <TableCell>
                              <div className="flex flex-wrap gap-1">
                                {(result.keywords || []).slice(0, 5).map((kw, kidx) => (
                                  <span
                                    key={kidx}
                                    className="text-xs bg-pink-700/50 text-pink-200 px-2 py-1 rounded font-semibold"
                                  >
                                    {kw}
                                  </span>
                                ))}
                              </div>
                            </TableCell>
                            <TableCell className="text-gray-300">{result.word_count}</TableCell>
                            <TableCell>
                              <img
                                src={`data:image/png;base64,${result.image_data}`}
                                alt={result.filename}
                                className="w-16 h-16 object-cover rounded border border-purple-500/30"
                              />
                            </TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </div>
                )}
              </CardContent>
            </Card>
          </TabsContent>

          {/* Generated Memes Tab */}
          <TabsContent value="generated">
            <Card className="bg-black/40 border-purple-500/30 backdrop-blur-sm">
              <CardHeader>
                <CardTitle className="text-purple-300">🔥 AI-Generated NSFW Memes</CardTitle>
                <CardDescription className="text-gray-400">
                  Explicit adult humor memes created from keyword patterns (profanity included!)
                </CardDescription>
              </CardHeader>
              <CardContent>
                {generatedMemes.length === 0 ? (
                  <div className="text-center py-12 text-gray-500">
                    No generated memes yet. Click "Generate New Memes" to create some!
                  </div>
                ) : (
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                    {generatedMemes.map((meme, idx) => (
                      <Card key={idx} className="bg-purple-900/20 border-purple-500/30" data-testid={`generated-meme-${idx}`}>
                        <CardContent className="p-4">
                          <img
                            src={`data:image/png;base64,${meme.image_data}`}
                            alt="Generated meme"
                            className="w-full h-48 object-cover rounded mb-4 border border-purple-500/30"
                          />
                          <div className="space-y-2">
                            <p className="text-white font-medium text-sm leading-relaxed">{meme.text}</p>
                            <div className="border-t border-purple-500/30 pt-2">
                              <p className="text-xs text-purple-300 mb-1">Keywords Used:</p>
                              <div className="flex flex-wrap gap-1">
                                {meme.source_words.map((word, widx) => (
                                  <span
                                    key={widx}
                                    className="text-xs bg-pink-700/70 text-pink-200 px-2 py-1 rounded font-bold"
                                  >
                                    {word}
                                  </span>
                                ))}
                              </div>
                            </div>
                          </div>
                        </CardContent>
                      </Card>
                    ))}
                  </div>
                )}
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
}

export default App;