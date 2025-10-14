import { useState, useEffect } from 'react';
import '@/App.css';
import axios from 'axios';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { Upload, Download, Sparkles, Trash2, RefreshCw, Edit } from 'lucide-react';
import { Slider } from '@/components/ui/slider';
import { Input } from '@/components/ui/input';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Label } from '@/components/ui/label';
import { toast } from 'sonner';
import { Toaster } from '@/components/ui/sonner';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

function App() {
  const [ocrResults, setOcrResults] = useState([]);
  const [generatedMemes, setGeneratedMemes] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [generating, setGenerating] = useState(false);
  const [activeTab, setActiveTab] = useState('builder');
  
  // Tone sliders
  const [naughty, setNaughty] = useState(5);
  const [sexy, setSexy] = useState(5);
  const [funny, setFunny] = useState(5);
  const [rude, setRude] = useState(5);
  
  // Tone presets
  const [tonePresets, setTonePresets] = useState([]);
  const [selectedPreset, setSelectedPreset] = useState('');
  
  // Style controls
  const [memeLength, setMemeLength] = useState('short');
  const [memeFormat, setMemeFormat] = useState('statement');
  
  // Meme builder
  const [builderMode, setBuilderMode] = useState(false);
  const [builderKeywords, setBuilderKeywords] = useState('');
  const [builderOptions, setBuilderOptions] = useState([]);
  const [builderRound, setBuilderRound] = useState(0);
  const [selectedMeme, setSelectedMeme] = useState(null);
  
  // Edit mode
  const [editingId, setEditingId] = useState(null);
  const [editText, setEditText] = useState('');

  useEffect(() => {
    loadOcrResults();
    loadGeneratedMemes();
    loadTonePresets();
  }, []);

  const loadOcrResults = async () => {
    try {
      const response = await axios.get(`${API}/ocr-results`);
      setOcrResults(response.data);
    } catch (error) {
      console.error('Failed to load OCR results:', error);
    }
  };

  const loadTonePresets = async () => {
    try {
      const response = await axios.get(`${API}/tone-presets`);
      setTonePresets(response.data.presets);
    } catch (error) {
      console.error('Failed to load tone presets:', error);
    }
  };

  const applyTonePreset = (presetName) => {
    const preset = tonePresets.find(p => p.name === presetName);
    if (preset) {
      setNaughty(preset.naughty);
      setSexy(preset.sexy);
      setFunny(preset.funny);
      setRude(preset.rude);
      setSelectedPreset(presetName);
      toast.success(`Applied: ${preset.name}! ${preset.description}`);
    }
  };

  const randomizeTone = () => {
    setNaughty(Math.floor(Math.random() * 10) + 1);
    setSexy(Math.floor(Math.random() * 10) + 1);
    setFunny(Math.floor(Math.random() * 10) + 1);
    setRude(Math.floor(Math.random() * 10) + 1);
    setSelectedPreset('');
    toast.success('🎲 Randomized! Let\'s see what happens!');
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
      
      toast.success(`Processed ${response.data.successful} meme(s)! Skipped ${response.data.skipped} without text.`);
      loadOcrResults();
      setActiveTab('results');
    } catch (error) {
      toast.error('Failed to upload memes: ' + error.message);
    } finally {
      setUploading(false);
    }
  };

  const handleGenerateMemes = async (keywords = null) => {
    if (!keywords && ocrResults.length === 0) {
      toast.error('Please upload some images first!');
      return;
    }

    setGenerating(true);
    try {
      const payload = {
        count: 4,
        tone: { naughty, sexy, funny, rude },
        style: { length: memeLength, format: memeFormat },
        keywords: keywords ? keywords.split(',').map(k => k.trim()) : null,
        generate_images: false,  // TEXT ONLY!
        custom_prompt: null
      };
      
      const response = await axios.post(`${API}/generate-new-memes`, payload);
      
      if (builderMode) {
        setBuilderOptions(response.data.memes);
        setBuilderRound(builderRound + 1);
        toast.success(`Round ${builderRound + 1}: Pick your favorite!`);
      } else {
        toast.success(`Generated ${response.data.generated} new memes!`);
        loadGeneratedMemes();
        setActiveTab('generated');
      }
    } catch (error) {
      toast.error('Failed to generate memes: ' + error.message);
    } finally {
      setGenerating(false);
    }
  };

  const handleGenerateSimilar = async (memeId) => {
    setGenerating(true);
    try {
      const response = await axios.post(`${API}/generate-similar`, {
        meme_id: memeId,
        tone: { naughty, sexy, funny, rude }
      });
      
      toast.success('Generated 3 similar memes!');
      loadGeneratedMemes();
    } catch (error) {
      toast.error('Failed to generate similar memes');
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

  const handleUpdateText = async (id) => {
    try {
      await axios.put(`${API}/update-text`, {
        id: id,
        corrected_text: editText
      });
      toast.success('Text updated and keywords re-extracted!');
      setEditingId(null);
      loadOcrResults();
    } catch (error) {
      toast.error('Failed to update text');
    }
  };

  const startBuilder = () => {
    setBuilderMode(true);
    setBuilderRound(0);
    setBuilderOptions([]);
    setActiveTab('builder');
  };

  const selectBuilderOption = (meme) => {
    setSelectedMeme(meme);
    setBuilderKeywords(meme.source_words.join(', '));
  };

  const finishBuilder = () => {
    if (selectedMeme) {
      toast.success('Meme created! Check Generated tab.');
      loadGeneratedMemes();
    }
    setBuilderMode(false);
    setBuilderOptions([]);
    setBuilderRound(0);
    setSelectedMeme(null);
    setActiveTab('generated');
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-pink-950 via-red-950 to-black">
      <Toaster richColors position="top-center" />
      
      {/* Header */}
      <div className="border-b-4 border-pink-500/50 bg-black/60 backdrop-blur-sm">
        <div className="max-w-7xl mx-auto px-4 py-6">
          <h1 className="text-5xl font-black text-transparent bg-clip-text bg-gradient-to-r from-pink-400 via-red-400 to-pink-500 drop-shadow-lg">
            💋 MISS TITTY SPRINKLES<br/>FUNNY FUCKING FACTORY
          </h1>
          <p className="text-pink-300 mt-3 text-lg font-bold">TEXT-ONLY AI Meme Generator - Upload Images, AI Writes Hilarious Text!</p>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-4 py-8">
        {/* Tone Controls */}
        <Card className="bg-black/60 border-pink-500/50 backdrop-blur-sm mb-6">
          <CardHeader className="flex flex-row items-center justify-between">
            <div>
              <CardTitle className="text-pink-300 text-xl">🎚️ Tone Controls</CardTitle>
              <CardDescription className="text-gray-400">Adjust the vibe - AI writes the text!</CardDescription>
            </div>
            <Button
              onClick={randomizeTone}
              className="bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700"
            >
              🎲 RANDOMIZE
            </Button>
          </CardHeader>
          <CardContent className="space-y-6">
            {/* Tone Sliders */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
              <div>
                <label className="text-pink-300 font-semibold mb-2 block">😈 Naughty: {naughty}</label>
                <Slider value={[naughty]} onValueChange={([v]) => setNaughty(v)} min={1} max={10} step={1} className="accent-pink-500" />
              </div>
              <div>
                <label className="text-pink-300 font-semibold mb-2 block">💋 Sexy: {sexy}</label>
                <Slider value={[sexy]} onValueChange={([v]) => setSexy(v)} min={1} max={10} step={1} className="accent-red-500" />
              </div>
              <div>
                <label className="text-pink-300 font-semibold mb-2 block">😂 Funny: {funny}</label>
                <Slider value={[funny]} onValueChange={([v]) => setFunny(v)} min={1} max={10} step={1} className="accent-yellow-500" />
              </div>
              <div>
                <label className="text-pink-300 font-semibold mb-2 block">🖕 Rude: {rude}</label>
                <Slider value={[rude]} onValueChange={([v]) => setRude(v)} min={1} max={10} step={1} className="accent-purple-500" />
              </div>
            </div>
          </CardContent>
        </Card>

        <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
          <TabsList className="bg-black/60 border border-pink-500/50">
            <TabsTrigger value="builder" className="data-[state=active]:bg-pink-600">
              <Sparkles className="w-4 h-4 mr-2" />
              Meme Builder
            </TabsTrigger>
            <TabsTrigger value="generated" className="data-[state=active]:bg-pink-600">
              Generated ({generatedMemes.length})
            </TabsTrigger>
            <TabsTrigger value="upload" className="data-[state=active]:bg-pink-600">
              <Upload className="w-4 h-4 mr-2" />
              Upload (Optional)
            </TabsTrigger>
          </TabsList>

          {/* Upload Tab - OPTIONAL */}
          <TabsContent value="upload">
            <Card className="bg-black/60 border-pink-500/50 backdrop-blur-sm">
              <CardHeader>
                <CardTitle className="text-pink-300">📤 Upload Images (Optional)</CardTitle>
                <CardDescription className="text-gray-400">
                  Upload your own images to use as backgrounds. Not required - you can use AI generation!
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="border-2 border-dashed border-pink-500/70 rounded-lg p-8 text-center hover:border-pink-400 transition-colors">
                  <Upload className="w-12 h-12 mx-auto mb-3 text-pink-400" />
                  <label htmlFor="file-upload" className="cursor-pointer">
                    <span className="text-lg text-pink-300 hover:text-pink-200 font-semibold">
                      Click to upload images
                    </span>
                    <p className="text-gray-500 mt-2">PNG, JPG, JPEG</p>
                  </label>
                  <input
                    id="file-upload"
                    type="file"
                    multiple
                    accept="image/*"
                    onChange={handleFileUpload}
                    className="hidden"
                    disabled={uploading}
                  />
                </div>
                
                {uploading && (
                  <div className="text-center text-pink-400">
                    <div className="animate-pulse">Processing...</div>
                  </div>
                )}

                {ocrResults.length > 0 && (
                  <div className="text-center p-4 bg-pink-900/20 rounded-lg">
                    <p className="text-pink-300 font-semibold">✅ {ocrResults.length} images uploaded!</p>
                    <p className="text-gray-400 text-sm mt-1">These can be used as meme backgrounds</p>
                  </div>
                )}
              </CardContent>
            </Card>
          </TabsContent>

          {/* Meme Builder Tab - NOW FIRST! */}
          <TabsContent value="builder">
            <Card className="bg-black/60 border-pink-500/50 backdrop-blur-sm">
              <CardHeader>
                <CardTitle className="text-pink-300 text-2xl">🎨 Meme Builder - Create From Scratch!</CardTitle>
                <CardDescription className="text-gray-400 text-lg">
                  Enter keywords, adjust tone, pick style, and generate! Get 4 options each round.
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                {/* Quick Start Keywords */}
                <div>
                  <Label className="text-pink-300 font-semibold mb-2 block text-lg">💬 Enter Keywords:</Label>
                  <div className="flex gap-2">
                    <Input
                      value={builderKeywords}
                      onChange={(e) => setBuilderKeywords(e.target.value)}
                      placeholder="e.g., drunk, work, Monday, fuck, tired..."
                      className="bg-black/40 text-white border-pink-500/50 flex-1 text-lg"
                    />
                    <Button
                      onClick={() => handleGenerateMemes(builderKeywords)}
                      disabled={generating || (!builderKeywords && imageMode === 'uploaded' && ocrResults.length === 0)}
                      className="bg-gradient-to-r from-pink-600 to-red-600 hover:from-pink-700 hover:to-red-700 px-8 text-lg"
                    >
                      <Sparkles className="w-5 h-5 mr-2" />
                      Generate 4 Options
                    </Button>
                  </div>
                  <p className="text-xs text-gray-500 mt-2">Tip: Use NSFW words freely - they work better!</p>
                </div>

                {/* Tone Presets - Quick Access */}
                <div>
                  <Label className="text-pink-300 font-semibold mb-2 block">🎭 Quick Tone Presets:</Label>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
                    {tonePresets.slice(0, 8).map((preset) => (
                      <Button
                        key={preset.name}
                        onClick={() => applyTonePreset(preset.name)}
                        variant="outline"
                        className={`border-pink-500/50 hover:bg-pink-600 text-sm ${selectedPreset === preset.name ? 'bg-pink-600' : ''}`}
                      >
                        {preset.name}
                      </Button>
                    ))}
                  </div>
                </div>

                {builderOptions.length > 0 && (
                  <>
                    <div className="border-t border-pink-500/30 pt-6">
                      <h3 className="text-pink-300 font-bold text-xl mb-4">Pick Your Favorite (Round {builderRound}):</h3>
                      <div className="grid grid-cols-2 gap-4">
                        {builderOptions.map((meme, idx) => (
                          <Card
                            key={idx}
                            className={`cursor-pointer transition-all ${
                              selectedMeme?.id === meme.id
                                ? 'bg-pink-900/50 border-pink-400 border-2'
                                : 'bg-pink-900/20 border-pink-500/30 hover:border-pink-400'
                            }`}
                            onClick={() => selectBuilderOption(meme)}
                          >
                            <CardContent className="p-4">
                              <img
                                src={`data:image/png;base64,${meme.image_data}`}
                                alt="Option"
                                className="w-full h-40 object-cover rounded mb-3"
                              />
                              <p className="text-white font-bold text-base leading-relaxed">{meme.text}</p>
                              <div className="flex flex-wrap gap-1 mt-2">
                                {meme.source_words.slice(0, 4).map((word, widx) => (
                                  <span key={widx} className="text-xs bg-pink-700/50 text-pink-200 px-2 py-1 rounded">
                                    {word}
                                  </span>
                                ))}
                              </div>
                            </CardContent>
                          </Card>
                        ))}
                      </div>
                    </div>

                    <div className="flex gap-4">
                      <Button
                        onClick={() => handleGenerateMemes(builderKeywords)}
                        disabled={generating}
                        className="flex-1 bg-pink-600 hover:bg-pink-700 text-lg"
                      >
                        <RefreshCw className="w-5 h-5 mr-2" />
                        Generate 4 More Options
                      </Button>
                      <Button
                        onClick={finishBuilder}
                        disabled={!selectedMeme}
                        className="flex-1 bg-green-600 hover:bg-green-700 text-lg font-bold"
                      >
                        ✅ Save Selected Meme
                      </Button>
                    </div>
                  </>
                )}

                {builderOptions.length === 0 && (
                  <div className="text-center py-12 border border-dashed border-pink-500/50 rounded-lg">
                    <Sparkles className="w-16 h-16 mx-auto mb-4 text-pink-400" />
                    <p className="text-pink-300 text-lg">Enter keywords above and click Generate to start building!</p>
                  </div>
                )}
              </CardContent>
            </Card>
          </TabsContent>

          {/* Generated Memes Tab */}
          <TabsContent value="generated">
            <Card className="bg-black/60 border-pink-500/50 backdrop-blur-sm">
              <CardHeader>
                <CardTitle className="text-pink-300">🔥 AI-Generated NSFW Memes</CardTitle>
                <CardDescription className="text-gray-400">
                  Explicit memes with tone controls. Click "Similar" for variations!
                </CardDescription>
              </CardHeader>
              <CardContent>
                {generatedMemes.length === 0 ? (
                  <div className="text-center py-12 text-gray-500">
                    No generated memes yet. Generate some!
                  </div>
                ) : (
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                    {generatedMemes.map((meme, idx) => (
                      <Card key={idx} className="bg-pink-900/20 border-pink-500/50" data-testid={`generated-meme-${idx}`}>
                        <CardContent className="p-4">
                          <img
                            src={`data:image/png;base64,${meme.image_data}`}
                            alt="Generated meme"
                            className="w-full h-48 object-cover rounded mb-4 border border-pink-500/50"
                          />
                          <div className="space-y-3">
                            <p className="text-white font-bold text-sm leading-relaxed">{meme.text}</p>
                            <div className="border-t border-pink-500/30 pt-2">
                              <p className="text-xs text-pink-300 mb-1 font-semibold">Keywords Used:</p>
                              <div className="flex flex-wrap gap-1 mb-3">
                                {meme.source_words.map((word, widx) => (
                                  <span
                                    key={widx}
                                    className="text-xs bg-pink-700/70 text-pink-200 px-2 py-1 rounded font-bold"
                                  >
                                    {word}
                                  </span>
                                ))}
                              </div>
                              <Button
                                size="sm"
                                onClick={() => handleGenerateSimilar(meme.id)}
                                disabled={generating}
                                className="w-full bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700"
                              >
                                <RefreshCw className="w-3 h-3 mr-2" />
                                Generate 3 Similar
                              </Button>
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