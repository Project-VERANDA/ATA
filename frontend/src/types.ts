export interface User {
  id: string;
  name: string;
  email: string;
  avatar?: string | null;
}

export interface AuthContextType {
  user: User | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  login: () => Promise<void>;
  logout: () => Promise<void>;
}

export interface ThemeContextType {
  theme: 'light' | 'dark';
  toggleTheme: () => void;
}

export interface Job {
  id: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  createdAt: string;
  completedAt?: string;
  files: number;
  type: 'audio' | 'video' | 'text' | 'batch';
}

export interface SubmitFormData {
  files: File[];
  folderPath: string;
  language: string;
  whisperModel: string;
  enableDiarization: boolean;
  enableAnonymization: boolean;
  anonymizationLevel: string;
  includeTags: string[];
  enableLLMRewrite: boolean;
  llmModel: string;
  enableAdversarial: boolean;
  enableTTS: boolean;
  ttsVoice: string;
  batchSize: number;
}
