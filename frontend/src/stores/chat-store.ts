import { create } from "zustand";

export interface Citation {
  document_id: string;
  document_title: string;
  chunk_index: number;
  title_chain?: string;
  content_snippet?: string;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  citations?: Citation[];
  created_at: string;
}

export interface ChatSession {
  id: string;
  last_active_at: string;
  is_expired: boolean;
  created_at: string;
  preview?: string;
}

interface ChatState {
  sessions: ChatSession[];
  currentSessionId: string | null;
  messages: ChatMessage[];
  isStreaming: boolean;
  streamingContent: string;
  error: string | null;
  isLoadingSessions: boolean;
  isLoadingHistory: boolean;

  setSessions: (sessions: ChatSession[]) => void;
  setCurrentSessionId: (id: string | null) => void;
  setMessages: (messages: ChatMessage[]) => void;
  addMessage: (message: ChatMessage) => void;
  setIsStreaming: (isStreaming: boolean) => void;
  setStreamingContent: (content: string) => void;
  appendStreamingContent: (chunk: string) => void;
  setError: (error: string | null) => void;
  setIsLoadingSessions: (loading: boolean) => void;
  setIsLoadingHistory: (loading: boolean) => void;
  reset: () => void;
}

export const useChatStore = create<ChatState>((set) => ({
  sessions: [],
  currentSessionId: null,
  messages: [],
  isStreaming: false,
  streamingContent: "",
  error: null,
  isLoadingSessions: false,
  isLoadingHistory: false,

  setSessions: (sessions) => set({ sessions }),
  setCurrentSessionId: (id) => set({ currentSessionId: id }),
  setMessages: (messages) => set({ messages }),
  addMessage: (message) =>
    set((state) => ({ messages: [...state.messages, message] })),
  setIsStreaming: (isStreaming) => set({ isStreaming }),
  setStreamingContent: (content) => set({ streamingContent: content }),
  appendStreamingContent: (chunk) =>
    set((state) => ({ streamingContent: state.streamingContent + chunk })),
  setError: (error) => set({ error }),
  setIsLoadingSessions: (loading) => set({ isLoadingSessions: loading }),
  setIsLoadingHistory: (loading) => set({ isLoadingHistory: loading }),
  reset: () =>
    set({
      currentSessionId: null,
      messages: [],
      isStreaming: false,
      streamingContent: "",
      error: null,
    }),
}));
