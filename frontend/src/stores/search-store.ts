import { create } from "zustand";

interface SearchResult {
  id: string;
  document_id: string;
  document_title: string;
  content: string;
  highlight: string;
  score: number;
  chunk_index: number;
}

interface SearchState {
  isOpen: boolean;
  query: string;
  results: SearchResult[];
  isSearching: boolean;
  open: () => void;
  close: () => void;
  setQuery: (query: string) => void;
  setResults: (results: SearchResult[]) => void;
  setIsSearching: (isSearching: boolean) => void;
  reset: () => void;
}

export const useSearchStore = create<SearchState>((set) => ({
  isOpen: false,
  query: "",
  results: [],
  isSearching: false,

  open: () => set({ isOpen: true }),
  close: () => set({ isOpen: false, query: "", results: [] }),
  setQuery: (query) => set({ query }),
  setResults: (results) => set({ results }),
  setIsSearching: (isSearching) => set({ isSearching }),
  reset: () => set({ query: "", results: [], isSearching: false }),
}));
