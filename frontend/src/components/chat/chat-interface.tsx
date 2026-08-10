"use client";

import * as React from "react";
import { Send, Loader2, AlertCircle, RotateCcw } from "lucide-react";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { MessageBubble, StreamingBubble } from "./message-bubble";
import { useChatStore, type ChatMessage, type Citation } from "@/stores/chat-store";
import { apiClient } from "@/lib/api-client";

const MAX_INPUT_LENGTH = 2000;
const STREAM_TIMEOUT_MS = 30000;

export function ChatInterface() {
  const {
    currentSessionId,
    messages,
    isStreaming,
    streamingContent,
    error,
    setCurrentSessionId,
    addMessage,
    setIsStreaming,
    setStreamingContent,
    appendStreamingContent,
    setError,
  } = useChatStore();

  const [input, setInput] = React.useState("");
  const messagesEndRef = React.useRef<HTMLDivElement>(null);
  const textareaRef = React.useRef<HTMLTextAreaElement>(null);
  const abortControllerRef = React.useRef<AbortController | null>(null);

  // Auto-scroll to bottom when new messages arrive
  React.useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, streamingContent]);

  // Auto-resize textarea
  React.useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 160)}px`;
    }
  }, [input]);

  const handleSend = async () => {
    const trimmed = input.trim();
    if (!trimmed || isStreaming) return;

    setError(null);
    setInput("");

    // Add user message
    const userMessage: ChatMessage = {
      id: crypto.randomUUID(),
      role: "user",
      content: trimmed,
      created_at: new Date().toISOString(),
    };
    addMessage(userMessage);

    // Start streaming
    setIsStreaming(true);
    setStreamingContent("");

    const abortController = new AbortController();
    abortControllerRef.current = abortController;

    // Set up timeout
    const timeoutId = setTimeout(() => {
      abortController.abort();
      setIsStreaming(false);
      setError("回答超时，请重试");
    }, STREAM_TIMEOUT_MS);

    try {
      let fullContent = "";
      let citations: Citation[] = [];
      let sessionId = currentSessionId;

      const stream = apiClient.stream("/api/rag/chat", {
        method: "POST",
        body: {
          question: trimmed,
          session_id: currentSessionId,
        },
        signal: abortController.signal,
      });

      for await (const chunk of stream) {
        if (abortController.signal.aborted) break;

        try {
          const parsed = JSON.parse(chunk);

          if (parsed.type === "token") {
            fullContent += parsed.content;
            appendStreamingContent(parsed.content);
          } else if (parsed.type === "citations") {
            citations = parsed.citations || [];
          } else if (parsed.type === "session_id") {
            sessionId = parsed.session_id;
            if (!currentSessionId) {
              setCurrentSessionId(sessionId);
            }
          } else if (parsed.type === "error") {
            setError(parsed.message || "生成回答时发生错误");
            break;
          }
        } catch {
          // If not JSON, treat as plain text token
          fullContent += chunk;
          appendStreamingContent(chunk);
        }
      }

      clearTimeout(timeoutId);

      // Add assistant message
      if (fullContent) {
        const assistantMessage: ChatMessage = {
          id: crypto.randomUUID(),
          role: "assistant",
          content: fullContent,
          citations: citations.length > 0 ? citations : undefined,
          created_at: new Date().toISOString(),
        };
        addMessage(assistantMessage);
      }
    } catch (err) {
      clearTimeout(timeoutId);
      if ((err as Error).name !== "AbortError") {
        setError("连接中断，请重试");
      }
    } finally {
      setIsStreaming(false);
      setStreamingContent("");
      abortControllerRef.current = null;
    }
  };

  const handleRetry = () => {
    // Remove the last user message and resend
    const lastUserMsg = [...messages].reverse().find((m) => m.role === "user");
    if (lastUserMsg) {
      setError(null);
      setInput(lastUserMsg.content);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="flex h-full flex-col">
      {/* Messages area */}
      <ScrollArea className="flex-1 px-4">
        <div className="mx-auto max-w-3xl space-y-6 py-6">
          {messages.length === 0 && !isStreaming && (
            <div className="flex flex-col items-center justify-center py-16 text-center">
              <div className="mb-4 rounded-full bg-muted p-4">
                <Send className="h-6 w-6 text-muted-foreground" />
              </div>
              <h3 className="text-lg font-medium">开始对话</h3>
              <p className="mt-1 text-sm text-muted-foreground">
                输入问题，AI 将基于知识库内容为您解答
              </p>
            </div>
          )}

          {messages.map((message) => (
            <MessageBubble key={message.id} message={message} />
          ))}

          {isStreaming && streamingContent && (
            <StreamingBubble content={streamingContent} />
          )}

          {isStreaming && !streamingContent && (
            <div className="flex gap-3">
              <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-muted text-muted-foreground">
                <Loader2 className="h-4 w-4 animate-spin" />
              </div>
              <div className="flex items-center">
                <span className="text-sm text-muted-foreground">
                  正在思考...
                </span>
              </div>
            </div>
          )}

          {/* Error state */}
          {error && (
            <div className="flex items-center gap-3 rounded-lg border border-destructive/50 bg-destructive/10 px-4 py-3">
              <AlertCircle className="h-4 w-4 shrink-0 text-destructive" />
              <span className="flex-1 text-sm text-destructive">{error}</span>
              <Button
                variant="outline"
                size="sm"
                onClick={handleRetry}
                className="shrink-0"
              >
                <RotateCcw className="mr-1.5 h-3.5 w-3.5" />
                重试
              </Button>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>
      </ScrollArea>

      {/* Input area */}
      <div className="border-t px-4 py-3">
        <div className="mx-auto max-w-3xl">
          <div className="flex items-end gap-2">
            <div className="relative flex-1">
              <textarea
                ref={textareaRef}
                value={input}
                onChange={(e) => {
                  if (e.target.value.length <= MAX_INPUT_LENGTH) {
                    setInput(e.target.value);
                  }
                }}
                onKeyDown={handleKeyDown}
                placeholder="输入问题，按 Enter 发送，Shift+Enter 换行..."
                rows={1}
                disabled={isStreaming}
                className="flex w-full resize-none rounded-lg border border-input bg-background px-4 py-3 pr-12 text-sm ring-offset-background placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50"
              />
              <span className="absolute bottom-1.5 right-3 text-xs text-muted-foreground">
                {input.length}/{MAX_INPUT_LENGTH}
              </span>
            </div>
            <Button
              onClick={handleSend}
              disabled={!input.trim() || isStreaming}
              size="icon"
              className="h-10 w-10 shrink-0"
            >
              {isStreaming ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <Send className="h-4 w-4" />
              )}
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}
