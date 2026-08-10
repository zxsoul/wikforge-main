"use client";

import { useState } from "react";
import { Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export function OidcButton() {
  const [isLoading, setIsLoading] = useState(false);

  const handleOidcLogin = () => {
    setIsLoading(true);
    // Redirect to the OIDC authorize endpoint which will redirect to the IdP
    window.location.href = `${API_BASE_URL}/api/auth/oidc/authorize`;
  };

  return (
    <Button
      type="button"
      variant="outline"
      className="w-full"
      onClick={handleOidcLogin}
      disabled={isLoading}
    >
      {isLoading ? (
        <Loader2 className="mr-2 h-4 w-4 animate-spin" />
      ) : (
        <svg
          className="mr-2 h-4 w-4"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
        >
          <path d="M15 3h4a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2h-4" />
          <polyline points="10 17 15 12 10 7" />
          <line x1="15" y1="12" x2="3" y2="12" />
        </svg>
      )}
      企业账号登录 (SSO)
    </Button>
  );
}
