import { ImageResponse } from "next/og";

// Next.js 14: 这个文件会被识别为站点 favicon,
// 自动注入到 <link rel="icon"> 而不需要手动 import。
// 改了之后下次 next build 就生效, 浏览器可能要 ctrl+shift+R 强刷。

export const size = { width: 32, height: 32 };
export const contentType = "image/png";

export default function Icon() {
  return new ImageResponse(
    (
      <div
        style={{
          width: "100%",
          height: "100%",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          background: "linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%)",
          borderRadius: 7,
          color: "white",
          fontSize: 22,
          fontWeight: 800,
          letterSpacing: -1,
          fontFamily: "system-ui, sans-serif",
        }}
      >
        W
      </div>
    ),
    { ...size }
  );
}
