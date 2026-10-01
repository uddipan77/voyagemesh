import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The dev server proxies /api to the gateway so the browser makes same-origin calls and CORS
// is a non-issue in local development. In production the frontend is served as static files and
// talks to the gateway via VITE_API_BASE_URL.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    proxy: {
      "/api": {
        target: process.env.VITE_API_BASE_URL || "http://localhost:8000",
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: "dist",
    sourcemap: true,
  },
});
